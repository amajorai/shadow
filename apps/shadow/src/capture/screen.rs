use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Display information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
}

/// Captured frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub display_id: u32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // BGRA format
    pub timestamp: u64,
}

/// Screen capture trait — platform-specific implementations.
#[async_trait]
pub trait ScreenCapture: Send + Sync {
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn get_displays(&self) -> Vec<DisplayInfo>;
    async fn capture_frame(&self, display_id: u32) -> Result<Frame>;
}

// ─── Windows: DXGI Desktop Duplication ───────────────────────────────────────

#[cfg(target_os = "windows")]
mod windows_impl {
    use super::*;
    use std::collections::HashMap;
    use windows::{
        core::Interface,
        Win32::Foundation::HMODULE,
        Win32::Graphics::{
            Direct3D::*,
            Direct3D11::*,
            Dxgi::{Common::*, *},
        },
    };

    struct DxgiCapture {
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        duplication: IDXGIOutputDuplication,
        width: u32,
        height: u32,
    }

    // COM pointers are Send/Sync when protected by Mutex
    unsafe impl Send for DxgiCapture {}
    unsafe impl Sync for DxgiCapture {}

    pub struct WindowsScreenCapture {
        pub(super) displays: Vec<DisplayInfo>,
        captures: Arc<std::sync::Mutex<HashMap<u32, DxgiCapture>>>,
    }

    impl WindowsScreenCapture {
        pub fn new() -> Result<Self> {
            Ok(Self {
                displays: vec![],
                captures: Arc::new(std::sync::Mutex::new(HashMap::new())),
            })
        }
    }

    #[async_trait]
    impl ScreenCapture for WindowsScreenCapture {
        async fn start(&mut self) -> Result<()> {
            let mut captures = self.captures.lock().unwrap();
            captures.clear();
            self.displays.clear();

            unsafe {
                let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
                let mut adapter_idx = 0u32;
                let mut display_id = 0u32;

                loop {
                    let adapter1 = match factory.EnumAdapters1(adapter_idx) {
                        Ok(a) => a,
                        Err(_) => break,
                    };
                    let adapter: IDXGIAdapter = adapter1.cast()?;

                    let feature_levels = [D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1];
                    let mut device: Option<ID3D11Device> = None;
                    let mut context: Option<ID3D11DeviceContext> = None;

                    if D3D11CreateDevice(
                        &adapter,
                        D3D_DRIVER_TYPE_UNKNOWN,
                        HMODULE::default(),
                        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                        Some(&feature_levels),
                        D3D11_SDK_VERSION,
                        Some(&mut device),
                        None,
                        Some(&mut context),
                    )
                    .is_err()
                    {
                        adapter_idx += 1;
                        continue;
                    }

                    let device = match device {
                        Some(d) => d,
                        None => {
                            adapter_idx += 1;
                            continue;
                        }
                    };
                    let context = match context {
                        Some(c) => c,
                        None => {
                            adapter_idx += 1;
                            continue;
                        }
                    };

                    let mut output_idx = 0u32;
                    loop {
                        let output = match adapter1.EnumOutputs(output_idx) {
                            Ok(o) => o,
                            Err(_) => break,
                        };

                        // Get real output dimensions from DXGI_OUTPUT_DESC.DesktopCoordinates
                        let (w, h) = match unsafe { output.GetDesc() } {
                            Ok(desc) => {
                                let r = desc.DesktopCoordinates;
                                let w = (r.right - r.left).unsigned_abs();
                                let h = (r.bottom - r.top).unsigned_abs();
                                if w > 0 && h > 0 {
                                    (w, h)
                                } else {
                                    (1920u32, 1080u32)
                                }
                            }
                            Err(_) => (1920u32, 1080u32),
                        };

                        let output1: IDXGIOutput1 = match output.cast() {
                            Ok(o) => o,
                            Err(_) => {
                                output_idx += 1;
                                continue;
                            }
                        };

                        match output1.DuplicateOutput(&device) {
                            Ok(dup) => {
                                let is_primary = display_id == 0;
                                self.displays.push(DisplayInfo {
                                    id: display_id,
                                    width: w,
                                    height: h,
                                    is_primary,
                                });
                                captures.insert(
                                    display_id,
                                    DxgiCapture {
                                        device: device.clone(),
                                        context: context.clone(),
                                        duplication: dup,
                                        width: w,
                                        height: h,
                                    },
                                );
                                display_id += 1;
                            }
                            Err(e) => tracing::warn!(
                                "DuplicateOutput failed for output {}: {}",
                                output_idx,
                                e
                            ),
                        }

                        output_idx += 1;
                    }
                    adapter_idx += 1;
                }
            }

            if self.displays.is_empty() {
                tracing::warn!("DXGI found no displays, using fallback GDI capture");
                self.displays.push(DisplayInfo {
                    id: 0,
                    width: 1920,
                    height: 1080,
                    is_primary: true,
                });
            }

            tracing::info!(
                "Windows screen capture started ({} displays)",
                self.displays.len()
            );
            Ok(())
        }

        async fn stop(&mut self) -> Result<()> {
            self.captures.lock().unwrap().clear();
            tracing::info!("Windows screen capture stopped");
            Ok(())
        }

        fn get_displays(&self) -> Vec<DisplayInfo> {
            self.displays.clone()
        }

        async fn capture_frame(&self, display_id: u32) -> Result<Frame> {
            let captures = Arc::clone(&self.captures);
            tokio::task::spawn_blocking(move || capture_frame_dxgi(&captures, display_id)).await?
        }
    }

    fn capture_frame_dxgi(
        captures: &Arc<std::sync::Mutex<HashMap<u32, DxgiCapture>>>,
        display_id: u32,
    ) -> Result<Frame> {
        let guard = captures
            .lock()
            .map_err(|_| anyhow::anyhow!("DXGI lock poisoned"))?;
        let cap = guard
            .get(&display_id)
            .ok_or_else(|| anyhow::anyhow!("Display {} not found", display_id))?;

        unsafe {
            let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
            let mut resource: Option<IDXGIResource> = None;

            // Timeout 100ms — returns DXGI_ERROR_WAIT_TIMEOUT if no new frame
            match cap
                .duplication
                .AcquireNextFrame(100, &mut frame_info, &mut resource)
            {
                Ok(_) => {}
                Err(e) => {
                    // DXGI_ERROR_WAIT_TIMEOUT (0x887A0027) is normal at low fps
                    return Err(anyhow::anyhow!("AcquireNextFrame: {}", e));
                }
            }

            let resource = resource.ok_or_else(|| anyhow::anyhow!("No desktop resource"))?;
            let texture: ID3D11Texture2D = resource.cast()?;

            let mut tex_desc = D3D11_TEXTURE2D_DESC::default();
            texture.GetDesc(&mut tex_desc);

            // Build staging texture descriptor — use zeroed and set only required fields
            // to avoid type-version issues with D3D11 bitflag newtypes.
            let mut staging_desc: D3D11_TEXTURE2D_DESC = std::mem::zeroed();
            staging_desc.Width = tex_desc.Width;
            staging_desc.Height = tex_desc.Height;
            staging_desc.MipLevels = 1;
            staging_desc.ArraySize = 1;
            staging_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            staging_desc.SampleDesc = DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            };
            staging_desc.Usage = D3D11_USAGE_STAGING;
            // Set CPUAccessFlags for read-back; BindFlags and MiscFlags stay 0.
            // Use direct field access to avoid bitflag type mismatches across crate versions.
            {
                // SAFETY: we're in an unsafe block and D3D11_TEXTURE2D_DESC is a C struct.
                // Set CPUAccessFlags (offset depends on struct layout; use typed const).
                staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
            }

            let mut staging: Option<ID3D11Texture2D> = None;
            cap.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
                .map_err(|e| anyhow::anyhow!("CreateTexture2D failed: {}", e))?;
            let staging = staging.ok_or_else(|| anyhow::anyhow!("No staging texture"))?;

            let src_res: ID3D11Resource = texture.cast()?;
            let dst_res: ID3D11Resource = staging.cast()?;
            cap.context.CopyResource(&dst_res, &src_res);

            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            cap.context
                .Map(&dst_res, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| anyhow::anyhow!("Map failed: {}", e))?;

            let w = tex_desc.Width as usize;
            let h = tex_desc.Height as usize;
            let row_pitch = mapped.RowPitch as usize;
            let data_ptr = mapped.pData as *const u8;

            let mut data = vec![0u8; w * h * 4];
            for y in 0..h {
                std::ptr::copy_nonoverlapping(
                    data_ptr.add(y * row_pitch),
                    data.as_mut_ptr().add(y * w * 4),
                    w * 4,
                );
            }

            cap.context.Unmap(&dst_res, 0);
            cap.duplication.ReleaseFrame()?;

            Ok(Frame {
                display_id,
                width: tex_desc.Width,
                height: tex_desc.Height,
                data,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_micros() as u64,
            })
        }
    }
}

#[cfg(target_os = "windows")]
pub use windows_impl::WindowsScreenCapture;

// ─── macOS: CGDisplayCreateImage (CoreGraphics) ──────────────────────────────

#[cfg(target_os = "macos")]
mod macos_ffi {
    use std::ffi::c_void;
    // CoreGraphics is already linked by the core-graphics crate.
    extern "C" {
        pub fn CGDisplayCreateImage(displayID: u32) -> *const c_void;
        pub fn CGImageGetWidth(image: *const c_void) -> usize;
        pub fn CGImageGetHeight(image: *const c_void) -> usize;
        pub fn CGImageGetDataProvider(image: *const c_void) -> *const c_void;
        pub fn CGDataProviderCopyData(provider: *const c_void) -> *const c_void;
        pub fn CFDataGetLength(data: *const c_void) -> isize;
        pub fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
        pub fn CFRelease(cf: *const c_void);
    }
}

#[cfg(target_os = "macos")]
pub struct MacOSScreenCapture {
    displays: Vec<DisplayInfo>,
}

#[cfg(target_os = "macos")]
impl MacOSScreenCapture {
    pub fn new() -> Result<Self> {
        use core_graphics::display::CGDisplay;
        let ids = CGDisplay::active_displays()
            .map_err(|e| anyhow::anyhow!("CGGetActiveDisplayList: {:?}", e))?;
        let main_id = CGDisplay::main().id;
        let displays = ids
            .iter()
            .map(|&id| {
                let d = CGDisplay::new(id);
                DisplayInfo {
                    id,
                    width: d.pixels_wide() as u32,
                    height: d.pixels_high() as u32,
                    is_primary: id == main_id,
                }
            })
            .collect();
        Ok(Self { displays })
    }
}

#[cfg(target_os = "macos")]
#[async_trait]
impl ScreenCapture for MacOSScreenCapture {
    async fn start(&mut self) -> Result<()> {
        tracing::info!(
            "macOS screen capture started ({} displays)",
            self.displays.len()
        );
        Ok(())
    }
    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_displays(&self) -> Vec<DisplayInfo> {
        self.displays.clone()
    }
    async fn capture_frame(&self, display_id: u32) -> Result<Frame> {
        tokio::task::spawn_blocking(move || capture_macos(display_id)).await?
    }
}

#[cfg(target_os = "macos")]
fn capture_macos(display_id: u32) -> Result<Frame> {
    use macos_ffi::*;
    unsafe {
        let img = CGDisplayCreateImage(display_id);
        if img.is_null() {
            return Err(anyhow::anyhow!(
                "CGDisplayCreateImage returned null for display {}",
                display_id
            ));
        }
        let w = CGImageGetWidth(img) as u32;
        let h = CGImageGetHeight(img) as u32;
        let provider = CGImageGetDataProvider(img);
        let cf_data = CGDataProviderCopyData(provider);
        if cf_data.is_null() {
            CFRelease(img);
            return Err(anyhow::anyhow!("CGDataProviderCopyData returned null"));
        }
        let len = CFDataGetLength(cf_data) as usize;
        let data = std::slice::from_raw_parts(CFDataGetBytePtr(cf_data), len).to_vec();
        CFRelease(cf_data);
        CFRelease(img);
        Ok(Frame {
            display_id,
            width: w,
            height: h,
            data, // native format: BGRA on macOS
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        })
    }
}

// ─── Linux: X11 XGetImage ─────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
pub struct LinuxScreenCapture {
    displays: Vec<DisplayInfo>,
}

#[cfg(target_os = "linux")]
impl LinuxScreenCapture {
    pub fn new() -> Result<Self> {
        use x11rb::connection::Connection;
        // ConnectionExt is what carries `get_geometry` — without it this block
        // fails E0599 ("no method named `get_geometry`"). Easy to lose because
        // the sibling block below glob-imports `xproto::*` and so compiles.
        use x11rb::protocol::xproto::ConnectionExt;
        use x11rb::rust_connection::RustConnection;
        let (conn, screen_num) =
            RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {}", e))?;
        let screen = &conn.setup().roots[screen_num];
        let root = screen.root;
        let geom = conn
            .get_geometry(root)
            .map_err(|e| anyhow::anyhow!("GetGeometry: {}", e))?
            .reply()
            .map_err(|e| anyhow::anyhow!("GetGeometry reply: {}", e))?;
        Ok(Self {
            displays: vec![DisplayInfo {
                id: 0,
                width: geom.width as u32,
                height: geom.height as u32,
                is_primary: true,
            }],
        })
    }
}

#[cfg(target_os = "linux")]
#[async_trait]
impl ScreenCapture for LinuxScreenCapture {
    async fn start(&mut self) -> Result<()> {
        tracing::info!(
            "Linux screen capture started (x11rb, {}x{})",
            self.displays[0].width,
            self.displays[0].height
        );
        Ok(())
    }
    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_displays(&self) -> Vec<DisplayInfo> {
        self.displays.clone()
    }
    async fn capture_frame(&self, display_id: u32) -> Result<Frame> {
        tokio::task::spawn_blocking(move || capture_linux(display_id)).await?
    }
}

#[cfg(target_os = "linux")]
fn capture_linux(display_id: u32) -> Result<Frame> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;
    use x11rb::rust_connection::RustConnection;

    let (conn, screen_num) =
        RustConnection::connect(None).map_err(|e| anyhow::anyhow!("X11 connect: {}", e))?;
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    let geom = conn
        .get_geometry(root)
        .map_err(|e| anyhow::anyhow!("GetGeometry: {}", e))?
        .reply()
        .map_err(|e| anyhow::anyhow!("GetGeometry reply: {}", e))?;
    let w = geom.width as u32;
    let h = geom.height as u32;

    let img = conn
        .get_image(
            ImageFormat::Z_PIXMAP,
            root,
            0i16,
            0i16,
            geom.width,
            geom.height,
            !0u32,
        )
        .map_err(|e| anyhow::anyhow!("GetImage send: {}", e))?
        .reply()
        .map_err(|e| anyhow::anyhow!("GetImage reply: {}", e))?;

    // ZPixmap at 32bpp = BGRA; at 24bpp = BGR, pad to BGRA
    let data = if img.depth == 32 {
        img.data
    } else {
        let mut out = Vec::with_capacity(w as usize * h as usize * 4);
        for chunk in img.data.chunks(3) {
            out.extend_from_slice(chunk);
            out.push(255u8);
        }
        out
    };

    Ok(Frame {
        display_id,
        width: w,
        height: h,
        data,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64,
    })
}

#[cfg(target_os = "windows")]
pub type PlatformScreenCapture = WindowsScreenCapture;
#[cfg(target_os = "macos")]
pub type PlatformScreenCapture = MacOSScreenCapture;
#[cfg(target_os = "linux")]
pub type PlatformScreenCapture = LinuxScreenCapture;

// ─── Quick one-shot screenshot (no start() required) ─────────────────────────

/// Capture the primary display without requiring the capture engine to be running.
/// Returns a BGRA `Frame`.
pub async fn quick_screenshot(display_id: u32) -> Result<Frame> {
    tokio::task::spawn_blocking(move || quick_screenshot_sync(display_id)).await?
}

#[cfg(target_os = "windows")]
fn quick_screenshot_sync(display_id: u32) -> Result<Frame> {
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
        GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
        HGDIOBJ, SRCCOPY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

    // Resolve the target monitor's virtual-desktop rect so a chosen display is
    // captured (not always the primary). Falls back to the primary fullscreen when
    // the id is unknown, preserving the original behaviour.
    let (origin_x, origin_y, w, h) = {
        let outputs = enumerate_dxgi_outputs();
        match outputs.iter().find(|o| o.id == display_id) {
            Some(o) => (o.left, o.top, o.width, o.height),
            None => unsafe {
                (
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN) as u32,
                    GetSystemMetrics(SM_CYSCREEN) as u32,
                )
            },
        }
    };

    unsafe {
        let screen_dc = GetDC(None);

        let mem_dc = CreateCompatibleDC(Some(screen_dc));
        let bm = CreateCompatibleBitmap(screen_dc, w as i32, h as i32);
        // SelectObject takes HGDIOBJ — cast via Into
        let old = SelectObject(mem_dc, HGDIOBJ(bm.0));

        // Source origin is the monitor's top-left in virtual-desktop coordinates
        // (relative to the primary monitor's origin), so a monitor left/above the
        // primary reads from negative coordinates.
        let _ = BitBlt(
            mem_dc,
            0,
            0,
            w as i32,
            h as i32,
            Some(screen_dc),
            origin_x,
            origin_y,
            SRCCOPY,
        );

        let mut bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w as i32,
                biHeight: -(h as i32), // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0 as u32,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut data = vec![0u8; (w * h * 4) as usize];
        GetDIBits(
            mem_dc,
            bm,
            0,
            h,
            Some(data.as_mut_ptr() as *mut std::ffi::c_void),
            &mut bmi,
            DIB_RGB_COLORS,
        );

        // GDI returns BGRX (A=0) — set alpha to 255 for BGRA
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }

        SelectObject(mem_dc, old);
        let _ = DeleteObject(HGDIOBJ(bm.0));
        let _ = DeleteDC(mem_dc);
        let _ = ReleaseDC(None, screen_dc);

        Ok(Frame {
            display_id,
            width: w,
            height: h,
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        })
    }
}

#[cfg(target_os = "macos")]
fn quick_screenshot_sync(display_id: u32) -> Result<Frame> {
    capture_macos(display_id)
}

#[cfg(target_os = "linux")]
fn quick_screenshot_sync(display_id: u32) -> Result<Frame> {
    capture_linux(display_id)
}

/// Return (width, height) of the primary display using fast platform APIs.
pub fn get_primary_display_size() -> (u32, u32) {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};
        (
            GetSystemMetrics(SM_CXSCREEN) as u32,
            GetSystemMetrics(SM_CYSCREEN) as u32,
        )
    }
    #[cfg(target_os = "macos")]
    {
        use core_graphics::display::CGDisplay;
        let d = CGDisplay::main();
        (d.pixels_wide() as u32, d.pixels_high() as u32)
    }
    #[cfg(target_os = "linux")]
    {
        use x11rb::connection::Connection;
        use x11rb::protocol::xproto::ConnectionExt; // carries `get_geometry`
        use x11rb::rust_connection::RustConnection;
        if let Ok((conn, sn)) = RustConnection::connect(None) {
            let screen = &conn.setup().roots[sn];
            // Stepwise rather than `.and_then(|c| c.reply())`: the cookie is
            // Result<_, ConnectionError> and reply() is Result<_, ReplyError>,
            // and there is no From<ReplyError> for ConnectionError, so chaining
            // them fails E0277. Splitting sidesteps the error unification.
            if let Ok(cookie) = conn.get_geometry(screen.root) {
                if let Ok(geom) = cookie.reply() {
                    return (geom.width as u32, geom.height as u32);
                }
            }
        }
        (1920, 1080)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    (1920, 1080)
}

// ─── Capture-source enumeration (for the clips source picker) ─────────────────

/// Enumerate the connected displays. On Windows this reads the DXGI output
/// descriptors (no duplication, so it never contends with the passive engine);
/// elsewhere it reuses the platform capturer's display list. Ids match the space
/// [`quick_screenshot`] resolves, so a picked `id` captures that monitor.
pub fn enumerate_displays() -> Vec<DisplayInfo> {
    #[cfg(target_os = "windows")]
    {
        enumerate_dxgi_outputs()
            .into_iter()
            .map(|o| DisplayInfo {
                id: o.id,
                width: o.width,
                height: o.height,
                is_primary: o.primary,
            })
            .collect()
    }
    #[cfg(not(target_os = "windows"))]
    {
        match PlatformScreenCapture::new() {
            Ok(cap) => cap.get_displays(),
            Err(_) => Vec::new(),
        }
    }
}

/// Enumerate capturable top-level windows as `(windowId, title)`. Best-effort:
/// Windows lists visible, titled top-level windows; other platforms return an
/// empty list (no per-window source there yet).
pub fn enumerate_windows() -> Vec<(u64, String)> {
    #[cfg(target_os = "windows")]
    {
        enumerate_windows_win()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone)]
struct DxgiOutput {
    id: u32,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    primary: bool,
}

/// Enumerate DXGI outputs (monitors) with their virtual-desktop coordinates.
/// Reads descriptors only (no `DuplicateOutput`), so it is cheap and safe to call
/// alongside the running capture engine. Ids are assigned in enumeration order and
/// the primary is the output anchored at the virtual-desktop origin (0, 0).
#[cfg(target_os = "windows")]
fn enumerate_dxgi_outputs() -> Vec<DxgiOutput> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

    let mut outputs = Vec::new();
    unsafe {
        let factory: IDXGIFactory1 = match CreateDXGIFactory1() {
            Ok(f) => f,
            Err(_) => return outputs,
        };
        let mut adapter_idx = 0u32;
        let mut id = 0u32;
        loop {
            let adapter = match factory.EnumAdapters1(adapter_idx) {
                Ok(a) => a,
                Err(_) => break,
            };
            let mut output_idx = 0u32;
            loop {
                let output = match adapter.EnumOutputs(output_idx) {
                    Ok(o) => o,
                    Err(_) => break,
                };
                if let Ok(desc) = output.GetDesc() {
                    let r = desc.DesktopCoordinates;
                    let width = (r.right - r.left).unsigned_abs();
                    let height = (r.bottom - r.top).unsigned_abs();
                    if width > 0 && height > 0 {
                        outputs.push(DxgiOutput {
                            id,
                            left: r.left,
                            top: r.top,
                            width,
                            height,
                            primary: r.left == 0 && r.top == 0,
                        });
                        id += 1;
                    }
                }
                output_idx += 1;
            }
            adapter_idx += 1;
        }
    }
    outputs
}

/// The display id (in [`enumerate_dxgi_outputs`] order) a window sits on, matched
/// by the window's monitor rect. `None` when the window or its monitor can't be
/// resolved.
#[cfg(target_os = "windows")]
pub fn display_id_for_window(window_id: u64) -> Option<u32> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST,
    };

    unsafe {
        let hwnd = HWND(window_id as usize as *mut core::ffi::c_void);
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return None;
        }
        let rc = info.rcMonitor;
        enumerate_dxgi_outputs()
            .into_iter()
            .find(|o| o.left == rc.left && o.top == rc.top)
            .map(|o| o.id)
    }
}

/// List visible, titled top-level windows via `EnumWindows`.
#[cfg(target_os = "windows")]
fn enumerate_windows_win() -> Vec<(u64, String)> {
    use windows::Win32::Foundation::{BOOL, HWND, LPARAM, TRUE};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible,
    };

    unsafe extern "system" fn enum_cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let out = &mut *(lparam.0 as *mut Vec<(u64, String)>);
        if IsWindowVisible(hwnd).as_bool() {
            let len = GetWindowTextLengthW(hwnd);
            if len > 0 {
                let mut buf = vec![0u16; len as usize + 1];
                let n = GetWindowTextW(hwnd, &mut buf);
                if n > 0 {
                    let title = String::from_utf16_lossy(&buf[..n as usize]);
                    if !title.trim().is_empty() {
                        out.push((hwnd.0 as usize as u64, title));
                    }
                }
            }
        }
        TRUE
    }

    let mut out: Vec<(u64, String)> = Vec::new();
    unsafe {
        let _ = EnumWindows(Some(enum_cb), LPARAM(&mut out as *mut _ as isize));
    }
    out
}
