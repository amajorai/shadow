use anyhow::Result;
use async_trait::async_trait;

/// Input event types.
///
/// Key events deliberately carry no key code: the hooks report only that a
/// key went down or up (see `input_event_record`).
#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyDown,
    KeyUp,
    MouseDown { x: i32, y: i32, button: u8 },
    MouseUp { x: i32, y: i32, button: u8 },
    MouseMove { x: i32, y: i32 },
}

/// Input monitor trait — platform-specific implementations.
#[async_trait]
pub trait InputMonitor: Send + Sync {
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
}

// ─── Shared: move throttle + shadow-core ingestion ───────────────────────────

/// Throttle pointer moves: emit only once the pointer is ≥50px from the last
/// emitted position. Each platform's hook has exactly one call site, so a single
/// `static` holds that hook's last position.
#[cfg(any(target_os = "windows", target_os = "macos"))]
fn throttled_mouse_move(x: i32, y: i32) -> Option<InputEvent> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_POS: AtomicU64 = AtomicU64::new(0);
    let packed = ((x as u32 as u64) << 32) | (y as u32 as u64);
    let last = LAST_POS.load(Ordering::Relaxed);
    let lx = (last >> 32) as i32;
    let ly = (last & 0xFFFF_FFFF) as i32;
    if (x - lx).pow(2) + (y - ly).pow(2) < 2500 {
        None
    } else {
        LAST_POS.store(packed, Ordering::Relaxed);
        Some(InputEvent::MouseMove { x, y })
    }
}

/// Ingest a monitor's events into shadow-core until the channel closes or the
/// monitor is stopped.
fn spawn_ingest_task(
    mut event_rx: tokio::sync::mpsc::UnboundedReceiver<InputEvent>,
    mut stop_rx: tokio::sync::oneshot::Receiver<()>,
) {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                event = event_rx.recv() => match event {
                    Some(ev) => ingest_input_event(ev),
                    None => break,
                },
                _ = &mut stop_rx => break,
            }
        }
    });
}

/// Write one event to shadow-core's input track (track 2) unless capture is paused.
fn ingest_input_event(event: InputEvent) {
    if crate::server::is_capture_paused() {
        return;
    }
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    let map = input_event_record(&event, now_us);

    if let Ok(data) = rmp_serde::to_vec(&map) {
        let _ = shadow_core::write_event(data);
    }
}

/// Build the persisted track-2 record for an input event. Every platform
/// monitor writes exactly this shape to the event log.
///
/// Key events record only that a key went down or up, never which key. The
/// global hooks see every keystroke system-wide (password fields and apps
/// outside the capture allowlist included) and the event log is plaintext, so
/// a stored key code would let anyone who can read the data dir replay typed
/// text, credentials included. Nothing downstream reads a key code: the index,
/// timeline, and workflow/behavioural analysis use only `ts`/`track`/`type`.
fn input_event_record(
    event: &InputEvent,
    now_us: u64,
) -> std::collections::HashMap<&'static str, rmpv::Value> {
    let mut map = std::collections::HashMap::new();
    map.insert("ts", rmpv::Value::from(now_us));
    map.insert("v", rmpv::Value::from(2u8));
    map.insert("track", rmpv::Value::from(2u8));

    match *event {
        InputEvent::KeyDown => {
            map.insert("type", rmpv::Value::from("key_down"));
        }
        InputEvent::KeyUp => {
            map.insert("type", rmpv::Value::from("key_up"));
        }
        InputEvent::MouseDown { x, y, button } => {
            map.insert("type", rmpv::Value::from("mouse_down"));
            map.insert("x", rmpv::Value::from(x));
            map.insert("y", rmpv::Value::from(y));
            map.insert("button", rmpv::Value::from(button));
        }
        InputEvent::MouseUp { x, y, button } => {
            map.insert("type", rmpv::Value::from("mouse_up"));
            map.insert("x", rmpv::Value::from(x));
            map.insert("y", rmpv::Value::from(y));
            map.insert("button", rmpv::Value::from(button));
        }
        InputEvent::MouseMove { x, y } => {
            map.insert("type", rmpv::Value::from("mouse_move"));
            map.insert("x", rmpv::Value::from(x));
            map.insert("y", rmpv::Value::from(y));
        }
    }
    map
}

// ─── Windows: SetWindowsHookEx (WH_KEYBOARD_LL + WH_MOUSE_LL) ────────────────

#[cfg(target_os = "windows")]
mod windows_impl {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::*;

    static INPUT_TX: OnceLock<Mutex<tokio::sync::mpsc::UnboundedSender<InputEvent>>> =
        OnceLock::new();

    unsafe extern "system" fn keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let event = if wparam.0 as u32 == WM_KEYDOWN || wparam.0 as u32 == WM_SYSKEYDOWN {
                InputEvent::KeyDown
            } else {
                InputEvent::KeyUp
            };
            if let Some(tx) = INPUT_TX.get() {
                if let Ok(tx) = tx.lock() {
                    let _ = tx.send(event);
                }
            }
        }
        unsafe { CallNextHookEx(None, code, wparam, lparam) }
    }

    unsafe extern "system" fn mouse_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let ms = &*(lparam.0 as *const MSLLHOOKSTRUCT);
            let x = ms.pt.x;
            let y = ms.pt.y;

            let event = match wparam.0 as u32 {
                WM_LBUTTONDOWN => Some(InputEvent::MouseDown { x, y, button: 0 }),
                WM_RBUTTONDOWN => Some(InputEvent::MouseDown { x, y, button: 1 }),
                WM_MBUTTONDOWN => Some(InputEvent::MouseDown { x, y, button: 2 }),
                WM_LBUTTONUP => Some(InputEvent::MouseUp { x, y, button: 0 }),
                WM_RBUTTONUP => Some(InputEvent::MouseUp { x, y, button: 1 }),
                WM_MBUTTONUP => Some(InputEvent::MouseUp { x, y, button: 2 }),
                WM_MOUSEMOVE => super::throttled_mouse_move(x, y),
                _ => None,
            };

            if let Some(ev) = event {
                if let Some(tx) = INPUT_TX.get() {
                    if let Ok(tx) = tx.lock() {
                        let _ = tx.send(ev);
                    }
                }
            }
        }
        unsafe { CallNextHookEx(None, code, wparam, lparam) }
    }

    pub struct WindowsInputMonitor {
        stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
    }

    impl WindowsInputMonitor {
        pub fn new() -> Result<Self> {
            Ok(Self { stop_tx: None })
        }
    }

    #[async_trait]
    impl super::InputMonitor for WindowsInputMonitor {
        async fn start(&mut self) -> Result<()> {
            let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<InputEvent>();
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();

            // Store the sender globally for the hook callbacks
            INPUT_TX.get_or_init(|| Mutex::new(event_tx));

            // Spawn hook thread (must have a message loop)
            std::thread::spawn(|| {
                unsafe {
                    let kb_hook = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), None, 0)
                        .expect("Failed to install keyboard hook");

                    let mouse_hook = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_proc), None, 0)
                        .expect("Failed to install mouse hook");

                    // Message loop — required for low-level hooks
                    let mut msg = MSG::default();
                    loop {
                        let ret = GetMessageW(&mut msg, None, 0, 0);
                        if ret.0 == 0 || ret.0 == -1 {
                            break;
                        }
                        let _ = TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }

                    let _ = UnhookWindowsHookEx(kb_hook);
                    let _ = UnhookWindowsHookEx(mouse_hook);
                }
            });

            super::spawn_ingest_task(event_rx, stop_rx);

            self.stop_tx = Some(stop_tx);
            tracing::info!("Windows input monitor started (WH_KEYBOARD_LL + WH_MOUSE_LL)");
            Ok(())
        }

        async fn stop(&mut self) -> Result<()> {
            if let Some(tx) = self.stop_tx.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "windows")]
pub use windows_impl::WindowsInputMonitor;

// ─── macOS: CGEventTap ────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};

    static INPUT_TX: OnceLock<Mutex<tokio::sync::mpsc::UnboundedSender<super::InputEvent>>> =
        OnceLock::new();

    // CGEventType constants
    const KCG_EVENT_LEFT_MOUSE_DOWN: u32 = 1;
    const KCG_EVENT_LEFT_MOUSE_UP: u32 = 2;
    const KCG_EVENT_RIGHT_MOUSE_DOWN: u32 = 3;
    const KCG_EVENT_RIGHT_MOUSE_UP: u32 = 4;
    const KCG_EVENT_MOUSE_MOVED: u32 = 5;
    const KCG_EVENT_OTHER_MOUSE_DOWN: u32 = 25;
    const KCG_EVENT_OTHER_MOUSE_UP: u32 = 26;
    const KCG_EVENT_KEY_DOWN: u32 = 10;
    const KCG_EVENT_KEY_UP: u32 = 11;

    // CGEventField constants
    const KCG_MOUSE_EVENT_BUTTON_NUMBER: u32 = 71;

    // CGEventTap parameters
    const KCG_SESSION_EVENT_TAP: u32 = 1;
    const KCG_HEAD_INSERT_EVENT_TAP: u32 = 0;
    const KCG_EVENT_TAP_OPTION_LISTEN_ONLY: u32 = 1;

    #[repr(C)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    extern "C" {
        fn CGEventGetLocation(event: *const c_void) -> CGPoint;
        fn CGEventGetIntegerValueField(event: *const c_void, field: u32) -> i64;
        fn CGEventTapCreate(
            tap: u32,
            place: u32,
            options: u32,
            events_of_interest: u64,
            callback: extern "C" fn(
                *const c_void,
                u32,
                *const c_void,
                *const c_void,
            ) -> *const c_void,
            user_info: *const c_void,
        ) -> *const c_void;
        fn CFMachPortCreateRunLoopSource(
            alloc: *const c_void,
            port: *const c_void,
            order: isize,
        ) -> *const c_void;
        fn CFRunLoopGetCurrent() -> *const c_void;
        fn CFRunLoopAddSource(rl: *const c_void, source: *const c_void, mode: *const c_void);
        fn CFRunLoopRun();
        fn CGEventTapEnable(tap: *const c_void, enable: bool);
        static kCFRunLoopCommonModes: *const c_void;
    }

    extern "C" fn tap_callback(
        _proxy: *const c_void,
        event_type: u32,
        event: *const c_void,
        _user_info: *const c_void,
    ) -> *const c_void {
        unsafe {
            let pt = CGEventGetLocation(event);
            let x = pt.x as i32;
            let y = pt.y as i32;

            let ev = match event_type {
                t if t == KCG_EVENT_KEY_DOWN => Some(super::InputEvent::KeyDown),
                t if t == KCG_EVENT_KEY_UP => Some(super::InputEvent::KeyUp),
                t if t == KCG_EVENT_LEFT_MOUSE_DOWN => {
                    Some(super::InputEvent::MouseDown { x, y, button: 0 })
                }
                t if t == KCG_EVENT_LEFT_MOUSE_UP => {
                    Some(super::InputEvent::MouseUp { x, y, button: 0 })
                }
                t if t == KCG_EVENT_RIGHT_MOUSE_DOWN => {
                    Some(super::InputEvent::MouseDown { x, y, button: 1 })
                }
                t if t == KCG_EVENT_RIGHT_MOUSE_UP => {
                    Some(super::InputEvent::MouseUp { x, y, button: 1 })
                }
                t if t == KCG_EVENT_OTHER_MOUSE_DOWN => {
                    let btn =
                        CGEventGetIntegerValueField(event, KCG_MOUSE_EVENT_BUTTON_NUMBER) as u8;
                    Some(super::InputEvent::MouseDown { x, y, button: btn })
                }
                t if t == KCG_EVENT_OTHER_MOUSE_UP => {
                    let btn =
                        CGEventGetIntegerValueField(event, KCG_MOUSE_EVENT_BUTTON_NUMBER) as u8;
                    Some(super::InputEvent::MouseUp { x, y, button: btn })
                }
                t if t == KCG_EVENT_MOUSE_MOVED => super::throttled_mouse_move(x, y),
                _ => None,
            };

            if let Some(e) = ev {
                if let Some(tx) = INPUT_TX.get() {
                    if let Ok(g) = tx.lock() {
                        let _ = g.send(e);
                    }
                }
            }
        }
        event
    }

    pub struct MacOSInputMonitor {
        stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
    }

    impl MacOSInputMonitor {
        pub fn new() -> Result<Self> {
            Ok(Self { stop_tx: None })
        }
    }

    #[async_trait]
    impl super::InputMonitor for MacOSInputMonitor {
        async fn start(&mut self) -> Result<()> {
            let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<super::InputEvent>();
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();

            INPUT_TX.get_or_init(|| Mutex::new(event_tx));

            // Spawn run-loop thread that hosts the event tap
            std::thread::spawn(|| unsafe {
                let mask: u64 = (1u64 << KCG_EVENT_KEY_DOWN)
                    | (1u64 << KCG_EVENT_KEY_UP)
                    | (1u64 << KCG_EVENT_LEFT_MOUSE_DOWN)
                    | (1u64 << KCG_EVENT_LEFT_MOUSE_UP)
                    | (1u64 << KCG_EVENT_RIGHT_MOUSE_DOWN)
                    | (1u64 << KCG_EVENT_RIGHT_MOUSE_UP)
                    | (1u64 << KCG_EVENT_OTHER_MOUSE_DOWN)
                    | (1u64 << KCG_EVENT_OTHER_MOUSE_UP)
                    | (1u64 << KCG_EVENT_MOUSE_MOVED);

                let tap = CGEventTapCreate(
                    KCG_SESSION_EVENT_TAP,
                    KCG_HEAD_INSERT_EVENT_TAP,
                    KCG_EVENT_TAP_OPTION_LISTEN_ONLY,
                    mask,
                    tap_callback,
                    std::ptr::null(),
                );
                if tap.is_null() {
                    tracing::warn!("CGEventTapCreate failed — grant Accessibility permission in System Preferences");
                    return;
                }
                let source = CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
                let rl = CFRunLoopGetCurrent();
                CFRunLoopAddSource(rl, source, kCFRunLoopCommonModes);
                CGEventTapEnable(tap, true);
                tracing::info!("macOS CGEventTap installed");
                CFRunLoopRun();
            });

            super::spawn_ingest_task(event_rx, stop_rx);

            self.stop_tx = Some(stop_tx);
            tracing::info!("macOS input monitor started");
            Ok(())
        }

        async fn stop(&mut self) -> Result<()> {
            if let Some(tx) = self.stop_tx.take() {
                let _ = tx.send(());
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos_impl::MacOSInputMonitor;

// ─── Linux: evdev ─────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
pub struct LinuxInputMonitor {
    stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

#[cfg(target_os = "linux")]
impl LinuxInputMonitor {
    pub fn new() -> Result<Self> {
        Ok(Self { stop_tx: None })
    }
}

#[cfg(target_os = "linux")]
#[async_trait]
impl InputMonitor for LinuxInputMonitor {
    async fn start(&mut self) -> Result<()> {
        use evdev::{Key, RelativeAxisType};

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<InputEvent>();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();

        // One blocking reader thread per keyboard/mouse device. evdev reports raw
        // codes globally; mouse motion is relative, so absolute coordinates are read
        // from the X server when emitting pointer events. Requires read access to
        // /dev/input/event* (root or the `input` group); the X11 pointer query only
        // resolves under X11/XWayland.
        tokio::task::spawn_blocking(move || {
            for (_path, dev) in evdev::enumerate() {
                let is_keyboard = dev
                    .supported_keys()
                    .is_some_and(|keys| keys.contains(Key::KEY_ENTER));
                let is_mouse = dev
                    .supported_relative_axes()
                    .is_some_and(|axes| axes.contains(RelativeAxisType::REL_X));
                if !(is_keyboard || is_mouse) {
                    continue;
                }
                let tx = event_tx.clone();
                std::thread::spawn(move || linux_read_device(dev, tx));
            }
        });

        spawn_ingest_task(event_rx, stop_rx);

        self.stop_tx = Some(stop_tx);
        tracing::info!("Linux input monitor started (evdev)");
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn linux_pointer_pos(x11: &Option<(x11rb::rust_connection::RustConnection, usize)>) -> (i32, i32) {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::ConnectionExt as _;
    if let Some((conn, sn)) = x11 {
        let root = conn.setup().roots[*sn].root;
        if let Some(reply) = conn.query_pointer(root).ok().and_then(|c| c.reply().ok()) {
            return (reply.root_x as i32, reply.root_y as i32);
        }
    }
    (0, 0)
}

#[cfg(target_os = "linux")]
fn linux_read_device(mut dev: evdev::Device, tx: tokio::sync::mpsc::UnboundedSender<InputEvent>) {
    use evdev::{InputEventKind, Key, RelativeAxisType};
    let x11 = x11rb::connect(None).ok();
    // Throttle mouse moves to ~50px like the Windows hook.
    let mut last_pos = (i32::MIN, i32::MIN);
    loop {
        let events = match dev.fetch_events() {
            Ok(e) => e,
            Err(_) => break,
        };
        for ev in events {
            let out = match ev.kind() {
                InputEventKind::Key(key) => match key {
                    Key::BTN_LEFT | Key::BTN_RIGHT | Key::BTN_MIDDLE => {
                        let (x, y) = linux_pointer_pos(&x11);
                        let button = match key {
                            Key::BTN_LEFT => 0,
                            Key::BTN_RIGHT => 1,
                            _ => 2,
                        };
                        match ev.value() {
                            1 => Some(InputEvent::MouseDown { x, y, button }),
                            0 => Some(InputEvent::MouseUp { x, y, button }),
                            _ => None,
                        }
                    }
                    _ => match ev.value() {
                        1 => Some(InputEvent::KeyDown),
                        0 => Some(InputEvent::KeyUp),
                        _ => None, // 2 = autorepeat
                    },
                },
                InputEventKind::RelAxis(axis) => match axis {
                    RelativeAxisType::REL_X | RelativeAxisType::REL_Y => {
                        let (x, y) = linux_pointer_pos(&x11);
                        let dist = (x - last_pos.0).pow(2) + (y - last_pos.1).pow(2);
                        if dist < 2500 {
                            None
                        } else {
                            last_pos = (x, y);
                            Some(InputEvent::MouseMove { x, y })
                        }
                    }
                    _ => None, // wheel: shadow's InputEvent has no scroll variant
                },
                _ => None,
            };
            if let Some(e) = out {
                if tx.send(e).is_err() {
                    return; // receiver dropped
                }
            }
        }
    }
}

#[cfg(target_os = "windows")]
pub type PlatformInputMonitor = WindowsInputMonitor;
#[cfg(target_os = "macos")]
pub type PlatformInputMonitor = MacOSInputMonitor;
#[cfg(target_os = "linux")]
pub type PlatformInputMonitor = LinuxInputMonitor;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    /// Round-trip through MessagePack exactly as the platform ingest paths do,
    /// so the assertions cover the bytes that reach the event log.
    fn persisted(event: InputEvent) -> HashMap<String, rmpv::Value> {
        let bytes = rmp_serde::to_vec(&input_event_record(&event, 1_700_000_000_000_000)).unwrap();
        rmp_serde::from_slice(&bytes).unwrap()
    }

    fn keys(record: &HashMap<String, rmpv::Value>) -> BTreeSet<&str> {
        record.keys().map(String::as_str).collect()
    }

    #[test]
    fn key_events_never_persist_the_key_code() {
        for (event, kind) in [
            (InputEvent::KeyDown, "key_down"),
            (InputEvent::KeyUp, "key_up"),
        ] {
            let record = persisted(event);
            assert!(
                !record.contains_key("key_code"),
                "{kind} record must not store a key code: {record:?}"
            );
            assert_eq!(keys(&record), BTreeSet::from(["track", "ts", "type", "v"]));
            assert_eq!(record["type"].as_str(), Some(kind));
            assert_eq!(record["track"].as_u64(), Some(2));
            assert_eq!(record["v"].as_u64(), Some(2));
            assert_eq!(record["ts"].as_u64(), Some(1_700_000_000_000_000));
        }
    }

    #[test]
    fn mouse_events_keep_their_coordinates_and_button() {
        let down = persisted(InputEvent::MouseDown {
            x: 10,
            y: -20,
            button: 1,
        });
        assert_eq!(
            keys(&down),
            BTreeSet::from(["button", "track", "ts", "type", "v", "x", "y"])
        );
        assert_eq!(down["type"].as_str(), Some("mouse_down"));
        assert_eq!(down["x"].as_i64(), Some(10));
        assert_eq!(down["y"].as_i64(), Some(-20));
        assert_eq!(down["button"].as_u64(), Some(1));

        let up = persisted(InputEvent::MouseUp {
            x: 3,
            y: 4,
            button: 2,
        });
        assert_eq!(up["type"].as_str(), Some("mouse_up"));
        assert_eq!(up["button"].as_u64(), Some(2));

        let moved = persisted(InputEvent::MouseMove { x: 300, y: 400 });
        assert_eq!(
            keys(&moved),
            BTreeSet::from(["track", "ts", "type", "v", "x", "y"])
        );
        assert_eq!(moved["type"].as_str(), Some("mouse_move"));
        assert_eq!(moved["track"].as_u64(), Some(2));
        assert_eq!(moved["x"].as_i64(), Some(300));
        assert_eq!(moved["y"].as_i64(), Some(400));
    }
}
