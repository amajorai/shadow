pub mod accessibility;
pub mod audio;
pub mod clip;
pub mod detect;
pub mod frames;
pub mod input;
pub mod meeting;
pub mod mixer;
pub mod passive;
pub mod screen;
pub mod segmenter;
pub mod window;

pub use accessibility::{AXTree, AXTreeNode, Bounds};
pub use audio::AudioCapture;
pub use input::InputMonitor;
pub use screen::{DisplayInfo, Frame, ScreenCapture};
pub use window::{AppInfo, WindowInfo, WindowTracker};

// Re-export platform-specific implementations
pub use accessibility::PlatformAXTree;
pub use audio::PlatformAudioCapture;
pub use input::PlatformInputMonitor;
pub use screen::PlatformScreenCapture;
pub use window::PlatformWindowTracker;
