//! Build script for scheduler_rs
//!
//! Automatically detects OR-Tools and protobuf installations from Homebrew
//! and sets the necessary linker flags for the cpsat feature.

fn main() {
    // Only add protobuf linking when cpsat feature is enabled
    #[cfg(feature = "cpsat")]
    {
        // Try to find protobuf library path
        // Priority: LIBRARY_PATH env var, then common Homebrew locations
        let lib_paths = [
            std::env::var("LIBRARY_PATH").ok(),
            Some("/opt/homebrew/lib".to_string()),  // Apple Silicon Homebrew
            Some("/usr/local/lib".to_string()),     // Intel Mac Homebrew
        ];

        for path in lib_paths.iter().flatten() {
            let protobuf_path = std::path::Path::new(path).join("libprotobuf.dylib");
            if protobuf_path.exists() {
                println!("cargo:rustc-link-search=native={}", path);
                println!("cargo:rustc-link-lib=protobuf");
                return;
            }
        }

        // Fallback: just try to link and let the linker figure it out
        println!("cargo:rustc-link-lib=protobuf");
    }
}
