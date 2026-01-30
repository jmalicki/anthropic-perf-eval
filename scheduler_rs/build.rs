//! Build script for scheduler_rs
//!
//! Automatically detects OR-Tools and protobuf installations from Homebrew
//! and sets the necessary linker flags for the cpsat feature.

fn main() {
    // Require BUILD_FROM_PROJECT_ROOT=1 so we don't install into scheduler_rs/.venv by mistake.
    // When you run maturin from project root, set it: BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let manifest_path = std::path::Path::new(&manifest_dir);
    let is_nested = manifest_path
        .file_name()
        .map_or(false, |n| n == "scheduler_rs")
        && manifest_path
            .parent()
            .map_or(false, |p| p.join("pyproject.toml").exists());
    if is_nested {
        let ok = std::env::var("BUILD_FROM_PROJECT_ROOT")
            .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true") || v == "yes");
        if !ok {
            panic!(
                "Do not run maturin/cargo from inside scheduler_rs/. \
                 Run from the project root with:\n  BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml"
            );
        }
    }

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
