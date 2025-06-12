use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/rtmo.rs");
    println!("cargo:rerun-if-changed=src/engine.cpp");
    println!("cargo:rerun-if-changed=src/engine.hpp");
    println!("cargo:rerun-if-changed=src/rtmo.cpp");
    println!("cargo:rerun-if-changed=src/rtmo.hpp");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir); // Assuming detector is one level down from workspace root

    // --- CUDA ---
    // These are common paths. Adjust if your CUDA installation is different.
    let cuda_include_path = PathBuf::from("/usr/local/cuda/include");
    let cuda_lib_path = PathBuf::from("/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart"); // Link against CUDA runtime

    // --- TensorRT ---
    // Path relative to workspace root.
    let tensorrt_base_path = workspace_root.join("/usr");
    let tensorrt_include_path = tensorrt_base_path.join("include/x86_64-linux-gnu");
    let tensorrt_lib_path = tensorrt_base_path.join("lib/x86_64-linux-gnu");

    println!("cargo:rustc-link-search=native={}", tensorrt_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin");

    // --- CVCUDA ---
    // Path relative to workspace root.
    let cvcuda_base_path = workspace_root.join("/usr");
    let cvcuda_include_path = cvcuda_base_path.join("include");
    let cvcuda_lib_path = cvcuda_base_path.join("lib");

    println!("cargo:rustc-link-search=native={}", cvcuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cvcuda");
    println!("cargo:rustc-link-lib=dylib=nvcv_types");


    // --- Build C++ files using cxx_build ---
    let mut build = cxx_build::bridge("src/rtmo.rs"); // Path to the Rust file with CXX bridge
    build
        .file("src/engine.cpp")
        .file("src/rtmo.cpp")
        .include(&manifest_dir.join("src")) // for engine.hpp, rtmo.hpp
        .include(&cuda_include_path)
        .include(&tensorrt_include_path)
        .include(&cvcuda_include_path)
        .flag_if_supported("-std=c++17") // Assuming C++17, adjust if necessary
        .flag_if_supported("-pthread"); // For TensorRT plugins

    // Add any other necessary compiler flags or definitions
    // e.g., .define("SOME_MACRO", "value")

    // Linker arguments for specific libraries if needed (beyond simple -l flags)
    // For TensorRT plugin, Bazel had: linkopts = ["-lpthread", "-Wl,--no-as-needed -ldl -lrt -Wl,--as-needed"]
    // The -lpthread is covered by cargo:rustc-link-lib=pthread if cc crate doesn't add it.
    // The cc crate usually handles pthread implicitly if specified with .flag("-pthread") for compiler and linker.
    // Let's ensure pthread is linked.
    println!("cargo:rustc-link-lib=dylib=pthread");
    // For dlsym, rt_timer related symbols which might be needed by TensorRT
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");


    build.compile("rtmo_cpp_bridge");

    println!("cargo:warning=This build.rs attempts to compile C++ dependencies. Ensure CUDA, TensorRT, and CVCUDA are correctly installed and paths are configured if issues arise.");
    println!("cargo:warning=CUDA include path: {}", cuda_include_path.display());
    println!("cargo:warning=CUDA lib path: {}", cuda_lib_path.display());
    println!("cargo:warning=TensorRT include path: {}", tensorrt_include_path.display());
    println!("cargo:warning=TensorRT lib path: {}", tensorrt_lib_path.display());
    println!("cargo:warning=CVCUDA include path: {}", cvcuda_include_path.display());
    println!("cargo:warning=CVCUDA lib path: {}", cvcuda_lib_path.display());
}
