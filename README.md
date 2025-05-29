# Real-Time Pose Detection and Tracking

This project implements a system for real-time human pose detection and tracking in images and videos. It leverages hardware acceleration for efficient processing.

## Features

*   Real-time pose detection (bounding boxes and keypoints).
*   Multi-object tracking using an adapted ByteTracker algorithm.
*   Hardware acceleration using NVIDIA TensorRT for inference and CVCUDA for vision pre/post-processing.
*   Cross-language implementation with C++ for performance-critical detection and Rust for tracking and application logic.
*   Example applications for processing both static images and video streams.

## Technologies Used

*   **Programming Languages:** C++, Rust
*   **Build System:** Bazel
*   **Core Libraries & Frameworks:**
    *   NVIDIA TensorRT: For high-performance deep learning inference.
    *   NVIDIA CVCUDA: For CUDA-accelerated computer vision tasks.
    *   OpenCV (indirectly via CVCUDA and for image/video handling concepts)
    *   `cxx.rs`: For seamless C++/Rust interoperability.
*   **Key Rust Crates:**
    *   `image`: For image loading, manipulation, and saving.
    *   `video-rs`: For video decoding.
    *   `minifb`: For simple GUI window creation to display video output.
    *   `nalgebra`: For numerical calculations, particularly in the tracking component.

## Prerequisites

Before building and running this project, please ensure you have the following installed:

*   **Bazel:** The build system used for this project. (Follow official Bazel installation instructions for your OS).
*   **NVIDIA TensorRT:**
    *   The detection engine relies on TensorRT for model inference.
    *   The project is configured to look for TensorRT in `/usr` (headers in `/usr/include/...` and libraries in `/usr/lib/...`). If your installation path differs, you may need to adjust the paths in `repositories/repositories.bzl` and potentially `detector/BUILD.bazel`.
*   **NVIDIA CVCUDA:**
    *   Used for CUDA-accelerated computer vision operations.
    *   Similar to TensorRT, it's expected to be in `/usr`. Adjust paths in `repositories/repositories.bzl` if needed.
*   **NVIDIA CUDA Toolkit:** Required by TensorRT and CVCUDA. Ensure your driver and toolkit versions are compatible.
*   **Rust Toolchain:** Install Rust and Cargo using `rustup`.
*   **C++ Compiler:** A modern C++ compiler (e.g., g++) compatible with your CUDA Toolkit and TensorRT versions.
*   **pkg-config:** Often used by build systems to find libraries.

**Note:** Specific versions of CUDA, TensorRT, and CVCUDA might be implicitly required for compatibility. While not explicitly versioned in the build files, it's recommended to use recent versions and ensure they are mutually compatible.

## Building the Project

Once all prerequisites are met, you can build the project using Bazel:

```bash
bazel build //...
```

This command will build all targets, including the detector library, tracker library, and the examples.

**Notes on Configuration:**

*   If your NVIDIA libraries (TensorRT, CVCUDA, CUDA) are not installed in standard system paths (e.g., `/usr/local` instead of `/usr`), you might need to:
    *   Adjust the `path` attribute in the `new_local_repository` rules within `repositories/repositories.bzl`.
    *   Ensure your `LD_LIBRARY_PATH` includes the paths to the shared libraries.
    *   Potentially modify include paths or library paths in the `cc_library` rules within `detector/BUILD.bazel` if the Bazel auto-configuration doesn't pick them up correctly.
*   The project uses `cc_library` rules which often rely on Bazel's C++ toolchain auto-configuration. If you have multiple compilers or specific toolchain requirements, you might need to configure your Bazel C++ toolchain (e.g., via `.bazelrc` or command-line flags).

## Running the Examples

The project includes examples to demonstrate pose detection on images and videos.

**Important:** The examples require a pre-built TensorRT engine plan file for the pose detection model. You will need to obtain or generate this file yourself. The specific model architecture and input/output tensor configurations will depend on the plan file you use. The example code expects the model to output bounding boxes and keypoints in a compatible format.

### Image Demo

The image demo processes a single image, performs pose detection, visualizes the results (bounding boxes and keypoints), and saves the output as `output.png`.

**Command:**

```bash
bazel run //examples:image_demo -- <path_to_tensorrt_plan_file> <path_to_your_image>
```

Replace `<path_to_tensorrt_plan_file>` with the actual path to your TensorRT model plan and `<path_to_your_image>` with the path to the image you want to process.

Example:
```bash
bazel run //examples:image_demo -- /path/to/model.plan /path/to/my_image.jpg
```

### Video Demo

The video demo processes a video file, performs real-time pose detection and tracking, and displays the annotated video in a window.

**Command:**

```bash
bazel run //examples:video_demo -- <path_to_tensorrt_plan_file> <path_to_your_video>
```

Replace `<path_to_tensorrt_plan_file>` with the path to your TensorRT model plan and `<path_to_your_video>` with the path to the video file.

Example:
```bash
bazel run //examples:video_demo -- /path/to/model.plan /path/to/my_video.mp4
```

Press 'q' in the window to quit the video demo.

## Project Structure

A brief overview of the key directories in this project:

*   `detector/`: Contains the core pose detection logic.
    *   `detector/src/engine.cpp, .hpp`: C++ code for the TensorRT inference engine.
    *   `detector/src/rtmo.cpp, .hpp`: C++ implementation details for the RTMO model processing.
    *   `detector/src/rtmo.rs`: Rust bindings for the C++ RTMO detector functions using `cxx.rs`.
    *   `detector/src/lib.rs`: Rust library module for the detector.
*   `tracker/`: Contains the object tracking logic.
    *   `tracker/src/byte_tracker.rs`: Implementation of the ByteTracker algorithm in Rust.
    *   `tracker/src/kalman_filter.rs`, `lapjv.rs`, etc.: Supporting components for the tracker.
*   `examples/`: Contains runnable demo applications.
    *   `examples/src/image_demo.rs`: Demonstrates pose detection on a static image.
    *   `examples/src/video_demo.rs`: Demonstrates pose detection and tracking on a video stream.
*   `repositories/`: Contains Bazel definitions for managing external dependencies like TensorRT and CVCUDA.
*   `BUILD.bazel`: Root Bazel build file.
*   `MODULE.bazel`: Bazel module file.

## License

The license for this project is not yet specified. Please assume it is proprietary unless a LICENSE file is added to the repository.
