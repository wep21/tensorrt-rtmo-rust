#[cxx::bridge(namespace = "rtmo")]
pub mod ffi {
    #[derive(Debug, Clone)]
    struct Point {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone)]
    struct Bbox {
        tl: Point,
        br: Point,
        score: f32,
        class_index: i32,
    }

    #[derive(Debug, Clone)]
    struct Keypoint {
        x: f32,
        y: f32,
        score: f32,
    }

    #[derive(Debug, Clone)]
    struct PoseResult {
        keypoints: Vec<Keypoint>,
        bbox: Bbox,
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("rtmo.hpp");

        type Rtmo;
        fn infer(
            self: Pin<&mut Rtmo>,
            image: Vec<u8>,
            boxes: Pin<&mut CxxVector<PoseResult>>,
        ) -> bool;

        pub fn make_rtmo(plan: &CxxString, image_width: i32, image_height: i32) -> UniquePtr<Rtmo>;
    }
}
