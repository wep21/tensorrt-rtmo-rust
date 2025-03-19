#[cxx::bridge(namespace = "yolo")]
pub mod ffi {
    #[derive(Debug)]
    struct Point {
        x: f32,
        y: f32,
    }

    #[derive(Debug)]
    struct Bbox {
        tl: Point,
        br: Point,
        score: f32,
        class_index: i32,
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("yolo.hpp");

        type Yolo;
        fn infer(self: Pin<&mut Yolo>, image: Vec<u8>, boxes: Pin<&mut CxxVector<Bbox>>) -> bool;

        pub fn make_yolo(plan: &CxxString, image_width: i32, image_height: i32) -> UniquePtr<Yolo>;
    }
}
