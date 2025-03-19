mod yolo;
use cxx::let_cxx_string;
use cxx::CxxVector;
use crate::yolo::ffi::Bbox;
use std::time::Instant;
use image::GenericImageView;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let_cxx_string!(plan = &args[1]);

    // Load image from args[2]
    let img = image::open(&args[2]).expect("Failed to open image");
    let (width, height) = img.dimensions();
    let image = img.to_rgb8().into_raw();

    let mut binding = yolo::ffi::make_yolo(&plan, width as i32, height as i32);
    let mut detector = binding.pin_mut();
    let mut boxes = CxxVector::<Bbox>::new();

    // Warm up inference
    let _ = detector.as_mut().infer(image.clone(), boxes.pin_mut());

    // Perform inference ten times and measure total time
    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..10 {
        let start = Instant::now();
        let status = detector.as_mut().infer(image.clone(), boxes.pin_mut());
        println!("{}", status);
        println!("{:?}", boxes);
        let duration = start.elapsed();
        total_duration += duration;

        println!("Inference status: {}", status);
        println!("Inference time: {:?}", duration);
    }

    // Visualize bounding boxes on the image
    let mut out_img = img.to_rgb8();
    for bbox in boxes.iter() {
        let x_min = bbox.tl.x as u32;
        let y_min = bbox.tl.y as u32;
        let x_max = bbox.br.x as u32;
        let y_max = bbox.br.y as u32;

        for x in x_min..=x_max {
            out_img.put_pixel(x, y_min, image::Rgb([255, 0, 0]));
            out_img.put_pixel(x, y_max, image::Rgb([255, 0, 0]));
        }
        for y in y_min..=y_max {
            out_img.put_pixel(x_min, y, image::Rgb([255, 0, 0]));
            out_img.put_pixel(x_max, y, image::Rgb([255, 0, 0]));
        }
    }

    // Save the image with bounding boxes
    out_img.save("output.png").expect("Failed to save image");



    // Calculate and print average inference time
    let average_duration = total_duration / 10;
    println!("Average inference time: {:?}", average_duration);
}
