mod rtmo;
use crate::rtmo::ffi::PoseResult;
use cxx::let_cxx_string;
use cxx::CxxVector;
use image::GenericImageView;
use std::time::Instant;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let_cxx_string!(plan = &args[1]);

    // Load image from args[2]
    let img = image::open(&args[2]).expect("Failed to open image");
    let (width, height) = img.dimensions();
    let image = img.to_rgb8().into_raw();

    let mut binding = rtmo::ffi::make_rtmo(&plan, width as i32, height as i32);
    let mut detector = binding.pin_mut();
    let mut pose_results = CxxVector::<PoseResult>::new();

    // Warm up inference
    let _ = detector
        .as_mut()
        .infer(image.clone(), pose_results.pin_mut());

    // Perform inference ten times and measure total time
    let mut total_duration = std::time::Duration::new(0, 0);
    for _ in 0..10 {
        let start = Instant::now();
        let status = detector
            .as_mut()
            .infer(image.clone(), pose_results.pin_mut());
        let duration = start.elapsed();
        total_duration += duration;

        println!("Inference status: {}", status);
        println!("Inference time: {:?}", duration);
    }

    // Visualize pose on the image
    let mut out_img = img.to_rgb8();
    for pose_result in pose_results.iter() {
        let x_min = pose_result.bbox.tl.x as u32;
        let y_min = pose_result.bbox.tl.y as u32;
        let x_max = std::cmp::min(pose_result.bbox.br.x as u32, width - 1);
        let y_max = std::cmp::min(pose_result.bbox.br.y as u32, height - 1);

        for x in x_min..=x_max {
            out_img.put_pixel(x, y_min, image::Rgb([255, 0, 0]));
            out_img.put_pixel(x, y_max, image::Rgb([255, 0, 0]));
        }
        for y in y_min..=y_max {
            out_img.put_pixel(x_min, y, image::Rgb([255, 0, 0]));
            out_img.put_pixel(x_max, y, image::Rgb([255, 0, 0]));
        }

        for keypoint in pose_result.keypoints.iter() {
            let x = keypoint.x as u32;
            let y = keypoint.y as u32;
            if x > width || y > height {
                continue;
            }
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height {
                        out_img.put_pixel(nx as u32, ny as u32, image::Rgb([0, 255, 0]));
                    }
                }
            }
        }
    }

    // Save the image with pose
    out_img.save("output.png").expect("Failed to save image");

    // Calculate and print average inference time
    let average_duration = total_duration / 10;
    println!("Average inference time: {:?}", average_duration);
}
