use cxx::let_cxx_string;
use cxx::CxxVector;
use image::{ImageBuffer, Rgb};
use minifb::{Key, Window, WindowOptions};
use rtmo::rtmo::ffi::{make_rtmo, PoseResult};
use video_rs::decode::Decoder;
use video_rs::location::Location;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<String>>();
    let_cxx_string!(plan = &args[1]);
    let video_path = std::path::Path::new(&args[2]);

    // Open video file
    video_rs::init().unwrap();
    let source: Location = video_path.into();
    let mut decoder = Decoder::new(&source).expect("failed to create decoder");
    let (width, height) = decoder.size();

    let mut binding = make_rtmo(
        &plan,
        (width as u32).try_into().unwrap(),
        (height as u32).try_into().unwrap(),
    );
    let mut detector = binding.pin_mut();
    let mut pose_results = CxxVector::<PoseResult>::new();

    // Create a window for displaying frames
    let mut window = Window::new(
        "Video Demo",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )?;

    for (_, frame) in decoder
        .decode_iter()
        .take_while(Result::is_ok)
        .map(Result::unwrap)
    {
        let loop_start_time = std::time::Instant::now();

        let (raw_vec, _) = frame.into_raw_vec_and_offset();
        let rgb_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width, height, raw_vec.clone())
            .expect("failed to convert frame to image");
        // Run inference
        let start_time = std::time::Instant::now();
        let status = detector.as_mut().infer(raw_vec, pose_results.pin_mut());
        let inference_time = start_time.elapsed();
        println!("Inference status: {}", status);
        println!("Inference time: {:?}", inference_time);

        // Draw inference results on the frame
        let mut annotated_image = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let pixel = rgb_image.get_pixel(x, y);
            *pixel
        });

        annotated_image = pose_results
            .iter()
            .fold(annotated_image, |mut img, pose_result| {
                let x_min = pose_result.bbox.tl.x as u32;
                let y_min = pose_result.bbox.tl.y as u32;
                let x_max = std::cmp::min(pose_result.bbox.br.x as u32, width - 1);
                let y_max = std::cmp::min(pose_result.bbox.br.y as u32, height - 1);

                // Draw bounding box
                for x in x_min..=x_max {
                    img.put_pixel(x, y_min, Rgb([255, 0, 0]));
                    img.put_pixel(x, y_max, Rgb([255, 0, 0]));
                }
                for y in y_min..=y_max {
                    img.put_pixel(x_min, y, Rgb([255, 0, 0]));
                    img.put_pixel(x_max, y, Rgb([255, 0, 0]));
                }

                // Draw keypoints
                for keypoint in &pose_result.keypoints {
                    let x = keypoint.x as u32;
                    let y = keypoint.y as u32;
                    if x < width && y < height {
                        for dx in -1..=1 {
                            for dy in -1..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height
                                {
                                    img.put_pixel(nx as u32, ny as u32, Rgb([0, 0, 0]));
                                    // Clear previous keypoint
                                }
                            }
                        }
                        for dx in -1..=1 {
                            for dy in -1..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height
                                {
                                    let pixel = img.get_pixel_mut(nx as u32, ny as u32);
                                    *pixel = Rgb([0, 255, 0]); // Draw new keypoint
                                }
                            }
                        }
                    }
                }

                img
            });

        // Display the frame
        let buffer: Vec<u32> = annotated_image
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
            })
            .collect();
        window.update_with_buffer(&buffer, width as usize, height as usize)?;

        // Exit if 'q' key is pressed
        if window.is_key_down(Key::Q) {
            break;
        }

        let loop_time = loop_start_time.elapsed();
        println!("Time for one loop iteration: {:?}", loop_time);
    }

    Ok(())
}
