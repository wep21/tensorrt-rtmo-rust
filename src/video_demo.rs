mod rtmo;
use crate::rtmo::ffi::PoseResult;
use cxx::let_cxx_string;
use cxx::CxxVector;
use image::{ImageBuffer, Rgb};
use minifb::{Key, Window, WindowOptions};
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

    let mut binding = rtmo::ffi::make_rtmo(
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
        let image = frame.into_raw_vec_and_offset();
        let rgb_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width, height, image.clone().0)
            .expect("failed to convert frame to image");
        // Run inference
        let status = detector
            .as_mut()
            .infer(image.clone().0, pose_results.pin_mut());
        println!("Inference status: {}", status);

        // Draw inference results on the frame
        let mut annotated_image = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let pixel = rgb_image.get_pixel(x, y);
            *pixel
        });

        for pose_result in pose_results.iter() {
            let x_min = pose_result.bbox.tl.x as u32;
            let y_min = pose_result.bbox.tl.y as u32;
            let x_max = pose_result.bbox.br.x as u32;
            let y_max = pose_result.bbox.br.y as u32;

            // Draw bounding box
            for x in x_min..x_max {
                if y_min < height as u32 {
                    annotated_image.put_pixel(x, y_min, image::Rgb([255, 0, 0]));
                }
                if y_max < height as u32 {
                    annotated_image.put_pixel(x, y_max, image::Rgb([255, 0, 0]));
                }
            }
            for y in y_min..y_max {
                if x_min < width as u32 {
                    annotated_image.put_pixel(x_min, y, image::Rgb([255, 0, 0]));
                }
                if x_max < width as u32 {
                    annotated_image.put_pixel(x_max, y, image::Rgb([255, 0, 0]));
                }
            }

            // Draw keypoints
            for keypoint in pose_result.keypoints.iter() {
                let x = keypoint.x as u32;
                let y = keypoint.y as u32;
                if x < width as u32 && y < height as u32 {
                    annotated_image.put_pixel(x, y, image::Rgb([0, 255, 0]));
                }
            }
        }

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
    }

    Ok(())
}
