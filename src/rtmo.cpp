#include "rtmo.hpp"

#include "rtmo.rs.h"

namespace rtmo {
Rtmo::~Rtmo() {
  CHECK_CUDA_ERROR(cudaFree(dets_));
  CHECK_CUDA_ERROR(cudaFree(keypoints_));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

Rtmo::Rtmo(std::string plan, const int image_width, const int image_height)
    : image_width_(image_width)
    , image_height_(image_height) {
  engine_ = std::make_unique<tensorrt::Engine>(plan);
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
  nvinfer1::Dims dims;
  dims.nbDims = 4;
  dims.d[0] = 1;
  dims.d[1] = 3;
  dims.d[2] = 640;
  dims.d[3] = 640;
  engine_->set_input_shape(engine_->get_io_tensor_name(0), dims);
  input_buffer_.strides[3] = sizeof(uint8_t);
  input_buffer_.strides[2] = 3 * input_buffer_.strides[3];
  input_buffer_.strides[1] = image_width_ * input_buffer_.strides[2];
  input_buffer_.strides[0] = image_height_ * input_buffer_.strides[1];
  CHECK_CUDA_ERROR(cudaMalloc(&input_buffer_.basePtr, input_buffer_.strides[0]));

  nvcv::Tensor::Requirements input_reqs =
    nvcv::Tensor::CalcRequirements(1, {image_width_, image_height_}, nvcv::FMT_BGR8);

  nvcv::TensorDataStridedCuda input_data(
    nvcv::TensorShape{input_reqs.shape, input_reqs.rank, input_reqs.layout},
    nvcv::DataType{input_reqs.dtype}, input_buffer_);

  input_tensor_ = nvcv::TensorWrapData(input_data);

  nvcv::Tensor::Requirements input_layer_reqs =
    nvcv::Tensor::CalcRequirements(1, {640, 640}, nvcv::FMT_RGBf32p);

  int64_t input_layer_size = CalcTotalSizeBytes(nvcv::Requirements{input_layer_reqs.mem}.cudaMem());
  std::copy(input_layer_reqs.strides, input_layer_reqs.strides + NVCV_TENSOR_MAX_RANK,
            input_layer_buffer_.strides);

  CHECK_CUDA_ERROR(cudaMalloc(&input_layer_buffer_.basePtr, input_layer_size));

  nvcv::TensorDataStridedCuda input_layer_data(
    nvcv::TensorShape{input_layer_reqs.shape, input_layer_reqs.rank, input_layer_reqs.layout},
    nvcv::DataType{input_layer_reqs.dtype}, input_layer_buffer_);
  input_layer_tensor_ = TensorWrapData(input_layer_data);
  engine_->set_tensor_address(engine_->get_io_tensor_name(0), input_layer_buffer_.basePtr);

  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&dets_, 1 * 2000 * 5 * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&keypoints_, 1 * 2000 * 17 * 3 * sizeof(float)));
  engine_->set_tensor_address("dets", dets_);
  engine_->set_tensor_address("keypoints", keypoints_);

  rgb_tensor_ = nvcv::Tensor(1, {image_width_, image_height_}, nvcv::FMT_RGB8);
  resized_tensor_ = nvcv::Tensor(1, {640, 640}, nvcv::FMT_RGB8);
  float_tensor_ = nvcv::Tensor(1, {640, 640}, nvcv::FMT_RGBf32);

  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

bool Rtmo::infer(const rust::Vec<uint8_t> image, std::vector<PoseResult>& pose_results) {
  cudaEventRecord(start_);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(input_buffer_.basePtr, image.data(), input_buffer_.strides[0],
                                   cudaMemcpyHostToDevice, stream_));
  cvt_color_op_(stream_, input_tensor_, rgb_tensor_, NVCV_COLOR_BGR2RGB);
  resize_op_(stream_, rgb_tensor_, resized_tensor_, NVCV_INTERP_LINEAR);
  convert_op_(stream_, resized_tensor_, float_tensor_, 1.0f, 0.0f);
  reformat_op_(stream_, float_tensor_, input_layer_tensor_);

  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);
  float operatorms = 0;
  cudaEventElapsedTime(&operatorms, start_, stop_);
  /* std::cout << "Time for Preprocess : " << operatorms << " ms" << std::endl; */
  cudaEventRecord(start_);
  if (!engine_->infer(stream_)) {
    return false;
  }
  cudaStreamSynchronize(stream_);
  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);
  operatorms = 0;
  cudaEventElapsedTime(&operatorms, start_, stop_);
  /* std::cout << "Time for Infer : " << operatorms << " ms" << std::endl; */
  cudaStreamSynchronize(stream_);
  pose_results.resize(2000);
  for (size_t i = 0; i < 2000; ++i) {
    const auto score = dets_[i * 5 + 4];
    if (score < 0.5) {
      break;
    }
    auto tl = Point();
    tl.x = dets_[i * 5] / 640 * image_width_;
    tl.y = dets_[i * 5 + 1] / 640 * image_height_;
    auto br = Point();
    br.x = dets_[i * 5 + 2] / 640 * image_width_;
    br.y = dets_[i * 5 + 3] / 640 * image_height_;
    pose_results.at(i).bbox.tl = tl;
    pose_results.at(i).bbox.br = br;
    pose_results.at(i).bbox.score = score;
    pose_results.at(i).bbox.class_index = 0;
    pose_results.at(i).keypoints.reserve(17);
    for (size_t j = 0; j < 17; ++j) {
      auto keypoint = Keypoint();
      keypoint.x = keypoints_[i * 17 * 3 + j * 3] / 640 * image_width_;
      keypoint.y = keypoints_[i * 17 * 3 + j * 3 + 1] / 640 * image_height_;
      keypoint.score = keypoints_[i * 17 * 3 + j * 3 + 2];
      pose_results.at(i).keypoints.push_back(keypoint);
    }
  }
  return true;
}

std::unique_ptr<Rtmo> make_rtmo(const std::string& plan, const int image_width,
                                const int image_height) {
  return std::make_unique<Rtmo>(plan, image_width, image_height);
}
}  // namespace rtmo
