#include "yolo.hpp"

#include "yolo.rs.h"

namespace yolo {
Yolo::~Yolo() {
  CHECK_CUDA_ERROR(cudaFree(num_detections_));
  CHECK_CUDA_ERROR(cudaFree(boxes_));
  CHECK_CUDA_ERROR(cudaFree(scores_));
  CHECK_CUDA_ERROR(cudaFree(classes_));
  CHECK_CUDA_ERROR(cudaFree(outputs_));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

Yolo::Yolo(std::string plan, const int image_width, const int image_height)
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

  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&num_detections_, sizeof(int)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&boxes_, 1 * 100 * 4 * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&scores_, 1 * 100 * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&classes_, 1 * 100 * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&outputs_, 1 * 8400 * 85 * sizeof(float)));
  engine_->set_tensor_address("num_detections", num_detections_);
  engine_->set_tensor_address("detection_boxes", boxes_);
  engine_->set_tensor_address("detection_scores", scores_);
  engine_->set_tensor_address("detection_classes", classes_);

  rgb_tensor_ = nvcv::Tensor(1, {image_width_, image_height_}, nvcv::FMT_RGB8);
  resized_tensor_ = nvcv::Tensor(1, {640, 640}, nvcv::FMT_RGB8);
  float_tensor_ = nvcv::Tensor(1, {640, 640}, nvcv::FMT_RGBf32);

  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

bool Yolo::infer(const rust::Vec<uint8_t> image, std::vector<Bbox>& boxes) {
  cudaEventRecord(start_);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(input_buffer_.basePtr, image.data(), input_buffer_.strides[0],
                                   cudaMemcpyHostToDevice, stream_));
  cvt_color_op_(stream_, input_tensor_, rgb_tensor_, NVCV_COLOR_BGR2RGB);
  resize_op_(stream_, rgb_tensor_, resized_tensor_, NVCV_INTERP_LINEAR);
  convert_op_(stream_, resized_tensor_, float_tensor_, 1.0f / 255.0f, 0.0f);
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
  boxes.resize(num_detections_[0]);
  for (int i = 0; i < num_detections_[0]; ++i) {
    auto tl = Point();
    tl.x = boxes_[i * 4] / 640 * image_width_;
    tl.y = boxes_[i * 4 + 1] / 640 * image_height_;
    auto br = Point();
    br.x = boxes_[i * 4 + 2] / 640 * image_width_;
    br.y = boxes_[i * 4 + 3] / 640 * image_height_;
    boxes.at(i).tl = tl;
    boxes.at(i).br = br;
    boxes.at(i).score = scores_[i];
    boxes.at(i).class_index = classes_[i];
  }
  return true;
}

std::unique_ptr<Yolo> make_yolo(const std::string& plan, const int image_width,
                                const int image_height) {
  return std::make_unique<Yolo>(plan, image_width, image_height);
}
}  // namespace yolo
