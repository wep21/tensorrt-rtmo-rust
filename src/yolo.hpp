#pragma once
#include <memory>
#include <vector>

#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include "engine.hpp"
#include "rust/cxx.h"

namespace yolo {

struct Bbox;

class Yolo {
private:
  int image_width_;
  int image_height_;

  std::unique_ptr<tensorrt::Engine> engine_;
  cudaStream_t stream_;
  nvcv::TensorDataStridedCuda::Buffer input_buffer_;
  nvcv::TensorDataStridedCuda::Buffer input_layer_buffer_;

  int* num_detections_;
  float* boxes_;
  float* scores_;
  int* classes_;
  float* outputs_;

  nvcv::Tensor input_tensor_;
  nvcv::Tensor rgb_tensor_;
  nvcv::Tensor resized_tensor_;
  nvcv::Tensor float_tensor_;
  nvcv::Tensor input_layer_tensor_;

  cvcuda::CvtColor cvt_color_op_;
  cvcuda::Resize resize_op_;
  cvcuda::ConvertTo convert_op_;
  cvcuda::Reformat reformat_op_;

  cudaEvent_t start_;
  cudaEvent_t stop_;

public:
  Yolo(std::string plan, const int image_width, const int image_height);
  ~Yolo();
  bool infer(const rust::Vec<uint8_t> image, std::vector<Bbox>& boxes);
};

std::unique_ptr<Yolo> make_yolo(const std::string& plan, const int image_width,
                                const int image_height);
}  // namespace yolo
