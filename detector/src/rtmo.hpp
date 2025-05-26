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

namespace rtmo {

struct PoseResult;

class Rtmo {
private:
  int image_width_;
  int image_height_;

  std::unique_ptr<tensorrt::Engine> engine_;
  cudaStream_t stream_;
  nvcv::TensorDataStridedCuda::Buffer input_buffer_;
  nvcv::TensorDataStridedCuda::Buffer input_layer_buffer_;

  float* dets_;
  float* keypoints_;

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
  Rtmo(std::string plan, const int image_width, const int image_height);
  ~Rtmo();
  bool infer(const rust::Vec<uint8_t> image, std::vector<PoseResult>& boxes);
};

std::unique_ptr<Rtmo> make_rtmo(const std::string& plan, const int image_width,
                                const int image_height);
}  // namespace rtmo
