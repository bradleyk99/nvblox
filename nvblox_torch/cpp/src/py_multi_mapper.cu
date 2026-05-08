/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include "nvblox_torch/py_multi_mapper.h"
#include "nvblox_torch/py_mapper.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "nvblox/utils/timing.h"
#include "nvblox_torch/check_utils.h"
#include "nvblox_torch/cuda_stream.h"

#include "nvblox_torch/sdf_query.cuh"

namespace pynvblox {

nvblox::MappingType MultiMapper::mappingTypeFromString(const std::string& s) {
  if (s == "static_tsdf") return nvblox::MappingType::kStaticTsdf;
  if (s == "static_occupancy") return nvblox::MappingType::kStaticOccupancy;
  if (s == "dynamic") return nvblox::MappingType::kDynamic;
  if (s == "human_with_static_tsdf")
    return nvblox::MappingType::kHumanWithStaticTsdf;
  if (s == "human_with_static_occupancy")
    return nvblox::MappingType::kHumanWithStaticOccupancy;
  LOG(FATAL) << "Invalid mapping_type: '" << s
             << "'. Expected one of: static_tsdf, static_occupancy, dynamic, "
                "human_with_static_tsdf, human_with_static_occupancy.";
  return nvblox::MappingType::kStaticTsdf;  // unreachable
}

nvblox::EsdfMode MultiMapper::esdfModeFromString(const std::string& s) {
  if (s == "3D") return nvblox::EsdfMode::k3D;
  if (s == "2D") return nvblox::EsdfMode::k2D;
  LOG(FATAL) << "Invalid esdf_mode: '" << s << "'. Expected '3D' or '2D'.";
  return nvblox::EsdfMode::kUnset;  // unreachable
}

MultiMapper::MultiMapper(
    double voxel_size_m, std::string mapping_type, std::string esdf_mode,
    c10::intrusive_ptr<MapperParams> background_mapper_params,
    c10::intrusive_ptr<MapperParams> foreground_mapper_params,
    c10::intrusive_ptr<MultiMapperParams> multi_mapper_params) {
  voxel_size_m_ = voxel_size_m;
  mapping_type_str_ = mapping_type;
  esdf_mode_str_ = esdf_mode;
  background_mapper_params_ = background_mapper_params;
  foreground_mapper_params_ = foreground_mapper_params;
  multi_mapper_params_ = multi_mapper_params;

  const nvblox::MappingType mt = mappingTypeFromString(mapping_type);
  const nvblox::EsdfMode em = esdfModeFromString(esdf_mode);

  multi_mapper_ = std::make_shared<nvblox::MultiMapper>(
      static_cast<float>(voxel_size_m), mt, em, nvblox::MemoryType::kDevice);

  // Apply per-mapper params to BOTH background and foreground mappers.
  // Mirrors isaac_ros_nvblox::initializeMultiMapper().
  multi_mapper_->setMapperParams(*background_mapper_params->params_,
                                 *foreground_mapper_params->params_);

  // Apply MultiMapper-level params (connected components, ground plane,
  // RANSAC). The freespace integrator params live on the per-Mapper
  // MapperParams and are applied by the call above.
  multi_mapper_->setMultiMapperParams(*multi_mapper_params->params_);
}

void MultiMapper::setForegroundMapperParams(
    c10::intrusive_ptr<MapperParams> params) {
  multi_mapper_->foreground_mapper()->setMapperParams(*params->params_);
  foreground_mapper_params_ = params;
}

// Legacy ctor — explicit body matching pre-patch wrapper behaviour.
// Does NOT call setMapperParams's two-arg form (foreground stays at its
// constructor-time defaults) and does NOT call setMultiMapperParams.
MultiMapper::MultiMapper(
    double voxel_size_m, std::string mapping_type, std::string esdf_mode,
    c10::intrusive_ptr<MapperParams> background_mapper_params) {
  voxel_size_m_ = voxel_size_m;
  mapping_type_str_ = mapping_type;
  esdf_mode_str_ = esdf_mode;
  background_mapper_params_ = background_mapper_params;
  foreground_mapper_params_ = c10::make_intrusive<MapperParams>();
  multi_mapper_params_ = c10::make_intrusive<MultiMapperParams>();

  const nvblox::MappingType mt = mappingTypeFromString(mapping_type);
  const nvblox::EsdfMode em = esdfModeFromString(esdf_mode);

  multi_mapper_ = std::make_shared<nvblox::MultiMapper>(
      static_cast<float>(voxel_size_m), mt, em, nvblox::MemoryType::kDevice);

  multi_mapper_->setMapperParams(*background_mapper_params->params_);
}

void MultiMapper::updateEsdf() {
  multi_mapper_->updateEsdf();
}

void MultiMapper::updateFreespace(int64_t update_time_ms) {
  multi_mapper_->background_mapper()->updateFreespace(
      static_cast<nvblox::Time>(update_time_ms),
      nvblox::UpdateFullLayer::kNo);
}

void MultiMapper::updateColorMesh() {
  multi_mapper_->updateColorMesh();
}

void MultiMapper::decayDynamicOccupancy() {
  // The foreground mapper is the dynamic occupancy mapper in dynamic mode.
  // decayOccupancyAllVoxels is templated only for the variant excluding
  // the last view; the all-voxels version is non-templated.
  multi_mapper_->foreground_mapper()->decayOccupancyAllVoxels();
}

void MultiMapper::decayStaticTsdf() {
  // Background mapper holds the static TSDF in dynamic mode.
  multi_mapper_->background_mapper()->decayTsdfAllVoxels();
}

// Layer accessors — return wrappers around the background mapper's layers.
c10::intrusive_ptr<PyTsdfLayer> MultiMapper::tsdf_layer() {
  auto bg = multi_mapper_->background_mapper();
  return c10::make_intrusive<PyTsdfLayer>(
      bg->layers().getSharedPtr<nvblox::TsdfLayer>());
}

c10::intrusive_ptr<PyColorLayer> MultiMapper::color_layer() {
  auto bg = multi_mapper_->background_mapper();
  return c10::make_intrusive<PyColorLayer>(
      bg->layers().getSharedPtr<nvblox::ColorLayer>());
}

// ============================================================================
// integrateDepth
// ============================================================================

template <typename SensorType>
static void multiIntegrateDepthWithSensorType(
    std::shared_ptr<nvblox::MultiMapper> mm, torch::Tensor depth_frame_t,
    torch::Tensor T_L_C_t, const SensorType& sensor, int64_t update_time_ms) {
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  const int rows = depth_frame_t.sizes()[0];
  const int cols = depth_frame_t.sizes()[1];

nvblox::DepthImage depth_image(rows, cols, nvblox::MemoryType::kDevice);
auto torch_stream = c10::cuda::getCurrentCUDAStream();

const size_t src_pitch = cols * sizeof(float);  // tightly-packed torch tensor
const size_t dst_pitch = depth_image.stride_bytes();
const size_t row_bytes = cols * sizeof(float);

cudaMemcpy2DAsync(depth_image.dataPtr(), dst_pitch,
                  depth_frame_t.data_ptr<float>(), src_pitch,
                  row_bytes, rows,
                  cudaMemcpyDeviceToDevice, torch_stream);
cudaDeviceSynchronize();

  std::optional<nvblox::Time> time = static_cast<nvblox::Time>(update_time_ms);
  mm->integrateDepth(depth_image, T_L_C, sensor, time);
  cudaDeviceSynchronize();
}

void MultiMapper::integrateDepth(torch::Tensor depth_frame_t,
                                 torch::Tensor T_L_C_t,
                                 c10::intrusive_ptr<PySensor> sensor,
                                 int64_t update_time_ms) {
  ALL_ON_GPU_OR_RETURN(depth_frame_t);

  if (!checkSizes(T_L_C_t, {4, 4})) {
    LOG(WARNING) << "Pose tensor size is not correct";
    return;
  }

  if (sensor->isSensorType<nvblox::Camera>()) {
    multiIntegrateDepthWithSensorType(multi_mapper_, depth_frame_t, T_L_C_t,
                                      sensor->getNvbloxSensor<nvblox::Camera>(),
                                      update_time_ms);
  } else if (sensor->isSensorType<nvblox::Lidar>()) {
    multiIntegrateDepthWithSensorType(multi_mapper_, depth_frame_t, T_L_C_t,
                                      sensor->getNvbloxSensor<nvblox::Lidar>(),
                                      update_time_ms);
  } else {
    LOG(ERROR) << "Unknown sensor type in MultiMapper::integrateDepth.";
  }
}

// ============================================================================
// integrateColor
// ============================================================================

template <typename SensorType>
static void multiIntegrateColorWithSensorType(
    std::shared_ptr<nvblox::MultiMapper> mm, torch::Tensor color_frame_t,
    torch::Tensor T_L_C_t, const SensorType& sensor) {
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);

  const int rows = color_frame_t.sizes()[0];
  const int cols = color_frame_t.sizes()[1];

nvblox::ColorImage color_image(rows, cols, nvblox::MemoryType::kDevice);
auto torch_stream = c10::cuda::getCurrentCUDAStream();

const size_t src_pitch = cols * sizeof(nvblox::Color);
const size_t dst_pitch = color_image.stride_bytes();
const size_t row_bytes = cols * sizeof(nvblox::Color);

cudaMemcpy2DAsync(color_image.dataPtr(),
                  dst_pitch,
                  reinterpret_cast<nvblox::Color*>(color_frame_t.data_ptr<uint8_t>()),
                  src_pitch,
                  row_bytes, rows,
                  cudaMemcpyDeviceToDevice, torch_stream);
cudaDeviceSynchronize();

  mm->integrateColor(color_image, T_L_C, sensor);
cudaDeviceSynchronize();
}

void MultiMapper::integrateColor(torch::Tensor color_frame_t,
                                 torch::Tensor T_L_C_t,
                                 c10::intrusive_ptr<PySensor> sensor) {
  ALL_ON_GPU_OR_RETURN(color_frame_t);

  if (!checkSizes(T_L_C_t, {4, 4})) {
    LOG(WARNING) << "Pose tensor size is not correct";
    return;
  }

  const int num_channels = color_frame_t.sizes()[2];
  CHECK_EQ(num_channels, nvblox::kRgbNumElements);

  if (sensor->isSensorType<nvblox::Camera>()) {
    multiIntegrateColorWithSensorType(multi_mapper_, color_frame_t, T_L_C_t,
                                      sensor->getNvbloxSensor<nvblox::Camera>());
  } else {
    LOG(ERROR) << "Color integration only supported for Camera sensors";
  }
}

// ============================================================================
// queryStaticEsdf — query the background mapper's ESDF
// ============================================================================

torch::Tensor MultiMapper::queryStaticEsdf(torch::Tensor output_tensor,
                                           const torch::Tensor query_sphere) {
  const int64_t num_queries = query_sphere.sizes()[0];

  if (!checkAllOnGPU(output_tensor, query_sphere)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }
  if (!checkSizes(query_sphere, {static_cast<int>(num_queries), 4})) {
    LOG(ERROR) << "query_sphere must be Nx4";
    return torch::empty({0});
  }
  if (!checkSizes(output_tensor, {static_cast<int>(num_queries), 4}) &&
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 1})) {
    LOG(ERROR) << "output_tensor must be Nx1 or Nx4";
    return torch::empty({0});
  }

  const bool extract_gradients = output_tensor.sizes()[1] == 4;
  auto bg_mapper = multi_mapper_->background_mapper();
  auto stream = getCurrentStream();

  nvblox::GPULayerView<nvblox::EsdfBlock>& gpu_layer_view =
      bg_mapper->esdf_layer().getGpuLayerView(stream);

  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  pynvblox::sdf::queryESDFKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_queries, extract_gradients, gpu_layer_view.getHash().impl_,
      bg_mapper->esdf_layer().block_size(), query_sphere.data_ptr<float>(),
      output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

// ============================================================================
// queryDynamicOccupancy — query the foreground mapper's occupancy log-odds
// ============================================================================

torch::Tensor MultiMapper::queryDynamicOccupancy(
    torch::Tensor output_tensor, const torch::Tensor query_positions) {
  const int64_t num_queries = query_positions.sizes()[0];

  if (!checkAllOnGPU(output_tensor, query_positions)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }
  if (!checkSizes(query_positions, {static_cast<int>(num_queries), 3}) ||
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 1})) {
    LOG(ERROR) << "query_positions must be Nx3 and output_tensor must be Nx1";
    return torch::empty({0});
  }

  auto fg_mapper = multi_mapper_->foreground_mapper();
  auto stream = getCurrentStream();

  nvblox::GPULayerView<nvblox::OccupancyBlock>& gpu_layer_view =
      fg_mapper->occupancy_layer().getGpuLayerView(stream);

  using OccupancyGPUHash =
      nvblox::Index3DDeviceHashMapType<nvblox::OccupancyBlock>;
  nvblox::host_vector<OccupancyGPUHash> host_hashes;
  host_hashes.push_back(gpu_layer_view.getHash().impl_);
  nvblox::device_vector<OccupancyGPUHash> device_hashes;
  device_hashes.copyFromAsync(host_hashes, stream);

  std::vector<float> block_sizes_host = {
      fg_mapper->occupancy_layer().block_size()};
  nvblox::device_vector<float> block_sizes_device;
  block_sizes_device.copyFromAsync(block_sizes_host, stream);

  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  pynvblox::sdf::queryOccupancyMultiMapperKernel<<<num_blocks, kNumThreads, 0,
                                                    stream>>>(
      1, num_queries, device_hashes.data(), block_sizes_device.data(),
      query_positions.data_ptr<float>(), output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

}  // namespace pynvblox
