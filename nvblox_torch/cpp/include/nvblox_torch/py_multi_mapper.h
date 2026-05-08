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
#pragma once

#include <torch/script.h>

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <nvblox/mapper/mapper.h>
#include <nvblox/mapper/multi_mapper.h>

#include "nvblox_torch/convert_tensors.h"
#include "nvblox_torch/py_layer.h"
#include "nvblox_torch/py_mapper_params.h"
#include "nvblox_torch/py_sensor.h"

namespace pynvblox {

/// Python wrapper around nvblox::MultiMapper for Dynablox-style
/// dynamic obstacle mapping. Internally holds a static (background) and
/// dynamic (foreground) nvblox::Mapper. Freespace integration and routing
/// of depth measurements between the two are handled by the underlying
/// nvblox::MultiMapper; the wrapper exposes a flat set of integrate/update/
/// query methods.
struct MultiMapper : torch::CustomClassHolder {
  /// Constructor.
  /// @param voxel_size_m Voxel size in meters for the contained layers.
  /// @param mapping_type One of: "static_tsdf", "static_occupancy",
  ///        "dynamic", "human_with_static_tsdf",
  ///        "human_with_static_occupancy". Use "dynamic" for Dynablox-style
  ///        freespace-consistency dynamic detection.
  /// @param esdf_mode One of: "3D", "2D".
  /// @param mapper_params Parameter struct, applied to the background mapper.
  /// Legacy single-MapperParams constructor. Foreground mapper params and
  /// MultiMapperParams default-construct. Kept for backward compatibility.
  MultiMapper(double voxel_size_m, std::string mapping_type,
              std::string esdf_mode,
              c10::intrusive_ptr<MapperParams> background_mapper_params);

  /// Full constructor matching the isaac_ros_nvblox usage pattern.
  /// @param background_mapper_params Params for the static (background) mapper.
  /// @param foreground_mapper_params Params for the dynamic (foreground)
  ///        mapper. For mapping_type "static_tsdf" / "static_occupancy" the
  ///        foreground mapper is inactive and these are unused.
  /// @param multi_mapper_params Multi-mapper-level params (mask connected
  ///        components, ground plane estimation, RANSAC).
  MultiMapper(double voxel_size_m, std::string mapping_type,
              std::string esdf_mode,
              c10::intrusive_ptr<MapperParams> background_mapper_params,
              c10::intrusive_ptr<MapperParams> foreground_mapper_params,
              c10::intrusive_ptr<MultiMapperParams> multi_mapper_params);

  ~MultiMapper() = default;

  /// Integrate a depth frame. For mapping_type == "dynamic", the
  /// underlying nvblox::MultiMapper internally separates static and dynamic
  /// portions of the frame using the freespace layer.
  /// @param depth_frame_t HxW float32 GPU tensor, depth in meters.
  /// @param T_L_C_t 4x4 GPU tensor, sensor-to-layer transform.
  /// @param sensor PySensor (Camera only currently supported).
  /// @param update_time_ms Current time in milliseconds. Required for
  ///        dynamic mapping; ignored for static modes.
  void integrateDepth(torch::Tensor depth_frame_t, torch::Tensor T_L_C_t,
                      c10::intrusive_ptr<PySensor> sensor,
                      int64_t update_time_ms);

  /// Integrate a color frame into the background mapper.
  void integrateColor(torch::Tensor color_frame_t, torch::Tensor T_L_C_t,
                      c10::intrusive_ptr<PySensor> sensor);

  /// Update the ESDF on both mappers as appropriate for the mapping type.
  void updateEsdf();

  void setForegroundMapperParams(c10::intrusive_ptr<MapperParams> params);

  /// Update the background mapper's freespace layer. Required for dynamic
  /// detection in mapping_type "dynamic" — without this the freespace layer
  /// stays empty and nothing is ever classified dynamic.
  /// @param update_time_ms Current time in milliseconds. Pass the same value
  ///        you passed to integrateDepth this frame.
  void updateFreespace(int64_t update_time_ms);

  /// Update the color mesh of the background mapper.
  void updateColorMesh();

  /// Decay the dynamic occupancy layer. Call once per frame to make
  /// dynamic obstacles fade when no longer observed.
  void decayDynamicOccupancy();

  // Decay static TSDF
  void decayStaticTsdf();

  /// Query the background mapper's ESDF. Same semantics as
  /// pynvblox::Mapper::queryEsdf.
  /// @param output_tensor Nx4 (gradient mode) or Nx1 GPU tensor.
  /// @param query_sphere Nx4 GPU tensor [x,y,z,radius].
  /// @return The output tensor (filled).
  torch::Tensor queryStaticEsdf(torch::Tensor output_tensor,
                                const torch::Tensor query_sphere);

  /// Query the foreground (dynamic) mapper's occupancy log-odds at points.
  /// @param output_tensor Nx1 GPU tensor.
  /// @param query_positions Nx3 GPU tensor.
  /// @return The output tensor (filled).
  torch::Tensor queryDynamicOccupancy(torch::Tensor output_tensor,
                                      const torch::Tensor query_positions);

  /// Layer accessors (background mapper).
  c10::intrusive_ptr<PyTsdfLayer> tsdf_layer();
  c10::intrusive_ptr<PyColorLayer> color_layer();

 protected:
  /// Convert a mapping_type string to the nvblox enum.
  static nvblox::MappingType mappingTypeFromString(const std::string& s);

  /// Convert an esdf_mode string to the nvblox enum.
  static nvblox::EsdfMode esdfModeFromString(const std::string& s);

  /// The underlying nvblox MultiMapper.
  std::shared_ptr<nvblox::MultiMapper> multi_mapper_;

  /// Cached parameters.
  double voxel_size_m_;
  std::string mapping_type_str_;
  std::string esdf_mode_str_;
  c10::intrusive_ptr<MapperParams> background_mapper_params_;
  c10::intrusive_ptr<MapperParams> foreground_mapper_params_;
  c10::intrusive_ptr<MultiMapperParams> multi_mapper_params_;
};

}  // namespace pynvblox