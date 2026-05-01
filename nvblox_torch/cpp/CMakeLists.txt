#[[
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
]]

# Torch messes up the architecture flags, so have to restore them
set(CMAKE_CUDA_ARCHITECTURES_OLD ${CMAKE_CUDA_ARCHITECTURES})
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_OLD})

# On Jetson, we need to link explicitly against cudnn
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND TORCH_LIBRARIES cudnn)
endif()

# Nvblox-torch python wrapper
add_nvblox_shared_library(
  py_nvblox
  SOURCE_FILES
  src/py_nvblox.cu
  src/py_mapper.cu
  src/py_multi_mapper.cu
  src/py_scene.cu
  src/py_sensor.cu
  src/convert_tensors.cpp
  src/sdf_query.cu
  src/py_layer.cpp
  src/py_mesh.cpp
  src/py_mapper_params.cpp
  src/cuda_stream.cpp
  src/py_rendering.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  ${TORCH_LIBRARIES}
  LINK_LIBRARIES_PRIVATE
  nvblox_stdgpu
  nvblox_gpu_hash
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Nvblox-torch python wrapper
add_nvblox_shared_library(
  py_nvblox_test
  SOURCE_FILES
  src/convert_tensors.cpp
  LINK_LIBRARIES_PUBLIC
  nvblox_lib
  ${TORCH_LIBRARIES}
  INCLUDE_DIRECTORIES_PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Tests
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
