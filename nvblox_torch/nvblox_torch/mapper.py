#
from __future__ import annotations
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List, Optional
from enum import Enum
import torch

from nvblox_torch.constants import constants
from nvblox_torch.lib.utils import get_nvblox_torch_class
from nvblox_torch.layer import TsdfLayer, FeatureLayer, ColorLayer, OccupancyLayer, EsdfLayer
from nvblox_torch.sdf_query import EsdfQuery
from nvblox_torch.mesh import ColorMesh, FeatureMesh
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor


class QueryType(Enum):
    """Enum used when querying layers."""
    TSDF = 'tsdf'
    FEATURE = 'feature'
    OCCUPANCY = 'occupancy'
    ESDF = 'esdf'
    ESDF_GRAD = 'esdf_with_gradients'
    COLOR = 'color'


class Mapper:
    """Mapper which accumulates data from depth images into a voxelized 3D map.

    There are two types of depth integrators supported:
    - TSDF: Accumulates depth data and uses a truncated signed distance function to build a map.
    - OCCUPANCY: Accumulates depth data and uses a occupancy grid to build a map.

    Notes:
    * The resulting reconstruction can be obtained as either a mesh or an ESDF.
    It is also possible to individually query the voxels in the map.

    * Input tensors are expected to be on the GPU and of type float32 if not otherwise specified.

    * Appearance data from images containing RGB or generic feature vectors can be integrated into
    the map.

    * Several functions accepts an optional mask. Pixels where the mask is zero will be masked out.

    * Building multiple maps is supported. This is useful for example when there is a need to
    distinguish dynamic object from static objects. Individual maps are identified by a mapper_id
    where -1 means all mappers.
    """

    def __init__(
            self,
            voxel_sizes_m: float | List[float],
            integrator_types: ProjectiveIntegratorType
        | List[ProjectiveIntegratorType] = ProjectiveIntegratorType.TSDF,
            mapper_parameters: MapperParams = MapperParams(),
    ) -> None:
        """Construct a mapper.

        Pass multiple arguments to create several maps with different voxel sizes and integrator
        types.

        Args:
            voxel_sizes_m: The size of the voxels in meters. One per map.
            integrator_types: The type of depth integrator to use (TSDF or OCCUPANCY). Oner per map.
            mapper_parameters: The parameters of the mapper.
        """
        voxel_sizes = [voxel_sizes_m] if isinstance(voxel_sizes_m, float) else voxel_sizes_m
        integrator_types = [integrator_types] if isinstance(
            integrator_types, ProjectiveIntegratorType) else integrator_types
        assert len(voxel_sizes) == len(integrator_types)
        # Initialize c_mapper with layers = len(voxel_sizes)
        integrator_types_str = [integrator_type.value for integrator_type in integrator_types]
        self._c_mapper = get_nvblox_torch_class('Mapper')(voxel_sizes, integrator_types_str,
                                                          mapper_parameters._c_params)
        self._voxel_sizes = voxel_sizes
        self._integrator_types = integrator_types

    def params(self) -> MapperParams:
        """Get the parameters of the mapper.

        Returns:
            MapperParams: Parameters of the mapper.
        """
        return MapperParams(self._c_mapper.params())

    def add_depth_frame(self,
                        depth_frame: torch.Tensor,
                        t_w_c: torch.Tensor,
                        sensor: Sensor,
                        mask_frame: Optional[torch.Tensor] = None,
                        mapper_id: int = 0) -> None:
        """Add a depth frame to the mapper.

        Unified entry point that works with both Sensor objects and intrinsics tensors.

        Args:
            depth_frame (H, W): Depth frame to integrate.
            t_w_c  (4, 4; device=CPU): Transform from sensor frame to world frame.
            sensor: Sensor object
            mask_frame (H, W; uint8): Mask frame.
            mapper_id: Map id.

        Examples:
            camera = Sensor.from_camera(fu=525, fv=525, cu=320, cv=240, width=640, height=480)
            mapper.add_depth_frame(depth, pose, camera)

        """
        assert 0 <= mapper_id < len(self._voxel_sizes)

        check_integrator_inputs(depth_frame, t_w_c, 'Depth', 2, torch.float32)
        self._c_mapper.integrate_depth(depth_frame, t_w_c, sensor.get_c_sensor(), mask_frame,
                                       mapper_id)

    def add_color_frame(self,
                        color_frame: torch.Tensor,
                        t_w_c: torch.Tensor,
                        sensor: Sensor,
                        mask_frame: Optional[torch.Tensor] = None,
                        mapper_id: int = 0) -> None:
        """Add a color frame to the mapper.

        Unified entry point that works with both Sensor objects and intrinsics tensors.

        Args:
            color_frame (H, W, 3; uint8): Color frame to integrate.
            t_w_c  (4, 4; device=CPU): Transform from sensor frame to world frame.
            sensor: Sensor object (Camera only)
            mask_frame (H, W; uint8): Mask frame.
            mapper_id: Map id.

        Note:
            Color integration only supports Camera sensors (not Lidar).
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)

        check_integrator_inputs(color_frame, t_w_c, 'Color', 3, torch.uint8, 3)
        self._c_mapper.integrate_color(color_frame, t_w_c, sensor.get_c_sensor(), mask_frame,
                                       mapper_id)

    def add_feature_frame(self,
                          feature_frame: torch.Tensor,
                          t_w_c: torch.Tensor,
                          sensor: Sensor,
                          mask_frame: Optional[torch.Tensor] = None,
                          mapper_id: int = 0) -> None:
        """Add a feature frame to the mapper.

        Unified entry point that works with both Sensor objects and intrinsics tensors.

        Args:
            feature_frame (H, W, F; float16): Feature frame to integrate.
            t_w_c  (4, 4; device=CPU): Transform from sensor frame to world frame.
            sensor: Sensor object (Camera only)
            mask_frame (H, W; uint8): Mask frame.
            mapper_id: Map id.

        Notes:
            F <= FeatureLayer.num_elements_per_voxel().
            If F < FeatureLayer.num_elements_per_voxel(), input features will be padded with zeros.
            Feature integration only supports Camera sensors (not Lidar).
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)

        check_integrator_inputs(feature_frame, t_w_c, 'Feature', 3, torch.float16,
                                constants.feature_array_num_elements())
        self._c_mapper.integrate_features(feature_frame, t_w_c, sensor.get_c_sensor(), mask_frame,
                                          mapper_id)

    def update_esdf(self, mapper_id: int = -1) -> None:
        """Update the ESDF for a given mapper.

        Args:
            mapper_id: The mapper to update.
        """
        assert -1 <= mapper_id < len(self._voxel_sizes)

        self._c_mapper.update_esdf(mapper_id)

    def update_color_mesh(self, mapper_id: int = -1) -> None:
        """Update the color mesh for a given mapper.

        Args:
            mapper_id: The mapper to update.
        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.update_color_mesh(mapper_id)

    def update_feature_mesh(self, mapper_id: int = -1) -> None:
        """Update the feature mesh for a given mapper.

        Args:
            mapper_id: The mapper to update.
        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.update_feature_mesh(mapper_id)

    def tsdf_layer_view(self, mapper_id: int = 0) -> TsdfLayer:
        """Get the TSDF layer for a given mapper.

        Args:
            mapper_id: The mapper to get the TSDF layer for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return TsdfLayer(voxel_size_m=self._voxel_sizes[mapper_id],
                         c_layer=self._c_mapper.tsdf_layer(mapper_id))

    def feature_layer_view(self, mapper_id: int = 0) -> FeatureLayer:
        """Get the feature layer for a given mapper.

        Args:
            mapper_id: The mapper to get the feature layer for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return FeatureLayer(voxel_size_m=self._voxel_sizes[mapper_id],
                            c_layer=self._c_mapper.feature_layer(mapper_id))

    def color_layer_view(self, mapper_id: int = 0) -> ColorLayer:
        """Get the color layer for a given mapper.

        Args:
            mapper_id: The mapper to get the color layer for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return ColorLayer(voxel_size_m=self._voxel_sizes[mapper_id],
                          c_layer=self._c_mapper.color_layer(mapper_id))

    def decay(self, mapper_id: int = -1) -> None:
        """Decay the map.

        Decrease the weights of voxels in the map and remove blocks for which
        all weights are zero.
        The decay behavior is governed by the following MapperParams:
          * tsdf_decay_integrator_params
          * occupancy_decay_integrator_params

        Args:
            mapper_id (int): The mapper to decay. Use -1 to decay all mappers.
        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        if self._integrator_types[mapper_id] == ProjectiveIntegratorType.TSDF:
            self._c_mapper.decay_tsdf(mapper_id)
        elif self._integrator_types[mapper_id] == ProjectiveIntegratorType.OCCUPANCY:
            self._c_mapper.decay_occupancy(mapper_id)
        else:
            raise ValueError(
                f'Layer type {self._integrator_types[mapper_id]} not supported for decay')

    def clear(self, mapper_id: int = -1) -> None:
        """Clear the map for a given mapper.

        Args:
            mapper_id: The mapper to clear.
        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.clear(mapper_id)

    def get_color_mesh(self, mapper_id: int = 0) -> ColorMesh:
        """Get the color mesh for a given mapper.

        Args:
            mapper_id: The mapper to get the color mesh for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return ColorMesh(c_mesh=self._c_mapper.get_color_mesh(mapper_id))

    def get_feature_mesh(self, mapper_id: int = 0) -> FeatureMesh:
        """Get the feature mesh for a given mapper.

        Args:
            mapper_id: The mapper to get the feature mesh for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return FeatureMesh(c_mesh=self._c_mapper.get_feature_mesh(mapper_id))

    def save_map(self, map_fname: str, mapper_id: int) -> None:
        """Save the map for a given mapper.

        Args:
            map_fname: The file name to save the map to.
            mapper_id: The mapper to save the map for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.output_blox_map(map_fname, mapper_id)

    def load_from_file(self, filename: str, mapper_id: int) -> None:
        """Load a map from a file for a given mapper.

        Args:
            filename: The file name to load the map from.
            mapper_id: The mapper to load the map for.
        """
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return self._c_mapper.load_from_file(filename, mapper_id)

    def num_mappers(self) -> int:
        """Get the number of mappers.

        Returns:
            int: The number of mappers.
        """
        return self._c_mapper.num_mappers()

    def _maybe_allocate(self,
                        size: torch.Size,
                        tensor: Optional[torch.Tensor] = None,
                        dtype: torch.dtype = torch.float32,
                        value: Optional[float | int] = None) -> torch.Tensor:
        """If tensor is none, it will be allocated to the specified size. Tensor is returned."""
        if tensor is None:
            if value is None:
                tensor = torch.zeros(size, dtype=dtype, device='cuda')
            else:
                tensor = torch.full(size, value, dtype=dtype, device='cuda')
        else:
            assert tensor.shape == size, f'Expected preallocated size: {size}.'
        return tensor

    def query_layer(self,
                    query_type: QueryType,
                    query: torch.Tensor,
                    output: Optional[torch.Tensor] = None,
                    mapper_id: int = -1) -> torch.Tensor:
        """Query a given layer at N specified positions.

        - The layer to query is governed by the query_type argument. See table below for expected
          output content for each layer type.

        - mapper_id governs which mapper to query. For certain layers (see table), mapper_id=-1
          can be provided to query all mappers. This will return the minimum SDF value across all
          mappers (or maximum log_odds for the Occupancy layer).

        - The query tensor contains 3D positions to be queried. For queries in the ESDF layer, an
          extra column containing point radii may be provided. This radii will be subtracted from
          the retrieved distances.


        query_type   mapper_id>=0  mapper_id=-1   Output size (S)  Output content
        -------------------------------------------------------------------------------
        TSDF            Y             Y                 2         [distance, weight]
        ESDF            Y             Y                 1         [distance]
        ESDF_GRAD       Y             Y                 4         [grad_x, grad_y, grad_z, distance]
        OCCUPANCY       N             Y                 1         [occupancy_log_odds]
        FEATURE         Y             N                 F         [f0, ..., fF-2, weight]

          where
              Y: Supported
              N: Not supported
              F: FeatureLayer.num_elements_per_voxel()

        Args:
           query_type: Type of layer to query.
           query: Nx3 device tensor containing query 3D points.
           output: NxS Optional pre-allocated output device tensor.
           mapper_id: ID of mapper to query. -1 will query all layers.

        Returns
            torch.Tensor: A NxS tensor containing the packed voxel values described in the table.

        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        num_queries = query.shape[0]

        if query_type == QueryType.TSDF:
            output = self._maybe_allocate((num_queries, TsdfLayer.num_elements_per_voxel()), output)
            if mapper_id == -1:
                result = self._c_mapper.query_multi_tsdf(output, query)
            else:
                result = self._c_mapper.query_tsdf(output, query, mapper_id)

        elif query_type == QueryType.OCCUPANCY:
            output = self._maybe_allocate((num_queries, OccupancyLayer.num_elements_per_voxel()),
                                          output)
            assert mapper_id == -1, 'Only multi mapper query is supported for occupancy'
            result = self._c_mapper.query_multi_occupancy(output, query)

        elif query_type == QueryType.FEATURE:
            output = self._maybe_allocate((num_queries, FeatureLayer.num_elements_per_voxel()),
                                          output,
                                          dtype=torch.float16)
            assert mapper_id >= 0, 'Only single mapper query is supported for features'
            result = self._c_mapper.query_features(output, query, mapper_id)

        elif query_type == QueryType.ESDF_GRAD:
            output = self._maybe_allocate((num_queries, EsdfLayer.num_elements_per_voxel()),
                                          output,
                                          value=constants.esdf_unknown_distance())
            if mapper_id == -1:
                result = self._c_mapper.query_multi_esdf(output, query)
            else:
                result = self._c_mapper.query_esdf(output, query, mapper_id)
        elif query_type == QueryType.ESDF:
            output = self._maybe_allocate((num_queries, 1),
                                          output,
                                          value=constants.esdf_unknown_distance())
            if mapper_id == -1:
                result = self._c_mapper.query_multi_esdf(output, query)
            else:
                result = self._c_mapper.query_esdf(output, query, mapper_id)
        else:
            raise NotImplementedError(f'Query type {query_type} not implemented')

        if len(result) == 0:
            raise ValueError(f'Query failed for: {query_type}')
        return result

    def query_differentiable_layer(self,
                                   query_type: QueryType,
                                   query: torch.Tensor,
                                   output: Optional[torch.Tensor] = None,
                                   mapper_id: int = -1) -> torch.Tensor:
        """Query a differentiable layer at N specified positions.

        - A differentiable layer returns queries that supports torch.autograd. This means that the
          gradients of the returned value wrt the input query position can be computed
          automatically.

        - Currently only implemented for the ESDF layer, for which the gradient represents the
          vector that points in the direction of the greatest rate of increse in the distance field,
          i.e.
            - Towards the surface for query points inside an object.
            - Away from the surface for query points outside ojects.

        Args:
           query_type (QueryType): Type of layer to query.
           query (torch.Tensor): Nx3 tensor containing query 3D points.
           output (torch.Tensor): NxS Optional pre-allocated output tensor.
           mapper_id (int): ID of mapper to query. -1 will query all layers.

        Returns
            torch.Tensor: A Nx1 differentiable tensor.

        """
        assert -1 <= mapper_id < len(self._voxel_sizes)
        num_queries = query.shape[0]

        if query_type == QueryType.ESDF:
            output = self._maybe_allocate((num_queries, EsdfLayer.num_elements_per_voxel()),
                                          output,
                                          value=constants.esdf_unknown_distance())
            result = EsdfQuery.apply(
                query,
                self._c_mapper,
                output,
                mapper_id,
            )
        else:
            raise NotImplementedError(
                f'Layer type {query_type} not implemented for differentiable queries')
        if len(result) == 0:
            raise ValueError(f'Differentiable query failed for layer: {query_type}')
        return result

    def get_c_mapper(self) -> torch.classes.Mapper:
        """Get the wrapped C++ mapper.

        Returns:
            The wrapped C++ mapper.
        """
        return self._c_mapper

    def print_timing(self) -> str:
        """Print information from internal nvblox timers.

            TODO(dtingdahl): Move out of mapper class

        Returns:
            str: Human readable timing information.
        """
        return self._c_mapper.print_timing()


def check_integrator_inputs(image: torch.Tensor,
                            t_w_c: torch.Tensor,
                            image_type: str,
                            expected_dim: int,
                            expected_type: torch.dtype,
                            expected_num_channels: Optional[int] = None) -> None:
    """Helper function to check sensor-based integrator function inputs.

    Args:
        image (torch.Tensor): The input image (depth, color, or mask)
        t_w_c  (torch.Tensor): The transform of the sensor.
        image_type (str): A string describing the image type (for debug message).
        expected_dim (int): The expected number of dimensions of the image.
        expected_type (torch.dtype): The type of image elements.
        expected_num_channels (Optional[int], optional): Expected number of image channels.
            Defaults to None.
    """
    assert image.dim() == expected_dim, f'{image_type} image should have dim == {expected_dim}.'
    assert image.is_cuda, f'{image_type} image should be on device.'
    assert image.dtype == expected_type, f'{image_type} image should have type {expected_type}.'
    # Note(Vik): Expensive operation, disable for now.
    # assert not torch.isnan(image).any().item(), f'{image_type} must not contain nan values.'
    assert t_w_c.is_cpu, f'{image_type} t_w_c  should be on the CPU.'
    assert t_w_c.dtype == torch.float32, f'{image_type} pose should have type torch.float32.'
    if expected_num_channels:
        assert image.shape[
            2] == expected_num_channels, \
            f'{image_type} should have {expected_num_channels} channels.'
