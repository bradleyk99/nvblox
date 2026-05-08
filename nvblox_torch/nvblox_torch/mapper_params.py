# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# pylint: disable=protected-access
from nvblox_torch.lib.utils import get_nvblox_torch_class
from typing import Optional, Type, Any, no_type_check

# // NOTE(alexmillane, 2024.11.14): The following sub-parameter structs are currently unwrapped.
# // If you need them wrapped, ask alex.
# // Unwrapped sub-parameter classes:
# // - ViewCalculatorParams view_calculator_params;
# // - OccupancyIntegratorParams occupancy_integrator_params;
# // - OccupancyDecayIntegratorParams occupancy_decay_integrator_params;
# // - FreespaceIntegratorParams freespace_integrator_params;


# noqa
class NvbloxParameterClass:
    """NvbloxParameterClass is a base class for Nvblox parameter classes."""

    def __init__(self) -> None:
        """Constructor that does nothing."""
        pass

    def wrap_getter_and_setters(self, parameter_class: Type, c_param_struct: Any) -> None:
        """Wrap the getter and setter methods of the C++ parameter struct."""
        attribute_names = [
            method_name[len('get_'):] for method_name in c_param_struct._method_names()
            if method_name.startswith('get')
        ]
        for name in attribute_names:

            # Create a getter function
            def getter(_: Any, name: str = name) -> object:
                print(f'Getting: {name}')
                getter_name = 'get_' + name
                method = getattr(c_param_struct, getter_name)
                return method()

            # Create a setter function
            def setter(_: Any, value: object, name: str = name) -> None:
                setter_name = 'set_' + name
                method = getattr(c_param_struct, setter_name)
                method(value)

            # Add an attribute to the class
            setattr(parameter_class, name, property(getter, setter))


@no_type_check
class ProjectiveIntegratorParams(NvbloxParameterClass):
    """Parameters governing the projective integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('ProjectiveIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(ProjectiveIntegratorParams, self._c_params)

class FreespaceIntegratorParams(NvbloxParameterClass):
    """Parameters governing the freespace integrator (Dynablox-style dynamic detection)."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('FreespaceIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(FreespaceIntegratorParams, self._c_params)

class MeshIntegratorParams(NvbloxParameterClass):
    """Parameters governing the mesh integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MeshIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(MeshIntegratorParams, self._c_params)


class DecayIntegratorBaseParams(NvbloxParameterClass):
    """Base parameters for the decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('DecayIntegratorBaseParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(DecayIntegratorBaseParams, self._c_params)


class TsdfDecayIntegratorParams(NvbloxParameterClass):
    """Parameters governing the TSDF decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('TsdfDecayIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(TsdfDecayIntegratorParams, self._c_params)


class OccupancyDecayIntegratorParams(NvbloxParameterClass):
    """Parameters governing the occupancy decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('OccupancyDecayIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(OccupancyDecayIntegratorParams, self._c_params)


class EsdfIntegratorParams(NvbloxParameterClass):
    """Parameters governing the ESDF integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('EsdfIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(EsdfIntegratorParams, self._c_params)


class ViewCalculatorParams(NvbloxParameterClass):
    """Parameters governing the view calculator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('ViewCalculatorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(ViewCalculatorParams, self._c_params)


class BlockMemoryPoolParams(NvbloxParameterClass):
    """Parameters governing memory allocation."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('BlockMemoryPoolParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(BlockMemoryPoolParams, self._c_params)

class GroundPlaneEstimatorParams(NvbloxParameterClass):
    """Parameters governing the ground-plane estimator candidate band."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        if c_params is None:
            self._c_params = get_nvblox_torch_class('GroundPlaneEstimatorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(GroundPlaneEstimatorParams, self._c_params)


class RansacPlaneFitterParams(NvbloxParameterClass):
    """Parameters governing RANSAC ground-plane fitting."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        if c_params is None:
            self._c_params = get_nvblox_torch_class('RansacPlaneFitterParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(RansacPlaneFitterParams, self._c_params)


class MultiMapperParams:
    """MultiMapperParams wraps the C++ MultiMapperParams class."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MultiMapperParams')()
        else:
            self._c_params = c_params

    # Scalar fields (manual property-style access, matching MapperParams)
    @property
    def connected_mask_component_size_threshold(self) -> int:
        return self._c_params.get_connected_mask_component_size_threshold()

    @connected_mask_component_size_threshold.setter
    def connected_mask_component_size_threshold(self, value: int) -> None:
        self._c_params.set_connected_mask_component_size_threshold(value)

    @property
    def remove_small_connected_components(self) -> bool:
        return self._c_params.get_remove_small_connected_components()

    @remove_small_connected_components.setter
    def remove_small_connected_components(self, value: bool) -> None:
        self._c_params.set_remove_small_connected_components(value)

    @property
    def experimental_use_ground_plane_estimation(self) -> bool:
        return self._c_params.get_experimental_use_ground_plane_estimation()

    @experimental_use_ground_plane_estimation.setter
    def experimental_use_ground_plane_estimation(self, value: bool) -> None:
        self._c_params.set_experimental_use_ground_plane_estimation(value)

    # Composed sub-params (get/set round-trip pattern, matching MapperParams)
    def get_ground_plane_estimator_params(self) -> GroundPlaneEstimatorParams:
        return GroundPlaneEstimatorParams(
            self._c_params.get_ground_plane_estimator_params())

    def set_ground_plane_estimator_params(
            self, params: GroundPlaneEstimatorParams) -> None:
        return self._c_params.set_ground_plane_estimator_params(params._c_params)

    def get_ransac_plane_fitter_params(self) -> RansacPlaneFitterParams:
        return RansacPlaneFitterParams(
            self._c_params.get_ransac_plane_fitter_params())

    def set_ransac_plane_fitter_params(
            self, params: RansacPlaneFitterParams) -> None:
        return self._c_params.set_ransac_plane_fitter_params(params._c_params)

class MapperParams:
    """MapperParams is a class that wraps the C++ MapperParams class."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MapperParams')()
        else:
            self._c_params = c_params
        # NOTE: We don't call the automatic wrapping function here because we
        # need to convert the subclasses to python objects manually.

    def get_projective_integrator_params(self) -> ProjectiveIntegratorParams:
        """Parameter getter."""
        return ProjectiveIntegratorParams(self._c_params.get_projective_integrator_params())

    def set_projective_integrator_params(self, params: ProjectiveIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_projective_integrator_params(params._c_params)

    def get_freespace_integrator_params(self) -> FreespaceIntegratorParams:
        """Parameter getter."""
        return FreespaceIntegratorParams(self._c_params.get_freespace_integrator_params())

    def set_freespace_integrator_params(self, params: FreespaceIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_freespace_integrator_params(params._c_params)

    def get_mesh_integrator_params(self) -> MeshIntegratorParams:
        """Parameter getter."""
        return MeshIntegratorParams(self._c_params.get_mesh_integrator_params())

    def set_mesh_integrator_params(self, params: MeshIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_mesh_integrator_params(params._c_params)

    def get_decay_integrator_base_params(self) -> DecayIntegratorBaseParams:
        """Parameter getter."""
        return DecayIntegratorBaseParams(self._c_params.get_decay_integrator_base_params())

    def set_decay_integrator_base_params(self, params: DecayIntegratorBaseParams) -> None:
        """Parameter setter."""
        return self._c_params.set_decay_integrator_base_params(params._c_params)

    def get_tsdf_decay_integrator_params(self) -> TsdfDecayIntegratorParams:
        """Parameter getter."""
        return TsdfDecayIntegratorParams(self._c_params.get_tsdf_decay_integrator_params())

    def set_tsdf_decay_integrator_params(self, params: TsdfDecayIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_tsdf_decay_integrator_params(params._c_params)

    def get_occupancy_decay_integrator_params(self) -> OccupancyDecayIntegratorParams:
        """Parameter getter."""
        return OccupancyDecayIntegratorParams(
            self._c_params.get_occupancy_decay_integrator_params())

    def set_occupancy_decay_integrator_params(self, params: OccupancyDecayIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_occupancy_decay_integrator_params(params._c_params)

    def get_esdf_integrator_params(self) -> EsdfIntegratorParams:
        """Parameter getter."""
        return EsdfIntegratorParams(self._c_params.get_esdf_integrator_params())

    def set_esdf_integrator_params(self, params: EsdfIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_esdf_integrator_params(params._c_params)

    def get_view_calculator_params(self) -> ViewCalculatorParams:
        """Parameter getter."""
        return ViewCalculatorParams(self._c_params.get_view_calculator_params())

    def set_view_calculator_params(self, params: ViewCalculatorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_view_calculator_params(params._c_params)

    def get_block_memory_pool_params(self) -> BlockMemoryPoolParams:
        """Parameter getter."""
        return BlockMemoryPoolParams(self._c_params.get_block_memory_pool_params())

    def set_block_memory_pool_params(self, params: BlockMemoryPoolParams) -> None:
        """Parameter setter."""
        return self._c_params.set_block_memory_pool_params(params._c_params)
