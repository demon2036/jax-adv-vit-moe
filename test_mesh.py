import functools

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from typing import Tuple

from utils.json_print import json_print

Device = jax.Device
TpuCoords = Tuple[int, int, int, int]


def get_device_coords_tpu(device: Device) -> TpuCoords:
    assert device.platform == 'tpu'
    # print(device)
    # print(device.core_on_chip)
    # # print(device.default_memory())
    # # print(device.memory_stats())
    # print(device.coords)
    # print()
    core_on_chip = int(device.core_on_chip)
    coords = tuple(map(int, device.coords))
    return core_on_chip, *coords


def get_hardware_mesh_tpu(devices):
    mesh_dict = {get_device_coords_tpu(device): device for device in devices}
    print(mesh_dict)

    nc, nx, ny, nz = map(lambda x: x + 1, sorted(mesh_dict.keys())[-1])
    print(nc, nx, ny, nz)
    mesh = np.empty((nc, nx, ny, nz), dtype=object)
    print(mesh)
    print(mesh_dict)
    for (c, x, y, z), device in mesh_dict.items():
        mesh[(c, x, y, z)] = device
    return mesh


def get_logical_mesh_default(partitions: Tuple[int, ...], replicas: Tuple[int, ...], hardware_mesh: np.ndarray):
    shape = functools.reduce(lambda a, b: a + b, zip(partitions, replicas))
    print(shape, partitions, replicas)
    devices = hardware_mesh.reshape(shape)
    print(devices.shape)
    devices = devices.transpose(tuple(range(0, 2 * hardware_mesh.ndim, 2))
                                + tuple(range(1, 2 * hardware_mesh.ndim, 2))
                                )
    print(devices.shape)
    num_partitions=np.prod(partitions)
    num_replicas=np.prod(replicas)
    devices=devices.reshape((num_partitions,num_replicas))
    print(f'{devices=}')
    return Mesh(devices=devices,axis_names=('expert','replica'))


def get_logical_mesh(partitions, hardware_mesh: np.ndarray):
    replicas = tuple(
        s // p for p, s in zip(partitions, hardware_mesh.shape, strict=True)
    )
    replicas = tuple(reversed(replicas))
    partitions = tuple(reversed(partitions))

    hardware_axes_order = tuple(reversed(range(hardware_mesh.ndim)))
    hardware_mesh = hardware_mesh.transpose(hardware_axes_order)
    print(hardware_mesh)
    logical_mesh = get_logical_mesh_default(partitions, replicas, hardware_mesh)
    print(logical_mesh)
    return logical_mesh


def get_auto_logical_mesh_tpu(num_partitions: int, hardware_mesh: np.ndarray):
    hardware_mesh_shape = hardware_mesh.shape
    z = min(num_partitions, hardware_mesh_shape[3])
    y = min(num_partitions // z, hardware_mesh_shape[2])
    x = min(num_partitions // (z * y), hardware_mesh_shape[1])
    c = min(num_partitions // (z * y * x), hardware_mesh_shape[0])

    return get_logical_mesh((c, x, y, z), hardware_mesh)


def go():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, ("data",))

    if jax.process_index() == 0:
        print(mesh)

        hardware_mesh = get_hardware_mesh_tpu(jax.devices())

        get_auto_logical_mesh_tpu(4, hardware_mesh)
        # print(hardware_mesh)
    pass


if __name__ == "__main__":
    jax.distributed.initialize()

    go()
