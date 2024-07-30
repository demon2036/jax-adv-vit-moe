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
    print(device)
    print(device.core_on_chip)
    # print(device.default_memory())
    # print(device.memory_stats())
    print(device.coords)
    print()
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



def get_logical_mesh(partitions,hardware_mesh):
    for p,s in zip(partitions,hardware_mesh.shape,strict=True):
        print(p,s)


def get_auto_logical_mesh_tpu(num_partitions:int,hardware_mesh:np.ndarray):
    z=min(num_partitions,hardware_mesh[3])
    y = min(num_partitions//z, hardware_mesh[2])
    x = min(num_partitions//(z*y), hardware_mesh[1])
    c = min(num_partitions//(z*y*x), hardware_mesh[0])

    return get_logical_mesh((c,x,y,z),hardware_mesh)



def go():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, ("data",))

    if jax.process_index() == 0:
        print(mesh)

        hardware_mesh = get_hardware_mesh_tpu(jax.devices())

        get_auto_logical_mesh_tpu(4,hardware_mesh)
        # print(hardware_mesh)
    pass


if __name__ == "__main__":
    jax.distributed.initialize()

    go()
