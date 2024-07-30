import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

Device = jax.Device


def get_device_coords_tpu(device: Device):
    assert device.platform == 'tpu'
    print(device)
    print(device.core_on_chip)
    print(device.default_memory())
    print(device.memory_stats())
    print(device.coords)
    print()
    core_on_chip = int(device.core_on_chip)
    coords = tuple(map(int, device.coords))
    return core_on_chip, *coords


def get_hardware_mesh_tpu(devices):
    mesh_dict = {get_device_coords_tpu(device): device for device in devices}
    print(mesh_dict)

    # mesh_dict=
    pass


def go():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, ("data",))

    if jax.process_index() == 0:
        print(mesh)

        get_hardware_mesh_tpu(jax.devices())

    pass


if __name__ == "__main__":
    jax.distributed.initialize()

    go()
