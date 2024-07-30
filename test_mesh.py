import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.sharding import Mesh,PartitionSpec,NamedSharding
from jax.experimental import mesh_utils


Device=jax.Device


def get_device_coords_tpu(device:Device):
    assert device.platform=='tpu'
    print(device)
    print(device.core_on_chip)
    print(device.default_memory())
    print(device.memory_stats())
    print(device.coords)


def get_hardware_mesh_tpu(devices):
    # mesh_dict=
    pass
def go():

    device_mesh=mesh_utils.create_device_mesh((jax.device_count(),))
    mesh=Mesh(device_mesh,("data",))

    if jax.process_index()==0:
        print(mesh)

        get_device_coords_tpu(jax.devices()[0])


    pass



if __name__=="__main__":

    jax.distributed.initialize()


    go()