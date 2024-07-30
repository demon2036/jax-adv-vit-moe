import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.sharding import Mesh,PartitionSpec,NamedSharding
from jax.experimental import mesh_utils

def go():

    device_mesh=mesh_utils.create_device_mesh((jax.device_count(),))
    mesh=Mesh(device_mesh,("data",))

    if jax.process_index()==0:
        print(mesh)


    pass



if __name__=="__main__":

    jax.distributed.initialize()


    go()