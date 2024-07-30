import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from train_state import create_train_state, EMATrainState


def block_all(xs):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
    return xs


def train_step(x, state: EMATrainState):
    def loss_fn(params):
        out = state.apply_fn({'params': params}, x)
        loss = (jnp.zeros_like(out) - out).mean()
        return loss

    grad = jax.grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grad)
    return state


def train():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    x_sharding = mesh_sharding(PartitionSpec('data'))
    rng = jax.random.PRNGKey(1)
    state, state_sharding = create_train_state(rng, x_sharding, mesh)

    # shape = (128, 256, 384)
    shape = (128, 32, 32, 3)
    batch, *res = shape

    # x = jnp.ones(shape)
    x = jax.random.normal(rng, shape)
    x_sharding = mesh_sharding(PartitionSpec('data'))

    global_batch_shape = (128 * jax.process_count(), *res)
    print(global_batch_shape)

    per_replica_batches = np.split(x, jax.local_device_count())

    global_batch_array = jax.make_array_from_single_device_arrays(
        global_batch_shape, sharding=x_sharding,
        arrays=[
            jax.device_put(batch, device)
            for batch, device in zip(per_replica_batches, x_sharding.addressable_devices)
        ]
    )

    train_step_jit = jax.jit(train_step, in_shardings=(x_sharding, state_sharding), out_shardings=state_sharding, )

    with mesh:
        # grad = block_all(train_step_jit(global_batch_array, state))
        #
        # for i in range(100):
        #     grad = block_all(train_step_jit(global_batch_array, state))
        #
        # start = time.time()
        # for i in range(1000):
        #     grad = block_all(train_step_jit(global_batch_array, state))
        # end = time.time()

        with tqdm.tqdm(range(1000), ) as pbar:
            state = block_all(train_step_jit(global_batch_array, state))

            pbar.update()

        # if jax.process_index() == 0:
        #     print(device_mesh)
        #     # print(x_sharding.addressable_devices)
        #     # print()
        #     # print(mesh)
        #     # jax.debug.visualize_sharding((shape[0], shape[1]), sharding=x_sharding)
        #     # jax.debug.visualize_array_sharding(global_batch_array[:, :, 0])
        #     #
        #     # print(x_sharding.addressable_devices)
        #     # print(state_sharding)
        #     # jax.debug.visualize_array_sharding(grad['Dense_0']['kernel'])
        #     # print(params)
        #     print(global_batch_array.shape)
        #     print(end - start)

            # print(grad)

    return state


if __name__ == "__main__":
    jax.distributed.initialize()

    # if jax.process_index() == 0:
    #     print(jax.devices())

    state = train()
