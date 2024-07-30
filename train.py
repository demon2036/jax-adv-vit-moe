import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from train_state import create_train_state, EMATrainState

from training import apply_model_trade


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


def convert_to_global_array(x, x_sharding):
    b, *res = x.shape

    per_replica_batches_x = np.split(x, jax.local_device_count())

    global_batch_shape_x = (b * jax.process_count(), *res)

    global_batch_array_x = jax.make_array_from_single_device_arrays(
        global_batch_shape_x, sharding=x_sharding,
        arrays=[
            jax.device_put(batch, device)
            for batch, device in zip(per_replica_batches_x, x_sharding.addressable_devices)
        ]
    )

    return global_batch_array_x


def train():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    x_sharding = mesh_sharding(PartitionSpec('data'))
    rng = jax.random.PRNGKey(1)

    # shape = (128, 256, 384)
    shape = (128, 3, 32, 32,)
    batch, *res = shape

    # x = jnp.ones(shape)
    x = jax.random.normal(rng, shape)
    y = jnp.ones(shape[0])
    """

    global_batch_shape_x = (128 * jax.process_count(), *res)
    global_batch_shape_y = (128 * jax.process_count(),)
    print(global_batch_shape_x, global_batch_shape_y)

    per_replica_batches_x = np.split(x, jax.local_device_count())
    per_replica_batches_y = np.split(y, jax.local_device_count())

    global_batch_array_x = jax.make_array_from_single_device_arrays(
        global_batch_shape_x, sharding=x_sharding,
        arrays=[
            jax.device_put(batch, device)
            for batch, device in zip(per_replica_batches_x, x_sharding.addressable_devices)
        ]
    )

    global_batch_array_y = jax.make_array_from_single_device_arrays(
        global_batch_shape_y, sharding=x_sharding,
        arrays=[
            jax.device_put(batch, device)
            for batch, device in zip(per_replica_batches_y, x_sharding.addressable_devices)
        ]
    )
    """
    # global_batch_array_x = convert_to_global_array(x, x_sharding)
    # global_batch_array_y = convert_to_global_array(y, x_sharding)
    # print(global_batch_array_x.shape,global_batch_array_y.shape)
    # return global_batch_array_x


    data=(x,y)
    data=jax.tree_util.tree_map(functools.partial(convert_to_global_array,x_sharding=x_sharding),data)

    for d in data:
        print(d.shape)
    return data
    while True:
        pass

    state, state_sharding = create_train_state(rng, x_sharding, mesh, dim=768)

    # train_step_jit = jax.jit(train_step, in_shardings=(x_sharding, state_sharding), out_shardings=state_sharding, )

    train_step_jit = jax.jit(apply_model_trade,
                             in_shardings=(state_sharding, (x_sharding, x_sharding), mesh_sharding(())),
                             out_shardings=state_sharding, )

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

        disable = not jax.process_index() == 0

        with tqdm.tqdm(range(1000), disable=disable) as pbar:
            for _ in pbar:
                state = block_all(train_step_jit(state, (global_batch_array_x, global_batch_array_y), rng))

                pbar.update()

    return state


if __name__ == "__main__":
    jax.config.update('jax_threefry_partitionable', True)
    jax.distributed.initialize()

    # if jax.process_index() == 0:
    #     print(jax.devices())

    state = train()
