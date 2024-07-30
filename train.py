import argparse
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
from dataset import get_train_dataloader


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


# grad = block_all(train_step_jit(global_batch_array, state))
#
# for i in range(100):
#     grad = block_all(train_step_jit(global_batch_array, state))
#
# start = time.time()
# for i in range(1000):
#     grad = block_all(train_step_jit(global_batch_array, state))
# end = time.time()


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


def train_and_evaluate(args):

    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)


    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    x_sharding = mesh_sharding(PartitionSpec('data'))



    train_dataloader_iter, test_dataloader = get_train_dataloader(args.train_batch_size,
                                                                  shard_path=args.train_dataset_shards,
                                                                  test_shard_path=args.valid_dataset_shards,
                                                                  origin_shard_path=args.train_origin_dataset_shards)

    data = next(train_dataloader_iter)

    data = jax.tree_util.tree_map(functools.partial(convert_to_global_array, x_sharding=x_sharding), data)
    return data





    """
    

    state, state_sharding = create_train_state(init_rng, x_sharding, mesh, dim=768)

    # train_step_jit = jax.jit(train_step, in_shardings=(x_sharding, state_sharding), out_shardings=state_sharding, )

    train_step_jit = jax.jit(apply_model_trade,
                             in_shardings=(state_sharding, (x_sharding, x_sharding), mesh_sharding(())),
                             out_shardings=(state_sharding, None), donate_argnums=0)

    with mesh:
        disable = not jax.process_index() == 0

        with tqdm.tqdm(range(1000), disable=disable) as pbar:
            for _ in pbar:
                state, metrics = block_all(train_step_jit(state, data, rng))

                pbar.update()
                break
    """

    return None

    return metrics


if __name__ == "__main__":
    jax.config.update('jax_threefry_partitionable', True)
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--train-origin-dataset-shards")
    parser.add_argument("--valid-dataset-shards")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    # parser.add_argument("--valid-batch-size", type=int, default=256)
    # parser.add_argument("--train-loader-workers", type=int, default=40)
    # parser.add_argument("--valid-loader-workers", type=int, default=5)

    # parser.add_argument("--random-crop", default="rrc")
    # parser.add_argument("--color-jitter", type=float, default=0.0)
    # parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    # parser.add_argument("--random-erasing", type=float, default=0.25)
    # parser.add_argument("--augment-repeats", type=int, default=3)
    # parser.add_argument("--test-crop-ratio", type=float, default=0.875)

    # parser.add_argument("--mixup", type=float, default=0.8)
    # parser.add_argument("--cutmix", type=float, default=1.0)
    # parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=5)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=10)
    parser.add_argument("--layerscale", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--posemb", default="learnable")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--droppath", type=float, default=0.1)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)
    parser.add_argument("--use-fc-norm", action="store_true", default=False)
    parser.add_argument("--reduce_include_prefix", action="store_true", default=False)

    # parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--pretrained-ckpt")
    # parser.add_argument("--label-mapping")
    #
    # parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-b1", type=float, default=0.95)
    parser.add_argument("--adam-b2", type=float, default=0.98)
    # parser.add_argument("--adam-eps", type=float, default=1e-8)
    # parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    # parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    #
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    #
    parser.add_argument("--project")
    parser.add_argument("--name")
    # parser.add_argument("--ipaddr")
    # parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
    data=train_and_evaluate(parser.parse_args())
    print(data.shape)