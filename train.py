import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from prefetch import prefetch_to_device
from train_state import create_train_state, EMATrainState

from training import apply_model_trade, eval_step
from dataset import get_train_dataloader
from utils.utils2 import AverageMeter


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
    x = np.array(x)

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
    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args.__dict__,
                   settings=wandb.Settings(_disable_stats=True),
                   config_exclude_keys=['train_dataset_shards', 'valid_dataset_shards', 'train_origin_dataset_shards'])
    average_meter = AverageMeter(use_latest=["learning_rate"])

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

    train_dataloader_iter = prefetch_to_device(train_dataloader_iter, 2, x_sharding)
    state, state_sharding = create_train_state(init_rng, x_sharding, mesh,
                                               layers=args.layers,
                                               dim=args.dim,
                                               heads=args.heads,
                                               labels=args.labels,
                                               layerscale=args.layerscale,
                                               patch_size=args.patch_size,
                                               image_size=args.image_size,
                                               posemb=args.posemb,
                                               pooling=args.pooling,
                                               dropout=args.dropout,
                                               droppath=args.droppath,
                                               warmup_steps=args.warmup_steps,
                                               training_steps=args.training_steps,
                                               learning_rate=args.learning_rate,
                                               weight_decay=args.weight_decay,
                                               ema_decay=args.ema_decay,
                                               trade_beta=args.beta,
                                               label_smoothing=args.label_smoothing,
                                               use_fc_norm=args.use_fc_norm,
                                               reduce_include_prefix=args.reduce_include_prefix,
                                               b1=args.adam_b1,
                                               b2=args.adam_b2,
                                               clip_grad=args.clip_grad,
                                               )

    # train_step_jit = jax.jit(train_step, in_shardings=(x_sharding, state_sharding), out_shardings=state_sharding, )

    train_step_jit = jax.jit(apply_model_trade,
                             in_shardings=(state_sharding, [x_sharding, x_sharding], mesh_sharding(())),
                             out_shardings=(state_sharding, None), donate_argnums=0)

    eval_step_jit = jax.jit(eval_step,
                            in_shardings=(state_sharding, [x_sharding, x_sharding],),
                            out_shardings=None, )

    with mesh:
        init_step = 1
        disable = not jax.process_index() == 0

        for step in tqdm.tqdm(range(init_step, args.training_steps), initial=init_step, total=args.training_steps):

            data = next(train_dataloader_iter)
            # data = jax.tree_util.tree_map(functools.partial(convert_to_global_array, x_sharding=x_sharding), data)
            rng, train_rng = jax.random.split(rng)

            state, metrics = train_step_jit(state, data, train_rng)

            average_meter.update(**metrics)
            metrics = average_meter.summary('train/')
            if jax.process_index() == 0:
                wandb.log(metrics, step)


            # if jax.process_index() == 0 and step % args.log_interval == 0:
            #     average_meter.update(**metrics)
            #     metrics = average_meter.summary('train/')
            #     wandb.log(metrics, step)
            #
            # if step % args.eval_interval == 0:
            #     for data in tqdm.tqdm(test_dataloader, leave=False, dynamic_ncols=True):
            #         data = jax.tree_util.tree_map(functools.partial(convert_to_global_array, x_sharding=x_sharding),
            #                                       data)
            #
            #         eval_metrics = eval_step_jit(state, data)
            #
            #         if jax.process_index() == 0:
            #             average_meter.update(**eval_metrics)
            #     if jax.process_index() == 0:
            #         metrics = average_meter.summary("val/")
            #         num_samples = metrics.pop("val/num_samples")
            #         metrics = jax.tree_util.tree_map(lambda x: x / num_samples, metrics)
            #         wandb.log(metrics, step)

    return eval_metrics


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

    metrics = train_and_evaluate(parser.parse_args())

    if jax.process_index() == 0:
        print(metrics)

    # data = train_and_evaluate(parser.parse_args())
    # jax.tree_util.tree_map(lambda x: print(x.shape), data)
