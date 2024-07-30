import jax

import orbax.checkpoint as ocp
import argparse
from typing import Any
from flax.serialization import msgpack_serialize
from attacks.pgd import pgd_attack3
from functools import partial
from torch.utils.data import DataLoader
import einops
import flax.jax_utils
import torchvision
import tqdm
from flax.training.common_utils import shard, shard_prng_key
from flax import linen as nn
from flax.training import train_state, orbax_utils

import jax.numpy as jnp
import numpy as np
import optax
from optax.losses import softmax_cross_entropy_with_integer_labels

from datasets_fork import get_train_dataloader
from model import ViT
import os
import wandb
from utils2 import AverageMeter, save_checkpoint_in_background

E

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'


@partial(jax.pmap, axis_name="batch", )
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = pgd_attack3(images, labels, state, )

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    return new_state, grads, loss, accuracy











class EMATrainState(flax.training.train_state.TrainState):
    label_smoothing: int
    trade_beta: int
    ema_decay: int = 0.995
    ema_params: Any = None


def create_train_state(rng,
                       layers=12,
                       dim=192,
                       heads=3,
                       labels=10,
                       layerscale=True,
                       patch_size=2,
                       image_size=32,
                       posemb="learnable",
                       pooling='cls',
                       dropout=0.0,
                       droppath=0.0,
                       clip_grad=1.0,
                       warmup_steps=None,
                       training_steps=None,
                       learning_rate=None,
                       weight_decay=None,
                       ema_decay=0.9999,
                       trade_beta=5.0,
                       label_smoothing=0.1,
                       use_fc_norm: bool = False,
                       reduce_include_prefix: bool = False,
                       b1=0.95,
                       b2=0.98

                       ):
    """Creates initial `TrainState`."""

    cnn = ViT(
        layers=layers,
        dim=dim,
        heads=heads,
        labels=labels,
        layerscale=layerscale,
        patch_size=patch_size,
        image_size=image_size,
        posemb=posemb,
        pooling=pooling,
        dropout=dropout,
        droppath=droppath,
        use_fc_norm=use_fc_norm,
        reduce_include_prefix=reduce_include_prefix
    )

    image_shape = [1, 32, 32, 3]
    if jax.process_index() == 0:
        print(cnn.tabulate(rng, jnp.ones(image_shape)))

    # cnn = CNN()

    # image_shape = [1, 28, 28, 1]

    params = cnn.init(rng, jnp.ones(image_shape))['params']

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.lion(
            learning_rate=learning_rate,
            b1=b1, b2=b2,
            # eps=args.adam_eps,
            weight_decay=weight_decay,
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        # if args.lr_decay < 1.0:
        #     layerwise_scales = {
        #         i: optax.scale(args.lr_decay ** (args.layers - i))
        #         for i in range(args.layers + 1)
        #     }
        #     label_fn = partial(get_layer_index_fn, num_layers=args.layers)
        #     label_fn = partial(tree_map_with_path, label_fn)
        #     tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=training_steps,
        end_value=1e-5,
    )

    # learning_rate = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-7,
    #     peak_value=LEARNING_RATE,
    #     warmup_steps=50000 * 5 // TRAIN_BATCH_SIZE,
    #     decay_steps=50000 * EPOCHS // TRAIN_BATCH_SIZE,
    #     end_value=1e-6,
    # )

    tx = create_optimizer_fn(learning_rate)

    return EMATrainState.create(apply_fn=cnn.apply, params=params, tx=tx, ema_params=params, ema_decay=ema_decay,
                                trade_beta=trade_beta, label_smoothing=label_smoothing)


@partial(jax.pmap, axis_name="batch", )
def accuracy(state, data):
    # inputs, labels = data

    inputs, labels = data
    inputs = inputs.astype(jnp.float32)
    labels = labels.astype(jnp.int64)

    # print(images)
    # while True:
    #     pass
    inputs = einops.rearrange(inputs, 'b c h w->b h w c')

    logits = state.apply_fn({"params": state.ema_params}, inputs)
    clean_accuracy = jnp.argmax(logits, axis=-1) == labels

    maxiter = 20

    adversarial_images = pgd_attack3(inputs, labels, state, epsilon=EPSILON, maxiter=maxiter,
                                     step_size=EPSILON * 2 / maxiter)
    logits_adv = state.apply_fn({"params": state.ema_params}, adversarial_images)
    adversarial_accuracy = jnp.argmax(logits_adv, axis=-1) == labels

    metrics = {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy, "num_samples": labels != -1}

    metrics = jax.tree_util.tree_map(lambda x: (x * (labels != -1)).sum(), metrics)

    metrics = jax.lax.psum(metrics, axis_name='batch')

    # metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


def train_and_evaluate(args
                       ) -> train_state.TrainState:
    """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """

    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args.__dict__,
                   settings=wandb.Settings(_disable_stats=True),
                   config_exclude_keys=['train_dataset_shards', 'valid_dataset_shards', 'train_origin_dataset_shards'])
        average_meter = AverageMeter(use_latest=["learning_rate"])

    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng,
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

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = {'model': state}
    postfix = "ema"
    name = args.name
    output_dir = args.output_dir
    filename = os.path.join(output_dir, f"{name}-{postfix}")
    print(filename)

    if args.pretrained_ckpt is not None:
        state = checkpointer.restore(filename, item=ckpt)['model']
        init_step = state.step + 1
    else:
        init_step = 1

    print(init_step)
    state = flax.jax_utils.replicate(state)
    train_dataloader_iter, test_dataloader = get_train_dataloader(args.train_batch_size,
                                                                  shard_path=args.train_dataset_shards,
                                                                  test_shard_path=args.valid_dataset_shards,
                                                                  origin_shard_path=args.train_origin_dataset_shards)

    def prepare_tf_data(xs):
        """Convert a input batch from tf Tensors to numpy arrays."""
        local_device_count = jax.local_device_count()

        def _prepare(x):
            x = np.asarray(x)

            return x.reshape((local_device_count, -1) + x.shape[1:])

        return jax.tree_util.tree_map(_prepare, xs)

    train_dataloader_iter = map(prepare_tf_data, train_dataloader_iter)

    train_dataloader_iter = flax.jax_utils.prefetch_to_device(train_dataloader_iter, 2)

    for step in tqdm.tqdm(range(init_step, args.training_steps), initial=init_step, total=args.training_steps):
        rng, input_rng = jax.random.split(rng)
        data = next(train_dataloader_iter)

        rng, train_step_key = jax.random.split(rng, num=2)
        train_step_key = shard_prng_key(train_step_key)

        state, metrics = apply_model_trade(state, data, train_step_key)

        if jax.process_index() == 0 and step % args.log_interval == 0:
            average_meter.update(**flax.jax_utils.unreplicate(metrics))
            metrics = average_meter.summary('train/')
            wandb.log(metrics, step)

        if step % args.eval_interval == 0:
            for data in tqdm.tqdm(test_dataloader, leave=False, dynamic_ncols=True):
                data = shard(jax.tree_util.tree_map(np.asarray, data))
                metrics = accuracy(state, data)

                if jax.process_index() == 0:
                    average_meter.update(**jax.device_get(flax.jax_utils.unreplicate(metrics)))
            if jax.process_index() == 0:
                metrics = average_meter.summary("val/")
                num_samples = metrics.pop("val/num_samples")
                metrics = jax.tree_util.tree_map(lambda x: x / num_samples, metrics)
                wandb.log(metrics, step)

                # params = flax.jax_utils.unreplicate(state.params)
                # params_bytes = msgpack_serialize(params)
                # save_checkpoint_in_background(params_bytes=params_bytes, postfix="last", name=args.name,
                #                               output_dir=os.getenv('GCS_DATASET_DIR'))

            ckpt = {'model': jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))}
            # orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpointer.save(filename, ckpt, save_args=save_args, force=True)

            # state=flax.jax_utils.replicate(state)

            # state_host =
            # checkpointer.save(filename, args=ocp.args.StandardSave(state_host),
            #                   force=True)

            # params = flax.jax_utils.unreplicate(state.ema_params)
            # params_bytes = msgpack_serialize(params )
            # save_checkpoint_in_background(params_bytes=params_bytes, postfix="ema", name=args.name,
            #                               output_dir=args.output_dir)

    return state


if __name__ == "__main__":
    # _, train_dataloader, test_dataloader = get_train_dataloader(TRAIN_BATCH_SIZE)
    # train_dataloader_iter = iter(train_dataloader)
    #
    # for _ in range(100):
    #     data=next(train_dataloader_iter)
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
    train_and_evaluate(parser.parse_args())
