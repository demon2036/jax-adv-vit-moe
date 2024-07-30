import jax

import orbax.checkpoint as ocp
import argparse
from typing import Any
from flax.serialization import msgpack_serialize
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
from models.model import ViT
import os


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

    model = ViT(
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
        print(model.tabulate(rng, jnp.ones(image_shape)))

    # cnn = CNN()

    # image_shape = [1, 28, 28, 1]

    # TODO: implement pjit train state init









    params = model.init(rng, jnp.ones(image_shape))['params']

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

    tx = create_optimizer_fn(learning_rate)

    return EMATrainState.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params, ema_decay=ema_decay,
                                trade_beta=trade_beta, label_smoothing=label_smoothing)