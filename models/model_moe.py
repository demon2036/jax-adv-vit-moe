# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal, Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax.experimental.pallas.ops.tpu.flash_attention
import jax.numpy as jnp
from chex import Array
from einops import einsum
from flax.typing import Initializer
from jax._src.typing import DType

from models.moe import PartitionSpec, _convert_partition_spec, with_sharding_constraint
from utils.utils2 import fixed_sincos2d_embeddings

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))

CeilOrRound = Literal["ceil", "round"]


def _dispatch(data: Array, partition_spec: Optional[PartitionSpec]) -> Array:
    """Dispatches data to experts using all_to_all."""
    partition_spec = PartitionSpec('data', )
    partition_spec = _convert_partition_spec(partition_spec)
    # partition_spec = mesh_sharding(partition_spec)
    num_groups, num_experts, capacity, *item_shape = data.shape
    data = with_sharding_constraint(data, partition_spec)
    if num_groups % num_experts == 0:
        data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
        data = jnp.swapaxes(data, 0, 2)
    else:
        data = jnp.swapaxes(data, 0, 1)
    data = data.reshape(-1, *item_shape)
    data = with_sharding_constraint(data, partition_spec)
    return data.reshape(num_experts, num_groups * capacity, *item_shape)


def _receive(data: Array, num_groups: int,
             partition_spec: Optional[PartitionSpec] = None) -> Array:
    """Receives data from experts using all_to_all."""
    partition_spec = ('data',)
    partition_spec = _convert_partition_spec(partition_spec)
    # partition_spec = mesh_sharding(partition_spec)

    num_experts, num_groups_time_capacity, *item_shape = data.shape
    capacity = num_groups_time_capacity // num_groups
    data = data.reshape(num_experts * num_groups, capacity, *item_shape)
    data = with_sharding_constraint(data, partition_spec)
    if num_groups % num_experts == 0:
        data = data.reshape(num_experts, -1, num_experts, capacity, *item_shape)
        data = jnp.swapaxes(data, 0, 2)
        data = data.reshape(num_groups, num_experts, capacity, *item_shape)
    else:
        data = data.reshape(num_experts, num_groups, capacity, *item_shape)
        data = jnp.swapaxes(data, 0, 1)
    data = with_sharding_constraint(data, partition_spec)
    return data


# def _dispatch(data: Array, partition_spec: Optional[PartitionSpec]) -> Array:
#     """Dispatches data to experts using all_to_all."""
#     # partition_spec = PartitionSpec('data', )
#     partition_spec = PartitionSpec('experts', 'replicate')
#     partition_spec = _convert_partition_spec(partition_spec)
#     # partition_spec = mesh_sharding(partition_spec)
#     num_groups, num_experts, *item_shape = data.shape
#     data = with_sharding_constraint(data, partition_spec)
#     if num_groups % num_experts == 0:
#         data = data.reshape(num_experts, -1, num_experts, *item_shape)
#         data = jnp.swapaxes(data, 0, 2)
#     else:
#         data = jnp.swapaxes(data, 0, 1)
#     data = data.reshape(-1, *item_shape)
#
#     data = with_sharding_constraint(data, partition_spec)
#     return data.reshape(num_experts, num_groups, *item_shape)
#
#
# def _receive(data: Array, num_groups: int,
#              partition_spec: Optional[PartitionSpec] = None) -> Array:
#     """Receives data from experts using all_to_all."""
#     # partition_spec = ('data',)
#     partition_spec = PartitionSpec('experts', 'replicate')
#     partition_spec = PartitionSpec('experts', 'replicate')
#     partition_spec = _convert_partition_spec(partition_spec)
#     # partition_spec = mesh_sharding(partition_spec)
#
#     num_experts, num_groups_time_capacity, *item_shape = data.shape
#     # capacity = num_groups_time_capacity // num_groups
#     data = data.reshape(num_experts * num_groups, *item_shape)
#     data = with_sharding_constraint(data, partition_spec)
#     if num_groups % num_experts == 0:
#         data = data.reshape(num_experts, -1, num_experts, *item_shape)
#         data = jnp.swapaxes(data, 0, 2)
#         data = data.reshape(num_groups, num_experts, *item_shape)
#     else:
#         data = data.reshape(num_experts, num_groups, *item_shape)
#         data = jnp.swapaxes(data, 0, 1)
#     data = with_sharding_constraint(data, partition_spec)
#     return data


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 2
    image_size: int = 32
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "gap"
    qk_norm: bool = False
    use_fc_norm: bool = True
    reduce_include_prefix: bool = False

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    use_kan: bool = False
    polynomial_degree: int = 8

    use_moe: bool = False

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


def normalize(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    m = jax.lax.rsqrt(jnp.square(x).sum(axis=axis, keepdims=True) + eps)
    return x * m


def compute_capacity(
        num_tokens: int,
        num_experts: int,
        capacity_factor: float,
        ceil_or_round: CeilOrRound = "ceil",
        multiple_of: Optional[int] = 4) -> int:
    """Returns the capacity per expert needed to distribute num_tokens among num_experts."""
    if ceil_or_round == "ceil":
        capacity = int(math.ceil(num_tokens * capacity_factor / num_experts))
    elif ceil_or_round == "round":
        capacity = int(round(num_tokens * capacity_factor / num_experts))
    else:
        raise ValueError(f"Unsupported {ceil_or_round=}")
    if capacity < 1:
        raise ValueError(f"The values num_tokens = f{num_tokens}, num_experts = "
                         f"{num_experts} and capacity_factor = {capacity_factor} "
                         f"lead to capacity = {capacity}, but it must be greater "
                         "than or equal to 1.")
    if multiple_of and multiple_of > 0:
        # Make capacity multiple of 4 to try to avoid padding.
        capacity += (-capacity) % multiple_of
    actual_capacity_factor = capacity * num_experts / num_tokens
    if abs(actual_capacity_factor - capacity_factor) > 1e-6:
        pass
        # logging.warning(
        #     "The target capacity_factor is %f, but with num_tokens=%d and "
        #     "num_experts=%d the actual capacity_factor is %f.",
        #     capacity_factor, num_tokens, num_experts, actual_capacity_factor)
    return capacity


class SoftRouter(ViTBase, nn.Module):
    """Soft router merging tokens as inputs/outputs of the experts."""
    dim: int
    num_experts: int = 32
    num_slots: Optional[int] = None
    capacity_factor: Optional[float] = 1.0
    noise_std: float = 0.0
    deterministic: bool = False
    dtype: Optional[DType] = None
    mu_init: Initializer = jax.nn.initializers.lecun_normal()
    expert_init: Initializer = init.truncated_normal(0.02)
    scale_init: Initializer = jax.nn.initializers.ones
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

    @nn.compact
    def __call__(self, inputs: Array):

        # y = nn.Dense(self.dim)(inputs)

        # Normalize inputs to have unit norm.
        dtype = self.dtype or inputs.dtype
        inputs = normalize(inputs.astype(dtype), axis=-1)
        # Create num_experts * num_slots parameters, normalized to have unit norm.
        batch_size, group_size, dim = inputs.shape
        if self.num_slots is None:
            num_slots = compute_capacity(
                group_size, self.num_experts, self.capacity_factor,
                ceil_or_round='round', multiple_of=1)
        else:
            num_slots = self.num_slots
            actual_capacity_factor = self.num_experts * num_slots / group_size
            pre = f'{self.capacity_factor=} ignored. ' if self.capacity_factor else ''
            logging.info(
                '%sWith num_tokens=%d, num_experts=%d and num_slots=%d, the actual '
                'capacity_factor is %f.', pre, group_size, self.num_experts,
                self.num_slots, actual_capacity_factor)
        mu = self.param('mu', self.mu_init, (dim, self.num_experts, num_slots))
        mu = normalize(mu.astype(dtype), axis=0)
        # self.sow('intermediates', 'mu_unit', mu)
        # Scale inputs/mu before computing the logits.
        scale = self.param('scale', self.scale_init, ()).astype(dtype)
        if inputs.size < mu.size:
            inputs = inputs * scale
        else:
            mu = mu * scale
        # Notation:
        # g = number of groups (typically batch size).
        # m = number of items per group (typically sequence length).
        # n = number of experts.
        # p = number of slots per expert.
        # n * p = number of total slots.
        # Compute router logits between pairs of items (m) and total slots (n * p),
        # independently on each group (g).
        logits = jnp.einsum('gmd,dnp->gmnp', inputs, mu, precision=self.precision)
        # Each slot takes a convex combination of the inputs.
        dispatch_weights = jax.nn.softmax(logits, axis=1)
        # Each item takes a convex combination of all the outputs of each slot.
        combine_weights = jax.nn.softmax(logits, axis=(2, 3))

        # w = self.param('w', self.expert_init, (self.num_experts, dim, self.dim))

        w1 = self.param('w1', nn.with_partitioning(self.expert_init, ('experts',)),
                        (self.num_experts, dim, self.hidden_dim))

        w2 = self.param('w2', nn.with_partitioning(self.expert_init, ('experts',)),
                        (self.num_experts, self.hidden_dim, self.dim))

        # print(inputs.shape,dispatch_weights.shape)

        x = einsum(inputs, dispatch_weights, 'b m d, b m n p->b n p d')

        x = _dispatch(x, None)
        # x = jnp.einsum('nbd,ndk->nbk', x, w, )

        x = jnp.einsum('nbd,ndk->nbk', x, w1, )
        x = nn.gelu(x)
        x = jnp.einsum('nbd,ndk->nbk', x, w2, )

        x = _receive(x, batch_size)

        # x = einsum(x, w, 'b n p d1,n d1 d2->b n p d2')
        x = einsum(x, combine_weights, 'b n p d,b m n p->b m d')
        return x


class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )
        # if self.pooling == "cls":
        self.cls_token = self.param(
            "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        # if self.pooling == "cls":
        # cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
        # x = jnp.concatenate((cls_token, x), axis=1)
        return x


class Identity(nn.Module):
    def __call__(self, x):
        return x


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
        self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
        z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
        return self.drop(self.wo(z), det)


class FeedForward(ViTBase, nn.Module):
    def setup(self):

        if self.use_moe:
            self.w1 = SoftRouter(self.hidden_dim)
            self.w2 = SoftRouter(self.dim)
        else:
            self.w1 = Dense(self.hidden_dim)
            self.w2 = Dense(self.dim)

        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)


# class ViTLayer(ViTBase, nn.Module):
#     def setup(self):
#         self.attn = Attention(**self.kwargs)
#         if self.use_kan:
#             self.ff = KANLayer(self.polynomial_degree)
#         else:
#             self.ff = FeedForward(**self.kwargs)
#
#         self.norm1 = nn.LayerNorm()
#         self.norm2 = nn.LayerNorm()
#         self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))
#
#         self.scale1 = self.scale2 = 1.0
#         if self.layerscale:
#             self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
#             self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))
#             # self.scale1 = self.param("scale1", init.constant(1e-6), (self.dim,))
#             # self.scale2 = self.param("scale2", init.constant(1e-6), (self.dim,))
#
#     def __call__(self, x: Array, det: bool = True) -> Array:
#         x = x + self.drop(self.scale1 * self.attn(self.norm1(x), det), det)
#         x = x + self.drop(self.scale2 * self.ff(self.norm2(x), det), det)
#         return x


# class ViTLayer(ViTBase, nn.Module):
#     """Soft router merging tokens as inputs/outputs of the experts."""
#
#     num_experts: int = 256
#     num_slots: Optional[int] = None
#     capacity_factor: Optional[float] = 1.0
#     noise_std: float = 0.0
#     deterministic: bool = False
#     dtype: Optional[DType] = jnp.bfloat16
#     mu_init: Initializer = jax.nn.initializers.lecun_normal()
#     expert_init: Initializer = jax.nn.initializers.lecun_normal()
#     scale_init: Initializer = jax.nn.initializers.ones
#     precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
#
#     @nn.compact
#     def __call__(self, inputs: Array):
#         _, group_size, dim = inputs.shape
#         # Normalize inputs to have unit norm.
#         # w = self.param('w', nn.with_partitioning(self.expert_init, ('model', None)),
#         #                (256, dim, dim))
#         x = inputs
#         for i in range(6):
#             w = self.param(f'w_{i}', self.expert_init,
#                            (self.num_experts, dim, 4 * dim))
#
#             w2 = self.param(f'w2_{i}', self.expert_init,
#                             (self.num_experts, 4 * dim, dim))
#
#             norm = nn.LayerNorm()
#             mha = Attention(**self.kwargs)
#
#             x = x + mha(norm(x))
#             # x = x + (norm(x))
#
#
#             y=x
#             x=nn.LayerNorm()
#
#             x = jnp.einsum('bnd,ndk->bnk', x, w, )
#             x = nn.gelu(x)
#             x = jnp.einsum('bnd,ndk->bnk', x, w2, )
#
#             x=x+y
#
#             # x = nn.Dense(dim)(x)
#
#         # jax.debug.visualize_array_sharding(inputs[:,:,0])
#
#         # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('data', 'model')))
#
#         return x


# class ViTLayer(ViTBase, nn.Module):
#     """Soft router merging tokens as inputs/outputs of the experts."""
#     num_experts: int = 256
#     num_slots: Optional[int] = None
#     capacity_factor: Optional[float] = 1.0
#     noise_std: float = 0.0
#     deterministic: bool = False
#     dtype: Optional[DType] = jnp.bfloat16
#     mu_init: Initializer = jax.nn.initializers.lecun_normal()
#     expert_init: Initializer = jax.nn.initializers.lecun_normal()
#     scale_init: Initializer = jax.nn.initializers.ones
#     precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
#
#     @nn.compact
#     def __call__(self, inputs: Array,*args,**kwargs):
#         batch_size, group_size, dim = inputs.shape
#
#         # inputs = nn.Dense(dim)(inputs)
#
#         x = inputs
#
#         for i in range(6):
#
#             # print(x.shape)
#
#             norm = nn.LayerNorm()
#             mha = Attention(**self.kwargs)
#
#             x = x + mha(norm(x))
#             # x = x + norm(x)
#
#             w = self.param(f'w_{i}', nn.with_partitioning(self.expert_init, ('data',)),
#                            (self.num_experts, dim, 4 * dim))
#
#             w2 = self.param(f'w2_{i}', nn.with_partitioning(self.expert_init, ('data',)),
#                             (self.num_experts, 4 * dim, dim))
#
#             # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model')))
#             # x=jax.lax.all
#
#             # x = einops.rearrange(x, 'b n d-> n b d')
#
#             # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec(None, 'model')))
#             # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model')))
#
#             # jax.debug.inspect_array_sharding(x, callback=print)
#
#             y = x
#             x = nn.LayerNorm()(x)
#
#             x = _dispatch(x, None)
#
#             # x = jnp.einsum('bnd,ndk->bnk', x, w, )
#             x = jnp.einsum('nbd,ndk->nbk', x, w, )
#             x = nn.gelu(x)
#             x = jnp.einsum('nbd,ndk->nbk', x, w2, )
#
#             x = _receive(x, batch_size)
#
#             x = x + y
#
#
#
#
#             # jax.debug.inspect_array_sharding(x, callback=print)
#
#         # def mul(xs, ws):
#         #     return xs @ ws
#         #
#         # x = jax.vmap(mul)(x, w)
#         # #
#
#         # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model')))
#
#         # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec(None, 'model')))
#         # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model', None)))
#
#         # x = nn.Dense(dim,)(x)
#
#         # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model', None)))
#         #
#         # print()
#         # print('inputs mesh')
#         # jax.debug.visualize_array_sharding(inputs[:, :, 0])
#         # print()
#         # print('output x mesh')
#         # jax.debug.visualize_array_sharding(x[:, :, 0])
#         # print()
#
#         return x


class ViTLayer(ViTBase, nn.Module):
    """Soft router merging tokens as inputs/outputs of the experts."""
    num_experts: int = 256
    num_slots: Optional[int] = None
    capacity_factor: Optional[float] = 1.0
    noise_std: float = 0.0
    deterministic: bool = False
    dtype: Optional[DType] = jnp.bfloat16
    mu_init: Initializer = jax.nn.initializers.lecun_normal()
    expert_init: Initializer = jax.nn.initializers.lecun_normal()
    scale_init: Initializer = jax.nn.initializers.ones
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

    @nn.compact
    def __call__(self, inputs: Array, *args, **kwargs):
        batch_size, group_size, dim = inputs.shape

        # inputs = nn.Dense(dim)(inputs)

        x = inputs

        for i in range(12):

            # print(x.shape)

            norm = nn.LayerNorm()
            mha = Attention(**self.kwargs)

            # x = x + mha(norm(x))
            # x = x + norm(x)

            # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model')))
            # x=jax.lax.all

            # x = einops.rearrange(x, 'b n d-> n b d')

            # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec(None, 'model')))
            # x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('model')))

            # jax.debug.inspect_array_sharding(x, callback=print)

            y = x
            x = nn.LayerNorm()(x)

            if i < 8:
                x = nn.Dense(4 * dim)(x)
                x = nn.gelu(x)
                x = nn.Dense(dim)(x)
            else:
                # w = self.param(f'w_{i}', nn.with_partitioning(self.expert_init, ('experts',)),
                #                (self.num_experts, dim, 4 * dim))
                #
                # w2 = self.param(f'w2_{i}', nn.with_partitioning(self.expert_init, ('experts',)),
                #                 (self.num_experts, 4 * dim, dim))
                #
                # x = _dispatch(x, None)
                # # x = jnp.einsum('bnd,ndk->bnk', x, w, )
                # x = jnp.einsum('nbd,ndk->nbk', x, w, )
                # x = nn.gelu(x)
                # x = jnp.einsum('nbd,ndk->nbk', x, w2, )
                #
                # x = _receive(x, batch_size)

                x = SoftRouter()(x)

            x = x + y

        return x


class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        # layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        # #use_moe=False if i < 6 else True,
        # self.layer = [layer_fn(**(self.kwargs | {'use_moe': False if i < 6 else True})) for i in range(self.layers)]

        self.layer = ViTLayer(**self.kwargs)

        # self.norm = nn.LayerNorm()

        self.norm = nn.LayerNorm() if not self.use_fc_norm else Identity()
        self.fc_norm = nn.LayerNorm() if self.use_fc_norm else Identity()

        # print(self.kwargs)

        self.head = nn.Dense(self.labels) if self.labels is not None else None

    def __call__(self, x: Array, det: bool = True) -> Array:
        # x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        x = self.drop(self.embed(x), det)
        # for layer in self.layer:
        #     x = layer(x, det)

        x = self.layer(x)

        x = self.norm(x)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None:
            return x
        """
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x[:, 0:].mean(1)
        return self.head(x)
        """

        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x if self.reduce_include_prefix else x[:, 1:]
            x = x.mean(1)
        else:
            raise NotImplemented()

        x = self.fc_norm(x)

        return self.head(x)
