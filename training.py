import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import einops

from attacks.pgd import pgd_attack3
from trade.trade import trade
import numpy as np

EPSILON = 8 / 255  # @param{type:"number"}


def apply_model_trade(state, data, key):
    images, labels = data

    images = einops.rearrange(images, 'b c h w->b h w c')

    images = images.astype(jnp.float32) / 255
    labels = labels.astype(jnp.float32)

    print(images.shape)

    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = trade(images, labels, state, key=key, epsilon=EPSILON, step_size=2 / 255)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        logits_adv = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        one_hot = optax.smooth_labels(one_hot, state.label_smoothing)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
        metrics = {'loss': loss, 'trade_loss': trade_loss, 'logits': logits, 'logits_adv': logits_adv}

        return loss + state.trade_beta * trade_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, metrics), grads = grad_fn(state.params)
    accuracy_std = jnp.mean(jnp.argmax(metrics['logits'], -1) == labels)
    accuracy_adv = jnp.mean(jnp.argmax(metrics['logits_adv'], -1) == labels)

    metrics['accuracy'] = accuracy_std
    metrics['adversarial accuracy'] = accuracy_adv

    state = state.apply_gradients(grads=grads)

    new_ema_params = jax.tree_util.tree_map(
        lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
        state.ema_params, state.params)
    state = state.replace(ema_params=new_ema_params)

    metrics = metrics | state.opt_state.hyperparams

    metrics=jax.tree_util.tree_map(jnp.mean,metrics)


    return state, metrics


def eval_step(state, data):
    # inputs, labels = data

    inputs, labels = data
    inputs = inputs.astype(jnp.float32)
    labels = labels.astype(jnp.int64)

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

    return metrics
