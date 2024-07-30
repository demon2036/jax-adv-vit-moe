import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn



def loss_fun_trade(state, data):
    """Compute the loss of the network."""
    inputs, logits = data
    x_adv = inputs.astype(jnp.float32)
    logits_adv = state.apply_fn({"params": state.params}, x_adv)
    return optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()


def trade(image, label, state, epsilon=0.1, maxiter=10, step_size=0.007, key=None):
    """PGD attack on the L-infinity ball with radius epsilon.

  Args:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    epsilon: float, radius of the L-infinity ball.
    maxiter: int, number of iterations of this algorithm.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball
      of radius epsilon and centered at image.

  Notes:
    PGD attack is described in (Madry et al. 2017),
    https://arxiv.org/pdf/1706.06083.pdf

    # image_perturbation = jnp.zeros_like(image)
    image_perturbation = 0.001 * jax.random.normal(key, shape=image.shape)

    def adversarial_loss(perturbation):
        return loss_fun_trade(params, (image, image + perturbation, label))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        # image_perturbation += (2 * epsilon / maxiter) * sign_grad

        image_perturbation += step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(image + image_perturbation, 0, 1)

     """

    logits = jax.lax.stop_gradient(state.apply_fn({"params": state.params}, image))

    # x_adv = 0.001 * jax.random.normal(key, shape=image.shape) + image

    x_adv = jax.random.uniform(key, shape=image.shape, minval=-epsilon, maxval=epsilon) + image
    x_adv = jnp.clip(x_adv, 0, 1)

    # def adversarial_loss(adv_image, image):
    #     return loss_fun_trade(state, (image, adv_image, label))

    def adversarial_loss(adv_image, logits):
        return loss_fun_trade(state, (adv_image, logits))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(jax.lax.stop_gradient(grad_adversarial(x_adv, logits)))
        # heuristic step-size 2 eps / maxiter
        # image_perturbation += step_size * sign_grad

        # delta = jnp.clip(image_perturbation - image, min=-epsilon, max=epsilon)

        x_adv = jax.lax.stop_gradient(x_adv) + step_size * sign_grad
        r1 = jnp.where(x_adv > image - epsilon, x_adv, image - epsilon)
        x_adv = jnp.where(r1 < image + epsilon, r1, image + epsilon)

        x_adv = jnp.clip(x_adv, min=0, max=1)

        # projection step onto the L-infinity ball centered at image
        # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return x_adv
