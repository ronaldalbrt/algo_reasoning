import haiku as hk
import jax 
import jax.numpy as jnp

@jax.jit
def binary_cross_entropy(pred, ground_truth):
    loss = jnp.sum(jnp.maximum(pred, 0) - pred * ground_truth + jnp.log(1 + jnp.exp(-jnp.abs(pred))))

    return loss

@jax.jit
def squared_error(pred, ground_truth):
    loss = jnp.mean(jnp.square(ground_truth - pred), axis=-1)

    return loss

@jax.jit
def cross_entropy(pred, ground_truth):
    loss = -jnp.sum(ground_truth * jax.nn.log_softmax(pred), axis=-1)

    return loss