import jax
import haiku as hk
import numpy as np
from numpy.random import random
import optax

# Generate random input data
x = random((30, 2)).astype('float32')
y = np.dot(x, [2., -3.]) + 1.
y = np.expand_dims(y, axis=1).astype('float32')

def mse_loss(y_pred, y_t):
    return jax.lax.integer_pow(y_pred - y_t, 2).sum()

def loss_fn(x, y_t):
    return mse_loss(hk.Linear(1)(x), y_t)

hk_loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
rng_key = jax.random.PRNGKey(42)
params = hk_loss_fn.init(x=x, y_t=y, rng=rng_key)
loss_fn = hk_loss_fn.apply

optimizer = optax.adam(learning_rate=1e-3)

# Training
opt_state = optimizer.init(params)
for epoch in range(5000):
    loss, grads = jax.value_and_grad(loss_fn)(params, x=x, y_t=y)
    print("progress:", "epoch: ", epoch, "loss: ", loss)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params  = optax.apply_updates(params, updates)

# After training
print("Estimated Parameters: ", params)