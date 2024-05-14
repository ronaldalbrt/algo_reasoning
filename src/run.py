import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np


from losses import losses
from models.GAT import GAT
from experiment import run_experiment


model_args = {
    'name': 'logistic_regression',
    'learning_rate': 1e-3,
    'epochs': 100
}

rng_key = jax.random.PRNGKey(7)
x_train = jax.random.uniform(rng_key, (10000, 15, 1))
y_train = jax.random.bernoulli(rng_key, 0.5, (10000, 1))
x_val = jax.random.uniform(rng_key, (10000, 15, 1))
y_val= jax.random.bernoulli(rng_key, 0.5, (10000, 1))
x_test = jax.random.uniform(rng_key, (10000, 15, 1))
y_test = jax.random.bernoulli(rng_key, 0.5, (10000, 1))

def full_adjacency_matrix(x):
    adj_matrix = jnp.ones((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            adj_matrix.at[i, j, j].set(0)

    return adj_matrix

loss_fn = lambda x, y: losses.binary_cross_entropy(hk.Linear(1)(GAT(32, 8)(x, full_adjacency_matrix(x))), y)

train_sampler = (x_train, x_train)
val_sampler = (x_val, x_val)
test_sampler = (x_test, x_test)

results, best_params = run_experiment(loss_fn, model_args, train_sampler, val_sampler, test_sampler)

print("Results: ", results)
print("Best params: ", best_params)