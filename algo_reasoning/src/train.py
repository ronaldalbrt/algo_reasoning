import jax
import optax

def train(params, sampler, loss_fn, optimizer, opt_init_state):
    optimizer_state = opt_init_state

    #for batch in sampler:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, sampler)

    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, optimizer_state, loss


def eval(params, sampler, loss_fn):
    metric = 0
    num_batches = 0

    #for batch in sampler:
    metric += loss_fn(params, sampler)
    num_batches += 1

    return metric / num_batches