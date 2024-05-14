import jax
import jax.numpy as jnp
import haiku as hk
import optax

from losses import losses
from train import train, eval

def run_experiment(loss_fn, model_args, train_sampler, val_sampler, test_sampler):
    best_params = None
    best_val_result = 0

    results = {}

    hk_loss_fn = hk.transform(loss_fn)
    rng_key = jax.random.PRNGKey(7)

    params = hk_loss_fn.init(rng_key, train_sampler[0], train_sampler[1])
    loss_fn = lambda params, sampler: hk_loss_fn.apply(params=params, x=sampler[0], y=sampler[1], rng=rng_key)

    optimizer = optax.adam(learning_rate=model_args['learning_rate'])
    opt_state = optimizer.init(params)

    for epoch in range(model_args['epochs']):
        print('Training...')
        params, opt_state, loss = train(params, train_sampler, loss_fn, optimizer, opt_state)

        print('Evaluating...')
        train_result = eval(params, train_sampler, loss_fn)
        val_result = eval(params, val_sampler, loss_fn)
        test_result = eval(params, test_sampler, loss_fn)
        
        results[epoch] = {
            'train_result': train_result, 
            'validation_result': val_result, 
            'test_result':test_result
        }

        if val_result > best_val_result:
            best_val_result = val_result
            best_params = params
            
        print(f'Model: {model_args["name"]} |'
                f'Epoch: {epoch:02d} | '
                f'Loss: {loss:.4f} | '
                f'Train: {100 * train_result:.2f} | '
                f'val: {100 * val_result:.2f} | '
                f'Test: {100 * test_result:.2f}')
    
    results['best'] = {
        'train_results': eval(best_params, train_sampler, loss_fn),
        'val_results': eval(best_params, val_sampler, loss_fn),
        'test_results': eval(best_params, test_sampler, loss_fn)
    }
        
    return results, best_params
