module Eval
    using Flux

    function param_update(model, data_loader, opt, loss_fn)
        opt_state = Flux.setup(opt, model)
        accum_loss = 0
        
        for data in data_loader
            input, label = data
            
            grads = Flux.gradient(model) do m
                pred = m(input)
                accum_loss += loss_fn(pred, label)
            end

            Flux.update!(opt_state, model, grads[1])
        end

        return accum_loss
    end

    function eval(model, data_loader, loss_fn)
        accum_loss = 0

        for data in data_loader
            input, label = data

            pred = model(input)
            accum_loss += loss_fn(pred, label)
        end

        return accum_loss
    end
end