module Train
    using .Eval
    using Flux, JLD2

    function train(model, model_args, opt, train_loader, valid_loader, test_loader, loss_fn)
        results = Dict()
        best_model_state = nothing
        best_val_result = Inf

        for epoch in 1:model_args["epochs"]
            println("Training...")
            loss = Eval.param_update(model, train_loader, opt, loss_fn)

            print("Evaluating...")
            train_result = Eval.eval(model, train_loader)
            val_result = Eval.eval(model, valid_loader)
            test_result = Eval.eval(model, test_loader)

            results[epoch] = Dict(:train=>train_result, :validation=>val_result, :test=>test_result)
            

            if best_val_result < val_result
                best_model_state = Flux.state(model)
                best_val_result = val_result
            end

            println("Model: $model_args[:name] |",
                "Epoch: $epoch | ",
                "Loss: $loss | ",
                "Train: $train_result | ",
                "Valid: $val_result | ",
                "Test: $test_result")
        end


        Flux.loadmodel!(model, best_model_state)
            results["best"] = Dict(
                :train=> eval(model, train_loader),
                valid=> eval(model, valid_loader),
                test=> eval(model, test_loader)
            )

        return results, best_model_state
    end

end