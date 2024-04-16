module Encoders

using Flux

    function construct_encoders(stage::String, loc::String, t::String, input_dim::Integer, hidden_dim::Integer, name::String)
        linear = hidden_dim -> Dense(input_dim => hidden_dim)

        return Sequential(linear(hidden_dim))
    end
end
