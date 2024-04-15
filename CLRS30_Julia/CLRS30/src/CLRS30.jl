module CLRS30
    using Base.Iterators, PyCall

    export create_dataset

    struct DataPoint
        name::String
        location::String
        type::String
        data::Array
    end

    struct Feedback
        inputs::Tuple{DataPoint}
        hints::Tuple{DataPoint}
        ouputs::Tuple{DataPoint}
        lengths::Vector{Float32}
    end

    function pyDataPoint_to_DataPoint(obj::PyCall.PyObject)
        return DataPoint(obj.name, obj.location, obj.type, obj.data.tolist())
    end

    function pyFeedback_to_Feedback(feedback::Tuple)
        @assert length(feedback) == 2
        @assert length(feedback[1]) == 3

        inputs = (pyDataPoint_to_DataPoint(dp) for dp in feedback[1][1])
        hints = (pyDataPoint_to_DataPoint(dp) for dp in feedback[1][2])
        outputs = (pyDataPoint_to_DataPoint(dp) for dp in feedback[2])
        lenghts = feedback[1][3].tolist()

        return Feedback(inputs, hints, outputs, lenghts)
    end

    function create_dataset(folder::String, algorithm::String, split::String, batch_size::Integer)
        clrs = pyimport("clrs")
        
        dataset, n_samples, spec = clrs.create_dataset(folder=folder, algorithm=algorithm, split=split, batch_size=batch_size)
        
        return Iterators.map(f -> pyFeedback_to_Feedback(f), dataset), n_samples, spec
    end
end