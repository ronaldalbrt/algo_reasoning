from algo_reasoning.src.specs import SPECS, Stage

def _get_algorithm_outputs(algorithm):
    """Get the expected output keys for the given algorithm."""

    algorithm_specs = SPECS[algorithm]
    for k, v in algorithm_specs.items():
        stage, _, _ = v

        if stage == Stage.OUTPUT:
            return k