class NoSampleError(Exception):
    def __init__(self, var: str) -> None:
        super().__init__(f'Both control and treatment groups need to have samples, current variable {var} does not satisfy the conditions')

class VariableError(Exception):
    pass

class VariableNoFoundError(Exception):
    def __init__(self, var_list: set) -> None:
        super().__init__(f'Some of the features are not found in the given covariates: {var_list}')

class SampleError(Exception):
    pass