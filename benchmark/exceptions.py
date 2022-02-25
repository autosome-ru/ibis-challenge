class IbisChallengeException(Exception):
    pass

class BenchmarkException(IbisChallengeException):
    pass

class WrongPathException(BenchmarkException):
    pass

class WrongDatasetModeException(BenchmarkException):
    pass

class WrongBecnhmarkModeException(BenchmarkException):
    pass

class WrongExperimentTypeException(BenchmarkException):
    pass

class WrongCurationStatusException(BenchmarkException):
    pass

class BenchmarkConfigException(BenchmarkException):
    pass