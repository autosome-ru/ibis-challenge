class IbisChallengeException(Exception):
    pass

class BenchmarkException(IbisChallengeException):
    pass

class WrongPathException(BenchmarkException):
    pass

class WrongDatasetTypeException(BenchmarkException):
    pass

class WrongBecnhmarkModeException(BenchmarkException):
    pass

class WrongExperimentTypeException(BenchmarkException):
    pass

class WrongCurationStatusException(BenchmarkException):
    pass

class BenchmarkConfigException(BenchmarkException):
    pass

class WrongProtocolException(BenchmarkException):
    pass

class WrongPRAUCTypeException(BenchmarkException):
    pass

class WrongScorerException(BenchmarkException):
    pass

class ModelNotTrainedException(BenchmarkException):
    pass

class ProtocolException(BenchmarkException):
    pass