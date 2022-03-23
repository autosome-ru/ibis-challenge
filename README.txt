# Current TODOs 
#TODO refactor dataset in a way it can be read in write in benchmark inner format (to avoid storing all datasets during model evaluation)
#TODO refactor benchmark and benchmarkconfig in the way not requiring to preprocess datasets each time benchmark is run 
#TODO how to write perfect submission? 
#TODO create UniqueTagger unique identifier to each object (required for dataset types different from PBM)
#TODO validate uniqueness of datasets names
#TODO implement automated install of PWMEval 
#TODO Possibly redesign ProtocolType and ExperimentalType in a way new protocol can be added by user without protocol.py/experiment.py modification
#     This can be done through main class knowing and instantiating it's subclasses