# Current TODOs 

#TODO implement BenchmarkInstance, which:
# 1. Processes datasets according to config (forming train and test dirs and datasets)
# 2. Score model on datasets 
#TODO implement Scorer and it's subclasses:
# 1. RandomScorer 
# 2. Sklearn-based ROC and PR scores
# 3. Jan Grau package-based ROC and PR scores
#TODO implement Model and its subclasses:
# 1. RandomPredictor (totally random and class-balance based)
# 2. PerfectPredictor
#TODO implement MotifModel (through binding of C-package) 
#TODO validator for MotifModel file.
#TODO *_protocol_* functions must be changed to Protocol class and it's subclasses
#TODO Allow benchmark optionally run model training on train dataset (not to implement now)
#TODO add unique identifier to each object  