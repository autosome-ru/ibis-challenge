# Current TODO 
#TODO add label to experiment - to tell the source, TF and etc? 
#TODO add unique identifier to each object
#TODO implement BenchmarkInstance, which:
# 1. Loads benchmark config file
# 2. Processes datasets according to config (formiing train and test dirs and datasets)
# 3. Optionally run model training on train dataset (not to implement now)  
# 4. Score model on datasets 
#TODO implement Scorer and it's subclasses:
# 1. RandomScorer 
# 2. Sklearn-based ROC and PR scores
# 3. Jan Grau package-based ROC and PR scores
#TODO implement Model and its subclasses:
# 1. RandomPredictor (totally random and class-balance based)
# 2. PerfectPredictor
#TODO implement MotifModel (through binding of C-package) 
#TODO validator for MotifModel file.