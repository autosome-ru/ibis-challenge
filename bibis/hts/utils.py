from copy import copy 
from ..logging import get_bibis_logger

logger = get_bibis_logger()

def dispatch_samples(cycle_counts: dict[str, int], 
                     sample_size: int):
    cycle_items = sorted(cycle_counts.items(), key=lambda x: x[1])
    
    cycle_assignments = {cycle: 0 for cycle in cycle_counts}
    
    total_size = sum(cycle_counts.values())
    if  total_size < sample_size:
        logger.warning(f"Can't sample request sample size: {sample_size}. Returning all counts")
        return copy(cycle_counts)
    elif total_size == sample_size:
        return copy(cycle_counts)
    
    rest = sample_size
    
    for ind, (cycle, size) in enumerate(cycle_items):
        to_sample, mod = divmod(rest, len(cycle_counts) - ind) 
        if size > to_sample:
            if mod > 0:
                to_sample = to_sample + 1
                mod -= 1     
        else: # size <= to_sample
            to_sample = size 
        
        cycle_assignments[cycle] = to_sample
        rest -= to_sample
        
            
    return cycle_assignments
    