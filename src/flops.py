TOTAL_FLOPS = 0 
OPTIMIZER_FLOPS = 0

def reset_flops():
    global TOTAL_FLOPS
    global OPTIMIZER_FLOPS
    TOTAL_FLOPS = 0
    OPTIMIZER_FLOPS = 0

def add_flops(n):
    global TOTAL_FLOPS
    TOTAL_FLOPS += n 
  
def get_flops():
    return TOTAL_FLOPS
  
def get_opt_flops():
    return OPTIMIZER_FLOPS
  
def add_opt_flops(n): 
    global OPTIMIZER_FLOPS 
    OPTIMIZER_FLOPS += n