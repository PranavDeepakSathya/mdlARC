TOTAL_FLOPS = 0 

def reset_flops():
    global TOTAL_FLOPS
    TOTAL_FLOPS = 0

def add_flops(n):
    global TOTAL_FLOPS
    TOTAL_FLOPS += n 
  
def get_flops():
    return TOTAL_FLOPS