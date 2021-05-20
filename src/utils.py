from time import time

def measure_exec_time(obj, *args, n_steps=100):
    """
    Measure the execution time of a function `obj`
    given args, when called `n_steps` times.
    """
    
    tic = time()
    for i in range(n_steps):
        obj(*args)
    diff = time() - tic
    print(f'Execution took {diff} seconds')