from time import time
from rich.console import Console

console = Console()

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

def timer(func):
    """
    Record execution time of any function with timer decorator
    Usage: just decorate a function when building it, the 
    decorator will be called every time the function is executed.
    # build the function
    @timer
    def some_function(some_arg):
        # do_something
        return 'foo'
        
    # call it
    some_function('boo')
    # output:
    >> Function 'some_function' finished after 0.01 seconds.
    """

    def wrapper(*args, **kwargs):
        start = time()
        results = func(*args, **kwargs)
        duration = time() - start
        console.print("Function [blue]{} [white]finished after {:.4f} seconds."\
              .format(func.__name__, duration))
        return results
    return wrapper