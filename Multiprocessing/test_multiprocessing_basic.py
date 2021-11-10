## This file contains basic multiprocessing examples for my learning purposes
## Note to self: multiprocessing cannot be run in interactive python enviromnent (such as jupyter notebook)

__date__ = '7/28/2021'

import multiprocessing

# functions
def square(n):
    return n**2

def square_plus(n, k):
    return n**2 + 5*k


# Map function
def map_function(n_workers, nums):
    # Can only receive one argument
    # memory intensive

    with multiprocessing.Pool(processes=n_workers) as p:
        print(p.map(square, nums))


# Imap function
def imap_function(n_workers, nums):
    # Similar to map function, except it reduces memory usage
    # return iterable object rather than list

    with multiprocessing.Pool(processes=n_workers) as p:
        result = p.imap(square, nums)
        print([i for i in result])


# Starmap Function
def starmap_function(n_workers, Tuple):
    # similar to map function, except it can accept multiple argument

    with multiprocessing.Pool(processes=n_workers) as p:
        print(p.starmap(square_plus, Tuple))


if __name__ == '__main__':

    # Define number of workers
    n_workers = 4

    nums = [i for i in range(100)]
    Tuple_nums = [(i, i*2) for i in range(100)]

    # run map function
    #map_function(n_workers, nums)

    # run imap function
    #imap_function(n_workers, nums)

    # run starmap function
    starmap_function(n_workers, Tuple_nums)
