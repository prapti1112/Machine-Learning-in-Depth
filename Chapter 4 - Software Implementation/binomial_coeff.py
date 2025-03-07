"""Implementation of code to find the binary coefficient of any term x^k from the expansion of (1+x)^n 
using the Dynamic Programing paradigm"""

import numpy as np
from logzero import logger

look_up = []

def nCk(n: int, k: int):
    global look_up
    if look_up[n][k]:
        return look_up[n][k]
    
    look_up[n][k] = nCk(n-1, k-1) + nCk(n-1, k)
    return look_up[n][k]


def binary_coeff(n: int, k: int):
    """Calculate the binary coefficient

    Arguments:
        n -- .
        k -- .

    Returns:
        _description_
    """
    return nCk(n, k)

if __name__ == "__main__":
    n, k = 5, 2

    look_up = np.zeros((n+1,n+1))
    np.fill_diagonal(look_up, 1)
    look_up[:, 0] = 1

    coeff = binary_coeff(n, k)    
    logger.info(f"Coefficient of x^{k} in (1 + x)^{n} = {coeff}\n\n")

    logger.debug(f"Look up table: \n{look_up}")