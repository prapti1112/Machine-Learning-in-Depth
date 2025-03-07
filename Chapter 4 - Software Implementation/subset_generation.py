"""Implementation of Subset Generation using Complete Search. This code generates all the possible subsets of [0, n]. """

from logzero import logger

subset = []

def search(k, n):
    if k == n:
        logger.info(subset)
        return

    search(k+1, n)
    subset.append(k)
    search(k+1, n)
    subset.pop()

def get_subsets(n: int):
    """Generate all possible subsets in [0, n]

    Arguments:
        n -- limit
    """
    logger.info(f"List of subsets of [0, {n}):")
    search(0, n)

if __name__ == "__main__":
    n = 2
    get_subsets(n)
    