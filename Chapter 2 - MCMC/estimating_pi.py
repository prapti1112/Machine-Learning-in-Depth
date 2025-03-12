"""This file tries to estimate the value of π using Monte Carlo integration"""


import random
from logzero import logger


def estimate_pi(num_samples: int, radius: float):
    samples = []
    for _ in range(num_samples):
        x, y = random.uniform(-1*radius, radius), random.uniform(-1*radius, radius)
        samples.append(int(x**2 + y**2 <= radius**2) * 4 * radius**2)
    
    integration = sum(samples)/num_samples
    pi_approx = integration / (radius**2)

    return pi_approx


if __name__ == "__main__":
    RADIUS = 5
    for n in [ 10**i for i in range(1, 10) ]:
        pi = estimate_pi(n, RADIUS)
        logger.info(f"Number of samples: {n}, radius: {RADIUS}, Pi: {pi}")
