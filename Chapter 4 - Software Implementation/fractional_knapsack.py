"""Implements fractional knapsack using greed approach"""
from queue import PriorityQueue
from logzero import logger

class Item:
    def __init__(self, profit: float, weight: float) -> None:
        self.profit = profit
        self.weight = weight
        self.pw_ratio = profit / weight
    
    def __lt__(self, obj: object):
        return self.pw_ratio > obj.pw_ratio # type: ignore
    
    def __str__(self) -> str:
        return f"Item(p/w: {round(self.pw_ratio, 2)} profit: {self.profit}, weight: {self.weight})"


def fractional_knapsack(profits: list[float], weights: list[float], capacity: float):
    """Solution to Greedy knapsack

    Arguments:
        profits -- .
        weights -- .
        capacity -- Total capacity of the sack

    Returns:
        total_profit -- Total profit achieved when greedy approach is used
    """
    item_queue = PriorityQueue(maxsize=len(profits)+1)
    for p, w in zip(profits, weights):
        item_queue.put(Item(p,w))
    
    current_item, total_profit = item_queue.get(), 0
    while capacity > current_item.weight:
        capacity -= current_item.weight
        total_profit += current_item.profit

        current_item = item_queue.get()
    
    total_profit +=  (capacity / current_item.weight) * current_item.profit
    return total_profit

if __name__ == "__main__":
    wt = [10.0, 20.0, 30.0] 
    val = [60.0, 100.0, 120.0] 
    capacity = 50

    final_profit = fractional_knapsack(val, wt, capacity)
    logger.info("Greedy Fractional Knapsack") 
    logger.info(f"Maximum profit: {final_profit}") 