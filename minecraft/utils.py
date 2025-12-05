import random


class Utils:
    """
    Utility functions for the Minecraft GFlowNet.
    """
    def __init__(self):
        pass

    @staticmethod
    def randint_even(min_val, max_val):
        """
        Generates a random EVEN number between min_val and max_val.
        """
        num = random.randint(min_val, max_val)
        return num if num % 2 == 0 else num + 1