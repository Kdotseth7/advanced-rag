import random
import os

class Utils:
    @staticmethod
    def check_dir(directory) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    @staticmethod
    def get_random_number() -> int:
        return random.randint(1, 100)