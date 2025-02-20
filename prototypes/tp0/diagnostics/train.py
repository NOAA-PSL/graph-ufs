import os
import logging

from graphufs.log import setup_simple_log

from config import TP0Emulator as Emulator
from prototypes.tp0.train import calc_stats, train

if __name__ == "__main__":

    
    setup_simple_log()

    stats_path = os.path.dirname(Emulator.norm_urls["mean"])
    if not os.path.isdir(stats_path):
        logging.info(f"Could not find {stats_path}, computing statistics...")
        calc_stats(Emulator)

    train(Emulator)
