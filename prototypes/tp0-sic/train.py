from prototypes.tp0.train import train
from graphufs.log import setup_simple_log
from config import SICEmulator

if __name__ == "__main__":
    setup_simple_log()
    train(SICEmulator)
