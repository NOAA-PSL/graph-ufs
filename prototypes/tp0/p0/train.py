from graphufs.log import setup_simple_log

from config import P0Emulator as Emulator
from prototypes.tp0.train import train

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    setup_simple_log()
    train(Emulator)
