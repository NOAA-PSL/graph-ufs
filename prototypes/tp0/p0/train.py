from graphufs.log import setup_simple_log

from config import P0Emulator as Emulator
from prototypes.tp0.train import train

if __name__ == "__main__":

    
    setup_simple_log()
    train(Emulator)
