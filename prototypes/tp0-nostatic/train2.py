from prototypes.tp0.train import train
from graphufs.log import setup_simple_log
from config import NoHeightEmulator as Emulator

if __name__ == "__main__":
    setup_simple_log()
    train(Emulator)
