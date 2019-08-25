import pyximport; pyximport.install()
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
from gctshogi.usi.usi import *
from gctshogi.player.parallel_mcts_player import *

def run():
    player = ParallelMCTSPlayer()
    usi(player)

if __name__ == '__main__':
    run()
