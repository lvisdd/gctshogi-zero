from gctshogi.usi.usi import *
from gctshogi.player.parallel_mcts_player import *

def run():
    player = ParallelMCTSPlayer()
    usi(player)

if __name__ == '__main__':
    run()
