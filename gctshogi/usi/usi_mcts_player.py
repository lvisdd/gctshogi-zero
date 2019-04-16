from gctshogi.usi.usi import *
from gctshogi.player.mcts_player import *

def run():
    player = MCTSPlayer()
    usi(player)

if __name__ == '__main__':
    run()
