# distutils: language = c++
# cython: language_level=3
import cython

import shogi
import shogi.CSA
import copy

from gctshogi.features import *
from cshogi import *

import pickle
import _pickle as cPickle

# read kifu
def read_kifu(kifu_list_file):
    positions = []
    # cdef line, move
    with open(kifu_list_file, 'r') as f:
        for line in f.readlines():
            filepath = line.rstrip('\r\n')
            # kifu = shogi.CSA.Parser.parse_file(filepath)[0]
            try:
                # print(filepath)
                with open(filepath, encoding="utf-8") as f:
                    kifu = shogi.CSA.Parser.parse_str(f.read())[0]
                    # print(kifu)
                    win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
                    board = shogi.Board()
                    for move in kifu['moves']:
                        if board.turn == shogi.BLACK:
                            # piece_bb = copy.deepcopy(board.piece_bb)
                            # occupied = copy.deepcopy((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
                            # pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
                            piece_bb = cPickle.loads(pickle.dumps(board.piece_bb))
                            occupied = cPickle.loads(pickle.dumps((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])))
                            pieces_in_hand = cPickle.loads(pickle.dumps((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])))
                            
                        else:
                            piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                            occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
                            # pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))
                            pieces_in_hand = cPickle.loads(pickle.dumps((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])))
                        
                        # move label
                        move_label = make_output_label(shogi.Move.from_usi(move), board.turn)
                        
                        # result
                        win = 1 if win_color == board.turn else 0
                        
                        positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
                        board.push_usi(move)
            except Exception as e:
                print(e)
                print("skip -> " + filepath)
    return positions

def read_kifu_from_hcpe(hcpe_path, split_num=0):
    hcpes = np.fromfile(hcpe_path, dtype=HuffmanCodedPosAndEval)
    positions = []

    if split_num==1:
        n = len(hcpes) // 2
        hcpes = hcpes[:n]
    elif split_num==2:
        n = len(hcpes) // 2
        hcpes = hcpes[n:]
    else:
        pass
    
    for i, hcpe in enumerate(hcpes):
        cboard = Board()        
        cboard.set_hcp(hcpes[i]['hcp'])
        
        sfen = cboard.sfen()
        board = shogi.Board(sfen=sfen.decode('utf-8'))
        if board.turn == shogi.BLACK:
            # piece_bb = copy.deepcopy(board.piece_bb)
            # occupied = copy.deepcopy((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
            # pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
            piece_bb = cPickle.loads(pickle.dumps(board.piece_bb))
            occupied = cPickle.loads(pickle.dumps((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])))
            pieces_in_hand = cPickle.loads(pickle.dumps((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])))

        else:
            piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
            occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
            # pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))
            pieces_in_hand = cPickle.loads(pickle.dumps((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])))
        
        # move label
        move = hcpes[i]['bestMove16']
        move_label = make_output_label(shogi.Move.from_usi(move_to_usi(move).decode('utf-8')), board.turn)
        
        # result
        gameResult = hcpes[i]['gameResult']
        if board.turn == BLACK:
            if gameResult == BLACK_WIN:
                win_color = 1
            if gameResult == WHITE_WIN:
                win_color = -1
            else:
                win_color = 0
        else:
            if gameResult == BLACK_WIN:
                win_color = -1
            if gameResult == WHITE_WIN:
                win_color = 1
            else:
                win_color = 0

        win = 1 if win_color == board.turn else 0
        
        positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
        
    return positions
