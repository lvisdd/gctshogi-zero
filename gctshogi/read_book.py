import re
import shogi

ptn_sfen = re.compile(r"^sfen (?P<sfen>.*)$")
ptn_move = re.compile(r"^(?P<bestmove>\S+)\s(?P<nextmove>\S+)\s(?P<value>\S+)\s(?P<depth>\S+)\s(?P<num>\S+)$")

# read book
def read_book(book_file):
    with open(book_file, 'r', encoding='utf-8') as f:
        book = {}
        for line in f.readlines():
            line = line.strip()
            # print(line)
            m = ptn_sfen.match(line)
            if m:
                sfen=m.group("sfen")
                # print(line)
                # board = shogi.Board(sfen=sfen)
                # print(board.kif_str())
            else:
                # print(line)
                m = ptn_move.match(line)
                if m:
                    book[sfen] = line
                    # print(sfen + ":" + book[sfen])
                    # print("この局面での指し手1:" + m.group("bestmove"))
                    # print("相手の応手1:" + m.group("nextmove"))
                    # print("この指し手を指したときの評価値:" + m.group("value"))
                    # print("そのときの探索深さ:" + m.group("depth"))
                    # print("その指し手が選択された回数:" + m.group("num"))
                    if int(m.group("value")) >= 0:
                        # board.push_usi(m.group(1))
                        # print(board.kif_str())
                        # print("move")
                        pass
                    else:
                        # print("skip")
                        pass
                else:
                    # print("skip")
                    pass
    return book

if __name__ == '__main__':
    book = read_book("../book/yaneura_book4.db")
    sfen = "ln1g3nl/1r3kg2/p2sppspp/2pp2p2/1p5P1/2PP1PP2/PPS1PSN1P/2G4R1/LN2KG2L w Bb 26"
    # print(book[sfen])
    line = book[sfen]
    m = ptn_move.match(line)
    if m:
        print(sfen + ":" + book[sfen])
        print("この局面での指し手1:" + m.group("bestmove"))
        print("相手の応手1:" + m.group("nextmove"))
        print("この指し手を指したときの評価値:" + m.group("value"))
        print("そのときの探索深さ:" + m.group("depth"))
        print("その指し手が選択された回数:" + m.group("num"))
