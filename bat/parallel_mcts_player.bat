@echo off
call activate gctshogi-zero
python -m gctshogi.usi.usi_parallel_mcts_player 2>NUL
REM python -O -m gctshogi.usi.usi_parallel_mcts_player 2>NUL
