def usi(player):
    while True:
        cmd_line = input()
        cmd = cmd_line.split(' ', 1)

        if cmd[0] == 'usi':
            player.usi()
        elif cmd[0] == 'setoption':
            option = cmd[1].split(' ')
            player.setoption(option)
        elif cmd[0] == 'isready':
            player.isready()
        elif cmd[0] == 'usinewgame':
            player.usinewgame()
        elif cmd[0] == 'position':
            moves = cmd[1].split(' ')
            player.position(moves)
        elif cmd[0] == 'go':
            # print(len(cmd))
            commands = cmd[1].split(' ') if len(cmd) > 1 else []
            player.go(commands)
        elif cmd[0] == 'ponderhit':
            player.ponderhit()
            # commands = []
            # player.go(commands)
        elif cmd[0] == 'stop':
            player.stop()
        elif cmd[0] == 'quit':
            player.quit()
            break
