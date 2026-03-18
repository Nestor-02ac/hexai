import sys
import os
import random
import time

# Añadimos las rutas para poder importar tus módulos
sys.path.insert(0, os.path.abspath('HexClassic'))
sys.path.insert(0, os.path.abspath('HexGumbel'))

try:
    from cmcts_hex import CMCTSHex as MCTS
    from mcts_hex import SimulationType
    print("Usando backend rapido Cython (CMCTSHex).")
except ImportError:
    from mcts_hex import MCTSHex as MCTS, SimulationType
    print("Usando backend Python (MCTSHex).")

try:
    from chex_board import CHexBoard as Board
except ImportError:
    from hex_board import HexBoard as Board

def play_games(num_games, board_size, sims, random_openings_range):
    wins_black = 0
    wins_white = 0

    for i in range(num_games):
        board = Board(board_size)
        current_player = 1

        # Apertura opcional
        n_random = random.choice(random_openings_range)
        for _ in range(n_random):
            legal = board.get_empty_cells()
            if not legal: break
            move = random.choice(legal)
            board.play(move, current_player)
            current_player = 3 - current_player

        # Partida MCTS vs MCTS
        # Dos bots con la misma inteligencia exacta
        bot_black = MCTS(board_size, c_uct=0.0, rave_bias=0.00025, use_rave=True, 
                         simulation_type=SimulationType.BRIDGES, num_simulations=sims)
        bot_white = MCTS(board_size, c_uct=0.0, rave_bias=0.00025, use_rave=True, 
                         simulation_type=SimulationType.BRIDGES, num_simulations=sims)

        while True:
            legal = board.get_empty_cells()
            if not legal: break

            if current_player == 1:
                move = bot_black.select_move(board, current_player)
            else:
                move = bot_white.select_move(board, current_player)

            board.play(move, current_player)

            if board.check_win(current_player):
                if current_player == 1:
                    wins_black += 1
                else:
                    wins_white += 1
                break
            current_player = 3 - current_player

    return wins_black, wins_white

if __name__ == "__main__":
    GAMES = 25
    SIZE = 7       # Tablero 7x7 (pequeño para que sea rápido)
    SIMS = 1000    # 1000 simulaciones para que jueguen de forma competente

    print("\nPRUEBA 1: EL MONOPOLIO (Tablero Vacio)")
    print(f"Enfrentando MCTS(Negras) vs MCTS(Blancas) - {GAMES} partidas...")
    b_wins, w_wins = play_games(GAMES, SIZE, SIMS, random_openings_range=[0])
    print(f"Victorias Negras (P1): {b_wins}")
    print(f"Victorias Blancas (P2): {w_wins}")
    print(f"Win Rate Blancas: {(w_wins/GAMES)*100}%")
    print("Conclusion: Sin aleatoriedad, las Blancas casi nunca ven como es ganar.")

    print("\nPRUEBA 2: DATOS EQUILIBRADOS (1 a 3 movs al azar)")
    print(f"Enfrentando MCTS(Negras) vs MCTS(Blancas) - {GAMES} partidas...")
    b_wins_r, w_wins_r = play_games(GAMES, SIZE, SIMS, random_openings_range=[1, 2, 3])
    print(f"Victorias Negras (P1): {b_wins_r}")
    print(f"Victorias Blancas (P2): {w_wins_r}")
    print(f"Win Rate Blancas: {(w_wins_r/GAMES)*100}%")
    print("Conclusion: Al inyectar caos inicial, las Blancas por fin obtienen partidas ganadas para aprender.\n")
