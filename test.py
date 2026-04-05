import board2048_ext
from board2048_ext import Board

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)

def show_board(board: Board) -> None:
    representation = [2**board.to_list()[i] if board.to_list()[i] > 0 else 0 for i in range(16)]
    for i in range(4):
        row = representation[i*4:(i+1)*4]
        print(' '.join(f"{v:4}" for v in row))

# 1. Init
separator("1. Create new board")
board = Board()
show_board(board)
print("list :", board.to_list())

# 2. Sweep right
separator("2. Sweep right")
board.sweep(RIGHT)
show_board(board)

# 3. Place a tile
separator("3. Place a tile")
board.place_tile()
show_board(board)