import sys
import tty
import termios
from game import Board2048

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow key escape sequences
        if ch == '\x1b':
            ch = sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def play_interactive():
    """Play 2048 with arrow keys"""
    board = Board2048()
    
    print("Use arrow keys to play. Press 'q' to quit.")
    print(board)
    
    key_map = {
        '[A': 0,  # Up
        '[B': 2,  # Down
        '[C': 1,  # Right
        '[D': 3,  # Left
    }
    
    while not board.is_game_over():
        key = get_key()
        
        if key == 'q':
            print("\nQuitting...")
            break
        
        if key in key_map:
            move = key_map[key]
            moved = board.move(move)
            
            if moved:
                score = board.get_score()
                print("\033[2J\033[H")  # Clear screen, move cursor to top
                print(f"Score: {score}")
                print(board)
            else:
                print("Invalid move!")
    
    if board.is_game_over():
        print("\nGame Over!")
        print(f"Final Score: {board.get_score()}")
        max_tile = 2 ** int(board.grid.max())
        print(f"Max Tile: {max_tile}")


if __name__ == "__main__":
    play_interactive()
