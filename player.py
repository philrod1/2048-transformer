import numpy as np
from game import Board2048

class MonotonicityPlayer:
    """Not great, but it works"""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    
    def evaluate_board(self, grid):

        score = 0.0  # Use float because tiles are uint8 and easily overflow
        
        # Monotonicity
        for row in grid:
            inc = sum(1 for i in range(3) if row[i] <= row[i+1])
            dec = sum(1 for i in range(3) if row[i] >= row[i+1])
            score += max(inc, dec) * 100.0
        
        for col in range(4):
            column = grid[:, col]
            inc = sum(1 for i in range(3) if column[i] <= column[i+1])
            dec = sum(1 for i in range(3) if column[i] >= column[i+1])
            score += max(inc, dec) * 100.0
        
        # Empty cells
        score += float(np.sum(grid == 0)) * 100.0
        
        # Max tile in corner
        corners = [grid[0,0], grid[0,3], grid[3,0], grid[3,3]]
        max_val = grid.max()
        if max_val in corners:
            score += 1000.0
        
        # Smoothness
        smoothness = 0.0
        for i in range(4):
            for j in range(3):
                if grid[i,j] > 0 and grid[i,j+1] > 0:
                    smoothness -= abs(int(grid[i,j]) - int(grid[i,j+1]))
                if grid[j,i] > 0 and grid[j+1,i] > 0:
                    smoothness -= abs(int(grid[j,i]) - int(grid[j+1,i]))
        score += smoothness * 10.0
        
        return score
    
    def expectimax(self, board, depth, is_player_turn=True):
        """Simple expectimax search"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board.grid), None
        
        if is_player_turn:
            # Max node - player chooses best move
            best_score = -float('inf')
            best_move = None
            
            for move in range(4):
                afterstate_grid, reward, moved = board.get_afterstate(move)
                
                if not moved:
                    score = -1e9  # Massive penalty for trying not to move
                else:
                    temp_board = board.copy()
                    temp_board.grid = afterstate_grid.copy()
                    score, _ = self.expectimax(temp_board, depth - 1, False)
                    score += reward
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_score, best_move
        
        else:
            # Chance node - average over possible tile spawns
            empty_cells = list(zip(*np.where(board.grid == 0)))
            if not empty_cells:
                return self.evaluate_board(board.grid), None

            total_score = 0.0
            total_probability = 0.0

            for (row, col) in empty_cells:
                for tile_value in [1, 2]:  # Exponents for tiles 2 and 4
                    prob = (0.9 if tile_value == 1 else 0.1) / len(empty_cells)
                    
                    temp_board = board.copy()
                    temp_board.grid[row, col] = tile_value
                    score, _ = self.expectimax(temp_board, depth - 1, True)
                    
                    total_score += prob * score
                    total_probability += prob

            return total_score, None
    
    def get_move(self, board):
        """Get best move for current board"""
        _, move = self.expectimax(board, self.max_depth, True)
        # print("Best move:", move)
        return move


def play_game_with_ai(player, verbose=False):
    board = Board2048()
    total_score = 0
    move_count = 0
    
    while not board.is_game_over():
        move = player.get_move(board)
        if move is None:
            break
        
        moved, reward = board.slide(move)
        if moved:
            board.add_random_tile()
            total_score += reward
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}, Score: {total_score}")
                print(board)
                print()
    
    max_tile = 2 ** int(board.grid.max())
    won = max_tile >= 2048
    
    if verbose:
        print(f"\nGame Over!")
        print(f"Moves: {move_count}")
        print(f"Score: {total_score}")
        print(f"Max tile: {max_tile}")
        print(f"Grid values: {board.grid}")
        print(f"Grid max: {board.grid.max()}")
        print(f"Won: {won}")
    
    return total_score, max_tile, won


if __name__ == "__main__":

    player = MonotonicityPlayer(max_depth=7)
    wins = 0
    scores = []
    max_tiles = []
    
    for i in range(10):
        score, max_tile, won = play_game_with_ai(player, verbose=(i==0))
        scores.append(score)
        max_tiles.append(max_tile)
        if won:
            wins += 1
        print(f"Game {i+1}: Score={score}, Max={max_tile}, Won={won}")
    
    print(f"\n--- Results over 10 games ---")
    print(f"Win rate: {wins/10:.1%}")
    print(f"Avg score: {np.mean(scores):.0f}")
    print(f"Max tile distribution: {np.unique(max_tiles, return_counts=True)}")