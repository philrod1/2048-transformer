import numpy as np

class Board2048:
    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=np.uint8)  # log2 values
        self.add_random_tile()
        self.add_random_tile()
        self.score = 0.0

    def __str__(self):
        """Show log2 values (internal representation)"""
        return "\n".join(" ".join(f"{cell:2d}" for cell in row) for row in self.grid)
    
    def add_random_tile(self):
        empty = list(zip(*np.where(self.grid == 0)))
        if not empty:
            return False
        pos = empty[np.random.randint(len(empty))]
        self.grid[pos] = 1 if np.random.random() < 0.9 else 2
        return True
    
    def slide_left(self, row):
        # Remove zeros
        row = row[row != 0]
        # Merge and track reward
        reward = 0.0
        i = 0
        while i < len(row) - 1:
            if row[i] == row[i+1]:
                row[i] += 1
                reward += 2 ** row[i]  # Reward is the value of the merged tile
                row = np.delete(row, i+1)
            i += 1
        # Pad with zeros
        return np.pad(row, (0, 4-len(row))), reward
    
    def slide(self, direction):
        old_grid = self.grid.copy()
        total_reward = 0.0
        
        if direction == 3:  # left
            for i in range(4):
                self.grid[i], reward = self.slide_left(self.grid[i])
                total_reward += reward
        elif direction == 1:  # right
            for i in range(4):
                row_reversed, reward = self.slide_left(self.grid[i][::-1])
                self.grid[i] = row_reversed[::-1]
                total_reward += reward
        elif direction == 0:  # up
            self.grid = self.grid.T
            for i in range(4):
                self.grid[i], reward = self.slide_left(self.grid[i])
                total_reward += reward
            self.grid = self.grid.T
        elif direction == 2:  # down
            self.grid = self.grid.T
            for i in range(4):
                row_reversed, reward = self.slide_left(self.grid[i][::-1])
                self.grid[i] = row_reversed[::-1]
                total_reward += reward
            self.grid = self.grid.T
        
        moved = not np.array_equal(old_grid, self.grid)
        return moved, total_reward
    
    def move(self, direction):
        moved, reward = self.slide(direction)
        if moved:
            self.add_random_tile()
            self.score += reward
        return moved
    
    def get_state(self):
        return self.grid.flatten()  # 16-element array for transformer
    
    def is_game_over(self):
        if 0 in self.grid:
            return False
        # Check if any moves possible
        for direction in range(4):
            _, _, moved = self.get_afterstate(direction)
            if moved:
                return False
        return True
    
    def get_score(self):
        # Score is the sum of all merged tiles (2^log2_value)
        # return np.sum((2 ** self.grid[self.grid > 1]) * (self.grid[self.grid > 1] - 1))
        return self.score

    def copy(self):
        """Deep copy for lookahead"""
        new_board = Board2048.__new__(Board2048)
        new_board.grid = self.grid.copy()
        new_board.score = self.score
        return new_board

    def get_afterstate(self, direction):
        """Get afterstate"""
        test_board = self.copy()
        moved, reward = test_board.slide(direction)
        return test_board.grid, reward, moved