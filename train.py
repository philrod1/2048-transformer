import torch
import torch.nn as nn
import numpy as np
from game import Board2048
from transformer import Board2048Transformer

def play_game(model, train=True, optimizer=None):
    """Play one game with TD-afterstate learning"""
    board = Board2048()
    game_score = 0
    move_count = 0
    
    while not board.is_game_over():
        # Evaluate all 4 possible moves
        best_value = -float('inf')
        best_move = None
        best_afterstate = None
        best_reward = 0
        
        for move in range(4):
            afterstate_grid, reward, moved = board.get_afterstate(move)
            if not moved:
                continue
            
            # Convert to tensor
            state_tensor = torch.tensor(afterstate_grid.flatten(), dtype=torch.long).unsqueeze(0)
            
            # Get value prediction
            with torch.no_grad():
                value = model(state_tensor).item()
            
            total_value = reward + value
            
            if total_value > best_value:
                best_value = total_value
                best_move = move
                best_afterstate = afterstate_grid
                best_reward = reward
        
        if best_move is None:
            break
        
        # Make the move
        current_afterstate = best_afterstate
        moved = board.move(best_move)
        game_score += best_reward
        move_count += 1
        
        # TD Learning update (if training)
        if train and optimizer is not None:
            # Get next best afterstate for TD target
            next_best_value = -float('inf')
            for next_move in range(4):
                next_afterstate_grid, next_reward, next_moved = board.get_afterstate(next_move)
                if not next_moved:
                    continue
                
                next_state_tensor = torch.tensor(next_afterstate_grid.flatten(), dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    next_value = model(next_state_tensor).item()
                
                if next_reward + next_value > next_best_value:
                    next_best_value = next_reward + next_value
            
            # If game over after this move, next value is 0
            if board.is_game_over():
                next_best_value = 0
            
            # TD update: target = reward + V(next_afterstate)
            current_state_tensor = torch.tensor(current_afterstate.flatten(), dtype=torch.long).unsqueeze(0)
            predicted_value = model(current_state_tensor)
            target_value = torch.tensor([best_reward + next_best_value], dtype=torch.float32)
            
            # Compute loss and update
            loss = nn.MSELoss()(predicted_value, target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    max_tile = 2 ** int(board.grid.max()) # Convert exponent back to tile value by first casting uint8 to int
    won = max_tile >= 2048
    
    return game_score, max_tile, won, move_count


def train_model(num_episodes=10000, learning_rate=0.001):
    """Train the transformer model"""
    model = Board2048Transformer(embedding_dim=64, num_heads=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scores = []
    win_count = 0
    
    for episode in range(num_episodes):
        score, max_tile, won, moves = play_game(model, train=True, optimizer=optimizer)
        scores.append(score)
        if won:
            win_count += 1
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            win_rate = win_count / (episode + 1)
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Score (last 100): {avg_score:.0f}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Last game: Score={score:.0f}, Max={max_tile}, Moves={moves}")
    
    return model


def evaluate_model(model, num_games=100):
    """Evaluate trained model"""
    scores = []
    max_tiles = []
    wins = 0
    
    for i in range(num_games):
        score, max_tile, won, moves = play_game(model, train=False, optimizer=None)
        scores.append(score)
        max_tiles.append(max_tile)
        if won:
            wins += 1
    
    print(f"\n--- Evaluation over {num_games} games ---")
    print(f"Win rate: {wins/num_games:.1%}")
    print(f"Avg score: {np.mean(scores):.0f}")
    print(f"Max tile distribution: {np.unique(max_tiles, return_counts=True)}")


if __name__ == "__main__":
    # Train
    print("Training transformer model...")
    model = train_model(num_episodes=10000, learning_rate=0.001)
    
    # Save model
    torch.save(model.state_dict(), '2048_transformer.pth')
    
    # Evaluate
    print("\nEvaluating trained model...")
    evaluate_model(model, num_games=100)