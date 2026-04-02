import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time
from game import Board2048
from transformer import Board2048Transformer

class ExperienceReplayBuffer:
    """Store transitions for batched training"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, reward, next_state, is_terminal):
        """Add a transition to the buffer"""
        self.buffer.append((state, reward, next_state, is_terminal))
    
    def sample(self, batch_size):
        """Sample a random batch"""
        import random
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        
        states, rewards, next_states, terminals = zip(*batch)
        return (
            np.array(states),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(terminals, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


def play_game_collect_experience(model, device='cpu', epsilon=0.0):
    """
    Play one game and collect experience transitions
    
    Args:
        epsilon: Probability of random action (for exploration)
    
    Returns:
        experiences: List of (state, reward, next_state, is_terminal)
        game_score: Final score
        max_tile: Maximum tile achieved
        won: Whether 2048 was reached
    """
    board = Board2048()
    experiences = []
    game_score = 0
    
    while not board.is_game_over():
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Random move
            valid_moves = []
            for move in range(4):
                _, _, moved = board.get_afterstate(move)
                if moved:
                    valid_moves.append(move)
            if not valid_moves:
                break
            best_move = np.random.choice(valid_moves)
            best_afterstate, best_reward, _ = board.get_afterstate(best_move)
        else:
            # Greedy move from model -- BATCHED VERSION
            valid_moves = []
            afterstates = []
            rewards = []
            
            # Collect all valid moves
            for move in range(4):
                afterstate_grid, reward, moved = board.get_afterstate(move)
                if moved:
                    valid_moves.append(move)
                    afterstates.append(afterstate_grid.flatten())
                    rewards.append(reward)
            
            if not valid_moves:
                break
            
            # Single batched forward pass for all moves
            batch_tensor = torch.from_numpy(np.array(afterstates)).long().to(device)
            with torch.no_grad():
                values = model(batch_tensor).cpu().numpy()
            
            # Find best move
            total_values = np.array(rewards) + values
            best_idx = total_values.argmax()
            best_move = valid_moves[best_idx]
            best_afterstate = afterstates[best_idx].reshape(4, 4)
            best_reward = rewards[best_idx]
        
        # Store the afterstate (before random tile)
        afterstate = best_afterstate.flatten()
        
        # Make the move (adds random tile)
        board.move(best_move)
        game_score += best_reward
        
        # Next state after random tile
        next_state = board.grid.copy().flatten()
        is_terminal = board.is_game_over()
        
        # Store transition: afterstate -> reward -> next_state
        experiences.append((afterstate, best_reward, next_state, is_terminal))
        
        if is_terminal:
            break
    
    max_tile = 2 ** int(board.grid.max())
    won = max_tile >= 2048
    
    return experiences, game_score, max_tile, won


def train_batch_from_buffer(model, optimizer, replay_buffer, batch_size, device):
    """Train on a batch from replay buffer"""
    if len(replay_buffer) < batch_size:
        return None
    
    states, rewards, next_states, terminals = replay_buffer.sample(batch_size)
    
    # Convert to tensors and move to device
    states_tensor = torch.tensor(states, dtype=torch.long).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.long).to(device)
    terminals_tensor = torch.tensor(terminals, dtype=torch.float32).to(device)
    
    # Current value predictions
    current_values = model(states_tensor)
    
    # Next state values (with no_grad for target)
    with torch.no_grad():
        next_values = model(next_states_tensor)
        # Zero out next values for terminal states
        next_values = next_values * (1 - terminals_tensor)
    
    # TD target: r + V(s')
    targets = rewards_tensor + next_values
    
    # MSE loss
    loss = nn.MSELoss()(current_values, targets)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (helps stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def train_gpu_batched(
    num_games=1000000,
    games_per_collection=100,
    replay_capacity=100000,
    batch_size=256,
    train_batches_per_collection=10,
    learning_rate=0.0001,
    epsilon_start=0.1,
    epsilon_end=0.01,
    epsilon_decay_games=100000,
    embedding_dim=128,
    num_heads=8,
    num_layers=4,
    device='cuda',
    checkpoint_every=10000,
    log_every=100
):
    """
    Main training loop with GPU batching
    
    Args:
        num_games: Total games to play
        games_per_collection: Games to play before training
        replay_capacity: Size of replay buffer
        batch_size: Training batch size
        train_batches_per_collection: How many training batches per collection phase
        learning_rate: Adam learning rate
        epsilon_start/end/decay: Exploration schedule
        embedding_dim, num_heads, num_layers: Model architecture
        device: 'cuda' or 'cpu'
        checkpoint_every: Save model every N games
        log_every: Print stats every N games
    """
    
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Model
    model = Board2048Transformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity)
    
    # Tracking
    games_played = 0
    total_scores = []
    total_max_tiles = []
    total_wins = 0
    losses = []
    
    start_time = time.time()
    
    print(f"\nStarting training for {num_games:,} games...")
    print(f"Collection batch size: {games_per_collection}")
    print(f"Training batch size: {batch_size}")
    print("-" * 80)
    
    while games_played < num_games:
        # === COLLECTION PHASE ===
        # Play games and collect experiences
        collection_scores = []
        collection_max_tiles = []
        collection_wins = 0
        
        # Epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (games_played / epsilon_decay_games)
        )
        
        for _ in range(games_per_collection):
            experiences, score, max_tile, won = play_game_collect_experience(
                model, device=device, epsilon=epsilon
            )
            
            # Add experiences to buffer
            for exp in experiences:
                replay_buffer.add(*exp)
            
            # Track stats
            collection_scores.append(score)
            collection_max_tiles.append(max_tile)
            if won:
                collection_wins += 1
            
            games_played += 1
            
            if games_played >= num_games:
                break
        
        # === TRAINING PHASE ===
        # Train on replay buffer
        batch_losses = []
        if len(replay_buffer) >= batch_size:
            for _ in range(train_batches_per_collection):
                loss = train_batch_from_buffer(
                    model, optimizer, replay_buffer, batch_size, device
                )
                if loss is not None:
                    batch_losses.append(loss)
        
        # Update tracking
        total_scores.extend(collection_scores)
        total_max_tiles.extend(collection_max_tiles)
        total_wins += collection_wins
        if batch_losses:
            losses.extend(batch_losses)
        
        # === LOGGING ===
        if games_played % log_every == 0 or games_played >= num_games:
            elapsed = time.time() - start_time
            games_per_sec = games_played / elapsed
            
            # Recent stats (last 1000 games)
            recent_window = min(1000, len(total_scores))
            recent_scores = total_scores[-recent_window:]
            recent_max_tiles = total_max_tiles[-recent_window:]
            recent_wins = sum(1 for t in recent_max_tiles if t >= 2048)
            
            # Get unique max tiles and their counts
            unique_tiles, counts = np.unique(recent_max_tiles, return_counts=True)
            tile_dist = {int(tile): int(count) for tile, count in zip(unique_tiles, counts)}
            
            # Model diagnostics
            with torch.no_grad():
                # Check empty board value
                empty_board = torch.zeros(1, 16, dtype=torch.long).to(device)
                empty_value = model(empty_board).item()
                
                # Check embedding stats
                embedding_weights = model.tile_embedding.weight
                embedding_mean = embedding_weights.mean().item()
                embedding_std = embedding_weights.std().item()
                embedding_norm = embedding_weights.norm().item()
            
            print(f"\n{'='*80}")
            print(f"Games: {games_played:,}/{num_games:,} ({games_played/num_games*100:.1f}%)")
            print(f"Time: {elapsed/3600:.2f}h | Speed: {games_per_sec:.1f} games/sec | ETA: {(num_games-games_played)/games_per_sec/3600:.2f}h")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"\nRecent Performance (last {recent_window} games):")
            print(f"  Avg Score: {np.mean(recent_scores):.0f} ± {np.std(recent_scores):.0f}")
            print(f"  Win Rate: {recent_wins/recent_window*100:.2f}% ({recent_wins}/{recent_window})")
            print(f"  Max Tile Distribution: {tile_dist}")
            print(f"\nOverall:")
            print(f"  Total Wins: {total_wins} ({total_wins/games_played*100:.2f}%)")
            print(f"  Avg Score: {np.mean(total_scores):.0f}")
            
            if losses:
                recent_losses = losses[-100:]
                print(f"\nTraining:")
                print(f"  Recent Loss: {np.mean(recent_losses):.6f}")
                print(f"  Replay Buffer: {len(replay_buffer):,}/{replay_capacity:,}")
            
            print(f"\nModel Diagnostics:")
            print(f"  Empty Board Value: {empty_value:.4f}")
            print(f"  Embedding Mean: {embedding_mean:.4f}, Std: {embedding_std:.4f}, Norm: {embedding_norm:.2f}")
            print(f"{'='*80}")
        
        # === CHECKPOINTING ===
        if games_played % checkpoint_every == 0:
            checkpoint_path = f'checkpoint_{games_played}.pth'
            torch.save({
                'games_played': games_played,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_wins': total_wins,
                'avg_score': np.mean(total_scores[-1000:]) if len(total_scores) >= 1000 else np.mean(total_scores),
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = 'final_model.pth'
    torch.save({
        'games_played': games_played,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_wins': total_wins,
        'avg_score': np.mean(total_scores[-1000:]) if len(total_scores) >= 1000 else np.mean(total_scores),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    return model


def evaluate_model(model_path, num_games=100, device='cuda'):
    """Evaluate a trained model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model (adjust these if you used different architecture)
    model = Board2048Transformer(embedding_dim=128, num_heads=8, num_layers=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Trained for {checkpoint['games_played']:,} games")
    print(f"\nEvaluating on {num_games} games...")
    
    scores = []
    max_tiles = []
    wins = 0
    
    for i in range(num_games):
        _, score, max_tile, won = play_game_collect_experience(model, device=device, epsilon=0.0)
        scores.append(score)
        max_tiles.append(max_tile)
        if won:
            wins += 1
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_games} games completed...")
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results:")
    print(f"  Win Rate: {wins/num_games*100:.2f}% ({wins}/{num_games})")
    print(f"  Avg Score: {np.mean(scores):.0f} ± {np.std(scores):.0f}")
    print(f"  Max Score: {max(scores):.0f}")
    print(f"  Max Tile Distribution: {np.unique(max_tiles, return_counts=True)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Train
    model = train_gpu_batched(
        num_games=1000000,
        games_per_collection=100,
        batch_size=256,
        train_batches_per_collection=10,
        learning_rate=0.0001,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay_games=100000,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        device='cuda',
        checkpoint_every=10000,
        log_every=1000
    )
    
    # Evaluate
    print("\n\nEvaluating final model...")
    evaluate_model('final_model.pth', num_games=100, device='cuda')