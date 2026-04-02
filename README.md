## What is it?

Just having a looksee if I can train a transformer model to play 2048
It's using TD afterstates described in (Szubert & Jaskowski, 2014)


## Files

- `game.py` - Board2048 implementation
- `transformer.py` - Transformer model architecture
- `train_gpu.py` - GPU-batched training loop
- `player.py` - Baseline player (monotonicity + expectimax)
- `interactive.py` - Keyboard controller for manual play


## Hypothesis

Transformers with learned embeddings might be able to:
1. Generalise to unseen tile values, unlike the N-Tuple work from (Szubert & Jaskowski, 2014)
2. Show smoother learning curves (no hard barriers at new tiles)
3. Learn spatial attention patterns that arise from being able to 'see' the whole board


## References

Szubert, M., & Jaskowski, W. (2014). Temporal Difference Learning of N-Tuple Networks for the Game 2048. *IEEE Conference on Computational Intelligence and Games*.


## Current status

- The board and basic player work.
- CPU training does something, but may or may not be learning anything.
- GPU training is as yet untried