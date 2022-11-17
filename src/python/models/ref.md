
### Model 1

- Actions: -1.0 - 0.5, -0.1, 0.0, 0.1, 0.5, 1.0
- Reward: 100 / (d_v^2 + d^2)
- Discount factor: .95
- NN Architecture: 32 (ELU) x 32 (ELU) x 6
- Learning rate: 1e-3
- Loss: MSE
- Batch size: 32
- Episodes: 2000
- Steps: 200
