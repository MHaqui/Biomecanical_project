
### Model 1

- Actions: - 0.5, -0.1, 0.0, 0.1, 0.5
- Reward:
```
if (pos_distance**2).sum() < self.proximity_threshold:
            reward = hmean(1 / (pos_distance)**2 + 1e-8) / (vel_distance**4 + 1e-8)
        else:
            reward = -1e6
```
- Discount factor: .95
- NN Architecture: 32 (ELU) x 32 (ELU) x 5
- Learning rate: 1e-3
- Loss: MSE
- Batch size: 32
- Episodes: 1800
- Steps: 200

### Model 2

- Actions: - 0.5, -0.1, 0.0, 0.1, 0.5
- Reward:
```
if (pos_distance**2).sum() < self.proximity_threshold:
            reward = hmean(1 / (pos_distance)**2 + 1e-8) / (
                (10 * vel_distance)**4 + 1e-8)
        else:
            reward = -1e6
```
- Discount factor: .95
- NN Architecture: 32 (ELU) x 32 (ELU) x 5
- Learning rate: 1e-3
- Loss: MSE
- Batch size: 32
- Episodes: 1800
- Steps: 200
