# 🚗 TRPO Agent for CarRacing-v3 🚗

This repository implements a **Trust Region Policy Optimization (TRPO)** agent with **RNN-based architecture** to solve the **CarRacing-v3** environment using `gymnasium`.


## 🌟 Overview
This project implements an **RNN-based TRPO agent** for reinforcement learning using PyTorch and Gymnasium. The agent is designed to navigate the CarRacing-v3 environment, learning to maximize the reward by improving driving strategies. 

### Key Features:
✅ **TRPO-based agent**: Uses trust region optimization for stable policy updates.  
✅ **Recurrent architecture**: Captures temporal dependencies using LSTM.  
✅ **CNN-based feature extraction**: Uses convolutional layers for spatial information processing.  
✅ **Frame stacking**: Stacks multiple frames to capture motion dynamics.  
✅ **Reward shaping**: Applies customized penalties and bonuses for better training.  
✅ **Exploration-exploitation trade-off**: Balances exploration using entropy regularization.  
✅ **Efficient off-track detection**: Uses image-based analysis to detect when the car is off-track.  

---

## 🚦 Environment: CarRacing-v3
We use the `CarRacing-v3` environment from OpenAI Gymnasium, which is a classic continuous control problem where an agent must learn to drive a car on a procedurally generated track.

### Installation:
```bash
pip install gymnasium
```

### **Objective**
- The goal is to complete the track while maximizing the total reward.
- The agent receives a negative reward for going off the track or making excessive turns.
- The agent receives a positive reward for staying on track, accelerating, and making smooth turns.

---

## 🛠️ State Space
The state is represented as an **RGB image** of the environment from a bird's-eye view.

### **Raw State:**  
- `(96, 96, 3)` → height, width, and color channels (RGB).  

### **Preprocessed State:**
1. Convert to grayscale (to reduce complexity):  
   $\text{Gray} = 0.299 R + 0.587 G + 0.114 B$
2. Normalize pixel values to `[0, 1]`:
   $
   x_{\text{normalized}} = \frac{x}{255.0}
   $
3. Stack last 4 frames to introduce temporal context:  
   Final state shape:
   $
   (4, 96, 96)
   $

### Why Frame Stacking?  
- A single frame does not provide motion information (e.g., velocity, direction).  
- Stacking frames helps the agent infer movement and momentum from changes across frames.

Example state tensor:
```
State shape: (4, 96, 96)
Frame 1 → t - 0
Frame 2 → t - 1
Frame 3 → t - 2
Frame 4 → t - 3
```

---

## 🎮 Action Space
The environment has **5 discrete actions**:

| Action | Description | Index |
|--------|-------------|-------|
| `Nothing` | No input | `0` |
| `Left` | Steer left | `1` |
| `Right` | Steer right | `2` |
| `Gas` | Accelerate | `3` |
| `Brake` | Apply brakes | `4` |

### **Action Encoding**  
Actions are represented as a **categorical distribution** over the 5 discrete actions.  
The policy outputs logits:
$
\pi(a|s) = \text{softmax}(W h_t)
$
where:
- $ W $ — policy weights
- $ h_t $ — hidden state from LSTM

Example action distribution:
```
[0.1, 0.2, 0.2, 0.4, 0.1]
```

### **Action Frequency Tracking**  
To avoid local minima where the agent relies too heavily on a small subset of actions, the agent:
- Tracks action frequencies.
- Applies entropy regularization to encourage action diversity.

---

## 🎯 Reward Structure
| Condition | Reward/Penalty |
|-----------|----------------|
| Stay on track | +0.3 per step |
| Off track | Penalty up to -3.0 |
| Driving straight | +0.2 |
| Correct turn at road edge | +0.8 |
| Excessive repetition of same action | -0.2 |
| Moderate speed | +0.3 |
| Braking | -0.1 |

### Reward Shaping Strategy:
1. **Positive reinforcement** for staying on the road.
2. **Penalties** for drifting off the track.
3. **Extra bonus** for making turns at the correct time.
4. **Entropy penalty** for over-exploration of the same action.

---

## 🏎️ Architecture

### 1. **CNN Feature Extractor** (Spatial Features)
The CNN processes the stacked frames and extracts spatial features:

| Layer | Output Shape | Kernel | Stride |
|-------|--------------|--------|--------|
| `Conv2d(4, 32)` | `(32, 32, 32)` | `6x6` | `3` |
| `Conv2d(32, 64)` | `(64, 15, 15)` | `4x4` | `2` |
| `Conv2d(64, 64)` | `(64, 13, 13)` | `3x3` | `1` |

**Attention Layer:**
- `Conv2d(64, 1)` → Applies a sigmoid-based attention mechanism:
$
\text{attention}(x) = \sigma(W x)
$

### 2. **LSTM Policy Network** (Temporal Features)
- Input: CNN-extracted features.
- Output: Action probabilities.
- LSTM hidden state:
$$
h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})
$$
- Output action probabilities:
$$
\pi(a|s) = \text{softmax}(W h_t)
$$

### 3. **Value Network** (State Value Estimation)
- Same architecture as policy network.
- Predicts the value of a state:
$$
V(s) = W_v h_t
$$

---

## 🔎 TRPO Optimization Strategy
TRPO solves the following constrained optimization problem:
$$
\max_{\theta} \mathbb{E}_{s, a} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s, a) \right]
$$

### 1. **Advantage Estimation (GAE)**
Advantage function using Generalized Advantage Estimation:
$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 2. **Conjugate Gradient**
Conjugate gradient solves:
$$
Ax = g
$$
where:
- $$ A $$ — Fisher Information Matrix
- $$ g $$  — Policy Gradient

### 3. **Backtracking Line Search**
Ensures that KL divergence constraint is satisfied:
$$
D_{KL}(\pi_{\theta} || \pi_{\theta_{\text{old}}}) < \delta
$$

### 4. **KL Divergence Calculation**
$$
D_{KL} = \sum_{i} p_i \log \frac{p_i}{q_i}
$$
where:
- $$ p_i $$ — current policy distribution
- $$ q_i $$ — old policy distribution

---

## 🏆 Training Procedure
### Hyperparameters:
| Hyperparameter | Value |
|---------------|-------|
| Learning rate | `1e-4` |
| Entropy coefficient | `0.01` |
| Discount factor (γ) | `0.99` |
| GAE lambda (λ) | `0.95` |
| KL constraint (δ) | `0.01` |
| Batch size | `32` |
| Max steps per episode | `1000` |

---

## 📊 Results
✅ Average reward: `> 900` after 100 episodes.  
✅ Completion rate: `> 95%`.  
✅ On-track percentage: `> 90%`.  

---

## 🎥 Demo
You can visualize the agent's behavior:
```bash
python visualize.py
```

---

## 🤝 Contributions
Feel free to open issues and submit pull requests!

---

## 📄 License
This project is licensed under the **MIT License**.
