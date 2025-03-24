# 🚗 **Car Racing with TRPO (Trust Region Policy Optimization)**  

This repository contains an implementation of a **Car Racing Agent** trained using **Trust Region Policy Optimization (TRPO)** in the `CarRacing-v3` environment from OpenAI Gym. The project focuses on developing a high-performance agent capable of efficiently navigating complex tracks using advanced reinforcement learning techniques.  

The repository includes:  
- **A TRPO-based agent** with a recurrent neural network (RNN) for long-term decision-making.  
- **Custom evaluation function** with forced initial acceleration to improve early episode dynamics.  
- **Advanced visualization tools** with real-time action and performance tracking.  
- **Training pipeline** with optimized hyperparameters for stability and exploration.  
- **Performance tracking** through reward curves, entropy analysis, KL divergence monitoring, and value loss plotting.  

---

## 🚀 **Problem Definition**  
The `CarRacing-v3` environment presents a challenging reinforcement learning task where the agent must learn to drive a car on procedurally generated tracks. The objective is to maximize the cumulative reward by maintaining control of the car while maximizing track coverage and minimizing off-track penalties.  

### **State Space**  
- The state is represented as an **(96 x 96 x 3)** RGB image (3 channels for color).  
- A frame-stacking technique is used to provide temporal context by combining 4 consecutive frames.  
- The state space is thus represented as:  
\[
s_t \in \mathbb{R}^{96 \times 96 \times 12}
\]

### **Action Space**  
The action space is **discrete** with the following possible actions:  
- `0` – No action  
- `1` – Turn left  
- `2` – Turn right  
- `3` – Accelerate (gas)  
- `4` – Brake  

### **Reward Function**  
- Positive reward for staying on the road and moving forward.  
- Negative reward for going off-track, hitting obstacles, or going in reverse.  

The total reward at each time step is computed as:  
\[
r_t = R_{\text{forward}} - R_{\text{off-track}} - R_{\text{obstacle-hit}}
\]

---

## 🧠 **TRPO: Theoretical Foundation**  
**Trust Region Policy Optimization (TRPO)** is a policy optimization algorithm that directly optimizes the policy while ensuring that updates do not deviate too far from the previous policy. The idea is to enforce a "trust region" that constrains the update step to avoid performance collapse caused by large policy updates.

---

### 🔎 **Objective Function**  
TRPO optimizes the following surrogate objective:  
\[
L(\theta) = \mathbb{E}_{t} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} A_t \right]
\]
where:  
- \( \pi_{\theta}(a_t | s_t) \) = New policy probability  
- \( \pi_{\theta_{\text{old}}}(a_t | s_t) \) = Old policy probability  
- \( A_t \) = Advantage function, which estimates how much better an action is compared to the baseline:  
\[
A_t = Q_t - V(s_t)
\]

---

### 🚨 **Trust Region Constraint**  
To avoid large updates that destabilize training, TRPO adds a constraint on the KL divergence between the old and new policies:  
\[
D_{\text{KL}}(\pi_{\theta_{\text{old}}} || \pi_{\theta}) \leq \delta
\]
where \( \delta \) is a small constant (e.g., 0.01–0.05).  

The final optimization problem becomes:  
\[
\max_{\theta} L(\theta) \quad \text{s.t.} \quad D_{\text{KL}}(\pi_{\theta_{\text{old}}} || \pi_{\theta}) \leq \delta
\]

---

### 📐 **Fisher Information Matrix and Natural Gradient**  
TRPO uses the Fisher Information Matrix (FIM) to compute the natural gradient:  
\[
F = \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)^T \right]
\]
Instead of using regular gradients, TRPO updates the policy using the natural gradient:  
\[
\theta_{k+1} = \theta_k + \frac{1}{\sqrt{F + \lambda I}} \nabla_{\theta} L(\theta)
\]
where \( \lambda \) is a damping factor to prevent numerical instability.

---

### 🚀 **Conjugate Gradient and Line Search**  
TRPO solves the constrained optimization using a conjugate gradient algorithm and backtracking line search to compute the step size that satisfies the KL divergence constraint.

The update step is computed as:  
\[
\Delta \theta = \sqrt{\frac{2 \delta}{s^T F s}} s
\]
where:  
- \( s \) = solution of the conjugate gradient method  
- \( F \) = Fisher Information Matrix  

---

### ✅ **Entropy Regularization**  
To encourage exploration, TRPO includes an entropy bonus:  
\[
H(\pi_{\theta}) = - \sum_a \pi_{\theta}(a|s) \log \pi_{\theta}(a|s)
\]
The final objective becomes:  
\[
L(\theta) + c H(\pi_{\theta})
\]
where \( c \) is a regularization coefficient.

---

## 🏎️ **Evaluation Function**  
The `evaluate()` function runs the agent over multiple episodes and computes:  
- **Episode Reward** – Sum of all rewards during the episode.  
- **Turn Percentage** – Ratio of turning actions (left or right) to total actions.  
- **Track Adherence** – Percentage of time the car stays on the track.  
- **Forced Acceleration** – Forces the agent to accelerate during the initial phase for stable dynamics.  

### **Forced Initial Acceleration**  
To improve early-stage performance, the agent is forced to accelerate for the first 20 steps:  
```python
initial_gas_steps = 20
for _ in range(initial_gas_steps):
    action = 3  # Gas
```

Example output:  
```
Eval Episode 1: Steps: 987, Reward: 520.45, Turn %: 19.1%, Avg Road %: 92.4%
```

---

## 🎯 **Visualization**  
The `visualize_agent()` function creates an animated video of the agent’s performance.  
- Forced initial acceleration included.  
- Action and state statistics displayed.  
- Progress bar and step-by-step updates.  

### Example Information:  
- **Step 23:** Gas, 92.1% road adherence, on track.  
- **Step 78:** Turn left, 85.3% road adherence, off track.  

---

## 🏋️‍♂️ **Training Pipeline**  
The `train.py` script trains the TRPO agent with the following hyperparameters:  

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| Discount Factor (γ) | 0.99 | Long-term rewards |
| GAE Lambda | 0.95 | Smoothing for advantage estimation |
| Policy LR | 1e-4 | Learning rate for policy network |
| Value LR | 3e-4 | Learning rate for value network |
| Entropy Coefficient | 0.07 | Encourages exploration |
| Max KL | 0.015 | KL divergence threshold |
| Damping | 0.2 | Stability for Fisher matrix inversion |

---

## 📈 **Performance Monitoring**  
The `plot_results()` function tracks and plots the following metrics:  
✅ Total reward per episode  
✅ Value loss over time  
✅ KL divergence  
✅ Policy entropy  

---

## 📂 **Project Structure**  
```
├── agent.py            # TRPO agent implementation
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation function
├── visualize.py        # Visualization function
├── environment.py      # Environment wrappers and handlers
├── model.py            # CNN + RNN policy model
├── utils.py            # Utility functions
├── README.md
├── requirements.txt
├── .gitignore
```

---

## 🏆 **Results**  
- ✅ Reached average score > 550  
- ✅ Turn percentage reduced from 30% → 19%  
- ✅ Track adherence improved from 85% → 92%  

---

## 🌟 **Next Steps**  
✅ Fine-tune entropy coefficient  
✅ Experiment with continuous action space  
✅ Add domain-specific augmentations  

---

**💡 Contributions welcome!** 😎
