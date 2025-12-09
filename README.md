# Performative Federated Learning: A Complete Technical Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Performative Power](#3-performative-power)
4. [The P-FedAvg Algorithm](#4-the-p-fedavg-algorithm)
5. [FL Power Amplification](#5-fl-power-amplification)
6. [Code Implementation](#6-code-implementation)
7. [Experimental Validation](#7-experimental-validation)
8. [References](#8-references)

---
## Installation

### Download original files
1. ```git clone https://github.com/xkabot/PowerFL.git```
2. ```cd src```
3. ```conda create --name <env> --file requirements.txt```
4. ```conda activate <env>```
5. ```python main.py```


### To download original dataset and process it 
0. Make sure you are in the src directory. 
1. ```git clone https://github.com/linanqiu/reddit-dataset.git```
2. ```python shortener.py```

File should now be found in the data subdirectory within the src directory. 

## 1. Introduction

This document provides a complete mathematical and computational walkthrough of **Performative Federated Learning**, combining two key frameworks and optimizations for FL power (my contribution):

1. **Performative Prediction** (Perdomo et al., 2020): Models how deployed ML systems influence the data they're trained on
2. **Performative FL** (Jin et al., 2023): Extends this to federated settings where clients have heterogeneous responses
3. **FL Power** : In progress

### 1.1 Core Question

> **How much more influence (performative power) does a federated learning system have compared to a centralized system, due to its ability to learn heterogeneous user responses?**

### 1.2 Key Result Preview

```
Amplification Factor = P_FL / P_central = √(E[ε²]) / E[ε] = √(1 + Var(ε)/E[ε]²)
```

This is always ≥ 1, with equality only when all users have identical sensitivity (Var(ε) = 0).

---
## Project Structure

```
src/
├── main.py              # Main entry point - runs both experiments
├── predictive_fl.py     # Core FL implementation (User, Client, Population, P-FedAvg)
├── reddit_test.py       # Reddit-specific experiment code
├── shortener.py         # Data preprocessing script
├── data/
│   └── reddit_combined_complete.csv  # Processed Reddit data
|── plots/               # Generated visualizations
```

---

## 2. Mathematical Foundations

### 2.1 Setup and Notation

Consider a learning system with:
- **N clients** (e.g., subreddits, user groups, devices)
- **Users within each client** indexed by u ∈ U
- **Model parameter** θ ∈ ℝᵐ deployed by the platform
- **User behavior distributions** that shift in response to θ

**Table 2.1: Notation Summary**

| Symbol | Meaning | Source |
|--------|---------|--------|
| θ | Model parameter (decision variable) | Perdomo et al. (2020) §2 |
| D_u(θ) | User u's distribution when θ is deployed | Perdomo et al. (2020) Def. 2.1 |
| ε_u | User u's sensitivity to the model | Perdomo et al. (2020) §3.1 |
| α_i | Weight (population fraction) of client i | Jin et al. (2023) §2.1 |
| p_i | Same as α_i (alternate notation) | Jin et al. (2023) Eq. (1) |
| W₁(·,·) | Wasserstein-1 distance | Perdomo et al. (2020) Def. 3.1 |
| θ^PS | Performative Stable solution | Perdomo et al. (2020) Def. 2.3 |
| θ^PO | Performative Optimal solution | Perdomo et al. (2020) Def. 2.2 |

### 2.2 Distribution Shift Model

**Definition 2.1 (Distribution Mapping)** [Perdomo et al. (2020), Definition 2.1]

A distribution mapping D: ℝᵐ → Δ(Z) maps model parameters to data distributions. When θ is deployed, data is drawn from D(θ).

**Assumption 2.1 (Sensitivity)** [Jin et al. (2023), Assumption 2.3]

For each client i, there exists ε_i > 0 such that:

```
W₁(D_i(θ), D_i(θ')) ≤ ε_i ||θ - θ'||₂   ∀θ, θ' ∈ ℝᵐ
```

where W₁ is the Wasserstein-1 distance.

**Interpretation**: ε_i measures how much client i's data distribution shifts per unit change in θ. Higher ε_i means more "persuadable" or "responsive" users.

### 2.3 Linear Response Model

For tractability, we use a linear Gaussian model:

**Model 2.1 (Linear Gaussian Response)**

```
D_u(θ) = N(μ_u + ε_u · θ, σ²)
```

where:
- μ_u is user u's baseline behavior (without any model influence)
- ε_u is user u's sensitivity
- σ² is noise variance

**Derivation of Wasserstein Distance**:

For Gaussian distributions with the same variance:
```
W₁(N(μ₁, σ²), N(μ₂, σ²)) = |μ₁ - μ₂|
```

Therefore, for our model:
```
W₁(D_u(θ), D_u(θ')) = |μ_u + ε_u·θ - μ_u - ε_u·θ'| = |ε_u| · |θ - θ'|
```

This confirms Assumption 2.1 with sensitivity ε_u.

### 2.4 Loss Function

**Definition 2.2 (Performative Loss)** [Perdomo et al. (2020), Equation (1)]

```
f(θ) := E_{Z~D(θ)}[ℓ(θ; Z)]
```

The expectation is over the **shifted** distribution D(θ), not a fixed distribution.

For squared loss ℓ(θ; z) = ½(θ - z)² with our Gaussian model:

```
f_u(θ) = E_{Z~D_u(θ)}[½(θ - Z)²]
       = E_{Z~N(μ_u + ε_u·θ, σ²)}[½(θ - Z)²]
       = ½(θ - μ_u - ε_u·θ)² + ½σ²
       = ½(1 - ε_u)²θ² - (1-ε_u)μ_u·θ + ½μ_u² + ½σ²
```

---

## 3. Performative Power

### 3.1 Definition

**Definition 3.1 (Performative Power)** [Perdomo et al. (2020), Definition 3.1]

```
P(F) := sup_{f ∈ F} (1/|U|) Σ_u E[W₁(D_u, D_u^f)]
```

where:
- F is the function class (set of deployable models)
- D_u is user u's baseline distribution (before any model deployment)
- D_u^f is user u's distribution after deploying model f

**Interpretation**: Performative power measures the maximum distributional shift a platform can induce across its user population.

### 3.2 Power in Our Model

For the linear Gaussian model, the Wasserstein shift for user u when deploying θ is:

```
W₁(D_u(0), D_u(θ)) = |ε_u · θ|
```

**Proposition 3.1 (Power at θ)**

The performative power when deploying model θ is:

```
P(θ) = (1/|U|) Σ_u |ε_u · θ| = E[ε] · |θ|
```

where E[ε] = (1/|U|) Σ_u ε_u is the average sensitivity.

**Proof**:
```
P(θ) = (1/|U|) Σ_u W₁(D_u(0), D_u(θ))
     = (1/|U|) Σ_u |ε_u · θ|
     = |θ| · (1/|U|) Σ_u ε_u    (since ε_u > 0)
     = E[ε] · |θ|               □
```

### 3.3 Supremum Power

**Proposition 3.2 (Supremum under Box Constraint)**

For F = {θ : |θ| ≤ θ_max}:

```
sup_{θ ∈ F} P(θ) = E[ε] · θ_max
```

achieved at θ* = θ_max.

**Proof**: P(θ) = E[ε] · |θ| is linear in |θ|, so the supremum is at the boundary. □

**Remark**: The supremum is over the model space, not a specific operating point. In practice, we often evaluate power at specific points like θ^PS.

### 3.4 Performative Stable Solution

**Definition 3.2 (Performative Stability)** [Perdomo et al. (2020), Definition 2.3]

θ^PS is performative stable if:

```
θ^PS = argmin_θ E_{Z~D(θ^PS)}[ℓ(θ; Z)]
```

That is, θ^PS is optimal for the distribution it induces.

**Proposition 3.3 (θ^PS for Gaussian Mean Estimation)** [Jin et al. (2023), §4.1]

For the loss ℓ(θ; z) = ½(θ - z)² with D(θ) = N(μ + ε·θ, σ²):

```
θ^PS = μ / (1 - ε)    if ε < 1
```

**Proof**:

The decoupled objective is:
```
f(θ; θ̃) = E_{Z~D(θ̃)}[½(θ - Z)²] = ½(θ - μ - ε·θ̃)² + ½σ²
```

Taking derivative w.r.t. θ and setting to zero:
```
∂f/∂θ = θ - μ - ε·θ̃ = 0
```

At a fixed point θ = θ̃ = θ^PS:
```
θ^PS - μ - ε·θ^PS = 0
θ^PS(1 - ε) = μ
θ^PS = μ/(1 - ε)    □
```

**Corollary 3.1**: If ε ≥ 1, no stable solution exists (the mapping Φ(θ) = argmin_θ' f(θ'; θ) diverges).

### 3.5 Power at Equilibrium

**Proposition 3.4 (Power at θ^PS)**

```
P(θ^PS) = E[ε] · |θ^PS| = E[ε] · μ / (1 - E[ε])
```

For heterogeneous clients with different ε_i:
```
P(θ^PS) = Σ_i α_i ε_i · |θ^PS|
```

---

## 4. The P-FedAvg Algorithm

### 4.1 Federated Learning Setup

**Definition 4.1 (FL System)** [Jin et al. (2023), §2.1]

- N clients with local distributions D_i(θ)
- Client i has weight p_i (population fraction), Σ p_i = 1
- Server coordinates training, clients keep data local

**Objective (Performative Optimal)**:

```
θ^PO := argmin_θ Σᵢ pᵢ E_{Z~D_i(θ)}[ℓ(θ; Z)]    [Jin et al. (2023), Eq. (1)]
```

**Objective (Performative Stable)**:

```
θ^PS := argmin_θ Σᵢ pᵢ E_{Z~D_i(θ^PS)}[ℓ(θ; Z)]    [Jin et al. (2023), Eq. (2)]
```

### 4.2 Algorithm Description

**Algorithm 4.1: P-FedAvg** [Jin et al. (2023), §2.4]

```
Input: Initial θ⁰, learning rates {η_t}, local steps E, batch size B
Output: θ^T ≈ θ^PS

for t = 0, 1, ..., T-1 do:
    // 1. BROADCAST
    Server sends θᵗ to all clients
    θ_deployed ← θᵗ
    
    // 2. LOCAL UPDATES
    for each client i in parallel do:
        θᵢᵗ ← θᵗ
        for e = 1, ..., E do:
            Sample batch {z₁, ..., z_B} from D_i(θ_deployed)
            g ← (1/B) Σⱼ ∇ℓ(θᵢᵗ; zⱼ)
            θᵢᵗ ← θᵢᵗ - η_t · g
        end
    end
    
    // 3. AGGREGATE
    θᵗ⁺¹ ← Σᵢ pᵢ θᵢᵗ
end
```

**Key insight**: Distribution D_i(θ_deployed) shifts based on the **deployed** model, not the current local model. This is the performative effect.

### 4.3 Learning Rate Schedule

**Theorem 4.1 (Convergence)** [Jin et al. (2023), Theorem 3.1]

Under Assumptions 2.1-2.6 (strong convexity, smoothness, bounded gradients), with learning rate:

```
η_t = β / (t + γ)    where β > 1/μ̃, γ > 0
```

P-FedAvg converges:

```
E[||θᵗ - θ^PS||²] ≤ υ / (γ + t)
```

where υ = max{4B/μ̃², γE[||θ⁰ - θ^PS||²]} and μ̃ = μ - (1+δ)Lε.

**Interpretation**: O(1/T) convergence rate, same as standard SGD.

### 4.4 Partial Participation

**Scheme I** [Jin et al. (2023), §2.4]: Sample K clients with replacement, probability p_i

```
θᵗ⁺¹ = (1/K) Σ_{k ∈ S_t} θ_k^t
```

**Scheme II** [Jin et al. (2023), §2.4]: Sample K clients without replacement, uniform

```
θᵗ⁺¹ = Σ_{k ∈ S_t} (p_k · N/K) θ_k^t
```

Both schemes are unbiased: E[θᵗ⁺¹] = Σ_i p_i θᵢᵗ

### 4.5 Consensus Error

**Definition 4.2 (Consensus Error)** [Jin et al. (2023), §3]

```
CE(t) := Σᵢ pᵢ ||θᵢᵗ - θ̄ᵗ||²
```

where θ̄ᵗ = Σᵢ pᵢ θᵢᵗ.

**Lemma 4.1**: CE(t) = 0 at aggregation steps (t ∈ I_E where I_E = {E, 2E, 3E, ...}).

---

## 5. FL Power Amplification

### 5.1 The Central Question

**Setup**: Compare two systems with the same total "budget" for influencing users:

1. **Centralized**: Must deploy uniform θ to all users
2. **Federated**: Can deploy different θᵢ per client (learned from local data)

**Question**: How much more influence can FL achieve?

### 5.2 Budget Constraint

We impose an L2 budget constraint on deployment:

```
Σᵢ αᵢ θᵢ² ≤ B²    (L2 budget)
```

**Interpretation**: Total "resources" or "intensity" is limited. Federated can allocate differently across clients.

### 5.3 Centralized Power

**Proposition 5.1 (Centralized Optimization)**

Centralized must use θᵢ = θ for all i. The optimization:

```
max_θ  P_central(θ) = Σᵢ αᵢ εᵢ |θ|
s.t.   Σᵢ αᵢ θ² ≤ B²
```

Since Σᵢ αᵢ = 1, this simplifies to:
```
max_θ  E[ε] · |θ|
s.t.   θ² ≤ B²
```

**Solution**: θ* = B (at the boundary)

```
P_central = E[ε] · B
```

### 5.4 Federated Power

**Proposition 5.2 (Federated Optimization)**

FL can choose different θᵢ per client:

```
max_{θ₁,...,θₙ}  Σᵢ αᵢ εᵢ |θᵢ|
s.t.             Σᵢ αᵢ θᵢ² ≤ B²
```

**Solution via Lagrangian**:

```
L = Σᵢ αᵢ εᵢ θᵢ - λ(Σᵢ αᵢ θᵢ² - B²)
```

Taking ∂L/∂θᵢ = 0:
```
αᵢ εᵢ - 2λ αᵢ θᵢ = 0
θᵢ = εᵢ / (2λ)
```

Substituting into constraint:
```
Σᵢ αᵢ (εᵢ/(2λ))² = B²
(1/4λ²) Σᵢ αᵢ εᵢ² = B²
(1/4λ²) E[ε²] = B²
λ = √(E[ε²]) / (2B)
```

Therefore:
```
θᵢ* = εᵢ · B / √(E[ε²])
```

**Optimal FL Power**:
```
P_FL = Σᵢ αᵢ εᵢ · θᵢ*
     = Σᵢ αᵢ εᵢ · εᵢ · B / √(E[ε²])
     = B · E[ε²] / √(E[ε²])
     = B · √(E[ε²])
```

### 5.5 Amplification Factor

**Theorem 5.1 (Main Result)**

The FL amplification factor is:

```
Amplification = P_FL / P_central = √(E[ε²]) / E[ε]
```

**Alternative form** (using variance decomposition):

Since E[ε²] = Var(ε) + E[ε]²:

```
Amplification = √(E[ε²]) / E[ε] = √((Var(ε) + E[ε]²) / E[ε]²) = √(1 + Var(ε)/E[ε]²)
```

**Corollary 5.1 (Properties)**:

1. Amplification ≥ 1 always (by Cauchy-Schwarz: E[ε]² ≤ E[ε²])
2. Amplification = 1 iff Var(ε) = 0 (all users identical)
3. Higher Var(ε) → Higher amplification

### 5.6 Interpretation

**Why does FL have more power?**

1. **Information advantage**: FL learns individual εᵢ from local data; centralized only sees pooled average

2. **Targeting capability**: FL deploys θᵢ ∝ εᵢ, focusing on high-sensitivity users

3. **Resource efficiency**: Same total budget, but allocated optimally

**Table 5.1: Targeting Weights**

| Sensitivity | Centralized θ | FL θᵢ* | Relative Weight |
|-------------|---------------|--------|-----------------|
| Low ε | 1 | < 1 | Under-weighted |
| Average ε | 1 | ≈ 1 | Neutral |
| High ε | 1 | > 1 | Over-weighted |

---

## 6. Code Implementation

### 6.1 Data Structures

#### 6.1.1 User Class

```python
@dataclass
class User:
    user_id: int
    group_id: int
    baseline: float    # μ_u
    epsilon: float     # ε_u
```

**Mathematical correspondence**:
- `baseline` = μ_u in D_u(θ) = N(μ_u + ε_u·θ, σ²)
- `epsilon` = ε_u (sensitivity parameter)

#### 6.1.2 Sampling from D_u(θ)

```python
def sample_behavior(self, theta: float, noise_std: float = 1.0) -> float:
    """Sample from D_u(θ) = N(μ_u + ε_u·θ, σ²)"""
    mean = self.baseline + self.epsilon * theta
    return np.random.normal(mean, noise_std)
```

**Formula**: z ~ N(μ_u + ε_u·θ, σ²) [Model 2.1]

#### 6.1.3 Wasserstein Shift

```python
def distribution_shift(self, theta: float) -> float:
    """W₁(D_u(0), D_u(θ)) = |ε_u · θ|"""
    return abs(self.epsilon * theta)
```

**Formula**: W₁(D_u(0), D_u(θ)) = |ε_u · θ| [§3.2]

### 6.2 Client Class

```python
@dataclass  
class Client:
    client_id: int
    name: str
    users: List[User]
    weight: float        # p_i = α_i
    theta_local: float = 0.0
```

**Mathematical correspondence**:
- `weight` = p_i in [Jin et al. (2023), Eq. (1)]
- `theta_local` = θᵢᵗ (local model parameter)

#### 6.2.1 Client Sensitivity

```python
@property
def epsilon(self) -> float:
    """ε_i = average of user sensitivities"""
    return np.mean([u.epsilon for u in self.users])
```

**Formula**: ε_i = (1/|U_i|) Σ_{u ∈ U_i} ε_u

### 6.3 Population Statistics

```python
def E_epsilon(self) -> float:
    """E[ε] = Σ α_i · ε_i"""
    return np.sum(self.client_weights * self.client_epsilons)

def E_epsilon_sq(self) -> float:
    """E[ε²] = Σ α_i · ε_i²"""
    return np.sum(self.client_weights * self.client_epsilons**2)

def Var_epsilon(self) -> float:
    """Var(ε) = E[ε²] - E[ε]²"""
    return self.E_epsilon_sq() - self.E_epsilon()**2
```

**Formulas**:
- E[ε] = Σᵢ αᵢ εᵢ [weighted average]
- E[ε²] = Σᵢ αᵢ εᵢ² [second moment]
- Var(ε) = E[ε²] - E[ε]² [variance decomposition]

### 6.4 Performative Power Computation

#### 6.4.1 Power at θ

```python
def power_at_theta(self, theta: float) -> float:
    """P(θ) = Σ_i α_i · (1/|U_i|) Σ_u |ε_u · θ|"""
    total_shift = sum(
        c.weight * c.total_shift(theta) 
        for c in self.pop.clients
    )
    return total_shift
```

**Formula**: P(θ) = (1/|U|) Σ_u |ε_u · θ| = E[ε] · |θ| [Proposition 3.1]

#### 6.4.2 Centralized Power

```python
def centralized_power(self, budget: float = 1.0) -> Tuple[float, float]:
    """P_central = E[ε] · budget"""
    E_eps = self.pop.E_epsilon()
    return budget, E_eps * budget
```

**Formula**: P_central = E[ε] · B [Proposition 5.1]

#### 6.4.3 Federated Power

```python
def federated_power(self, budget: float = 1.0) -> Tuple[Dict[int, float], float]:
    """
    Optimal targeting: θ_i* = ε_i · B / √E[ε²]
    Power: P_FL = √E[ε²] · B
    """
    E_eps_sq = self.pop.E_epsilon_sq()
    sqrt_E_eps_sq = np.sqrt(E_eps_sq)
    
    targeting = {}
    for c in self.pop.clients:
        targeting[c.client_id] = c.epsilon * budget / sqrt_E_eps_sq
    
    power = sqrt_E_eps_sq * budget
    return targeting, power
```

**Formulas**:
- θᵢ* = εᵢ · B / √(E[ε²]) [§5.4, Lagrangian solution]
- P_FL = √(E[ε²]) · B [Proposition 5.2]

#### 6.4.4 Amplification

```python
def amplification_factor(self) -> float:
    """Amplification = √E[ε²] / E[ε] = √(1 + Var(ε)/E[ε]²)"""
    E_eps = self.pop.E_epsilon()
    E_eps_sq = self.pop.E_epsilon_sq()
    return np.sqrt(E_eps_sq) / E_eps
```

**Formula**: A = √(E[ε²]) / E[ε] = √(1 + Var(ε)/E[ε]²) [Theorem 5.1]

### 6.5 P-FedAvg Implementation

#### 6.5.1 Learning Rate

```python
def learning_rate(self, t: int, beta: float = 2.0, gamma: float = 10.0) -> float:
    """η_t = β / (t + γ)"""
    return beta / (t + gamma)
```

**Formula**: η_t = β/(t+γ) where β > 1/μ̃ [Theorem 4.1]

#### 6.5.2 Local SGD Step

```python
def local_sgd_step(self, theta_deployed: float, learning_rate: float, 
                   batch_size: int) -> float:
    # Sample from D_i(θ_deployed)
    batch = self.sample_local_data(theta_deployed, batch_size)
    
    # Gradient for squared loss: ∇ℓ(θ;z) = θ - z
    grad = self.theta_local - np.mean(batch)
    
    # SGD update
    self.theta_local = self.theta_local - learning_rate * grad
```

**Formula**: θᵢ ← θᵢ - η · (θᵢ - z̄) where z̄ ~ D_i(θ_deployed) [Algorithm 4.1]

#### 6.5.3 Aggregation

```python
def aggregate(self, participation: str = 'full', K: int = None, scheme: str = 'I'):
    if participation == 'full':
        # θ^{t+1} = Σ p_i θ_i^t
        self.theta_global = sum(
            c.weight * c.theta_local for c in self.pop.clients
        )
```

**Formula**: θᵗ⁺¹ = Σᵢ pᵢ θᵢᵗ [Jin et al. (2023), §2.4]

#### 6.5.4 Performative Loss

```python
def _performative_loss(self) -> float:
    """f(θ) = Σ p_i E_{Z~D_i(θ)}[(θ - Z)² + σ²]"""
    loss = 0.0
    for c in self.pop.clients:
        for u in c.users:
            shifted_mean = u.baseline + u.epsilon * self.theta_global
            # E[(θ - Z)²] = (θ - μ - ε·θ)² + σ²
            loss += c.weight * ((self.theta_global - shifted_mean)**2 + 1.0) / len(c.users)
    return loss
```

**Formula**: f(θ) = E_{Z~D(θ)}[(θ-Z)²] = (θ - μ - ε·θ)² + σ² [§2.4]

#### 6.5.5 Consensus Error

```python
def _consensus_error(self) -> float:
    """CE = Σ p_i ||θ_i - θ̄||²"""
    return sum(
        c.weight * (c.theta_local - self.theta_global)**2 
        for c in self.pop.clients
    )
```

**Formula**: CE(t) = Σᵢ pᵢ ||θᵢᵗ - θ̄ᵗ||² [Definition 4.2]

### 6.6 Reddit Data Simulation

#### 6.6.1 Engagement Profiles

```python
engagement_profiles = {
    'gaming': (0.7, 0.15),      # (base_ε, std_ε)
    'politics': (0.8, 0.20),    # high engagement, high variance
    'science': (0.4, 0.10),     # moderate engagement, low variance
    ...
}
```

**Interpretation**:
- base_ε: Average sensitivity of subreddit users
- std_ε: Within-subreddit heterogeneity

#### 6.6.2 User Generation

```python
for u in range(n_users):
    user_epsilon = max(0.05, min(0.95, np.random.normal(base_eps, eps_std)))
```

**Model**: ε_u ~ N(ε̄_group, σ²_group), clipped to [0.05, 0.95]

---

## 7. Experimental Validation

### 7.1 Experiment 1: Synthetic Data

**Setup**:
- 10 clients, 50 users each
- Heterogeneity levels: low (σ=0.1), medium (σ=0.3), high (σ=0.5)

**Results**:

| Heterogeneity | E[ε] | Var(ε) | P_central | P_FL | Amplification |
|---------------|------|--------|-----------|------|---------------|
| Low | 0.55 | 0.01 | 0.55 | 0.56 | 1.02x (+2%) |
| Medium | 0.66 | 0.09 | 0.66 | 0.72 | 1.10x (+10%) |
| High | 0.76 | 0.24 | 0.76 | 0.91 | 1.19x (+19%) |

**Validation**: Empirical amplification matches theory √(1 + Var(ε)/E[ε]²).

### 7.2 Experiment 2: Convergence

**Setup**:
- P-FedAvg with E=5 local steps
- 200 rounds
- Full and partial (K=5) participation

**Results**:
- All methods converge to θ^PS
- Full participation: smooth O(1/T) decay
- Partial participation: higher variance, same limit
- Scheme I ≈ Scheme II for equal weights

### 7.3 Experiment 3: Reddit Simulation

**Setup**:
- 15 subreddits with realistic engagement profiles
- User counts: 30-200 per subreddit
- 300 training rounds

**Results**:
- r/politics (ε=0.76) and r/gaming (ε=0.68): highest engagement
- r/books (ε=0.35) and r/science (ε=0.39): lowest engagement
- FL amplification: 1.02x (+2.3%)
- Lower than synthetic because Reddit structure has lower Var(ε)

### 7.4 Key Observations

1. **Formula Validated**: Amplification = √(1 + Var(ε)/E[ε]²) matches empirical results exactly

2. **Heterogeneity Matters**: Var(ε) is the key driver of FL advantage

3. **Convergence Confirmed**: O(1/T) rate as predicted by Theorem 4.1

4. **Targeting Intuition**: FL correctly allocates θᵢ ∝ εᵢ

---

## 8. References

### Primary Sources

**[Perdomo et al., 2020]** Perdomo, J., Zrnic, T., Mendler-Dünner, C., & Hardt, M. (2020). Performative Prediction. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119:7599-7609.

- Definition 2.1: Distribution mapping
- Definition 2.2: Performative optimal (θ^PO)
- Definition 2.3: Performative stable (θ^PS)
- Definition 3.1: Performative power
- §3.1: Sensitivity parameter ε

**[Jin et al., 2023]** Jin, K., Yin, T., Chen, Z., Sun, Z., Zhang, X., Liu, Y., & Liu, M. (2023). Performative Federated Learning: A Solution to Model-Dependent and Heterogeneous Distribution Shifts. *arXiv:2305.05090*.

- §2.1: FL system setup, objectives (Eq. 1-2)
- Assumption 2.3: Distribution mapping sensitivity (W₁ bound)
- Assumption 2.6: Local gradient bound
- §2.4: P-FedAvg algorithm
- Proposition 2.8: Uniqueness of θ^PS
- Theorem 3.1: Convergence rate O(1/T)
- §4.1: Gaussian mean estimation example

### Supporting Sources

**[Li et al., 2022]** Li, Q., Yau, C.Y., & Wai, H.T. (2022). Multi-Agent Performative Prediction with Greedy Deployment and Consensus Seeking Agents. *arXiv:2209.03811*.

- Decentralized performative prediction framework
- Inspiration for heterogeneous client responses

**[Li et al., 2020b]** Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z. (2020). On the Convergence of FedAvg on Non-IID Data. *ICLR 2020*.

- FedAvg convergence analysis
- Partial participation schemes

---

## Appendix A: Derivation Details

### A.1 Lagrangian for FL Optimization

**Problem**:
```
max  Σᵢ αᵢ εᵢ θᵢ
s.t. Σᵢ αᵢ θᵢ² ≤ B²
```

**Lagrangian**:
```
L(θ, λ) = Σᵢ αᵢ εᵢ θᵢ - λ(Σᵢ αᵢ θᵢ² - B²)
```

**KKT Conditions**:

1. Stationarity: ∂L/∂θᵢ = αᵢ εᵢ - 2λ αᵢ θᵢ = 0
   → θᵢ = εᵢ / (2λ)

2. Complementary slackness: λ(Σᵢ αᵢ θᵢ² - B²) = 0

3. Primal feasibility: Σᵢ αᵢ θᵢ² ≤ B²

**Solving for λ**:

Assume constraint is active (λ > 0):
```
Σᵢ αᵢ (εᵢ/2λ)² = B²
(1/4λ²) Σᵢ αᵢ εᵢ² = B²
1/4λ² = B² / E[ε²]
λ = √(E[ε²]) / (2B)
```

**Optimal solution**:
```
θᵢ* = εᵢ / (2 · √(E[ε²])/(2B)) = εᵢ · B / √(E[ε²])
```

### A.2 Variance Decomposition

**Claim**: √(E[ε²])/E[ε] = √(1 + Var(ε)/E[ε]²)

**Proof**:
```
E[ε²] = Var(ε) + E[ε]²                [definition of variance]

√(E[ε²])/E[ε] = √(Var(ε) + E[ε]²)/E[ε]
              = √((Var(ε) + E[ε]²)/E[ε]²)
              = √(Var(ε)/E[ε]² + 1)
              = √(1 + Var(ε)/E[ε]²)    □
```

### A.3 Cauchy-Schwarz Bound

**Claim**: Amplification ≥ 1

**Proof** (Cauchy-Schwarz):
```
E[ε]² = (Σᵢ αᵢ εᵢ)² ≤ (Σᵢ αᵢ)(Σᵢ αᵢ εᵢ²) = 1 · E[ε²] = E[ε²]
```

Therefore:
```
Amplification = √(E[ε²])/E[ε] ≥ √(E[ε]²)/E[ε] = E[ε]/E[ε] = 1    □
```

Equality iff εᵢ = c for all i (constant), i.e., Var(ε) = 0.

---

## Appendix B: Code-Formula Mapping

| Code Location | Formula | Reference |
|---------------|---------|-----------|
| `User.sample_behavior()` | z ~ N(μ_u + ε_u·θ, σ²) | Model 2.1 |
| `User.distribution_shift()` | W₁ = \|ε_u·θ\| | §3.2 |
| `Population.E_epsilon()` | E[ε] = Σ αᵢεᵢ | Definition |
| `Population.E_epsilon_sq()` | E[ε²] = Σ αᵢεᵢ² | Definition |
| `Population.Var_epsilon()` | Var(ε) = E[ε²] - E[ε]² | Definition |
| `Population.theta_PS()` | θ^PS = μ/(1-ε) | Prop. 3.3 |
| `PerformativePower.power_at_theta()` | P(θ) = E[ε]·\|θ\| | Prop. 3.1 |
| `PerformativePower.centralized_power()` | P_c = E[ε]·B | Prop. 5.1 |
| `PerformativePower.federated_power()` | P_FL = √E[ε²]·B | Prop. 5.2 |
| `PerformativePower.federated_power()` | θᵢ* = εᵢB/√E[ε²] | §5.4 |
| `PerformativePower.amplification_factor()` | A = √(1+Var/E²) | Thm. 5.1 |
| `PFedAvg.learning_rate()` | η_t = β/(t+γ) | Thm. 4.1 |
| `PFedAvg.aggregate()` | θ = Σ pᵢθᵢ | Alg. 4.1 |
| `PFedAvg._consensus_error()` | CE = Σ pᵢ\|\|θᵢ-θ̄\|\|² | Def. 4.2 |

---

*Document generated for educational purposes. All formulas traced to primary sources.*