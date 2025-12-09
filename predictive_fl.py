import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar, minimize
import copy
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class User: 
    """Individual user with baseline behavior and sensitivity to model deployment."""
    user_id: int 
    group_id: int
    baseline: float  # μ_u: mean behavior (e.g., typical score)
    epsilon: float   # ε_u: sensitivity to deployed model (how impacted a user is by a change in content or recommendation)
    sigma_sq: float = 1.0  # noise variance

    def sample_behavior(self, theta: float, noise: float = 1.0) -> float:
        """Sample from D_u(θ): behavior ~ N(baseline + ε·θ, σ²)"""
        return np.random.normal(self.baseline + self.epsilon * theta, noise)
    
    def distribution_shift(self, theta: float) -> float:
        """Wasserstein-1 distance: W_1(D_u, D_u^θ) = |ε_u · θ|"""
        return abs(self.epsilon * theta)
    

@dataclass
class Client:
    """
    FL Client representing a sub-population (e.g., subreddit).
    
    In the Reddit context, each client is a subreddit with its own
    user base and engagement characteristics.
    """
    client_id: int
    name: str
    users: List[User] 
    weight: float            # p_i: population fraction
    epsilon: float           # mean sensitivity for this client
    theta_local: float = 0.0 # local model parameter

    # optional metadata (for Reddit data)
    num_comments: int = 0
    num_authors: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    votes_per_comment: float = 0.0

    @property
    def epsilon_std(self) -> float:
        """Within-client heterogeneity."""
        if len(self.users) == 0:
            return 0.0
        return np.std([u.epsilon for u in self.users])
    
    def sample_local_data(self, theta_deployed: float, batch_size: int, noise: float = 1.0) -> np.ndarray:
        """Sample batch from local distribution D_i(θ_deployed)."""
        if len(self.users) == 0:
            # fallback if no users
            return np.array([self.epsilon * theta_deployed])
        return np.array([
            np.random.choice(self.users).sample_behavior(theta_deployed, noise)
            for _ in range(batch_size)
        ])
    
    def total_shift(self, theta: float) -> float:
        """Total distributional shift for this client."""
        if len(self.users) == 0:
            return abs(self.epsilon * theta)
        return sum(u.distribution_shift(theta) for u in self.users) / len(self.users)


@dataclass
class Population:
    """
    Complete population with all clients/users.
    
    Can be created from synthetic data or real Reddit data.
    """
    clients: List[Client]
    source: str = "synthetic"  # "synthetic" or "Reddit"

    @property
    def all_users(self) -> List[User]:
        return [u for c in self.clients for u in c.users]

    @property
    def all_epsilons(self) -> np.ndarray:
        return np.array([u.epsilon for u in self.all_users])

    @property
    def client_epsilons(self) -> np.ndarray:
        return np.array([c.epsilon for c in self.clients])

    @property
    def client_weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.clients])

    def E_epsilon(self) -> float:
        """E[ε] = Σ α_i · ε_i"""
        return np.sum(self.client_weights * self.client_epsilons)

    def E_epsilon_sq(self) -> float:
        """E[ε²] = Σ α_i · ε_i²"""
        return np.sum(self.client_weights * self.client_epsilons**2)

    def Var_epsilon(self) -> float:
        """Var(ε) = E[ε²] - E[ε]²"""
        return self.E_epsilon_sq() - self.E_epsilon()**2

    def theta_PS(self) -> float:
        """
        Performative Stable solution for Gaussian mean estimation.
        θ^PS = Σ p_i m_i / (1 - ε̄)
        """
        if len(self.all_users) > 0:
            avg_baseline = np.mean([u.baseline for u in self.all_users])
        else:
            # fallback for clients without explicit users
            avg_baseline = 10.0
            
        avg_epsilon = self.E_epsilon()
        if avg_epsilon < 1.0:
            return avg_baseline / (1 - avg_epsilon)
        
        # divergence if our sensitivity >= 1 
        return float('inf')


# =============================================================================
# PERFORMATIVE POWER ANALYSIS
# =============================================================================

class PerformativePower:
    """ 
    Performative power as seen in Perdomo et al. (2020).
    Amplification is an extension of the linear case that we explore. 
    
    Handles three types of power computation:
    1. Power at specific θ (e.g., θ^PS)
    2. Supremum over θ ∈ Θ (constrained optimization)
    3. Optimal targeting power (FL advantage)
    """

    def __init__(self, population: Population):
        self.pop = population

    # power at a specific theta value distribution 
    def power_at_theta(self, theta: float) -> float:
        """
        P(θ) = Σ α_i · ε_i · |θ| / |u|

        Power when deploying model with parameter θ uniformly.
        """
        return sum(c.weight * c.epsilon * abs(theta) for c in self.pop.clients)

    def power_at_theta_PS(self) -> float:
        """Power at the performative stable solution."""
        theta_ps = self.pop.theta_PS()
        # if theta value diverges 
        if np.isinf(theta_ps):
            return float('inf')
        # return the power at that theta value 
        return self.power_at_theta(theta_ps)   

    def supremum_power(self, theta_max: float = 10.0, constraint: str = 'box') -> Tuple[float, float]:
        """
        P(F) = sup_{θ ∈ Θ} Σ α_i · ε_i · |θ|

        Args:
            theta_max: Upper bound on |θ| for box constraint
            constraint: 'box' (|θ| ≤ θ_max) or 'l2' (||θ||² ≤ 1)

        Returns:
            (optimal_theta, supremum_power)

        For linear shift model, power is linear in |θ|, so:
        - Box constraint: supremum at θ = ±θ_max
        - L2 constraint: supremum at θ = ±1
        """
        E_eps = self.pop.E_epsilon()

        if constraint == 'box':
            # Power = E[ε] · |θ|, maximized at θ = θ_max
            opt_theta = theta_max
            sup_power = E_eps * theta_max
        else:  # L2: ||θ||² ≤ 1 means |θ| ≤ 1
            opt_theta = 1.0
            sup_power = E_eps

        return opt_theta, sup_power 

    # Centralized Power for linear definition 
    def centralized_power(self, budget: float = 1.0) -> Tuple[float, float]:
        """
        Centralized: Must deploy uniform θ to all users.

        P_central = max_{|θ|≤budget} E[ε] · |θ| = E[ε] · budget

        Returns:
            (optimal_theta, power)
        """
        eps = self.pop.E_epsilon()
        return budget, eps * budget
    
    # Federated Learning Power for linear definition 
    def federated_power(self, budget: float = 1.0) -> Tuple[Dict[int, float], float]:
        """
        FL: Can deploy different θ_i per client.

        Optimization problem:
            max  Σ α_i · ε_i · |θ_i|
            s.t. Σ α_i · θ_i² ≤ budget² # L2 budget constraint  

        Solution via Lagrangian:
            θ_i* = ε_i · budget / √E[ε²]

        Achieves power:
            P_FL = √E[ε²] · budget

        Returns:
            (targeting_weights, power)
        """
        # E[ε²]
        eps_squared = self.pop.E_epsilon_sq()

        # sqrt(E[ε²])
        sqrt_eps_sq = np.sqrt(eps_squared)

        # client -> optimized θ_i
        targeting = {}
        for c in self.pop.clients:
            targeting[c.client_id] = c.epsilon * budget / sqrt_eps_sq

        # total performative power achieved under optimal θ_i
        power = sqrt_eps_sq * budget
        return targeting, power
    
    def amplification_factor(self) -> float:
        """
        Amplification = P_FL / P_central = sqrt(E[ε²]) / E[ε]
                      = sqrt(1 + Var(ε)/E[ε]²)
        """
        # E[ε]
        E_eps = self.pop.E_epsilon()
        # E[ε²]
        E_eps_sq = self.pop.E_epsilon_sq()
        return np.sqrt(E_eps_sq) / E_eps if E_eps > 0 else float('inf')

    def power_curve(self, theta_range: np.ndarray) -> np.ndarray:
        """Compute power at each θ in range."""
        return np.array([self.power_at_theta(t) for t in theta_range])

    def full_comparison(self, budget: float = 1.0) -> Dict:
        """Complete power analysis."""
        theta_ps = self.pop.theta_PS()

        # supremum (box constraint with budget)
        theta_sup, P_sup = self.supremum_power(theta_max=budget)

        # At θ^PS
        P_at_PS = self.power_at_theta(theta_ps) if not np.isinf(theta_ps) else float('inf')

        # Centralized vs FL
        theta_central, P_central = self.centralized_power(budget)
        targeting_fl, P_fl = self.federated_power(budget)

        # Theory
        E_eps = self.pop.E_epsilon()
        Var_eps = self.pop.Var_epsilon()
        amp_theory = np.sqrt(1 + Var_eps / E_eps**2) if E_eps > 0 else float('inf')

        return {
            'theta_PS': theta_ps,
            'P_at_PS': P_at_PS,
            'theta_supremum': theta_sup,
            'P_supremum': P_sup,
            'theta_central': theta_central,
            'P_central': P_central,
            'targeting_FL': targeting_fl,
            'P_FL': P_fl,
            'amplification_empirical': P_fl / P_central if P_central > 0 else float('inf'),
            'amplification_theory': amp_theory,
            'E_epsilon': E_eps,
            'Var_epsilon': Var_eps,
            'E_epsilon_sq': self.pop.E_epsilon_sq(),
            'fl_advantage_percent': (P_fl / P_central - 1) * 100 if P_central > 0 else float('inf')
        }
    

# =============================================================================
# TRAINING STATE AND P-FEDAVG
# =============================================================================

@dataclass
class TrainingState:
    """Track training metrics over rounds."""
    theta_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    distance_to_ps: List[float] = field(default_factory=list)
    consensus_error: List[float] = field(default_factory=list)
    power_history: List[float] = field(default_factory=list)


class PFedAvg:
    """
    Performative FedAvg (Jin et al. 2023).

    Converges to θ^PS at rate O(1/T).
    """

    def __init__(self, population: Population, E: int = 5, batch_size: int = 4):
        self.pop = population
        self.E = E
        self.batch_size = batch_size
        self.theta_global = 0.0
        self.theta_ps = population.theta_PS()
        self.power_calc = PerformativePower(population)
        self.state = TrainingState()
        self._record_metrics()

    # Theorem 3.1 from Jin et al. 2023 guarantees fasted convergence O(1/t) to O(1/T)
    def learning_rate(self, t: int, beta: float = 2.0, gamma: float = 10.0) -> float:
        """η_t = β / (t + γ) — diminishing for convergence."""
        return beta / (t + gamma)
    
    def _record_metrics(self):
        """Record current metrics."""
        self.state.theta_history.append(self.theta_global)
        
        # only record distance if theta_ps is finite
        if not np.isinf(self.theta_ps):
            self.state.distance_to_ps.append((self.theta_global - self.theta_ps)**2)
            
        self.state.loss_history.append(self._performative_loss())
        self.state.consensus_error.append(self._consensus_error())
        self.state.power_history.append(self.power_calc.power_at_theta(self.theta_global))

    # from section 4.1 in FedAvg
    def _performative_loss(self) -> float:
        """Σ p_i E_{Z~D_i(θ)}[1/2(θ - Z)²]"""
        loss = 0.0
        for c in self.pop.clients:
            # calculate client (subreddit) individual loss 
            client_loss = 0.0

            # for each user in client 
            for u in c.users:
                # µ = u + ε * θ
                shifted_mean = u.baseline + u.epsilon * self.theta_global

                # check u.sigma_sq if available, else 1.0
                variance = getattr(u, 'sigma_sq', 1.0) 
                
                # (θ - µ)² + σ²
                client_loss += (self.theta_global - shifted_mean)**2 + variance
                
            # Average over users in the client
            if len(c.users) > 0:
                client_loss /= len(c.users)
            
            # Apply client weight (p_i)
            loss += c.weight * client_loss

        # Apply the 1/2 factor from Section 4.1
        return 0.5 * loss

    # Section 3 Lemma B.2 
    def _consensus_error(self) -> float:
        """Σ p_i ||θ_i - θ̄||²"""
        return sum(
            c.weight * (c.theta_local - self.theta_global)**2
            for c in self.pop.clients
        )
    
    def broadcast(self):
        """Server broadcasts global model to local clients."""
        for c in self.pop.clients:
            c.theta_local = self.theta_global

    # Section 2.4 
    def aggregate(self, participation: str = "full", K: int = None, scheme: str = "I"):
        """
        Aggregate local models to update global model.
        
        Args:
            participation: 'full' or 'partial'
            K: number of clients to sample (for partial participation)
            scheme: 'I' (with replacement) or 'II' (without replacement)
        """
        # full participation
        if participation == "full":
            # θ^(t+1) = Σ (j=1 to N) p_j * θ^(t+1)_j
            self.theta_global = sum(
                c.weight * c.theta_local for c in self.pop.clients
            )

        # partial participation 
        else:
            # Scheme I vs II 
            if scheme == 'I':
                # θ^(t+1) = (1/K) * Σ (k in S_(t+1)) θ^(t+1)_k
                probs = [c.weight for c in self.pop.clients]
                indices = np.random.choice(len(self.pop.clients), K, replace=True, p=probs)
                self.theta_global = np.mean([self.pop.clients[i].theta_local for i in indices])
            else:  # Scheme II 
                # θ^(t+1) = Σ (k in S_(t+1)) (p_k * N / K) * θ^(t+1)_k
                indices = np.random.choice(len(self.pop.clients), K, replace=False)
                weights = np.array([self.pop.clients[i].weight for i in indices])
                weights /= weights.sum()
                self.theta_global = sum(
                    weights[j] * self.pop.clients[indices[j]].theta_local
                    for j in range(K)
                )

    # Section 2.4
    def run_round(self, round_num: int, participation: str = 'full',
                  K: int = None, scheme: str = 'I'):
        """One round of P-FedAvg."""
        # 1. Broadcast
        self.broadcast()
        theta_deployed = self.theta_global

        # 2. Local SGD (E steps)
        lr = self.learning_rate(round_num * self.E)
        for _ in range(self.E):
            for c in self.pop.clients:
                batch = c.sample_local_data(theta_deployed, self.batch_size)
                grad = c.theta_local - np.mean(batch)
                c.theta_local -= lr * grad

        # 3. Aggregate
        self.aggregate(participation, K, scheme)

        # 4. Record
        self._record_metrics()

    def train(self, num_rounds: int, participation: str = 'full',
              K: int = None, scheme: str = 'I', verbose: bool = False) -> TrainingState:
        """
        Train for num_rounds.
        
        Args:
            num_rounds: number of communication rounds
            participation: 'full' or 'partial'
            K: clients per round (for partial)
            scheme: 'I' or 'II' (for partial)
            verbose: print progress
        """
        if verbose:
            print(f"\nP-FedAvg: E={self.E}, batch={self.batch_size}")
            print(f"Target: θ^PS = {self.theta_ps:.4f}")
            print("-" * 50)
            
        # for each round 
        for r in range(num_rounds):
            # run the round 
            self.run_round(r, participation, K, scheme)
            
            # progress logging
            if verbose and (r + 1) % max(1, num_rounds // 5) == 0:
                d = self.state.distance_to_ps[-1] if self.state.distance_to_ps else 0
                print(f"Round {r+1:4d}: θ = {self.theta_global:8.4f}, ||θ-θ^PS||² = {d:.4f}")

        return self.state


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def create_synthetic_population(n_clients: int = 10, 
                                users_per_client: int = 50,
                                heterogeneity: str = 'medium',
                                within_client_var: float = 0.05,
                                seed: int = None) -> Population:
    """
    Create synthetic population with controlled heterogeneity.
    
    Args:
        n_clients: number of clients (subreddits)
        users_per_client: users per client
        heterogeneity: 'low', 'medium', or 'high'
        within_client_var: variance of user ε around client ε
        seed: random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # set to low sensitivity -> high sensitivity 
    het_scale = {'low': 0.1, 'medium': 0.3, 'high': 0.5}[heterogeneity]

    # clients 
    clients = []
    
    # for each client in the clients
    for c in range(n_clients):
        # set the sensitivity of the client
        client_epsilon = max(0.1, 0.5 + het_scale * np.random.randn())

        # for each user in user / client
        users = []
        for u in range(users_per_client):
            # set the user individual epsilon 
            user_epsilon = max(0.01, client_epsilon + within_client_var * np.random.randn())
            # add the user to our user list
            users.append(User(
                user_id=c * users_per_client + u,
                group_id=c,
                baseline=5.0 + 0.5 * np.random.randn(),
                epsilon=user_epsilon
            ))
        
        # add the client with the user group to the list of clients 
        clients.append(Client(
            client_id=c,
            name=f"Client_{c}",
            users=users,
            weight=1.0 / n_clients,
            epsilon=client_epsilon
        ))

    # return the population with the clients 
    return Population(clients=clients, source="synthetic")


def copy_population(population: Population) -> Population:
    """Create a deep copy of a population for independent experiments."""
    new_clients = []
    for c in population.clients:
        new_users = [
            User(
                user_id=u.user_id,
                group_id=u.group_id,
                baseline=u.baseline,
                epsilon=u.epsilon,
                sigma_sq=u.sigma_sq
            )
            for u in c.users
        ]
        new_clients.append(Client(
            client_id=c.client_id,
            name=c.name,
            users=new_users,
            weight=c.weight,
            epsilon=c.epsilon,
            theta_local=0.0,  # reset local theta
            num_comments=c.num_comments,
            num_authors=c.num_authors,
            mean_score=c.mean_score,
            std_score=c.std_score,
            votes_per_comment=c.votes_per_comment
        ))
    return Population(clients=new_clients, source=population.source)


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def experiment_power_analysis(population: Population, verbose: bool = True) -> Dict:
    """
    Complete performative power analysis.

    Computes:
    1. Power at θ^PS (equilibrium power) where the distribution is stable and converges 
    2. Supremum power (max achievable)
    3. Centralized vs FL power comparison
    """
    # compute the power using performative power 
    power = PerformativePower(population)
    # call comparison to see FL vs Central with our budget set to 1
    results = power.full_comparison(budget=1.0)

    # print results 
    if verbose:
        print("\n" + "=" * 70)
        print(" PERFORMATIVE POWER ANALYSIS")
        print("=" * 70)

        print(f"\n  Population Statistics:")
        print(f"    E[ε] = {results['E_epsilon']:.4f}")
        print(f"    E[ε²] = {results['E_epsilon_sq']:.4f}")
        print(f"    Var(ε) = {results['Var_epsilon']:.4f}")
        print(f"    θ^PS = {results['theta_PS']:.4f}")

        print(f"\n  Power at Different Operating Points:")
        print(f"    P(θ^PS) = {results['P_at_PS']:.4f}  [at equilibrium]")
        print(f"    P(θ_sup) = {results['P_supremum']:.4f}  [supremum, θ={results['theta_supremum']:.2f}]")

        print(f"\n  Centralized vs Federated (budget=1):")
        print(f"    P_central = {results['P_central']:.4f}  [uniform θ={results['theta_central']:.2f}]")
        print(f"    P_FL      = {results['P_FL']:.4f}  [optimal targeting]")
        print(f"    Amplification: {results['amplification_empirical']:.4f}x " +
              f"(theory: {results['amplification_theory']:.4f}x)")
        print(f"    FL advantage: +{results['fl_advantage_percent']:.1f}%")

        print("\n  FL Targeting Weights (θ_i = ε_i / √E[ε²]):")
        
        # print first 5 clients 
        for c in population.clients[:5]:  
            theta_i = results['targeting_FL'].get(c.client_id, 0)
            print(f"    {c.name:<20} ε={c.epsilon:.3f} → θ={theta_i:.3f}")
        
        # print that if there are more clients n more clients below 
        if len(population.clients) > 5:
            print(f"    ... and {len(population.clients)-5} more")

    return results


def experiment_convergence(population: Population, num_rounds: int = 200,
                           E: int = 5) -> Dict:
    """
    Compare P-FedAvg convergence: full vs partial participation.
    
    Uses the SAME population (via deep copies) for all three experiments
    to ensure fair comparison.
    """
    print("\n" + "=" * 70)
    print(" CONVERGENCE TO θ^PS (recreated using algorithm from Jin et al. 2023)")
    print("=" * 70)

    results = {}

    # full participation - use a copy of the passed population
    np.random.seed(42)
    pop_full = copy_population(population)
    
    # get pfed avg algo at full participation 
    pfedavg_full = PFedAvg(pop_full, E=E)
    # call training on the num of rounds 
    results['full'] = pfedavg_full.train(num_rounds, participation='full')
    # get the theta ps value converged to 
    results['theta_ps'] = pfedavg_full.theta_ps

    # Partial participation (Scheme I) - use a copy of the SAME population
    np.random.seed(42)
    pop_partial = copy_population(population)
    
    # calculate the partial p fed avg
    pfedavg_partial = PFedAvg(pop_partial, E=E)

    # run training with multiple rounds 
    results['partial_I'] = pfedavg_partial.train(num_rounds, participation='partial',
                                                  K=5, scheme='I')

    # Partial participation (Scheme II) - use a copy of the SAME population
    np.random.seed(42)
    pop_partial2 = copy_population(population)
    
    # scheme II training
    pfedavg_partial2 = PFedAvg(pop_partial2, E=E)
    results['partial_II'] = pfedavg_partial2.train(num_rounds, participation='partial',
                                                    K=5, scheme='II')

    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_power_comparison(power_results: Dict, population: Population, save_path: str = None):
    """
    Visualize performative power analysis results.
    
    Generates four subplots:
    1. Bar chart comparing power at different operating points
    2. Power curve P(θ) over parameter space
    3. Client sensitivities ε_i vs FL targeting weights θ_i
    4. Amplification factor breakdown with formula derivation
    """
    plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performative Power Analysis', fontsize=16)

    # ---------------------------------------------------------
    # Plot 1: Power at different operating points (Bar Chart)
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    
    points = ['P(θ^PS)', 'P_central', 'P_FL', 'P_supremum']
    values = [
        power_results['P_at_PS'],
        power_results['P_central'],
        power_results['P_FL'],
        power_results['P_supremum']
    ]
    colors = ['gray', 'steelblue', 'coral', 'gold']

    bars = ax1.bar(points, values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Performative Power')
    ax1.set_title('Power at Different Operating Points')
    
    ax1.axhline(
        y=power_results['E_epsilon'], 
        color='red', linestyle='--', alpha=0.6,
        label=f"E[ε]={power_results['E_epsilon']:.3f}"
    )
    ax1.legend()

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # ---------------------------------------------------------
    # Plot 2: Power curve over θ
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    
    thetas = np.linspace(-2, 2, 100)
    power_calc = PerformativePower(population)
    powers = np.array([power_calc.power_at_theta(t) for t in thetas])

    ax2.plot(thetas, powers, 'b-', linewidth=2, label='P(θ) = E[ε]·|θ|')
    
    theta_ps = power_results['theta_PS']
    if not np.isinf(theta_ps) and abs(theta_ps) <= 2:
        ax2.axvline(x=theta_ps, color='gray', linestyle='--',
                    label=f'θ^PS={theta_ps:.2f}')
    
    ax2.axvline(x=1.0, color='steelblue', linestyle=':', label='θ_central=1')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Curve P(θ)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ---------------------------------------------------------
    # Plot 3: Client sensitivities vs FL Targeting
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    
    display_limit = min(10, len(population.clients))
    client_names = [c.name for c in population.clients[:display_limit]]
    client_eps = [c.epsilon for c in population.clients[:display_limit]]

    x = np.arange(len(client_names))
    width = 0.35

    ax3.bar(x - width/2, client_eps, width, label='ε_i (sensitivity)', color='steelblue')

    targeting = power_results['targeting_FL']
    target_weights = [targeting.get(c.client_id, 0) for c in population.clients[:display_limit]]
    ax3.bar(x + width/2, target_weights, width, label='θ_i (FL targeting)', color='coral')

    ax3.axhline(y=power_results['E_epsilon'], color='red', linestyle='--', alpha=0.5,
                label=f'E[ε]={power_results["E_epsilon"]:.3f}')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(client_names, rotation=45, ha='right')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Client Sensitivities & FL Targeting (First {display_limit})')
    ax3.legend()

    # ---------------------------------------------------------
    # Plot 4: Amplification breakdown
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    
    E_eps = power_results['E_epsilon']
    Var_eps = power_results['Var_epsilon']
    amp = power_results['amplification_empirical']
    E_eps_sq_sqrt = np.sqrt(power_results['E_epsilon_sq'])

    comps = ['E[ε]\n(Centralized)', '√E[ε²]\n(Federated)']
    comp_vals = [E_eps, E_eps_sq_sqrt]
    
    ax4.bar(comps, comp_vals, color=['steelblue', 'coral'], edgecolor='black', width=0.5)
    ax4.set_ylabel('Power Factor')
    ax4.set_title(f'Amplification Factor: {amp:.3f}x')
    
    formula_text = (
        f"Amplification = P_FL / P_central\n"
        f"= √E[ε²] / E[ε]\n"
        f"= √(1 + Var(ε)/E[ε]²)\n"
        f"= √(1 + {Var_eps:.3f}/{E_eps**2:.3f})\n"
        f"= {amp:.4f}"
    )
    
    ax4.text(0.95, 0.5, formula_text, transform=ax4.transAxes, 
             ha='right', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax4.annotate('', xy=(1, E_eps_sq_sqrt), xytext=(0, E_eps),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5, ls='--'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved power analysis plot to {save_path}")
    else:
        plt.show()


def plot_convergence(results: Dict, save_path: str = None):
    """
    Plot convergence comparison for P-FedAvg (Jin et al. 2023).
    
    Generates three subplots:
    1. Distance to θ^PS over rounds (log scale)
    2. θ trajectory
    3. Power during training
    """
    plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('P-FedAvg Convergence (Jin et al. 2023)', fontsize=16)

    theta_ps = results.get('theta_ps', 0)
    
    full = results.get('full')
    part_I = results.get('partial_I')
    part_II = results.get('partial_II')
            
    # ---------------------------------------------------------
    # Plot 1: Distance to θ^PS (Log Scale)
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    if full and full.distance_to_ps: 
        ax1.semilogy(full.distance_to_ps, label='Full participation', linewidth=2)
    if part_I and part_I.distance_to_ps: 
        ax1.semilogy(part_I.distance_to_ps, label='Partial (Scheme I)', linewidth=2, alpha=0.8)
    if part_II and part_II.distance_to_ps: 
        ax1.semilogy(part_II.distance_to_ps, label='Partial (Scheme II)', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('||θ - θ^PS||² (Log Scale)')
    ax1.set_title('Convergence to Stable Solution')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # ---------------------------------------------------------
    # Plot 2: θ trajectory
    # ---------------------------------------------------------
    ax2 = axes[1]
    
    if full: 
        ax2.plot(full.theta_history, label='Full', linewidth=2)
    if part_I: 
        ax2.plot(part_I.theta_history, label='Scheme I', linewidth=2, alpha=0.8)
    if part_II: 
        ax2.plot(part_II.theta_history, label='Scheme II', linewidth=2, alpha=0.8)
    
    ax2.axhline(y=theta_ps, color='red', linestyle='--', linewidth=2, label='θ^PS (Target)')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('θ Global')
    ax2.set_title('Parameter Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Plot 3: Power during training
    # ---------------------------------------------------------
    ax3 = axes[2]
    
    if full: 
        ax3.plot(full.power_history, label='Full', linewidth=2)
    if part_I: 
        ax3.plot(part_I.power_history, label='Scheme I', linewidth=2, alpha=0.8)
    if part_II: 
        ax3.plot(part_II.power_history, label='Scheme II', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Performative Power P(θ)')
    ax3.set_title('Power Optimization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    else:
        plt.show()


# =============================================================================
# MAIN EXECUTION (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("\n" + "=" * 70)
    print(" PERFORMATIVE FEDERATED LEARNING: SYNTHETIC DATA TEST")
    print("=" * 70)

    # create synthetic population
    pop = create_synthetic_population(
        n_clients=10, 
        users_per_client=50,
        heterogeneity='medium'
    )
    
    # run power analysis
    results = experiment_power_analysis(pop)
    
    # run convergence experiment
    convergence_results = experiment_convergence(pop, num_rounds=200, E=5)
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  Final θ values after 200 rounds:")
    print(f"    Full participation: {convergence_results['full'].theta_history[-1]:.4f}")
    print(f"    Scheme I:          {convergence_results['partial_I'].theta_history[-1]:.4f}")
    print(f"    Scheme II:         {convergence_results['partial_II'].theta_history[-1]:.4f}")
    print(f"    Target θ^PS:       {convergence_results['theta_ps']:.4f}")