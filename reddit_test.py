"""
Performative FL on Reddit Data
Perdomo et al. (2020) + Jin et al. (2023)

Dataset: reddit_combined_complete.csv (https://github.com/linanqiu/reddit-dataset.git)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from predictive_fl import (
    User, Client, Population, PerformativePower, PFedAvg, TrainingState,
    experiment_power_analysis, copy_population
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reddit_data(filepath: str, 
                     min_comments: int = 50, 
                     max_subreddits: int = 50) -> pd.DataFrame:
    """Load and filter Reddit data."""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # compute engagement metrics
    df['score'] = df['ups'] - df['downs']
    df['total_votes'] = df['ups'] + df['downs']
    df['upvote_ratio'] = np.where(df['total_votes'] > 0, df['ups'] / df['total_votes'], 0.5)

    # filter to active subreddits
    counts = df['subreddit'].value_counts()
    active = counts[counts >= min_comments].head(max_subreddits).index
    df = df[df['subreddit'].isin(active)].copy()

    print(f"  {len(df):,} comments | {df['subreddit'].nunique()} subreddits | {df['author'].nunique():,} authors")
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subreddit engagement statistics."""
    stats = df.groupby('subreddit').agg({
        'score': ['mean', 'std', 'median'],
        'ups': ['mean', 'sum'],
        'downs': ['mean', 'sum'],
        'total_votes': ['mean', 'sum'],
        'upvote_ratio': 'mean',
        'author': 'nunique',
        'authorkarma': 'mean',
        'authorlinkkarma': 'mean',
        'authorisgold': ['mean', 'sum'],
        'id': 'count'
    }).reset_index()

    stats.columns = [
        'subreddit', 'mean_score', 'std_score', 'median_score',
        'mean_ups', 'total_ups', 'mean_downs', 'total_downs',
        'mean_votes', 'total_votes', 'mean_upvote_ratio', 'num_authors',
        'mean_author_karma', 'mean_author_link_karma', 'gold_ratio', 'num_gold', 'num_comments'
    ]
    
    stats['std_score'] = stats['std_score'].fillna(0)
    stats['comments_per_author'] = stats['num_comments'] / stats['num_authors']
    stats['votes_per_comment'] = stats['total_votes'] / stats['num_comments']
    stats['score_variance'] = stats['std_score'] ** 2
    stats['weight'] = stats['num_comments'] / stats['num_comments'].sum()
    
    return stats


def derive_sensitivity(stats: pd.DataFrame, method: str = 'combined') -> pd.DataFrame:
    """Derive ε from engagement metrics. High ε = easily influenced."""
    def norm(x, lo=0.1, hi=0.9):
        return lo + (hi - lo) * (x - x.min()) / (x.max() - x.min() + 1e-9)

    if method == 'variance':
        stats['epsilon'] = norm(stats['score_variance'])
    elif method == 'engagement':
        stats['epsilon'] = norm(stats['votes_per_comment'])
    elif method == 'combined':
        raw = (0.4 * norm(stats['score_variance'], 0, 1) +
               0.3 * norm(stats['votes_per_comment'], 0, 1) +
               0.2 * norm(stats['comments_per_author'], 0, 1) +
               0.1 * norm(stats['mean_downs'] / (stats['mean_ups'] + 1), 0, 1))
        stats['epsilon'] = 0.1 + 0.8 * raw
    elif method == 'karma':
        stats['epsilon'] = norm(1 / (stats['mean_author_karma'] + 1))
    
    stats['epsilon'] = stats['epsilon'].clip(0.05, 0.95)
    return stats


# =============================================================================
# POPULATION CREATION
# =============================================================================

def create_population(stats: pd.DataFrame, df: pd.DataFrame, 
                      users_per_sub: int = 1000, within_var: float = 0.05) -> Population:
    """Create Population from Reddit data. 50 subs × 1000 users = 50,000 users."""
    clients = []
    
    for idx, row in stats.iterrows():
        sub_df = df[df['subreddit'] == row['subreddit']]
        base_eps = row['epsilon']
        
        # sample scores as user baselines
        scores = sub_df['score'].sample(min(users_per_sub, len(sub_df)), replace=True).values
        if len(scores) < users_per_sub:
            scores = np.concatenate([scores, np.random.normal(row['mean_score'], max(row['std_score'], 1), users_per_sub - len(scores))])

        users = [
            User(
                user_id=idx * users_per_sub + u,
                group_id=idx,
                baseline=float(scores[u]),
                epsilon=np.clip(base_eps + within_var * np.random.randn(), 0.05, 0.95)
            )
            for u in range(users_per_sub)
        ]
        
        clients.append(Client(
            client_id=idx, name=f"r/{row['subreddit']}", users=users,
            weight=row['weight'], epsilon=base_eps,
            num_comments=int(row['num_comments']), num_authors=int(row['num_authors']),
            mean_score=row['mean_score'], std_score=row['std_score'],
            votes_per_comment=row['votes_per_comment']
        ))

    return Population(clients=clients, source="Reddit")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(pop: Population, pwr: Dict, training: TrainingState = None, save_path: str = None):
    """6-panel visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Performative FL on Reddit', fontsize=14)
    
    sorted_c = sorted(pop.clients, key=lambda c: c.epsilon, reverse=True)[:15]
    names, eps = [c.name for c in sorted_c], [c.epsilon for c in sorted_c]
    tgt = pwr['targeting_FL']
    
    # 1. Top subreddits by ε
    ax = axes[0, 0]
    ax.barh(names, eps, color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names))), edgecolor='k')
    ax.axvline(pwr['E_epsilon'], color='red', ls='--', lw=2, label=f"E[ε]={pwr['E_epsilon']:.3f}")
    ax.set_xlabel('Sensitivity (ε)'); ax.set_title('Top 15 by Sensitivity'); ax.legend(); ax.invert_yaxis()

    # 2. ε vs FL targeting
    ax = axes[0, 1]
    x, w = np.arange(len(names)), 0.35
    ax.barh(x - w/2, eps, w, label='ε', color='steelblue')
    ax.barh(x + w/2, [tgt.get(c.client_id, 0) for c in sorted_c], w, label='θ* (FL)', color='coral')
    ax.set_yticks(x); ax.set_yticklabels(names); ax.set_xlabel('Value')
    ax.set_title('ε vs FL Targeting'); ax.legend(); ax.invert_yaxis()

    # 3. Power comparison
    ax = axes[0, 2]
    bars = ax.bar(['Centralized', 'Federated'], [pwr['P_central'], pwr['P_FL']], 
                  color=['steelblue', 'coral'], edgecolor='k', lw=2)
    ax.set_ylabel('Power'); ax.set_title(f"Amp: {pwr['amplification_empirical']:.3f}x (+{pwr['fl_advantage_percent']:.1f}%)")
    for b, v in zip(bars, [pwr['P_central'], pwr['P_FL']]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

    # 4. Scatter ε vs θ*
    ax = axes[1, 0]
    all_eps = [c.epsilon for c in pop.clients]
    all_theta = [tgt.get(c.client_id, 0) for c in pop.clients]
    sc = ax.scatter(all_eps, all_theta, s=[c.weight * 3000 for c in pop.clients], 
                    c=all_eps, cmap='RdYlBu_r', alpha=0.7, edgecolors='k')
    ax.plot([0, 1], [0, max(all_theta)], 'k--', alpha=0.3)
    ax.set_xlabel('ε'); ax.set_ylabel('θ*'); ax.set_title('FL Targets High-ε')
    plt.colorbar(sc, ax=ax, label='ε')

    # 5. Amplification formula
    ax = axes[1, 1]
    E_e, E_e2, V = pwr['E_epsilon'], pwr['E_epsilon_sq'], pwr['Var_epsilon']
    ax.bar(['E[ε]', '√E[ε²]'], [E_e, np.sqrt(E_e2)], color=['steelblue', 'coral'], edgecolor='k', lw=2)
    ax.set_ylabel('Value'); ax.set_title('Amp = √E[ε²] / E[ε]')
    ax.text(0.5, 0.95, f"E[ε]={E_e:.4f}\nE[ε²]={E_e2:.4f}\nVar={V:.4f}\nAmp={pwr['amplification_empirical']:.4f}",
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 6. Convergence
    ax = axes[1, 2]
    if training and training.distance_to_ps:
        ax.semilogy(training.distance_to_ps, 'b-', lw=2)
        ax.set_xlabel('Round'); ax.set_ylabel('||θ - θ^PS||²'); ax.set_title('Convergence'); ax.grid(alpha=0.3)
    else:
        ax.pie([E_e**2, V], labels=['E[ε]²', 'Var(ε)'], autopct='%1.1f%%', colors=['steelblue', 'coral'])
        ax.set_title(f'Var/E² = {V/E_e**2:.4f}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def print_summary(pop: Population, pwr: Dict):
    """Print results summary."""
    print(f"""
{'='*70}
 RESULTS: {pop.source} | {len(pop.clients)} subreddits | {len(pop.all_users):,} users
{'='*70}
 E[ε]={pwr['E_epsilon']:.4f}  E[ε²]={pwr['E_epsilon_sq']:.4f}  Var(ε)={pwr['Var_epsilon']:.4f}
 
 P_central = {pwr['P_central']:.4f}  |  P_FL = {pwr['P_FL']:.4f}
 Amplification: {pwr['amplification_empirical']:.4f}x (theory: {pwr['amplification_theory']:.4f}x)
 FL Advantage: +{pwr['fl_advantage_percent']:.2f}%
{'='*70}
 TOP TARGETED SUBREDDITS:""")
    for c in sorted(pop.clients, key=lambda x: x.epsilon, reverse=True)[:10]:
        print(f"   {c.name:<20} ε={c.epsilon:.3f}  θ*={pwr['targeting_FL'].get(c.client_id, 0):.3f}  votes/cmt={c.votes_per_comment:.1f}")


# =============================================================================
# CONVERGENCE EXPERIMENT
# =============================================================================

def reddit_convergence_experiment(population: Population, num_rounds: int = 200, E: int = 5) -> Dict:
    """
    Compare P-FedAvg convergence on Reddit data: full vs partial participation.
    
    Uses deep copies of the SAME population for fair comparison across:
    - Full participation
    - Partial participation Scheme I (with replacement)
    - Partial participation Scheme II (without replacement)
    """
    print("\n" + "=" * 70)
    print(" CONVERGENCE EXPERIMENT (Jin et al. 2023)")
    print("=" * 70)

    results = {}

    # full participation
    np.random.seed(42)
    pop_full = copy_population(population)
    pfedavg_full = PFedAvg(pop_full, E=E, batch_size=8)
    results['full'] = pfedavg_full.train(num_rounds, participation='full', verbose=True)
    results['theta_ps'] = pfedavg_full.theta_ps

    # partial participation (Scheme I) - sample with replacement
    np.random.seed(42)
    pop_partial_I = copy_population(population)
    pfedavg_partial_I = PFedAvg(pop_partial_I, E=E, batch_size=8)
    print("\nScheme I (partial, with replacement):")
    results['partial_I'] = pfedavg_partial_I.train(num_rounds, participation='partial', K=10, scheme='I', verbose=True)

    # partial participation (Scheme II) - sample without replacement
    np.random.seed(42)
    pop_partial_II = copy_population(population)
    pfedavg_partial_II = PFedAvg(pop_partial_II, E=E, batch_size=8)
    print("\nScheme II (partial, without replacement):")
    results['partial_II'] = pfedavg_partial_II.train(num_rounds, participation='partial', K=10, scheme='II', verbose=True)

    # summary
    print("\n" + "-" * 50)
    print(" Final θ values:")
    print(f"   Full:      {results['full'].theta_history[-1]:.4f}")
    print(f"   Scheme I:  {results['partial_I'].theta_history[-1]:.4f}")
    print(f"   Scheme II: {results['partial_II'].theta_history[-1]:.4f}")
    print(f"   Target:    {results['theta_ps']:.4f}")

    return results


def plot_reddit_convergence(results: Dict, save_path: str = None):
    """Plot convergence comparison for Reddit P-FedAvg."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('P-FedAvg Convergence (Jin et al. 2023) on Reddit Data', fontsize=14)

    theta_ps = results.get('theta_ps', 0)
    full = results.get('full')
    part_I = results.get('partial_I')
    part_II = results.get('partial_II')

    # 1. Distance to θ^PS
    ax = axes[0]
    if full and full.distance_to_ps:
        ax.semilogy(full.distance_to_ps, label='Full', lw=2)
    if part_I and part_I.distance_to_ps:
        ax.semilogy(part_I.distance_to_ps, label='Scheme I', lw=2, alpha=0.8)
    if part_II and part_II.distance_to_ps:
        ax.semilogy(part_II.distance_to_ps, label='Scheme II', lw=2, alpha=0.8)
    ax.set_xlabel('Round'); ax.set_ylabel('||θ - θ^PS||²')
    ax.set_title('Convergence (Log Scale)'); ax.legend(); ax.grid(alpha=0.3)

    # 2. θ trajectory
    ax = axes[1]
    if full: ax.plot(full.theta_history, label='Full', lw=2)
    if part_I: ax.plot(part_I.theta_history, label='Scheme I', lw=2, alpha=0.8)
    if part_II: ax.plot(part_II.theta_history, label='Scheme II', lw=2, alpha=0.8)
    ax.axhline(y=theta_ps, color='red', ls='--', lw=2, label='θ^PS')
    ax.set_xlabel('Round'); ax.set_ylabel('θ')
    ax.set_title('Parameter Trajectory'); ax.legend(); ax.grid(alpha=0.3)

    # 3. Power
    ax = axes[2]
    if full: ax.plot(full.power_history, label='Full', lw=2)
    if part_I: ax.plot(part_I.power_history, label='Scheme I', lw=2, alpha=0.8)
    if part_II: ax.plot(part_II.power_history, label='Scheme II', lw=2, alpha=0.8)
    ax.set_xlabel('Round'); ax.set_ylabel('Power P(θ)')
    ax.set_title('Performative Power'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(filepath: str, min_comments: int = 30, max_subreddits: int = 50,
                   users_per_sub: int = 1000, method: str = 'combined', 
                   rounds: int = 200, save_plots: bool = True,
                   run_convergence: bool = True) -> Dict:
    """Run full experiment."""
    # load and process
    df = load_reddit_data(filepath, min_comments, max_subreddits)
    stats = derive_sensitivity(compute_statistics(df), method)
    pop = create_population(stats, df, users_per_sub)
    
    # power analysis
    pwr = PerformativePower(pop).full_comparison()
    print_summary(pop, pwr)
    
    # convergence experiment (full vs partial participation)
    if run_convergence:
        convergence = reddit_convergence_experiment(pop, num_rounds=rounds, E=5)
        training = convergence['full']  # use full participation for main plot
        
        # plot convergence comparison
        if save_plots:
            plot_reddit_convergence(convergence, 'plots/reddit_convergence.png')
    else:
        # single training run (full participation only)
        print(f"\nP-FedAvg Training ({rounds} rounds)...")
        training = PFedAvg(pop, E=5, batch_size=8).train(rounds, verbose=True)
        convergence = None
    
    # plot power analysis
    plot_results(pop, pwr, training, 'plots/reddit_fl_analysis.png' if save_plots else None)
    
    return {
        'df': df, 
        'stats': stats, 
        'population': pop, 
        'power': pwr, 
        'training': training,
        'convergence': convergence
    }


if __name__ == "__main__":
    results = run_experiment(
        filepath='data/reddit_combined_complete.csv',
        min_comments=30,
        max_subreddits=50,      # all 50 subreddits
        users_per_sub=1000,     # 50 × 1000 = 50,000 users
        method='combined',
        rounds=200,
        run_convergence=True    # compare full vs partial participation
    )
    
    p = results['power']
    print(f"\n>>> FL achieves {p['amplification_empirical']:.2f}x power (+{p['fl_advantage_percent']:.1f}%)")
    print(f">>> Formula: Amp = √(1 + Var/E²) = {p['amplification_theory']:.4f}")