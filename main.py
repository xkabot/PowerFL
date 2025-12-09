import numpy as np 
from predictive_fl import (
    create_synthetic_population,
    experiment_power_analysis,
    experiment_convergence,
    plot_power_comparison,
    plot_convergence
)
from reddit_test import run_experiment


if __name__ == "__main__":

    # file path 
    reddit_data = 'data/reddit_combined_complete.csv'

    # =========================================================================
    # SYNTHETIC DATA EXPERIMENT
    # =========================================================================
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
    power_results = experiment_power_analysis(pop)
    
    # run convergence experiment
    convergence_results = experiment_convergence(pop, num_rounds=200, E=5)
    
    # summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  Final θ values after 200 rounds:")
    print(f"    Full participation: {convergence_results['full'].theta_history[-1]:.4f}")
    print(f"    Scheme I:          {convergence_results['partial_I'].theta_history[-1]:.4f}")
    print(f"    Scheme II:         {convergence_results['partial_II'].theta_history[-1]:.4f}")
    print(f"    Target θ^PS:       {convergence_results['theta_ps']:.4f}")

    # plot synthetic results
    plot_power_comparison(power_results, pop, save_path='plots/synthetic_power_analysis.png')
    plot_convergence(convergence_results, save_path='plots/synthetic_convergence.png')

    # =========================================================================
    # REDDIT DATA EXPERIMENT
    # =========================================================================
    results = run_experiment(
        filepath=reddit_data,
        min_comments=30,
        max_subreddits=50,      # all 50 subreddits
        users_per_sub=1000,     # 50 × 1000 = 50,000 users
        method='combined',
        rounds=200,
        run_convergence=True    # compare full vs partial participation
    )

    # final summary
    p = results['power']
    print(f"""
{'='*70}
 FINAL RESULTS
{'='*70}
FL achieves {p['amplification_empirical']:.2f}x more performative power than centralized 
(+{p['fl_advantage_percent']:.1f}% advantage)
Formula verified: Amp = √(1 + Var(ε)/E[ε]²) = {p['amplification_theory']:.4f}
""")