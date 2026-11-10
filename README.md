# Contextual Non-Stationary Bandits

Comparing three bandit algorithms on a simulated recommendation task where the environment changes over time, including one sudden shock halfway through.

**Algorithms:** Epsilon-Greedy, LinUCB, Sliding-Window Linear Thompson Sampling.

## The setup
- 20,000 steps, 10 arms (e.g. 10 ads or offers), 5-dimensional context
- Reward is a click (1) or no click (0), drawn from a logistic model
- Parameters drift slowly and are fully replaced at step 10,000 (concept drift)
- Regret is measured against an oracle that knows the true click probabilities

## Results
| Algorithm | Total regret (lower is better) | Mean CTR |
|---|---:|---:|
| Thompson Sampling (window=400) | **1411** | **0.882** |
| Epsilon-Greedy | 1552 | 0.877 |
| LinUCB | 1752 | 0.868 |

Thompson Sampling starts slower, but after the shock it adapts best (CTR 0.931 vs 0.866 for Epsilon-Greedy), because its sliding window forgets outdated data.

## Notes
- Matrix inverses are updated incrementally (Sherman–Morrison), not recomputed.
- No random seed is set, so exact numbers change between runs.
- The model has no explicit intercept term, so it is a simplified approximation of the true environment.

## Run it
pip install numpy pandas matplotlib, then open the notebook.
