"""Command-line evaluation for the SafarGo RL MVP."""
from core import train_agent, simulate_policy
import pandas as pd

def main():
    agent, rewards = train_agent(episodes=100, seed=42, alpha=0.15, gamma=0.90)
    frames = [
        simulate_policy("Q-learning", agent, days=10, seed=777),
        simulate_policy("Busiest zone", days=10, seed=777),
        simulate_policy("Random", days=10, seed=777),
        simulate_policy("Stay in place", days=10, seed=777),
    ]
    results = pd.concat(frames, ignore_index=True)
    summary = results.groupby("policy", as_index=False).agg(
        avg_wait_min=("avg_wait_min", "mean"),
        rides_completed=("rides_completed", "mean"),
        fuel_units=("fuel_units", "mean"),
        mean_reward=("mean_reward_per_vehicle_step", "mean"),
        zone_coverage_pct=("zone_coverage_pct", "mean"),
    )
    print("Final 10-episode moving average reward:", round(sum(rewards[-10:]) / min(10, len(rewards)), 4))
    print(summary.round(2).to_string(index=False))
    summary.to_csv("evaluation_summary.csv", index=False)
    pd.DataFrame({"episode": range(1, len(rewards)+1), "mean_reward": rewards}).to_csv("reward_curve.csv", index=False)

if __name__ == "__main__":
    main()
