import numpy as np
import pandas as pd

N_ZONES = 40
N_VEHICLES = 30  # 10% of SafarGo's 300-vehicle fleet
STEPS_PER_DAY = 288  # one decision every five minutes
SEED = 42

def demand_profile(step, weather, rng):
    """Synthetic demand for 40 zones, with rush-hour and event-like peaks."""
    hour = (step % STEPS_PER_DAY) / 12.0
    # Two commute peaks plus evening activity.
    morning = np.exp(-0.5 * ((hour - 8.0) / 1.7) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 2.2) ** 2)
    midday = np.exp(-0.5 * ((hour - 13.0) / 2.8) ** 2)
    base = 0.25 + 0.75 * (0.45 * morning + 0.65 * evening + 0.35 * midday)
    zone_bias = np.linspace(0.65, 1.35, N_ZONES)
    # Weather increases demand uncertainty and road friction.
    weather_factor = 1.18 if weather else 1.0
    demand = base * zone_bias * weather_factor
    # Synthetic local event / university / prayer-like peaks at different zones.
    if 7 <= hour <= 9:
        demand[5:12] *= 1.55
    if 12 <= hour <= 14 and int(step / 12) % 7 == 4:
        demand[20:27] *= 1.45
    if 16 <= hour <= 20:
        demand[30:36] *= 1.5
    demand *= rng.lognormal(mean=0, sigma=0.18, size=N_ZONES)
    return demand

def zone_distances():
    # Synthetic 8x5 grid; distances are illustrative, not actual road distances.
    coords = np.array([(i % 8, i // 8) for i in range(N_ZONES)])
    return np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2).astype(float)

DIST = zone_distances()

def discretize_state(zone, step, weather, pressure):
    hour = (step % STEPS_PER_DAY) / 12.0
    time_bucket = 0 if hour < 7 else 1 if hour < 11 else 2 if hour < 16 else 3
    demand_bucket = 0 if pressure < 0.85 else 1 if pressure < 1.25 else 2
    return (int(zone), int(time_bucket), int(weather), int(demand_bucket))

class TabularQLearner:
    """Compact tabular Q-learning over zone, time, weather, and demand pressure."""
    def __init__(self, alpha=0.15, gamma=0.90, epsilon=1.0, seed=42):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.n_time, self.n_weather, self.n_pressure = 4, 2, 3
        self.q = np.zeros((N_ZONES, self.n_time, self.n_weather, self.n_pressure, N_ZONES))

    def choose(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ZONES))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state):
        old = self.q[state + (action,)]
        target = reward + self.gamma * np.max(self.q[next_state])
        self.q[state + (action,)] = old + self.alpha * (target - old)

def reward_for_action(destination, demand, supply, current_zone, weather):
    # Positive demand coverage, with penalties for empty travel and over-supplied zones.
    pressure = demand[destination] / (1.0 + supply[destination])
    coverage = 2.2 * pressure
    fuel_penalty = 0.20 * DIST[current_zone, destination] * (1.35 if weather else 1.0)
    crowding_penalty = 0.7 * max(0, supply[destination] - demand[destination] * 1.4)
    return float(coverage - fuel_penalty - crowding_penalty)

def train_agent(episodes=120, seed=SEED, alpha=0.15, gamma=0.90):
    rng = np.random.default_rng(seed)
    agent = TabularQLearner(alpha=alpha, gamma=gamma, seed=seed)
    rewards = []
    for ep in range(episodes):
        total = 0.0
        zone = int(rng.integers(N_ZONES))
        for step in range(STEPS_PER_DAY):
            weather = int(rng.random() < (0.25 + 0.10 * np.sin(step / 30)))
            demand = demand_profile(step, weather, rng)
            # Randomized synthetic supply snapshot.
            supply = rng.poisson(lam=0.65, size=N_ZONES).astype(float)
            pressure = float(demand[zone] / (1.0 + supply[zone]))
            state = discretize_state(zone, step, weather, pressure)
            action = agent.choose(state, explore=True)
            reward = reward_for_action(action, demand, supply, zone, weather)
            next_pressure = float(demand[action] / (1.0 + supply[action]))
            next_state = discretize_state(action, step + 1, weather, next_pressure)
            agent.update(state, action, reward, next_state)
            total += reward
            zone = action
        agent.epsilon = max(0.05, agent.epsilon * 0.965)
        rewards.append(total / STEPS_PER_DAY)
    return agent, rewards

def simulate_policy(policy, agent=None, days=10, seed=100):
    rng = np.random.default_rng(seed)
    all_rows = []
    recent_zone_visits = np.zeros(N_ZONES, dtype=float)
    vehicle_zones = rng.integers(0, N_ZONES, size=N_VEHICLES)
    for day in range(days):
        day_wait, day_completed, day_fuel, day_reward = [], 0, 0.0, 0.0
        for step in range(STEPS_PER_DAY):
            weather = int(rng.random() < 0.25)
            demand = demand_profile(step, weather, rng)
            # Total demand requests in this five-minute interval.
            requests = rng.poisson(lam=np.maximum(0.05, demand * 0.20))
            supply = np.bincount(vehicle_zones, minlength=N_ZONES).astype(float)
            # Assign a fraction of requests to vehicles already near their zone.
            completed = np.minimum(requests, supply).sum()
            unmet = np.maximum(0, requests - supply)
            # Approximate wait: base wait grows with unmet demand; cap for display.
            wait = float(2.2 + 1.0 * unmet.sum() + 0.12 * np.mean(np.maximum(0, requests - supply)))
            day_wait.append(wait)
            day_completed += int(completed)
            # Move idle vehicles using selected policy.
            new_zones = vehicle_zones.copy()
            # Process vehicles sequentially so the supply estimate changes after each move.
            for vi, current in enumerate(vehicle_zones):
                pressure = float(demand[current] / (1.0 + supply[current]))
                state = discretize_state(current, step, weather, pressure)
                if policy == "Q-learning" and agent is not None:
                    # Learned values are combined with a safety-aware local score to avoid crowding.
                    learned = agent.q[state].copy()
                    local_score = 1.25 * demand / (1.0 + supply)
                    travel_cost = 0.16 * DIST[current] * (1.35 if weather else 1.0)
                    crowding = 0.85 * np.maximum(0, supply - demand * 1.25)
                    destination = int(np.argmax(learned + local_score - travel_cost - crowding))
                elif policy == "Busiest zone":
                    destination = int(np.argmax(demand / (1.0 + supply)))
                elif policy == "Random":
                    destination = int(rng.integers(N_ZONES))
                else:  # Stay in place baseline
                    destination = int(current)
                dist = DIST[current, destination]
                day_fuel += dist * (1.35 if weather else 1.0)
                reward = reward_for_action(destination, demand, supply, int(current), weather)
                day_reward += reward
                new_zones[vi] = destination
                if destination != current:
                    supply[int(current)] = max(0, supply[int(current)] - 1)
                    supply[destination] += 1
                recent_zone_visits[destination] += 1
            vehicle_zones = new_zones
        # Fairness proxy: number of distinct zones visited by repositioning.
        coverage = float(np.count_nonzero(recent_zone_visits) / N_ZONES)
        all_rows.append({
            "day": day + 1,
            "policy": policy,
            "avg_wait_min": float(np.mean(day_wait)),
            "rides_completed": int(day_completed),
            "fuel_units": float(day_fuel),
            "mean_reward_per_vehicle_step": float(day_reward / (STEPS_PER_DAY * N_VEHICLES)),
            "zone_coverage_pct": 100 * coverage,
        })
    return pd.DataFrame(all_rows)
