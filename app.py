import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core import N_ZONES, N_VEHICLES, STEPS_PER_DAY, train_agent, simulate_policy

@st.cache_data(show_spinner=False)
def get_results(episodes, eval_days, alpha, gamma):
    agent, rewards = train_agent(episodes=episodes, alpha=alpha, gamma=gamma)
    q_eval = simulate_policy("Q-learning", agent, days=eval_days, seed=777)
    busiest = simulate_policy("Busiest zone", days=eval_days, seed=777)
    random = simulate_policy("Random", days=eval_days, seed=777)
    stay = simulate_policy("Stay in place", days=eval_days, seed=777)
    results = pd.concat([q_eval, busiest, random, stay], ignore_index=True)
    return agent, rewards, results


st.set_page_config(page_title="SafarGo RL Fleet MVP", page_icon="🚕", layout="wide")
st.title("🚕 SafarGo — RL Fleet Repositioning MVP")
st.caption("Educational simulation only • Synthetic data • Not connected to live vehicles")

with st.sidebar:
    st.header("Simulation settings")
    episodes = st.slider("Training episodes", 20, 200, 100, 20)
    eval_days = st.slider("Evaluation days", 3, 20, 10, 1)
    alpha = st.slider("Learning rate (α)", 0.05, 0.40, 0.15, 0.05)
    gamma = st.slider("Discount factor (γ)", 0.50, 0.99, 0.90, 0.05)
    run = st.button("Run simulation", type="primary")

if "results_bundle" not in st.session_state or run:
    with st.spinner("Training Q-learning agent and evaluating baselines..."):
        st.session_state.results_bundle = get_results(episodes, eval_days, alpha, gamma)
agent, rewards, results = st.session_state.results_bundle

summary = results.groupby("policy", as_index=False).agg(
    avg_wait_min=("avg_wait_min", "mean"),
    rides_completed=("rides_completed", "mean"),
    fuel_units=("fuel_units", "mean"),
    mean_reward=("mean_reward_per_vehicle_step", "mean"),
    zone_coverage_pct=("zone_coverage_pct", "mean"),
)
summary["rides_completed"] = summary["rides_completed"].round(0).astype(int)
summary[["avg_wait_min", "fuel_units", "mean_reward", "zone_coverage_pct"]] = summary[
    ["avg_wait_min", "fuel_units", "mean_reward", "zone_coverage_pct"]
].round(2)

m1, m2, m3 = st.columns(3)
qrow = summary[summary.policy == "Q-learning"].iloc[0]
m1.metric("Q-learning average wait", f"{qrow.avg_wait_min:.2f} min")
m2.metric("Q-learning completed rides/day", f"{qrow.rides_completed:,}")
m3.metric("Q-learning fuel units/day", f"{qrow.fuel_units:,.1f}")

st.subheader("Baseline comparison")
st.dataframe(summary.rename(columns={
    "policy": "Policy", "avg_wait_min": "Average wait (min)",
    "rides_completed": "Completed rides/day", "fuel_units": "Fuel units/day",
    "mean_reward": "Mean reward", "zone_coverage_pct": "Zone coverage (%)"
}), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Training reward curve")
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(rewards)+1), rewards)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Q-learning training reward")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Operational metrics by policy")
    fig, ax = plt.subplots()
    summary.set_index("policy")[["avg_wait_min", "fuel_units"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Metric value (different units)")
    ax.set_title("Waiting time and fuel use")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Completed rides by policy")
fig, ax = plt.subplots()
ax.bar(summary["policy"], summary["rides_completed"])
ax.set_ylabel("Mean completed rides per day")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("Ethics and safety controls")
st.markdown("""
- **No live control:** this prototype uses synthetic data only.
- **Pilot limit:** a real pilot would be limited to 30 of 300 vehicles (10%).
- **Fallback:** stop RL repositioning and return to the existing dispatcher or a conservative rule-based policy.
- **Fairness monitoring:** review waiting time and service coverage for every zone, including low-demand areas.
- **Privacy:** use aggregated zone/time counts; remove passenger and driver identifiers.
""")
st.info("Important: this simplified simulation is for an academic MVP. Its results are synthetic and must not be presented as measured SafarGo business outcomes.")
