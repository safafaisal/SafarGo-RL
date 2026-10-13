# SafarGo RL Fleet Repositioning MVP

A two-week academic MVP using tabular Q-learning in a synthetic 40-zone simulation.

## Features
- Tabular Q-learning with configurable learning rate and discount factor.
- Epsilon-greedy exploration during training; greedy actions during evaluation.
- Baselines: busiest zone, random destination, and stay in place.
- Reward curve and metrics: average waiting time, completed rides, fuel-use proxy, and zone coverage.
- Safety/ethics notes and explicit synthetic-data disclaimer.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run a reproducible evaluation
```bash
python evaluate.py
```
This writes `evaluation_summary.csv` and `reward_curve.csv`.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `core.py`, `evaluate.py`, `requirements.txt`, and `README.md`.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Choose the repository, branch, and `app.py` as the main file.
4. Click Deploy. The app does not need an API key.
5. Submit the public app URL after the deployment succeeds.

## Model settings
- Alpha (learning rate): 0.15
- Gamma (discount factor): 0.90
- Epsilon: starts at 1.0, decays by 0.965 per episode, and has a 0.05 minimum.
- State: current zone (40), time bucket (4), weather (2), demand pressure (3).
- Action: destination zone (40).
- Q-table size: 40 × 4 × 2 × 3 × 40 = 38,400 action values.

## Limitations
All demand, distances, waiting times, fuel values, and results are synthetic illustrative proxies, not SafarGo records or verified Rawalpindi road data. The simulation does not model full passenger-driver matching, individual driver shifts, or actual traffic routes. Do not connect this prototype directly to live dispatch.
