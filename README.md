# PokerBots — IIT Pokerbots Entry

An ML-powered Texas Hold'em poker bot built for the [MIT Pokerbots](https://pokerbots.org/) competition format. The bot uses a trained **equity MLP** (compressed and embedded directly in the source) combined with a **Bayesian opponent model** to make real-time betting decisions.

---

## Repository Structure

```
PokerBots/
├── bot/
│   ├── __init__.py          # Package entry, exports CustomPokerBot
│   ├── constants.py         # Deck/rank/suit globals + embedded MLP weights
│   ├── equity.py            # MLP loader, predictor, and feature extractor
│   └── bot.py               # CustomPokerBot — main game-playing agent
├── training/
│   ├── Pokerbot.ipynb       # Colab notebook: MLP training pipeline (GPU)
│   └── train_equity_colab.py # Standalone training script (extracted from notebook)
├── docs/
│   └── PokerNoobs_IIT_Pokerbots-2.pdf  # Reference / design document
├── main.py                  # Entry point + simulation sanity-check
├── requirements.txt
└── .gitignore
```

---

## How It Works

### 1. Equity Model (`bot/equity.py`)
A **28-feature MLP** predicts win equity for five game states:

| State       | Hole | Board | Opp Revealed |
|-------------|------|-------|--------------|
| `preflop`   | 2    | 0     | 0            |
| `flop`      | 2    | 3     | 0            |
| `turn`      | 2    | 4     | 0            |
| `flop_opp`  | 2    | 3     | 1            |
| `turn_opp`  | 2    | 4     | 1            |

The trained numpy weights are **compressed (zlib) + base64-encoded** and embedded directly into `bot/constants.py` — no external weight file needed at runtime.

### 2. Opponent Model (`bot/bot.py`)
EMA-tracked opponent stats drive bluff-adjusted decisions:

```
W  = E_adj*(1−β) + β           # true win prob (bluff-weighted)
X  = max(0, 2W−1)               # advantage score
R_cap = S_eff * I_mod * X^3.5/(1.05−X) * (1−0.5β)   # raise ceiling
C_cap = R_cap + β * μ_bluff * 1.5                     # call ceiling
x*  = P * ((1−W)/(2W−1) + β)   # optimal opening raise size
```

`β` = opponent bluff rate · `μ_bluff` = EMA of their bet sizes · `I_mod` = auction-result modifier.

### 3. Auction Bidding
A trap-aware bid formula scales bids near the nuts:

```
B_true   = P_implied * (V_gain + V_leak)
α(E)     = 0 if E ≤ 0.78, else ((E−0.78)/0.22)²
B_target = max(0, μ_opp − λ*σ_opp)
B_final  = B_true + α(E) * max(0, B_target − B_true)
```

---

## Quick Start

```bash
pip install -r requirements.txt
# Run with the pkbot engine:
python main.py
# Sanity-check the math:
python main.py --simulate
```

---

## Training the Equity Model (Google Colab)

Open `training/Pokerbot.ipynb` in Colab (GPU runtime recommended) or run:

```bash
pip install eval7 torch
python training/train_equity_colab.py \
    --samples 200000 --val-samples 25000 \
    --epochs 300 --hidden 256,128,64 \
    --mc-iters 1000
```

Best achieved validation MAE: **~1.79%** overall (0.60% preflop → 2.91% turn+opp).

---

## Dependencies

| Package   | Purpose                        |
|-----------|-------------------------------|
| `eval7`   | Fast poker hand evaluation     |
| `numpy`   | Feature computation & MLP math |
| `torch`   | Training only (not at runtime) |
