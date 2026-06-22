"""Entry point for the PokerBot."""
import sys
import numpy as np

from bot.bot import CustomPokerBot
from pkbot.runner import parse_args, run_bot

def _simulate():
    """Quick sanity check of the math without a real game engine."""
    print("=" * 60)
    print("SIMULATION: Raise Cap / Call Cap / x* verification")
    print("=" * 60)

    scenarios = [
        # (label,          E,    beta, auction_won, P,    S_eff, mu_bluff)
        ("65%E Won β=0.2",  0.65, 0.2,  True,        400,  5000, 100),
        ("75%E Lost β=0.6", 0.75, 0.6,  False,       400,  5000, 150),
        ("90%E Won β=0.2",  0.90, 0.2,  True,        400,  5000, 100),
        ("45%E Won β=0.2",  0.45, 0.2,  True,        400,  5000, 100),
        ("65%E Lost β=0.6", 0.65, 0.6,  False,       400,  5000, 150),
    ]

    for label, E, beta, won, P, S_eff, mu_bluff in scenarios:
        W = E * (1.0 - beta) + 1.0 * beta
        W = max(0.0, min(1.0, W))
        I_mod = 1.0 if won else (0.5 + 0.5 * beta)
        X = max(0.0, 2.0 * W - 1.0)
        denom = max(0.001, 1.05 - X)
        R_cap = min(S_eff, S_eff * I_mod * (X ** 3.5 / denom) * (1.0 - 0.5 * beta))
        C_cap = min(S_eff, R_cap + beta * mu_bluff * 1.5)
        if W > 0.5 and (2.0 * W - 1.0) > 1e-6:
            x_star = P * ((1.0 - W) / (2.0 * W - 1.0) + beta)
        else:
            x_star = 0.0

        print(f"\n[{label}]")
        print(f"  W={W:.3f}  X={X:.3f}  I_mod={I_mod:.2f}")
        print(f"  R_cap  = {R_cap:8.1f}  chips  (max you raise)")
        print(f"  C_cap  = {C_cap:8.1f}  chips  (max you call)")
        print(f"  x*     = {x_star:8.1f}  chips  (opening raise size)")
        print(f"  → Raise up to {min(R_cap, x_star):.0f}, Call up to {C_cap:.0f}, Fold above that")

    print("\n" + "=" * 60)
    print("AUCTION TRAP SIMULATION")
    print("=" * 60)
    mu_opp = 300.0; sigma_opp = 80.0; P_c = 100; S_eff = 4900
    for E_a, beta_a, label in [(0.60, 0.4, "Decent hand"), (0.85, 0.4, "Monster hand"), (0.95, 0.4, "Near-nuts")]:
        V_gain, V_leak = 0.07, 0.04
        P_implied = P_c + S_eff * (E_a ** 3) * (1.0 + beta_a / 2.0)
        B_true = P_implied * (V_gain + V_leak)
        E_min = 0.78
        alpha_t = max(0.0, ((E_a - E_min) / (1.0 - E_min)) ** 2) if E_a > E_min else 0.0
        lam = 2.0  # fixed for deterministic sim
        B_target = max(0.0, mu_opp - lam * sigma_opp)
        B_final = B_true + alpha_t * max(0.0, B_target - B_true)
        print(f"\n[{label} E={E_a}]  P_implied={P_implied:.0f}  B_true={B_true:.0f}"
              f"  α={alpha_t:.2f}  B_target={B_target:.0f}  → B_final={B_final:.0f}")


if __name__ == '__main__':
    if '--simulate' in sys.argv:
        _simulate()
    else:
        run_bot(CustomPokerBot(), parse_args())