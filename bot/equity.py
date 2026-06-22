"""MLP equity model: weight loading, prediction, and feature extraction."""
import sys
import base64
import zlib
import pickle
import numpy as np
import eval7
from collections import Counter

from bot.constants import (
    EMBEDDED_WEIGHTS, ALL_CARDS, RANKS_STR, SUITS_STR, FULL_DECK_STR
)

def _load_mlp_weights():
    """Decode embedded MLP weights - no external files needed"""
    try:
        compressed = base64.b64decode(EMBEDDED_WEIGHTS)
        pickled = zlib.decompress(compressed)
        data = pickle.loads(pickled)
        return data['W'], data['b']
    except Exception as e:
        print(f"Failed to load embedded MLP: {e}", file=sys.stderr)
        return None

_MLP_DATA = _load_mlp_weights()

def _mlp_predict(x_features):
    if _MLP_DATA is None:
        return 0.5
    W_list, b_list = _MLP_DATA
    h = x_features
    for i in range(len(W_list) - 1):
        h = np.maximum(0, h @ W_list[i] + b_list[i])
    z = (h @ W_list[-1] + b_list[-1]).item()
    return 1.0 / (1.0 + np.exp(-max(-15.0, min(15.0, z))))


def _extract_equity_features(my_hand, board, opp_revealed):
    feats = []
    r0 = RANKS_STR.index(my_hand[0][0])
    r1 = RANKS_STR.index(my_hand[1][0])
    hi, lo = max(r0, r1), min(r0, r1)

    feats.append(hi / 12.0)
    feats.append(lo / 12.0)
    feats.append(1.0 if my_hand[0][1] == my_hand[1][1] else 0.0)
    feats.append(1.0 if r0 == r1 else 0.0)
    feats.append(min(hi - lo - 1, 4) / 4.0)

    board_ranks = [RANKS_STR.index(c[0]) for c in board]
    board_suits = [SUITS_STR.index(c[1]) for c in board]
    n_board = len(board)
    feats.append(n_board / 4.0)
    if board:
        feats.append(max(board_ranks) / 12.0)
        feats.append(min(board_ranks) / 12.0)
        feats.append(np.mean(board_ranks) / 12.0)
    else:
        feats.extend([0.0, 0.0, 0.0])

    all_ranks = [r0, r1] + board_ranks
    rc = Counter(all_ranks)
    max_kind = max(rc.values())
    feats.append((max_kind - 1) / 3.0)
    feats.append(sum(1 for r in [r0, r1] if r in board_ranks) / 2.0)
    if board_ranks:
        feats.append(1.0 if max(board_ranks) in [r0, r1] else 0.0)
        feats.append(1.0 if r0 == r1 and r0 > max(board_ranks) else 0.0)
    else:
        feats.extend([0.0, 0.0])

    all_suits = [c[1] for c in my_hand] + [c[1] for c in board]
    sc = Counter(all_suits)
    max_flush = max(sc.values())
    feats.append(max_flush / 5.0)
    feats.append(1.0 if max_flush >= 5 else 0.0)
    feats.append(1.0 if max_flush == 4 else 0.0)

    unique_ranks = sorted(set(all_ranks))
    max_consec = 1; cur = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i - 1] + 1:
            cur += 1; max_consec = max(max_consec, cur)
        else:
            cur = 1
    if 12 in unique_ranks and 0 in unique_ranks:
        w = 1 + sum(1 for r in [1, 2, 3] if r in unique_ranks)
        max_consec = max(max_consec, w)
    feats.append(max_consec / 5.0)
    feats.append(1.0 if max_consec >= 5 else 0.0)
    feats.append(1.0 if max_consec == 4 else 0.0)

    if board_ranks:
        brc = Counter(board_ranks)
        feats.append(1.0 if max(brc.values()) >= 2 else 0.0)
        bsc = Counter(board_suits)
        feats.append(max(bsc.values()) / 4.0)
        ubr = sorted(set(board_ranks)); mc3 = 1; c3 = 1
        for i in range(1, len(ubr)):
            if ubr[i] == ubr[i - 1] + 1: c3 += 1; mc3 = max(mc3, c3)
            else: c3 = 1
        feats.append(mc3 / 4.0)
    else:
        feats.extend([0.0, 0.0, 0.0])

    has_opp = len(opp_revealed) > 0
    feats.append(1.0 if has_opp else 0.0)
    if has_opp:
        opp_r = RANKS_STR.index(opp_revealed[0][0])
        feats.append(opp_r / 12.0)
        feats.append(1.0 if opp_r in board_ranks else 0.0)
        feats.append(1.0 if opp_r > hi else 0.0)
        opp_suit = opp_revealed[0][1]
        feats.append(1.0 if sc.get(opp_suit, 0) >= 3 else 0.0)
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    if n_board >= 3:
        e7cards = [ALL_CARDS[c] for c in my_hand + board]
        if len(e7cards) >= 5:
            feats.append((eval7.evaluate(e7cards) >> 24) / 8.0)
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)

# =============================================================================
# MATHEMATICAL FRAMEWORK SUMMARY
# =============================================================================
# W  = E_adj*(1-β) + 1.0*β          — True Win Probability (bluff-weighted)
# X  = max(0, 2W-1)                  — Advantage score, 0 at coin-flip
#
# I_mod = 1.0 if won/tied auction
#       = 0.5 + 0.5*β if lost         — Maniac's "strong hand" signal is noise
#
# R_cap = S_eff * I_mod * X^3.5/(1.05-X) * (1 - 0.5*β)
#       — Raise cap: shrinks vs maniacs (keep bluffs in), bloated at nuts
#
# C_cap = R_cap + β * μ_bluff * 1.5  — Call cap: wider when they bluff often
#
# x*  = P * ((1-W)/(2W-1) + β)       — Optimal opening raise (don't overbet)
#
# Auction:
# B_true = P_implied * (V_gain + V_leak)
# B_final = B_true + α(E) * max(0, B_target - B_true)
# α(E) = 0 if E ≤ 0.78, else ((E-0.78)/0.22)^2   — only trap with monsters
# B_target = max(0, μ_opp - λ*σ_opp), λ~U(1.5,3.0)
# P_implied = P + S_eff * E^3 * (1 + β/2)          — equity-anchored implied pot
# =============================================================================

import sys
import traceback
import math
import random
import numpy as np
import eval7
from collections import Counter

# Assuming these are defined in the environment:
# BaseBot, GameInfo, PokerState
# ActionCheck, ActionCall, ActionFold, ActionRaise, ActionBid
# _MLP_DATA, _extract_equity_features, _mlp_predict
# ALL_CARDS, FULL_DECK_STR, RANKS_STR, SUITS_STR


