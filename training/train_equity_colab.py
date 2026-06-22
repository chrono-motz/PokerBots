#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  GPU-Optimized Poker Equity MLP Training — Google Colab Ready
═══════════════════════════════════════════════════════════════════════════════

Trains on ALL 5 game states:
  (2,0,0) preflop    — 2 hole cards, no board, no opp
  (2,3,0) flop       — 2 hole + 3 board
  (2,4,0) turn       — 2 hole + 4 board
  (2,3,1) flop_opp   — 2 hole + 3 board + 1 opp revealed
  (2,4,1) turn_opp   — 2 hole + 4 board + 1 opp revealed

Colab usage:
  !pip install eval7
  !python train_equity_colab.py --samples 200000 --val-samples 25000 --epochs 300

Or run cells individually — see section markers.
"""

import numpy as np
import time
import random
import pickle
import os
import argparse
from collections import Counter
from multiprocessing import Pool, cpu_count

import eval7

# ─── GPU detection ───────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ PyTorch {torch.__version__} | Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("⚠ PyTorch not found. Install: pip install torch")

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]
ALL_E7 = {c: eval7.Card(c) for c in FULL_DECK}

# 5 state types: (label, n_hole, n_board, n_opp)
STATE_TYPES = [
    ('preflop',  2, 0, 0),
    ('flop',     2, 3, 0),
    ('turn',     2, 4, 0),
    ('flop_opp', 2, 3, 1),
    ('turn_opp', 2, 4, 1),
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE EXTRACTION (28 dims — must match bot exactly)
# ═════════════════════════════════════════════════════════════════════════════
def extract_features(my_hand, board, opp_revealed):
    """
    28 poker-meaningful features from game state.

    [0]  hi_rank/12         [1]  lo_rank/12         [2]  suited
    [3]  pair               [4]  gap/4              [5]  n_board/4
    [6]  board_hi/12        [7]  board_lo/12        [8]  board_avg/12
    [9]  (max_kind-1)/3     [10] paired_w_board/2   [11] top_pair
    [12] overpair           [13] flush_prog/5       [14] flush_made
    [15] flush_draw         [16] straight_prog/5    [17] straight_made
    [18] straight_draw      [19] board_paired       [20] board_flush_dens
    [21] board_connectivity [22] has_opp            [23] opp_rank/12
    [24] opp_pairs_board    [25] opp_beats_hi       [26] opp_flush_zone
    [27] eval7_tier/8
    """
    feats = []
    r0 = RANKS.index(my_hand[0][0])
    r1 = RANKS.index(my_hand[1][0])
    hi, lo = max(r0, r1), min(r0, r1)

    feats.append(hi / 12.0)
    feats.append(lo / 12.0)
    feats.append(1.0 if my_hand[0][1] == my_hand[1][1] else 0.0)
    feats.append(1.0 if r0 == r1 else 0.0)
    feats.append(min(hi - lo - 1, 4) / 4.0)

    board_ranks = [RANKS.index(c[0]) for c in board]
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
    feats.append((max(rc.values()) - 1) / 3.0)
    feats.append(sum(1 for r in [r0, r1] if r in board_ranks) / 2.0)
    feats.append(1.0 if board_ranks and max(board_ranks) in [r0, r1] else 0.0)
    feats.append(1.0 if r0 == r1 and board_ranks and r0 > max(board_ranks) else 0.0)

    all_suits = [c[1] for c in my_hand] + [c[1] for c in board]
    sc = Counter(all_suits)
    mf = max(sc.values()) if sc else 0
    feats.append(mf / 5.0)
    feats.append(1.0 if mf >= 5 else 0.0)
    feats.append(1.0 if mf == 4 else 0.0)

    ur = sorted(set(all_ranks))
    mc = 1; c = 1
    for i in range(1, len(ur)):
        if ur[i] == ur[i - 1] + 1: c += 1; mc = max(mc, c)
        else: c = 1
    if 12 in ur and 0 in ur:
        mc = max(mc, 1 + sum(1 for r in [1, 2, 3] if r in ur))
    feats.append(mc / 5.0)
    feats.append(1.0 if mc >= 5 else 0.0)
    feats.append(1.0 if mc == 4 else 0.0)

    if board_ranks:
        brc = Counter(board_ranks)
        bsc = Counter([SUITS.index(c2[1]) for c2 in board])
        feats.append(1.0 if max(brc.values()) >= 2 else 0.0)
        feats.append(max(bsc.values()) / 4.0)
        ubr = sorted(set(board_ranks)); m3 = 1; c3 = 1
        for i in range(1, len(ubr)):
            if ubr[i] == ubr[i - 1] + 1: c3 += 1; m3 = max(m3, c3)
            else: c3 = 1
        feats.append(m3 / 4.0)
    else:
        feats.extend([0.0, 0.0, 0.0])

    has_opp = len(opp_revealed) > 0
    feats.append(1.0 if has_opp else 0.0)
    if has_opp:
        opp_r = RANKS.index(opp_revealed[0][0])
        feats.append(opp_r / 12.0)
        feats.append(1.0 if opp_r in board_ranks else 0.0)
        feats.append(1.0 if opp_r > hi else 0.0)
        feats.append(1.0 if sc.get(opp_revealed[0][1], 0) >= 3 else 0.0)
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    if n_board >= 3:
        e7c = [ALL_E7[c2] for c2 in my_hand + board]
        feats.append((eval7.evaluate(e7c) >> 24) / 8.0 if len(e7c) >= 5 else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: MONTE CARLO EQUITY (label generation)
# ═════════════════════════════════════════════════════════════════════════════
def mc_equity(my_hand, board, opp_revealed, n_iters=10000):
    """
    MC equity vs random opponent. Board completed to 5 cards
    (matches engine: eval7.evaluate(deck.peek(5) + hand)).
    """
    mc = [ALL_E7[c] for c in my_hand]
    bc = [ALL_E7[c] for c in board]
    oc = [ALL_E7[c] for c in opp_revealed]
    dead = set(my_hand + board + opp_revealed)
    dk = [ALL_E7[c] for c in FULL_DECK if c not in dead]
    nb = 5 - len(board)
    no = 2 - len(opp_revealed)
    w = 0.0
    for _ in range(n_iters):
        d = random.sample(dk, nb + no)
        mv = eval7.evaluate(mc + bc + d[:nb])
        ov = eval7.evaluate(oc + d[nb:] + bc + d[:nb])
        if mv > ov: w += 1.0
        elif mv == ov: w += 0.5
    return w / n_iters


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: PARALLEL DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def _gen_one(args):
    """Worker: generate one (features, equity, type_label) sample."""
    st_label, n_board, n_opp, mc_iters, seed = args
    random.seed(seed)
    dk = FULL_DECK[:]
    random.shuffle(dk)
    hand = [dk[0], dk[1]]
    board = dk[2:2 + n_board]
    opp = dk[2 + n_board:2 + n_board + n_opp]
    eq = mc_equity(hand, board, opp, mc_iters)
    feats = extract_features(hand, board, opp)
    return feats, eq, st_label


def generate_dataset(n_samples, mc_iters=1000, n_workers=None, seed=42):
    """
    Generate balanced dataset. Returns (X, Y, type_labels).
    type_labels is a string array: ['preflop', 'flop', 'turn', 'flop_opp', 'turn_opp'].
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    # Balanced: equal samples per type
    per_type = n_samples // len(STATE_TYPES)
    tasks = []
    rng = random.Random(seed)
    for label, _, n_b, n_o in STATE_TYPES:
        for _ in range(per_type):
            tasks.append((label, n_b, n_o, mc_iters, rng.randint(0, 2**31)))
    # Fill remainder
    for i in range(n_samples - len(tasks)):
        label, _, n_b, n_o = STATE_TYPES[i % len(STATE_TYPES)]
        tasks.append((label, n_b, n_o, mc_iters, rng.randint(0, 2**31)))

    random.Random(seed + 1).shuffle(tasks)

    print(f"  Samples: {len(tasks)} | MC: {mc_iters} | Workers: {n_workers}")
    t0 = time.time()

    X, Y, T = [], [], []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (f, e, t) in enumerate(pool.imap_unordered(_gen_one, tasks, chunksize=128)):
                X.append(f); Y.append(e); T.append(t)
                if (i + 1) % 5000 == 0:
                    el = time.time() - t0; rate = (i + 1) / el
                    print(f"  [{i+1:>7}/{len(tasks)}] {rate:.0f}/s ETA:{(len(tasks)-i-1)/rate:.0f}s")
    else:
        for i, task in enumerate(tasks):
            f, e, t = _gen_one(task)
            X.append(f); Y.append(e); T.append(t)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0; rate = (i + 1) / el
                print(f"  [{i+1:>7}/{len(tasks)}] {rate:.0f}/s ETA:{(len(tasks)-i-1)/rate:.0f}s")

    el = time.time() - t0
    print(f"  Done: {el:.0f}s ({len(tasks)/el:.0f}/s)")
    print(f"  Types: {Counter(T)}")

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(T)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: PyTorch MODEL + TRAINING
# ═════════════════════════════════════════════════════════════════════════════
if HAS_TORCH:
    class EquityNet(nn.Module):
        def __init__(self, input_dim=28, hidden_dims=[128, 64], dropout=0.1):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x).squeeze(-1)


    def train_model(X_train, Y_train, X_val, Y_val, val_types,
                    hidden_dims=[128, 64], epochs=300, batch_size=512,
                    lr=1e-3, weight_decay=1e-5, dropout=0.1):

        model = EquityNet(X_train.shape[1], hidden_dims, dropout).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Arch: {X_train.shape[1]} → {' → '.join(map(str, hidden_dims))} → 1")
        print(f"  Params: {n_params:,} | Device: {DEVICE}\n")

        pin = (DEVICE.type == 'cuda')
        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
        val_dl = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),
            batch_size=4096, shuffle=False, num_workers=2, pin_memory=pin)

        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
        loss_fn = nn.MSELoss()

        best_mae = float('inf')
        best_ep = 0
        best_state = None

        for ep in range(epochs):
            model.train()
            tloss = 0.0; nb = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tloss += loss.item(); nb += 1
            sched.step()
            tloss /= nb

            if (ep + 1) % 5 == 0 or ep == 0:
                model.eval()
                ps, ls = [], []
                with torch.no_grad():
                    for xb, yb in val_dl:
                        ps.append(model(xb.to(DEVICE)).cpu().numpy())
                        ls.append(yb.numpy())
                ps = np.concatenate(ps); ls = np.concatenate(ls)
                mae = np.mean(np.abs(ps - ls))
                tag = ""
                if mae < best_mae:
                    best_mae = mae; best_ep = ep + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    tag = " ★"
                clr = opt.param_groups[0]['lr']
                print(f"  Ep {ep+1:>3}: loss={tloss:.6f} MAE={mae*100:.2f}% lr={clr:.6f}{tag}")

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # Per-type eval
        print(f"\n  ── Validation by state type (best epoch {best_ep}) ──")
        with torch.no_grad():
            all_p = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()
        _eval_per_type(all_p, Y_val, val_types)

        return model, best_mae, best_ep


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: NUMPY MLP (for bot inference — no PyTorch needed)
# ═════════════════════════════════════════════════════════════════════════════
class NumpyMLP:
    """Minimal numpy MLP. BN is folded into weights at export time."""
    def __init__(self, W, b):
        self.W = W  # list of (in, out) arrays
        self.b = b  # list of (out,) arrays

    def predict(self, x):
        h = x
        for i in range(len(self.W) - 1):
            h = np.maximum(0, h @ self.W[i] + self.b[i])
        z = h @ self.W[-1] + self.b[-1]
        return (1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))).squeeze(-1)

    def predict_single(self, x):
        h = x
        for i in range(len(self.W) - 1):
            h = np.maximum(0, h @ self.W[i] + self.b[i])
        z = float(h @ self.W[-1] + self.b[-1])
        return 1.0 / (1.0 + np.exp(-max(-15.0, min(15.0, z))))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return cls(d['W'], d['b'])


def torch_to_numpy(model):
    """
    Convert PyTorch EquityNet → NumpyMLP.
    Folds BatchNorm running stats into preceding Linear layer:
      BN(Wx+b) = γ(Wx+b-μ)/√(σ²+ε) + β  →  W'x + b'
    """
    model.eval()
    W_list, b_list = [], []
    mods = list(model.net)
    i = 0
    while i < len(mods):
        if isinstance(mods[i], nn.Linear):
            W = mods[i].weight.detach().cpu().numpy()  # (out, in)
            b = mods[i].bias.detach().cpu().numpy()
            if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm1d):
                bn = mods[i + 1]
                γ = bn.weight.detach().cpu().numpy()
                β = bn.bias.detach().cpu().numpy()
                μ = bn.running_mean.detach().cpu().numpy()
                σ2 = bn.running_var.detach().cpu().numpy()
                s = γ / np.sqrt(σ2 + bn.eps)
                W = W * s[:, None]
                b = s * (b - μ) + β
                i += 1
            W_list.append(W.T)  # (in, out) for x @ W
            b_list.append(b)
        i += 1
    return NumpyMLP(W_list, b_list)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: PER-TYPE EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def _eval_per_type(preds, labels, types):
    errs = np.abs(preds - labels)
    print(f"\n  {'Type':<12} {'N':>6} {'MAE':>7} {'<2%':>5} {'<5%':>5} {'<10%':>5}")
    print(f"  {'─'*48}")
    for st in ['preflop', 'flop', 'turn', 'flop_opp', 'turn_opp']:
        m = types == st
        if not m.any(): continue
        e = errs[m]
        print(f"  {st:<12} {m.sum():>6} {np.mean(e)*100:>6.2f}% "
              f"{np.mean(e<.02)*100:>4.1f}% {np.mean(e<.05)*100:>4.1f}% {np.mean(e<.10)*100:>4.1f}%")
    print(f"  {'─'*48}")
    print(f"  {'OVERALL':<12} {len(labels):>6} {np.mean(errs)*100:>6.2f}% "
          f"{np.mean(errs<.02)*100:>4.1f}% {np.mean(errs<.05)*100:>4.1f}% {np.mean(errs<.10)*100:>4.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples',     type=int,   default=200000)
    p.add_argument('--val-samples', type=int,   default=25000)
    p.add_argument('--mc-iters',    type=int,   default=1000)
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--hidden',      type=str,   default='128,64')
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--batch-size',  type=int,   default=512)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--workers',     type=int,   default=None)
    p.add_argument('--output',      type=str,   default='equity_mlp.pkl')
    p.add_argument('--data-dir',    type=str,   default='./data')
    p.add_argument('--no-cache',    action='store_true')
    args = p.parse_args()

    hidden = [int(x) for x in args.hidden.split(',')]
    os.makedirs(args.data_dir, exist_ok=True)

    # ── Data ──
    tcache = os.path.join(args.data_dir, f'train_{args.samples}_mc{args.mc_iters}.npz')
    vcache = os.path.join(args.data_dir, f'val_{args.val_samples}_mc{args.mc_iters}.npz')

    if not args.no_cache and os.path.exists(tcache) and os.path.exists(vcache):
        print("Loading cached data...")
        d = np.load(tcache, allow_pickle=True)
        X_train, Y_train, train_types = d['X'], d['Y'], d['types']
        d = np.load(vcache, allow_pickle=True)
        X_val, Y_val, val_types = d['X'], d['Y'], d['types']
    else:
        print(f"\n{'='*60}")
        print(f"TRAINING DATA ({args.samples} samples)")
        print(f"{'='*60}")
        X_train, Y_train, train_types = generate_dataset(
            args.samples, args.mc_iters, args.workers, seed=42)

        print(f"\n{'='*60}")
        print(f"VALIDATION DATA ({args.val_samples} samples)")
        print(f"{'='*60}")
        X_val, Y_val, val_types = generate_dataset(
            args.val_samples, args.mc_iters, args.workers, seed=99999)

        np.savez(tcache, X=X_train, Y=Y_train, types=train_types)
        np.savez(vcache, X=X_val, Y=Y_val, types=val_types)
        print(f"  Cached → {args.data_dir}/")

    print(f"\n  Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"  Y: {Y_train.mean():.3f} ± {Y_train.std():.3f}")

    # ── Train ──
    if not HAS_TORCH:
        print("\n⚠ No PyTorch. Upload this script to Colab with GPU runtime.")
        print(f"  Data cached in {args.data_dir}/ — rerun will load instantly.")
        return

    print(f"\n{'='*60}")
    print(f"TRAINING (GPU: {DEVICE})")
    print(f"{'='*60}")

    model, best_mae, best_ep = train_model(
        X_train, Y_train, X_val, Y_val, val_types,
        hidden_dims=hidden, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout)

    print(f"\n  Best MAE: {best_mae*100:.2f}% @ epoch {best_ep}")
    baseline = np.mean(np.abs(Y_val - Y_train.mean()))
    print(f"  Baseline: {baseline*100:.2f}% | Improvement: {(baseline-best_mae)*100:.2f}%")

    # ── Export ──
    print(f"\n{'='*60}")
    print(f"EXPORT")
    print(f"{'='*60}")

    np_mlp = torch_to_numpy(model)
    np_mlp.save(args.output)
    print(f"  numpy weights → {args.output}")

    torch.save(model.state_dict(), args.output.replace('.pkl', '_torch.pt'))
    print(f"  torch checkpoint → {args.output.replace('.pkl', '_torch.pt')}")

    # Verify numpy matches torch
    np_pred = np_mlp.predict(X_val).squeeze()
    with torch.no_grad():
        t_pred = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()
    diff = np.max(np.abs(np_pred - t_pred))
    print(f"  numpy vs torch max diff: {diff:.8f} {'✅' if diff < 0.01 else '⚠ MISMATCH'}")

    np_mae = np.mean(np.abs(np_pred - Y_val))
    print(f"  numpy MAE: {np_mae*100:.2f}%")

    # Speed
    x1 = X_val[:1]
    t0 = time.time()
    for _ in range(10000): np_mlp.predict(x1)
    spd = (time.time() - t0) / 10000 * 1000
    print(f"  numpy speed: {spd:.4f}ms/call")

    print(f"\n  ✅ Copy {args.output} to your bot directory.")
    print(f"  Load with: NumpyMLP.load('{args.output}')")


if __name__ == '__main__':
    main()
