"""CustomPokerBot: the main game-playing agent."""
import sys
import traceback
import math
import random
import numpy as np
import eval7
from collections import Counter

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot

from bot.constants import ALL_CARDS, FULL_DECK_STR, RANKS_STR, SUITS_STR, RANK_VALS
from bot.equity import _MLP_DATA, _mlp_predict, _extract_equity_features

class CustomPokerBot(BaseBot):
    def __init__(self):
        super().__init__()

        # ── Core opponent model (EMA-updated) ──────────────────────────────
        self.beta = 0.4          # Bluff rate: 0.0 = pure nit, 1.0 = pure maniac
        self.mu_bluff = 100.0    # EMA of opponent bet sizes when bluffing
        self.in_hand_aggression = 0.0

        # ── Auction bid tracking (EMA) ──────────────────────────────────────
        self.mu_opp_bid = 50.0   # EMA mean of opponent auction bids
        self.sigma_opp_bid = 30.0  # EMA std-dev of opponent auction bids
        self._EMA_BID_ALPHA = 0.15  # Learning rate for bid EMA

        # ── Hand-level state ───────────────────────────────────────────────
        self.last_opp_wager = 0
        self.last_pot = 0
        self.auction_won = True   # True = won/tied, False = lost
        self.my_bid = 0
        self.opp_bid_this_hand = None  # Observed after auction resolves

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.last_opp_wager = current_state.opp_wager
        self.last_pot = current_state.pot
        self.auction_won = True   # default until auction resolves
        self.my_bid = 0
        self.opp_bid_this_hand = None
        self.in_hand_aggression = 0.0

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """
        Update β (bluff rate), μ_bluff, and opponent bid EMA at showdown.
        """
        opp_revealed = [c.strip() for c in current_state.opp_revealed_cards if c.strip()]
        board = [c.strip() for c in current_state.board if c.strip()]

        # ── Update β from showdown ground truth ───────────────────────────
        if len(opp_revealed) == 2 and len(board) >= 3:
            try:
                opp_cards_eval = [ALL_CARDS[c] for c in opp_revealed + board]
                opp_tier = eval7.evaluate(opp_cards_eval) >> 24  # 0=HighCard … 8=StraightFlush
                # Proxy for opponent equity at showdown
                e_opp_proxy = min(1.0, opp_tier / 8.0)
                # Bluff magnitude: bet big relative to pot + had weak hand = bluff
                b_ratio = current_state.opp_wager / max(1.0, float(current_state.pot))
                m_bluff = max(0.0, min(1.0, b_ratio * (1.0 - e_opp_proxy)))
                # EMA update
                alpha_ema = 0.05
                self.beta = alpha_ema * m_bluff + (1.0 - alpha_ema) * self.beta
                self.beta = max(0.05, min(0.95, self.beta))  # keep in [0.05, 0.95]

                # Update μ_bluff (EMA of how much they bet when bluffing)
                if m_bluff > 0.3:
                    bet_size = current_state.opp_wager
                    self.mu_bluff = 0.1 * bet_size + 0.9 * self.mu_bluff
            except Exception:
                pass

        # ── Update opponent bid EMA if we observed their bid ──────────────
        # The engine typically exposes the auction result in opp_wager delta
        # or via a dedicated field; we infer it from context here.
        if self.opp_bid_this_hand is not None:
            bid = float(self.opp_bid_this_hand)
            old_mu = self.mu_opp_bid
            self.mu_opp_bid = self._EMA_BID_ALPHA * bid + (1.0 - self._EMA_BID_ALPHA) * old_mu
            diff = bid - old_mu
            self.sigma_opp_bid = math.sqrt(
                self._EMA_BID_ALPHA * diff**2 + (1.0 - self._EMA_BID_ALPHA) * self.sigma_opp_bid**2
            )
            self.sigma_opp_bid = max(5.0, self.sigma_opp_bid)  # floor to avoid collapse

    # =========================================================================
    # ENTRY POINT
    # =========================================================================

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        try:
            return self._decide(game_info, current_state)
        except Exception as e:
            print(f"CRASH AVOIDED: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            if current_state.street == 'auction':
                return ActionBid(0)
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

    # =========================================================================
    # MAIN DECISION ROUTER
    # =========================================================================

    def _decide(self, game_info: GameInfo, current_state: PokerState):
        my_hand = [c.strip() for c in current_state.my_hand if c.strip()]
        board = [c.strip() for c in current_state.board if c.strip()]
        opp_revealed = [c.strip() for c in current_state.opp_revealed_cards] \
                       if current_state.opp_revealed_cards else []

        P_current = current_state.pot
        S_eff = min(current_state.my_chips + current_state.my_wager,
                    current_state.opp_chips + current_state.opp_wager)
        C = current_state.cost_to_call
        bounds = current_state.raise_bounds
        min_raise, max_raise = (bounds if bounds is not None else (0, 0))

        # ── Track in-hand aggression ───────────────────────────────────────
        if current_state.opp_wager > self.last_opp_wager and self.last_pot > 0:
            wager_diff = current_state.opp_wager - self.last_opp_wager
            b_ratio = wager_diff / float(self.last_pot)
            self.in_hand_aggression = 0.8 * self.in_hand_aggression + 0.2 * b_ratio
        self.last_opp_wager = current_state.opp_wager
        self.last_pot = P_current

        # ── Infer auction outcome ─────────────────────────────────────────
        # If opp_revealed has a card after the auction street, we won.
        # We track whether we won by comparing our bid to known wager changes.
        if current_state.street != 'auction' and not self.auction_processed:
            self.auction_won = len(opp_revealed) > 0
            # Infer opponent bid from pot delta minus our bid
            # (heuristic; exact method depends on engine API)
            pot_delta = P_current - (current_state.my_wager + current_state.opp_wager)
            inferred_opp_bid = max(0, pot_delta - self.my_bid) if self.auction_won else self.my_bid
            if inferred_opp_bid > 0:
                self.opp_bid_this_hand = inferred_opp_bid
            self.auction_processed = True

        # ── Base equity from MLP ──────────────────────────────────────────
        E_0 = 0.5
        if _MLP_DATA is not None:
            feats = _extract_equity_features(my_hand, board, opp_revealed)
            E_0 = _mlp_predict(feats)

        # ── Board volatility (σ_board) ────────────────────────────────────
        E_adj = E_0
        sigma_board = 0.05
        if board and current_state.street not in ('pre-flop', 'auction'):
            known_cards = set(my_hand + board + opp_revealed)
            unknown_cards = [c for c in FULL_DECK_STR if c not in known_cards]
            sample_equities = []
            for card in random.sample(unknown_cards, min(5, len(unknown_cards))):
                feats = _extract_equity_features(my_hand, board + [card], opp_revealed)
                sample_equities.append(_mlp_predict(feats))
            if sample_equities:
                sigma_board = float(np.std(sample_equities))
            E_adj = max(0.01, E_0 - 0.5 * sigma_board)

        # ── Route ─────────────────────────────────────────────────────────
        if current_state.street == 'auction':
            return self._handle_auction(E_0, my_hand, board, S_eff, P_current)
        else:
            return self._handle_street(
                E_adj, S_eff, P_current, C, min_raise, max_raise, current_state
            )

    # =========================================================================
    # AUCTION MODULE
    # =========================================================================

    def _handle_auction(self, E_0, my_hand, board, S_eff, P_current):
        self.auction_processed = False  # reset for this hand's auction

        known_cards = set(my_hand + board)
        unknown_cards = [c for c in FULL_DECK_STR if c not in known_cards]

        # ── V_gain: expected |ΔE| from seeing one opponent card ───────────
        equity_shifts = []
        for card in random.sample(unknown_cards, min(12, len(unknown_cards))):
            feats = _extract_equity_features(my_hand, board, [card])
            E_i = _mlp_predict(feats)
            equity_shifts.append(abs(E_i - E_0))
        V_gain = float(np.mean(equity_shifts)) if equity_shifts else 0.05

        # ── V_leak: proxy via self-removal (2 MLP calls) ──────────────────
        V_leak = 0.0
        if len(my_hand) == 2 and len(unknown_cards) > 0:
            dummy = unknown_cards[0]
            feats_c1 = _extract_equity_features([my_hand[0], dummy], board, [])
            feats_c2 = _extract_equity_features([my_hand[1], dummy], board, [])
            E_c1 = _mlp_predict(feats_c1)
            E_c2 = _mlp_predict(feats_c2)
            V_leak = (abs(E_0 - E_c1) + abs(E_0 - E_c2)) / 2.0

        # ── P_implied: equity-anchored, not naive stack ───────────────────
        # E^3 crushes weak hands; (1 + β/2) expands vs maniacs
        P_implied = P_current + S_eff * (E_0 ** 3) * (1.0 + self.beta / 2.0)

        # ── B_true: honest Vickrey bid ────────────────────────────────────
        B_true = P_implied * (V_gain + V_leak)

        # ── Trap activation: only for monsters (E > 0.78) ─────────────────
        E_min_trap = 0.78
        if E_0 > E_min_trap:
            alpha_trap = ((E_0 - E_min_trap) / (1.0 - E_min_trap)) ** 2
        else:
            alpha_trap = 0.0

        # Randomised λ so opponent cannot reverse-engineer our threshold
        lam = random.uniform(1.5, 3.0)
        B_target = max(0.0, self.mu_opp_bid - lam * self.sigma_opp_bid)

        # ── Final bid: blend B_true toward B_target for strong hands ─────
        B_final = B_true + alpha_trap * max(0.0, B_target - B_true)
        B_final = int(max(0, min(B_final, S_eff)))

        self.my_bid = B_final
        return ActionBid(B_final)

    # =========================================================================
    # STREET BETTING MODULE
    # =========================================================================

    def _handle_street(self, E_adj, S_eff, P_current, C, min_raise, max_raise, state):
        beta = self.beta

        # ── Step 1: True Win Probability W ───────────────────────────────
        # Bluff-weighted: if opponent bluffs often, our equity improves
        W = E_adj * (1.0 - beta) + 1.0 * beta
        W = max(0.0, min(1.0, W))

        # ── Step 2: Auction modifier I_mod ───────────────────────────────
        # If lost auction, we have an information gap.
        # Against a maniac (high β), their "strong signal" means little → partial recovery
        if self.auction_won:
            I_mod = 1.0
        else:
            # Recovers from 0.5 toward 1.0 as β → 1.0 (maniac signals are unreliable)
            I_mod = 0.5 + 0.5 * beta

        # ── Step 3: Advantage score X ─────────────────────────────────────
        X = max(0.0, 2.0 * W - 1.0)  # 0 at W=0.5, 1.0 at W=1.0

        # ── Step 4: Raise Cap R_cap ───────────────────────────────────────
        # Asymptote at X→1.05 ensures exponential blow-up near the nuts.
        # (1 - 0.5*β): REDUCE raise vs maniacs — keep their bluffs alive
        denom = max(0.001, 1.05 - X)
        raw_curve = (X ** 3.5) / denom
        R_cap = S_eff * I_mod * raw_curve * (1.0 - 0.5 * beta)
        R_cap = min(R_cap, float(S_eff))  # hard cap at stack

        # ── Step 5: Call Cap C_cap ────────────────────────────────────────
        # Always wider than raise cap — catch bluffs they make after we check
        C_cap = R_cap + beta * self.mu_bluff * 1.5
        C_cap = min(C_cap, float(S_eff))

        # ── Step 6: Optimal opening raise size x* ─────────────────────────
        # x* = P*((1-W)/(2W-1) + β): balances fold equity vs call equity
        # Only meaningful when W > 0.5
        if W > 0.5 and (2.0 * W - 1.0) > 1e-6:
            x_star = P_current * ((1.0 - W) / (2.0 * W - 1.0) + beta)
        else:
            x_star = 0.0
        # Clamp x* to legal raise window and R_cap
        x_star = max(float(min_raise), min(x_star, R_cap, float(max_raise))) if max_raise > 0 else 0.0

        # ── Action routing ─────────────────────────────────────────────────
        if C > 0:
            # Facing a bet from the opponent
            if C <= R_cap:
                # Cost is within raise territory → re-raise to x* (or call if x* ≤ C)
                if state.can_act(ActionRaise) and x_star > C and max_raise > 0:
                    raise_amt = int(max(min_raise, min(max_raise, x_star)))
                    return ActionRaise(raise_amt)
                # Otherwise just call
                if state.can_act(ActionCall):
                    return ActionCall()
            elif C <= C_cap:
                # Too expensive to raise but within call range → call
                if state.can_act(ActionCall):
                    return ActionCall()
            # Cost exceeds C_cap → fold
            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()  # shouldn't reach here normally

        else:
            # We have the initiative (no bet to face, C == 0)
            if x_star >= min_raise and min_raise > 0 and state.can_act(ActionRaise):
                raise_amt = int(max(min_raise, min(max_raise, x_star)))
                return ActionRaise(raise_amt)
            # No profitable raise size → check
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()


# =============================================================================
# SIMULATION VERIFICATION (run with: python bot_rewrite.py --simulate)
# =============================================================================

