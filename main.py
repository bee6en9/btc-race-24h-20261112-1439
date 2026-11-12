#!/usr/bin/env python3
"""BTC race bot — agent-fun/bars-replay/v1, single file, stdlib only.

Strategy (momentum ignition + regime filter + take-profit / trailing stop):
  * ENTER (long only, all-in) when, on a 1-minute close:
      - the 30-bar return is > 25 bps (momentum ignition), AND
      - the close is above its 120-bar EMA (short-term trend up), AND
      - the 240-bar return is > 0 — or, before 240 bars exist, the return
        since the first close is > 0 (medium-term regime up).
    A fresh ignition is required after any exit: the signal must have been
    false for 30 consecutive bars first (no re-entry on a stale signal), AND
    a re-entry must break above the highest close since the exit (the old
    rally's high must be exceeded — bounces below it are not chased).
  * EXIT on whichever comes first:
      - take-profit: the position's value at the close is >= 0.8% above its
        cost basis, or
      - trailing stop: the close falls >= 1.0% below the highest close
        since entry.
  * No new entries in the last 60 bars (a trade needs time to develop; the
    window flattens everything at the end).
  * Gap bars (a batch with no CBBTC bar) carry no new close: no indicator
    update, no new orders on that batch.

Deterministic: integer math for returns/thresholds, one float EMA (IEEE754,
same input -> same output), no clock, no network, no randomness, no files.
One reply per batch, echoing its seq. Diagnostics: stderr only.
"""
import json
import sys

SYMBOL = "CBBTC"
# --- strategy parameters (validated on the dev window; see dev/research6.py) ---
MOM_LOOK = 30          # momentum lookback (bars)
MOM_THR_BPS = 25       # momentum ignition threshold (bps)
EMA_SPAN = 120         # trend EMA span (bars)
REGIME_LOOK = 240      # medium-term regime lookback (bars)
TP_BPS = 80            # take-profit above cost basis (bps)
STOP_BPS = 100         # trailing stop below peak close (bps)
RESET_BARS = 30        # consecutive false-signal bars required after an exit
MAX_SLIP_BPS = 100     # max slippage allowed on fills (bps)
NO_ENTRY_AFTER_SEQ = 1380  # no new entries in the last 60 bars

# price_scale 5, qty_scale 8, quote_scale 6 (from hello; fixed by the protocol)
PRICE_SCALE = 5
QTY_SCALE = 8
QUOTE_SCALE = 6


def value_quote(qty, price):
    """Value of qty at price, in quote units (truncating), as the engine does."""
    return qty * price * (10 ** QUOTE_SCALE) // (10 ** (QTY_SCALE + PRICE_SCALE))


class Bot:
    def __init__(self):
        self.closes = []          # closes seen so far (ints, price_scale 5)
        self.ema = None          # 120-bar EMA of closes (float)
        self.cash = 0            # quote units
        self.qty = 0             # held qty (qty_scale 8)
        self.cost = 0            # cost basis of the position (quote units)
        self.peak = None         # highest close since entry (int)
        self.entry_close = None  # close at the bar the buy was decided on
        # consecutive bars the signal has been false (counted through the
        # previous bar); starts high so the first trade is never blocked
        self.false_run = 10 ** 9
        self.ever_exited = False
        # highest close since the last exit (None until the first exit);
        # a re-entry must break above it
        self.high_since_exit = None
        self.retry = False       # the previous buy was rejected: allow a retry
        self.pending_buy = None  # client_id of the buy awaiting its fill
        self.next_id = 1
        self.seq = -1

    def client_id(self):
        cid = "o%d" % self.next_id
        self.next_id += 1
        return cid

    def signal(self):
        """True when the entry conditions hold on the latest close."""
        n = len(self.closes)
        if n <= MOM_LOOK:
            return False
        c = self.closes[-1]
        # 30-bar return in bps (integer, truncating)
        r30 = (c - self.closes[-1 - MOM_LOOK]) * 10000 // self.closes[-1 - MOM_LOOK]
        if r30 <= MOM_THR_BPS:
            return False
        # above the 120-bar EMA
        if self.ema is None or c <= self.ema:
            return False
        # medium-term regime: 240-bar return > 0 (or since-start return > 0 early)
        if n > REGIME_LOOK:
            if c <= self.closes[-1 - REGIME_LOOK]:
                return False
        else:
            if c <= self.closes[0]:
                return False
        return True

    def on_batch(self, batch):
        seq = batch["seq"]
        self.seq = seq
        # --- fold the account in (authoritative cash / position) ---
        acct = batch["account"]
        self.cash = int(acct["cash_quote_units"])
        pos = None
        for p in acct["positions"]:
            if p["symbol"] == SYMBOL:
                pos = p
                break
        was_in = self.qty > 0
        if pos is not None:
            self.qty = int(pos["qty"])
            self.cost = int(pos["cost_quote_units"])
        else:
            self.qty = 0
            self.cost = 0
        in_pos = self.qty > 0
        if was_in and not in_pos:
            # position closed (sold / flattened): arm the reset rule
            self.ever_exited = True
            self.false_run = 0
            self.peak = None
            self.entry_close = None
        if in_pos and self.peak is None:
            # just entered: seed the trailing-stop peak from the decision close
            self.peak = self.entry_close if self.entry_close is not None else self.closes[-1]

        # --- learn the pending buy's outcome from this batch's fills ---
        if self.pending_buy is not None:
            for f in batch.get("fills", []):
                if f.get("client_id") == self.pending_buy:
                    if f.get("status") == "rejected":
                        # buy never filled (e.g. its fill bar was a gap):
                        # allow one retry on the next true signal
                        self.retry = True
                    self.pending_buy = None
                    break

        # --- fold the new bar in (a gap batch has no bar: nothing to update) ---
        bar = None
        for b in batch["bars"]:
            if b["symbol"] == SYMBOL:
                bar = b
                break
        if bar is not None:
            c = int(bar["c"])
            self.closes.append(c)
            k = 2.0 / (EMA_SPAN + 1)
            self.ema = c if self.ema is None else self.ema + k * (c - self.ema)
            if in_pos and self.peak is not None and c > self.peak:
                self.peak = c

        orders = []
        # last batch: orders would be rejected (after_end); never send any
        if seq >= 1439 or not self.closes:
            return orders
        if bar is None:
            # gap bar: no new information, no new orders
            return orders

        sig = self.signal()
        # Reset gate, judged on the false-run BEFORE this bar: after an exit,
        # the signal must have been false for RESET_BARS consecutive bars
        # before a new ignition may trigger an entry. (Judging it after the
        # update would make re-entry impossible, since a true signal resets
        # the run to zero.)
        reset_ok = (not self.ever_exited) or (self.false_run >= RESET_BARS) or self.retry
        # Breakout gate, judged on the post-exit high BEFORE this bar: a
        # re-entry must close above the highest close since the exit.
        newhigh_ok = (not self.ever_exited) or (
            self.high_since_exit is not None and self.closes[-1] > self.high_since_exit)
        if sig:
            self.false_run = 0
        else:
            self.false_run += 1

        if in_pos:
            c = self.closes[-1]
            # take-profit: position value >= cost * (1 + TP_BPS/10000)
            tp_hit = value_quote(self.qty, c) * 10000 >= self.cost * (10000 + TP_BPS)
            # trailing stop: close >= STOP_BPS below the peak since entry
            stop_hit = self.peak is not None and c * 10000 <= self.peak * (10000 - STOP_BPS)
            if tp_hit or stop_hit:
                orders.append({
                    "side": "sell",
                    "client_id": self.client_id(),
                    "symbol": SYMBOL,
                    "qty": str(self.qty),
                    "max_slippage_bps": MAX_SLIP_BPS,
                })
                # the post-exit high starts at this exit bar's close
                self.high_since_exit = c
        else:
            # Track the post-exit high. This runs after newhigh_ok was judged
            # above, so a breakout bar still sees the pre-bar high.
            if self.ever_exited and self.high_since_exit is not None:
                if self.closes[-1] > self.high_since_exit:
                    self.high_since_exit = self.closes[-1]
            if (sig and reset_ok and newhigh_ok
                    and seq < NO_ENTRY_AFTER_SEQ and self.cash >= 1000000):
                cid = self.client_id()
                orders.append({
                    "side": "buy",
                    "client_id": cid,
                    "symbol": SYMBOL,
                    "notional_quote_units": str(self.cash),
                    "max_slippage_bps": MAX_SLIP_BPS,
                })
                self.entry_close = self.closes[-1]
                self.pending_buy = cid
                self.retry = False
        return orders


def main():
    bot = Bot()
    out = sys.stdout
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            # nothing to configure: scales and symbol are fixed by the protocol
            continue
        if kind == "batch":
            orders = bot.on_batch(msg)
            out.write(json.dumps(
                {"type": "reply", "seq": msg["seq"], "orders": orders},
                separators=(",", ":"), ensure_ascii=False) + "\n")
            out.flush()
        elif kind == "done":
            break


if __name__ == "__main__":
    main()
