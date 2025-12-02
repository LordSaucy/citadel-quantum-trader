import asyncio
import time
from decimal import Decimal
from .config_loader import Config
from .broker_interface import BrokerInterface
from .utils import pips_to_price  # helper that converts pips → price delta

cfg = Config().settings
LATENCY_COST = cfg["arb"]["latency_allowance_secs"]
LATENCY_MULT = cfg["arb"]["latency_multiplier"]
DEPTH_MULT = cfg["arb"]["min_depth_multiplier"]
SPREAD_BUFFER = cfg["arb"]["spread_buffer_pips"]
PARTIAL_TIMEOUT = cfg["arb"]["partial_fill_timeout_secs"]

class ArbExecutionError(RuntimeError):
    pass

async def execute_triangular_arb(
    broker: BrokerInterface,
    legs: list[dict],   # each leg: {"symbol": "...", "side": "buy"/"sell", "volume": float}
    gross_profit_pips: float,
) -> None:
    """
    *legs* must be ordered as they will be sent (first → second → third).
    The function raises `ArbExecutionError` on any guard violation.
    """

    # -----------------------------------------------------------------
    # 1️⃣ Latency allowance check
    # -----------------------------------------------------------------
    # Estimated latency cost (pips) = latency (sec) * (price‑movement‑per‑sec)
    # For FX we approximate 1 pips ≈ 0.0001 price change.
    latency_cost_pips = LATENCY_COST * 10_000   # 0.15 s → ~1.5 pips (rough)
    required_profit = max(gross_profit_pips, LATENCY_MULT * latency_cost_pips)

    if gross_profit_pips < required_profit:
        raise ArbExecutionError(
            f"Arb profit {gross_profit_pips:.2f} pips < latency‑adjusted "
            f"requirement {required_profit:.2f} pips"
        )

    # -----------------------------------------------------------------
    # 2️⃣ Depth check for each leg
    # -----------------------------------------------------------------
    for leg in legs:
        # Query the order‑book at the *mid* price (or the price we intend to trade)
        # For simplicity we use the current market price:
        market_price = await broker.get_current_price(leg["symbol"])
        depth = await broker.get_book_depth(leg["symbol"], market_price)

        if depth < DEPTH_MULT * leg["volume"]:
            raise ArbExecutionError(
                f"Insufficient depth on {leg['symbol']} (have {depth:.2f} lots, "
                f"need {DEPTH_MULT * leg['volume']:.2f})"
            )

    # -----------------------------------------------------------------
    # 3️⃣ Submit first leg, wait for fill confirmation
    # -----------------------------------------------------------------
    tickets = []
    for i, leg in enumerate(legs):
        ticket = await broker.submit_order_async(
            symbol=leg["symbol"],
            volume=leg["volume"],
            side=leg["side"],
        )
        tickets.append(ticket)

        # After the *first* leg we must verify it is fully filled before
        # proceeding to the next leg.
        if i == 0:
            filled = await _wait_for_fill(broker, ticket, PARTIAL_TIMEOUT)
            if not filled:
                # Abort the remaining legs – cancel any open orders
                await _cancel_pending_orders(broker, tickets[1:])
                raise ArbExecutionError(
                    f"First leg {leg['symbol']} not fully filled after {PARTIAL_TIMEOUT}s"
                )
        # Small pause to give the broker a breath (avoid hammering)
        await asyncio.sleep(0.05)

    # -----------------------------------------------------------------
    # 4️⃣ Spread‑adjusted profit verification (post‑trade)
    # -----------------------------------------------------------------
    net_profit_pips = await _compute_net_arb_profit(broker, legs)
    if net_profit_pips < SPREAD_BUFFER:
        # Undo the whole arb – cancel all three legs (they should already be filled,
        # but we can send opposite‑direction orders to neutralise)
        await _neutralise_arb(broker, legs)
        raise ArbExecutionError(
            f"Net arb profit {net_profit_pips:.2f} pips < spread buffer {SPREAD_BUFFER} pips"
        )

    # If we reach here the arb is considered successful
    return None


# -----------------------------------------------------------------
# Helper utilities (private to this module)
# -----------------------------------------------------------------
async def _wait_for_fill(broker: BrokerInterface, ticket: str, timeout: float) -> bool:
    """Poll the broker until the order is fully filled or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = await broker.get_order_status(ticket)   # should return "filled", "partial", "rejected"
        if status == "filled":
            return True
        if status == "rejected":
            return False
        await asyncio.sleep(0.1)
    return False


async def _cancel_pending_orders(broker: BrokerInterface, tickets: list[str]) -> None:
    for t in tickets:
        await broker.cancel_order(t)


async def _compute_net_arb_profit(broker: BrokerInterface, legs: list[dict]) -> float:
    """
    Pull the *actual* execution price for each leg, compute the
    P&L in pips, then subtract the *real* spread that was in effect at
    the moment of each fill.
    """
    total_pips = 0.0
    for leg in legs:
        fill = await broker.get_last_fill(leg["symbol"], leg["side"])
        exec_price = fill["price"]
        market_price = await broker.get_current_price(leg["symbol"])
        # profit in pips = (exec – market) * 10 000 for FX
        pips = (exec_price - market_price) * 10_000
        if leg["side"] == "sell":
            pips = -pips
        # Subtract the spread that existed at fill time
        spread = await broker.get_spread_at_timestamp(leg["symbol"], fill["timestamp"])
        spread_pips = spread * 10_000
        total_pips += pips - spread_pips
    return total_pips
