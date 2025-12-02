from ib_insync import IB, util
import pandas as pd

def connect_ibkr(host="127.0.0.1", port=7497, client_id=1):
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

def get_dom(ib: IB, symbol: str, exchange: str = "IDEALPRO", depth: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: price, bid_volume, ask_volume
    """
    contract = ib.qualifyContracts(util.contract(symbol, exchange=exchange))[0]
    # Subscribe to market depth (Level II)
    md = ib.reqMktDepth(contract, numRows=depth, isSmartDepth=False)
    # md is a list of MktDepthData objects
    rows = []
    for tick in md:
        rows.append({
            "price": tick.price,
            "bid_volume": tick.size if tick.side == 0 else 0,   # 0 = bid
            "ask_volume": tick.size if tick.side == 1 else 0,   # 1 = ask
        })
    df = pd.DataFrame(rows)
    df = df.groupby("price", as_index=False).sum()
    return df.sort_values("price", ascending=False)
