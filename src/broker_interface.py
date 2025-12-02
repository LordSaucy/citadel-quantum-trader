# broker_interface.py
class BrokerInterface(ABC):
    ...
    @abstractmethod
    def get_market_depth(self, symbol: str, depth: int = 30) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts:
        [
            {"price": 1.2345, "bid_vol": 1200, "ask_vol": 950},
            ...
        ]
        The list should be ordered by price (ascending or descending – your choice,
        as long as you treat it consistently).
        """
        raise NotImplementedError

 @abstractmethod
    def send_order(self, **kwargs) -> dict:
        """Send order to the *primary* broker. Returns a dict with at least:
        {'order_id': <id>, 'timestamp': <epoch>, 'price': <price>}
        """

    @abstractmethod
    def send_order_secondary(self, **kwargs) -> dict:
        """Send order to the *secondary* broker. Same contract as send_order."""

 @abstractmethod
    async def get_book_depth(self, symbol: str, price: float) -> float:
        """Return the total available volume (in lots) at the given price level."""
        ...

    @abstractmethod
    async def submit_order_async(
        self, symbol: str, volume: float, side: str, price: float | None = None
    ) -> str:
        """Submit an order and immediately return a *ticket* identifier.
        The concrete implementation must fire a websocket / callback when the order
        is fully filled, partially filled or rejected."""
        ...
