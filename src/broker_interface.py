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
