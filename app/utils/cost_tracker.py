class QueryCostTracker:
    def __init__(self, usd_to_zar_conversion_rate: float = 18.0):
        """
        Initializes the QueryCostTracker.
        
        Args:
            usd_to_zar_conversion_rate (float): Conversion rate from USD to ZAR.
        """
        self.total_cost_usd = 0.0
        self.query_count = 0
        self.usd_to_zar_conversion_rate = usd_to_zar_conversion_rate
        self.query_costs = []  # Stores individual query costs for statistics
    
    def record_query_cost(self, cost_usd: float):
        """
        Records the cost of a single query.
        
        Args:
            cost_usd (float): Cost of the query in USD.
        """
        self.total_cost_usd += cost_usd
        self.query_count += 1
        self.query_costs.append(cost_usd)
        print(f"Total cost of query: {cost_usd:.6f} USD")
    
    def get_full_amount(self) -> float:
        """
        Returns the total cost of all queries in USD.
        
        Returns:
            float: Total cost in USD.
        """
        return self.total_cost_usd
    
    def get_cost_in_rands(self) -> float:
        """
        Converts the total cost from USD to ZAR using the fixed conversion rate.
        
        Returns:
            float: Total cost in ZAR.
        """
        return self.total_cost_usd * self.usd_to_zar_conversion_rate
    
    def get_average_cost_per_query(self) -> float:
        """
        Calculates the average cost per query.
        
        Returns:
            float: Average cost in USD.
        """
        if self.query_count == 0:
            return 0.0
        return self.total_cost_usd / self.query_count
    
    def get_query_stats(self) -> dict:
        """
        Provides statistics for the queries made in this session.
        
        Returns:
            dict: Dictionary containing total cost, average cost, and total cost in ZAR.
        """
        return {
            "total_cost_usd": round(self.get_full_amount(), 6),
            "total_cost_zar": round(self.get_cost_in_rands(), 6),
            "average_cost_per_query_usd": round(self.get_average_cost_per_query(), 6),
            "query_count": self.query_count,
        }