import commune as c
class CompareTokenPrice(c.Module):
    description = """"Compare the price of 2 tokens"""
    def call(self, token1_name: str, token1_price: float, token2_name: str, token2_price: float, find_cheaper: bool) -> str:
        """
        Compares the prices of two tokens and returns the name of the cheaper or more expensive token based on the boolean input.

        :param token1_name: Name of the first token.
        :param token1_price: Price of the first token.
        :param token2_name: Name of the second token.
        :param token2_price: Price of the second token.
        :param find_cheaper: Boolean value. If True, the function returns the cheaper token; if False, it returns the more expensive token.
        :return: Name of the cheaper or more expensive token.
        """
        
        # Check if the prices are equal
        if token1_price == token2_price:
            return "Both tokens have the same price."
        
        # Find and return the cheaper or more expensive token based on the boolean input
        if find_cheaper:
            return token1_name if token1_price < token2_price else token2_name
        else:
            return token1_name if token1_price > token2_price else token2_name



# if __name__ == "__main__":


#     # Example usage:
#     token1_name = "ETH"
#     token1_price = 3000.0
#     token2_name = "BTC"
#     token2_price = 45000.0
    
#     cheaper_token = CompareTokenPrice().call(token1_name, token1_price, token2_name, token2_price, find_cheaper=True)
#     print(f"The cheaper token is {cheaper_token}")

#     more_expensive_token = CompareTokenPrice().call(token1_name, token1_price, token2_name, token2_price, find_cheaper=False)
#     print(f"The more expensive token is {more_expensive_token}")