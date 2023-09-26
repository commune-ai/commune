import requests
from pprint import pprint
import commune as c

class GraphQuery(c.Module):
    
    description = "Query the Graph"

    @staticmethod
    def run_query(endpoint, q):
        request = requests.post(endpoint, json={'query': q})
        if request.status_code == 200:
            return request.json()
        else:
            raise Exception('Query failed. Return code is {}. {}'.format(request.status_code, q))

    @classmethod
    def aave_query(cls):
        endpoint = 'https://api.thegraph.com/subgraphs/name/aave/protocol'
        query = """
        {
        flashLoans (first: 10, orderBy: timestamp, orderDirection: desc,){
          id
          reserve {
            name
            symbol
          }
          amount
          timestamp
        }
        }
        """
        return cls.run_query(endpoint, query)

    @classmethod
    def uniswap_query(cls):
        endpoint = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2'
        query = """
        {
          pairs(first: 10, where: {reserveUSD_gt: "1000000", volumeUSD_gt: "50000"}, orderBy: reserveUSD, orderDirection: desc) {
            id
            token0 {
              id
              symbol
            }
            token1 {
              id
              symbol
            }
            reserveUSD
            volumeUSD
          }
        }
        """
        return cls.run_query(endpoint, query)

    @classmethod
    def test(cls):
        print("Running Aave Query")
        result = cls.aave_query()
        pprint(result)
        print("#############")

        print("Running Uniswap Query")
        result = cls.uniswap_query()
        pprint(result)
        print("#############")


if __name__ == "__main__":
    GraphQLQuerier.test_queries()
