import commune as c
from apify_client import ApifyClient
from dotenv import load_dotenv
import os

load_dotenv()

class Apify(c.Module):
    def __init__(self):

        self.client = ApifyClient(os.getenv("apify_token"))

    def scrape(self, twitterFilter):
        twitter_actor_id = os.getenv("twitter_actor_id")
        twitter_run_input = {
            "filter:blue_verified": False,
            "filter:has_engagement": False,
            "filter:images": False,
            "filter:media": False,
            "filter:nativeretweets": False,
            "filter:quote": False,
            "filter:replies": False,
            "filter:retweets": False,
            "filter:safe": False,
            "filter:twimg": False,
            "filter:verified": False,
            "filter:videos": False,
            "max_tweets": 10,
            "only_tweets": False,
            "queries": ["bit"],
            "use_experimental_scraper": False,
            "language": "any",
            "user_info": "user info and replying info",
            "max_attempts": 5,
            "actor_id": twitter_actor_id
        }
        twitter_run_input.update(twitterFilter)

        run = self.client.actor(twitter_run_input["actor_id"]).call(run_input=twitter_run_input)

        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            print(item)


# if __name__ == '__main__':
#     scraper = Apify(os.getenv("apify_token"))
#     twitterFilter = {"queries":["bitcoin"]}
#     scraper.scrape(twitterFilter)