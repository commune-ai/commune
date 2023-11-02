import commune as c
from apify_client import ApifyClient
from dotenv import load_dotenv
import os

load_dotenv()

class Apify(c.Module):
    def __init__(self):

        self.client = ApifyClient(os.getenv("apify_token"))

    def scrape(self, 
               blue_verified = False, has_engagement = False, images = False, media = False, nativeretewwtes = False, quote = False, replies = False, retweets = False,
               safe = False, twimg = False, verified = False, videos = False, max_tweets = 10, only_tweets = False,
               keyword = "bitcoin", experimental_scraper = False, language = "any", user_info = "user info and replying info", max_attempts = 5):
        run_input = {
            "filter:blue_verified": blue_verified,
            "filter:has_engagement": has_engagement,
            "filter:images": images,
            "filter:media": media,
            "filter:nativeretweets": nativeretewwtes,
            "filter:quote": quote,
            "filter:replies": replies,
            "filter:retweets": retweets,
            "filter:safe": safe,
            "filter:twimg": twimg,
            "filter:verified": verified,
            "filter:videos": videos,
            "max_tweets": max_tweets,
            "only_tweets": only_tweets,
            "queries": [keyword],
            "use_experimental_scraper": experimental_scraper,
            "language": language,
            "user_info": user_info,
            "max_attempts": max_attempts
        }

        run = self.client.actor("wHMoznVs94gOcxcZl").call(run_input=run_input)

        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            print(item)


if __name__ == '__main__':
    keyword = "bitcoin"
    scraper = TweetScraper(os.getenv("apify_token"))
    scraper.scrape_tweets(keyword)