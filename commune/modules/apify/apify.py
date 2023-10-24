from apify_client import ApifyClient


class TweetScraper:
    def __init__(self, api_token):
        self.client = ApifyClient(api_token)

    def scrape_tweets(self):
        run_input = {
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
            "queries": [
                "bitcoin"
            ],
            "use_experimental_scraper": False,
            "language": "any",
            "user_info": "user info and replying info",
            "max_attempts": 5
        }

        run = self.client.actor("wHMoznVs94gOcxcZl").call(run_input=run_input)

        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            print(item)


if __name__ == '__main__':
    api_token = "apify_api_PWSZ5jVZhtpANm6hPDVTFdPja4Gnqc4kfdd3"
    scraper = TweetScraper(api_token)
    scraper.scrape_tweets()
