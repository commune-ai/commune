# Twitter Scraping with Apify - README

This Python code is designed for scraping Twitter data using the Apify platform. It provides a `scrape` method that allows users to customize and filter the tweets you want to retrieve. Below are explanations for each of the parameters used in the `scrape` method:

### Parameters for the `scrape` Method:

- `blue_verified` (boolean): If set to `True`, this parameter filters tweets from users with a blue verification badge. Default is `False`.

- `has_engagement` (boolean): If set to `True`, this parameter filters tweets that have user engagement, including likes, retweets, and replies. Default is `False`.

- `images` (boolean): If set to `True`, this parameter filters tweets that contain images. Default is `False`.

- `media` (boolean): If set to `True`, this parameter filters tweets that contain media, such as images and videos. Default is `False`.

- `nativeretweets` (boolean): If set to `True`, this parameter filters native retweets (retweets of original tweets). Default is `False`.

- `quote` (boolean): If set to `True`, this parameter filters tweets that are quoted. Default is `False`.

- `replies` (boolean): If set to `True`, this parameter filters replies to other tweets. Default is `False`.

- `retweets` (boolean): If set to `True`, this parameter filters retweets of other tweets. Default is `False`.

- `safe` (boolean): If set to `True`, this parameter filters tweets that are considered safe. Default is `False`.

- `twimg` (boolean): If set to `True`, this parameter filters tweets with the "twimg.com" domain, which typically host images. Default is `False`.

- `verified` (boolean): If set to `True`, this parameter filters tweets from verified Twitter accounts. Default is `False`.

- `videos` (boolean): If set to `True`, this parameter filters tweets that contain videos. Default is `False`.

- `max_tweets` (integer): The maximum number of tweets to retrieve. The default is set to 10.

- `only_tweets` (boolean): If set to `True`, this parameter filters only tweets and excludes other types of Twitter data, such as user profiles. Default is `False`.

- `keyword` (string): The search keyword or hashtag to use when scraping tweets. Default is set to "bitcoin."

- `experimental_scraper` (boolean): If set to `True`, this parameter enables the use of an experimental scraper. Default is `False`.

- `language` (string): Specifies the language to filter tweets by. The default value is "any."

- `user_info` (string): Specifies what user information to include in the scraped data. The default value is "user info and replying info."

- `max_attempts` (integer): The maximum number of attempts to scrape data. The default value is 5.

### Usage Example:

To scrape tweets related to a specific keyword (e.g., "bitcoin"), you can use the following code:

## Call the scrape method with custom parameters
scraper.scrape(
    blue_verified=True,
    has_engagement=True,
    images=True,
    media=True,
    max_tweets=20,  # You can specify the number of tweets you want to retrieve
    keyword="bitcoin"
)
