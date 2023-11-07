# Twitter Scraping with Apify - README

This Python code is designed for scraping Twitter data using the Apify platform. It provides a `scrape` method that allows users to customize and filter the tweets you want to retrieve. Below are explanations for each of the parameters used in the `scrape` method:

## Here's a detailed explanation of parameters for twitter scrapping:

- `filter:blue_verified`: A boolean value to filter tweets based on whether they are from Twitter accounts with blue verification badges.

- `filter:has_engagement`: A boolean value to filter tweets based on whether they have user engagement (e.g., likes, retweets).

- `filter:images`: A boolean value to filter tweets that contain images.

- `filter:media`: A boolean value to filter tweets that contain media content.

- `filter:nativeretweets`: A boolean value to filter native retweets.

- `filter:quote`: A boolean value to filter tweets that are quotes.

- `filter:replies`: A boolean value to filter tweets that are replies to other tweets.

- `filter:retweets`: A boolean value to filter retweets.

- `filter:safe`: A boolean value to filter tweets that are considered safe.

- `filter:twimg`: A boolean value to filter tweets that contain "twimg" content.

- `filter:verified`: A boolean value to filter tweets from verified Twitter accounts.

- `filter:videos`: A boolean value to filter tweets that contain videos.

- `max_tweets`: An integer representing the maximum number of tweets to retrieve.

- `only_tweets`: A boolean value to specify whether to retrieve only tweets (True) or a combination of tweets and other content (False).

- `queries`: A list of search queries to filter tweets based on keywords.

- `use_experimental_scraper`: A boolean value to enable or disable the use of an experimental web scraper.

- `language`: A string specifying the desired language for the retrieved content (e.g., "en" for English, "any" for any language).

- `user_info`: A string specifying the level of user information and replying information to retrieve.

- `max_attempts`: An integer representing the maximum number of attempts to retrieve tweets.

- `actor_id`: The identifier of the Twitter scraping actor.

### Usage Example:

To run this script, you can pass a JSON-formatted string as the twitterFilter parameter. Here's an example of how to use the command:

`c model.apify scrape twitterFilter="{"max_attempts":10, "queries":["bitcoin"], "only_tweets":True}"`

In this example, we pass a JSON-formatted string as the twitterFilter parameter, which specifies the maximum number of attempts, search queries ("bitcoin"), and that only tweets should be retrieved.

You can customize the JSON string to include any combination of the parameters mentioned above, adjusting them to suit your specific scraping requirements. The provided example is just one way to use the command with different parameters.