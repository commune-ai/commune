# Twitter Scraping with Apify - README

This Python code is designed for scraping Twitter data using the Apify platform. It provides a `scrape` method that allows users to customize and filter the tweets you want to retrieve. Below are explanations for each of the parameters used in the `scrape` method:

## Here's a detailed explanation of parameters for `Twitter` scrapping:

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


## Here's a detailed explanation of parameters for `Reddit` scrapping:

- `debugMode (Boolean)`: If set to True, it enables debug mode, which may provide additional logging or debugging information during the scraping process.
- `maxComments (Integer)`: This parameter specifies the maximum number of comments to scrape. In this case, it is set to 10, meaning that the scraper will attempt to scrape up to 10 comments for the specified search query.
- `maxCommunitiesCount (Integer)`: This parameter sets the maximum number of communities (subreddits) to scrape. In this example, it is set to 2, meaning that the scraper will attempt to scrape posts and information from up to 2 different communities.
- `maxItems (Integer)`:The maxItems parameter defines the maximum number of items to scrape. Items can include posts, comments, users, or other entities. In this case, it is set to 10.
- `maxPostCount (Integer)`: This parameter specifies the maximum number of posts to scrape. With a value of 10, the scraper will try to scrape up to 10 posts.
- `maxUserCount (Integer)`: The maxUserCount parameter defines the maximum number of users to scrape. It is set to 2 in this example, meaning that the scraper will attempt to scrape information for up to 2 users.
- `proxy (Object)`: This object contains settings related to using a proxy server for scraping.
- `useApifyProxy (Boolean)`: If set to True, it indicates that the scraper should use the Apify proxy service to anonymize requests.
- `scrollTimeout (Integer)`: The scrollTimeout parameter sets the time, in seconds, that the scraper should wait for content to load when scrolling a webpage. In this case, it is set to 40 seconds.
- `searchComments (Boolean)`: If set to True, it enables searching and scraping comments related to the specified search query (in this case, "bitcoin").
- `searchCommunities (Boolean)`: When True, it allows the scraper to search and scrape information from different Reddit communities (subreddits).
- `searchPosts (Boolean)`: Enabling this parameter allows the scraper to search and scrape posts related to the specified query.
- `searchUsers (Boolean)`: When set to True, it enables the scraper to search and retrieve user information related to the specified search query.
- `searches (List of search keyward strings)`: This list contains the search queries that the scraper should use to search for content on Reddit. In this case, there is a single search query:"bitcoin"
- `skipComments (Boolean)`: If True, it skips the scraping of comments. If set to False, it will scrape comments when searchComments is also True.

### Usage Example:

To run this script, you can pass a JSON-formatted string as the filter parameter. The filter parameter allows you to customize the scraping criteria. If you want to scrape data from Twitter, you can specify type as "twitter," and if you want to scrape data from Reddit, you can set type as "reddit."

Here's an example of how to use the command:

`c model.apify scrape filter="{"queries":["bitcoin"], "max_attempts":10, "only_tweets":True}" type="twitter"`

In this example, we pass a JSON-formatted string as the filter parameter, which specifies the maximum number of attempts, search queries ("bitcoin"), and the data source ("twitter") to scrape. If you were to change type to "reddit," it would instruct the script to scrape data from Reddit.

You can customize the JSON string to include any combination of the parameters mentioned above, adjusting them to suit your specific scraping requirements. The provided example is just one way to use the command with different parameters and data sources.