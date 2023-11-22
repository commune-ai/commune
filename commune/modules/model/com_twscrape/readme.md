# twscrape

Twitter GraphQL API implementation with [SNScrape](https://github.com/JustAnotherArchivist/snscrape) data models.

## Features
- Async/Await functions (can run multiple scrapers in parallel at the same time)
- Login accounts flow
- Saving/restoring account sessions
- Automatic account switching to scrape

### Add accounts & login

First add accounts

```sh
# Create "accounts.txt" file in the same directory. 
# Add twitter credentials in this format. username:password:email:email_password

c model.com_twscrape add_accounts
```

Then call login:

```sh
c model.com_twscrape login_accounts
```

### Get list of accounts and their statuses

```sh
c model.com_twscrape list_accounts

# Output:
# username  logged_in  active  last_used            total_req  error_msg
# user1     1          1       2023-11-14 13:20:40  100        None
# user2     1          1       2023-05-20 08:52:35  120        None
# user3     0          0       None                 120        Login error
```

### Re-login accounts

It is possible to re-login specific accounts:

```sh
c model.com_twscrape relogin user1 user2
```

### Search commands

```sh
c model.com_twscrape search "QUERY" 
c model.com_twscrape tweet_details TWEET_ID
c model.com_twscrape retweeters TWEET_ID 
c model.com_twscrape favoriters TWEET_ID 
c model.com_twscrape user_by_id USER_ID
c model.com_twscrape user_by_login USERNAME
c model.com_twscrape followers USER_ID 
c model.com_twscrape following USER_ID 
c model.com_twscrape user_tweets USER_ID 
c model.com_twscrape user_tweets_and_replies USER_ID 
```


## Screenshots of com_twscrape module usage

```sh
c model.com_twscrape search "commune" 
``````
<div align="center">
    <img src="https://i.ibb.co/HnhZp5T/commune-search1.png" alt="example of search usage" width="560px">   
    <img src="https://i.ibb.co/bND3bsN/commune-search2.png" alt="example of search usage" width="560px">
</div>

```sh
c model.com_twscrape tweet_details 1724619757721444670
``````
<div align="center">
    <img src="https://i.ibb.co/F6YJL2J/tweet-details.png" alt="example of tweet_details" width="560px">
</div>
