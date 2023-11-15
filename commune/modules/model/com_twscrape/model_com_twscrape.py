import commune as c
import asyncio

from typing import Any

from .api import API, AccountsPool
from .db import get_sqlite_version
from .logger import logger, set_log_level
from .models import Tweet, User
from .utils import print_table

class com_twscrape(c.Module):
    def __init__(self, db: str = "accounts.db", debug: bool = True):

        config = self.set_config(kwargs=locals())

        self.scrape_pool = AccountsPool(self.config.db)
        self.tw_api = API(self.scrape_pool, debug=self.config.debug)

    def list_accounts(self) -> Any:
        loop = asyncio.get_event_loop()
        accounts = loop.run_until_complete(self.scrape_pool.accounts_info())
        loop.close()
        print_table(accounts)

    def add_accounts(self):
        file_path = './accounts.txt'
        line_format:str = 'username:password:email:email_password'
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.scrape_pool.load_from_file(file_path, line_format))
        loop.close()

    def login_accounts(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.scrape_pool.login_all())
        loop.close()

    def del_accounts(self, username):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.scrape_pool.delete_accounts(username))
        loop.close()

    def search(self, q: str, limit=10, kv=None) -> Any:
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self._collect_search_results(self.tw_api.search(q, limit, kv)))
        loop.close()
        print(res)
        return res
    
    async def _collect_search_results(self, async_gen):
        return [item async for item in async_gen]
    
    def tweet_details(self, tweetId: int):
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.tw_api.tweet_details(twid=tweetId))
        loop.close()
        print(res)

    def retweeters(self, tweetId: int):
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self._collect_retweeters_results(self.tw_api.retweeters(tweetId, 10)))
        loop.close()
        print(res)
        return res
    
    async def _collect_retweeters_results(self, async_gen):
        return [item async for item in async_gen]
    
    

    def test(self):
        print("hello, i'm twscape")