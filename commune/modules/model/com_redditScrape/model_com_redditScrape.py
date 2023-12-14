from .core.selenium_scraper import SeleniumScraper
from .core.soup_scraper import SoupScraper
from .core.progress_bar import ProgressBar
from .core.sql_access import SqlAccess
import time
import commune as c

class RedditScraper(c.Module):
    def __init__(self, subreddit="DataScience", sort_by='/hot/', scroll_n_times=10, scrape_comments=True, erase_db_first=True):
        self.reddit_home = 'https://www.reddit.com'
        self.slash = '/r/'
        self.subreddit = subreddit
        self.sort_by = sort_by
        self.scroll_n_times = scroll_n_times
        self.scrape_comments = scrape_comments
        self.erase_db_first = erase_db_first

        self.SQL = SqlAccess()
        self.SelScraper = SeleniumScraper()
        self.BSS = SoupScraper(self.reddit_home, self.slash, self.subreddit)

    def run(self):
        start = time.time()

        self.setup_chrome_browser()

        # Collect links from subreddit
        links = self.collect_links()

        # Process the data
        self.process_data(links)

        # Save to database
        self.save_to_database(links)

        end = time.time()
        print(f'\nIt took {round(end - start, 1)} seconds to scrape')

    def setup_chrome_browser(self):
        self.SelScraper.setup_chrome_browser()

    def collect_links(self):
        page = self.reddit_home + self.slash + self.subreddit + self.sort_by
        return self.SelScraper.collect_links(page=page, scroll_n_times=self.scroll_n_times)

    def process_data(self, links):
        # Find the <script> with id='data' for each link
        script_data = self.BSS.get_scripts(urls=links)

        # Transforms each script with data into a Python dict
        self.BSS.data = self.SelScraper.reddit_data_to_dict(script_data=script_data)

        print('Scraping data...')
        progress = ProgressBar(len(links))
        for i, current_data in enumerate(self.BSS.data):
            progress.update()
            self.process_each_post(current_data, i)

    def process_each_post(self, current_data, index):
        self.BSS.get_url_id_and_url_title(self.BSS.urls[index], current_data, index)
        self.BSS.get_title()
        # ... (add all other get methods here)

    def save_to_database(self, links):
        try:
            self.SQL.create_or_connect_db(erase_first=self.erase_db_first)
            for i in range(len(self.BSS.post_data)):
                self.SQL.insert('post', data=self.BSS.post_data[i])
                self.SQL.insert('link', data=self.BSS.link_data[i])
                if self.scrape_comments:
                    self.SQL.insert('comment', data=self.BSS.comment_data[i])
        except Exception as ex:
            print(ex)
        finally:
            self.SQL.save_changes()
