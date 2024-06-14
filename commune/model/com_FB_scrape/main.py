import commune as c
from facebook_page_scraper import Facebook_scraper

class FacebookPageDataScraper:
    def __init__(self, page_name = "metaai", posts_count = 10, browser = "chrome", proxy = "IP:PORT", timeout = 20, headless = True):
        self.page_name = page_name
        self.posts_count = posts_count
        self.browser = browser
        self.proxy = proxy
        self.timeout = timeout
        self.headless = headless
        self.scraper = Facebook_scraper(self.page_name, self.posts_count, self.browser, proxy=self.proxy, timeout=self.timeout, headless=self.headless)

    def scrap_to_json(self):
        print(self.scraper.scrap_to_json())

    def scrap_to_csv(self, filename = "scraped_csv", directory="./"):
        self.scraper.scrap_to_csv(filename, directory)