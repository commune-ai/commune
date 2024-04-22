import time
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from .progress_bar import ProgressBar

class SeleniumScraper():
    def __init__(self):
        self.driver = None
        self.page = ''
        self.links = []
    
    def setup_chrome_browser(self):
        '''
            This function allows for setting up a chrome driver for use with
            Selenium. It expects a path to a chromedriver, available for 
            download on this link: https://chromedriver.chromium.org/home
        '''

        if os.name == 'posix':
            chromedriver = '/chromedriver'
            print("-------------chromedriver")
        elif os.name == 'nt':
            chromedriver = '/chromedriver.exe'

        options = webdriver.ChromeOptions()
        prefs = {"profile.default_content_setting_values.notifications" : 2}
        options.add_experimental_option("prefs", prefs)
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument("--headless")

        # path = os.path.dirname(os.path.abspath(__file__))
        # service = Service(executable_path=path + chromedriver)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        chromedriver_path = os.path.join(script_directory, 'chromedriver')
        service = Service(executable_path=chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=options)

        
    def collect_links(self,
                      page,
                      scroll_n_times):
        '''
            This function opens a page in a browser and scrolls n times to the
            bottom of the page. After that, it finds all the elements as
            specified by the xpath, then finds the href attribute.
            
            Parameters:
                page : string
                    This is the URL you want to collect links for.
                
                scroll_n_times : int
                    How many times you want to scroll to the bottom of page
                
                xpath_element : string
                    In the xpath style, you should define which element and
                    value you want the value of the href attribute from.
            
            Returns:
                links : array
                    An array of links to the URLs scraped.
        '''
        if(scroll_n_times < 0):
            raise ValueError('scroll_n_times must be greater' +
                             'than or equal to 0')
        
        self.page = page.lower()
        self.driver.get(page)
        
        # Selects all the a elements that have a "data-click-id" attribute with a value of "body"
        # https://stackoverflow.com/questions/36019544/if-double-slash-is-used-2-times-in-xpath-what-does-it-mean
        xpath = "//a[@data-click-id='body']"
        
        sleep_time = 0.5
        if(scroll_n_times != 0):
            print(('Opening reddit and scrolling: takes approximately {0} seconds'
                   ).format(sleep_time*scroll_n_times))
        else:
            print('Opening reddit and scrolling.. done')
        
        try:
            # When scroll_n_times = 0, loop stops
            while scroll_n_times:
                # Scrolls browser to the bottom of the page
                self.driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(sleep_time)
                scroll_n_times -= 1
            
            elements = self.driver.find_elements("xpath", xpath)
            
            # Get the link from the href attribute
            self.links = [tag.get_attribute('href') for tag in elements]
            
        finally:
            self.driver.quit()
        
        
        
        return self.links

    
    def reddit_data_to_dict(self,
                            script_data = []):
        '''
            Takes id='data' as input and outputs a dict with all ids from 
            page input
        '''
        
        pure_dicts = []
        
        print('Making Python dicts out of script data')
        
        progress = ProgressBar(len(script_data))
        for data in script_data:
            progress.update()
            
            first_index = data.index('{')
            last_index = data.rfind('}') + 1
            
            json_str = data[first_index:last_index]
            
            pure_dicts.append(json.loads(json_str))
        
        return pure_dicts