from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class GoogleAgent:
    def __init__(self):
        # Initialize the Chrome driver
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        
    def start(self):
        """Start the browser and go to Google"""
        self.driver.get("https://www.google.com")
        
    def search(self, query):
        """Perform a Google search"""
        try:
            # Find the search box
            search_box = self.wait.until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            
            # Clear any existing text and enter the search query
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            # Wait for results to load
            self.wait.until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            
            return True
        except TimeoutException:
            print("Timeout while searching")
            return False
            
    def get_search_results(self, num_results=5):
        """Get search results"""
        try:
            # Wait for results to be present
            results = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g"))
            )
            
            # Extract and return results
            search_results = []
            for i, result in enumerate(results):
                if i >= num_results:
                    break
                    
                try:
                    title = result.find_element(By.CSS_SELECTOR, "h3").text
                    link = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    search_results.append({"title": title, "link": link})
                except:
                    continue
                    
            return search_results
            
        except TimeoutException:
            print("Timeout while getting results")
            return []
            
    def click_result(self, index):
        """Click on a search result by index"""
        try:
            results = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g h3"))
            )
            
            if 0 <= index < len(results):
                results.click()
                return True
            return False
            
        except TimeoutException:
            print("Timeout while clicking result")
            return False
            
    def go_back(self):
        """Go back to the previous page"""
        self.driver.back()
        
    def close(self):
        """Close the browser"""
        self.driver.quit()

# Example usage:
if __name__ == "__main__":
    # Create an instance of the GoogleAgent
    agent = GoogleAgent()
    
    try:
        # Start the browser
        agent.start()
        
        # Perform a search
        agent.search("Python programming")
        
        # Get search results
        results = agent.get_search_results(3)
        
        # Print results
        for i, result in enumerate(results):
            print(f"Result {i + 1}:")
            print(f"Title: {result['title']}")
            print(f"Link: {result['link']}")
            print("---")
        
        # Click the first result
        agent.click_result(0)
        
        # Wait a few seconds to see the page
        import time
        time.sleep(3)
        
        # Go back to search results
        agent.go_back()
        
        # Wait a few more seconds
        time.sleep(3)
        
    finally:
        # Close the browser
        agent.close()

class EnhancedGoogleAgent(GoogleAgent):
    def scroll_down(self):
        """Scroll down the page"""
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
    def scroll_up(self):
        """Scroll up the page"""
        self.driver.execute_script("window.scrollTo(0, 0);")
        
    def take_screenshot(self, filename):
        """Take a screenshot of the current page"""
        self.driver.save_screenshot(filename)
        
    def get_page_text(self):
        """Get all text from the current page"""
        return self.driver.find_element(By.TAG_NAME, "body").text
    
    def navigate_to(self, url):
        """Navigate to a specific URL"""
        self.driver.get(url)
        
    def get_current_url(self):
        """Get the current URL"""
        return self.driver.current_url

    def run(self):
        # Example usage with enhanced features:
        agent = EnhancedGoogleAgent()
        
        try:
            # Start the browser
            agent.start()
            
            # Perform a search
            agent.search("Python web scraping")
            
            # Take a screenshot of search results
            agent.take_screenshot("search_results.png")
            
            # Scroll down and up
            agent.scroll_down()
            time.sleep(1)
            agent.scroll_up()
            
            # Get current URL
            current_url = agent.get_current_url()
            print(f"Current URL: {current_url}")
            
            # Navigate to a specific website
            agent.navigate_to("https://python.org")
            
            # Get page text
            page_text = agent.get_page_text()
            print("First 200 characters of page text:")
            print(page_text[:200])
            
        finally:
            agent.close()