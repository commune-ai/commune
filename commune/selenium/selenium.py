import commune as c

class Selenium(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

    def install(self):
        return c.cmd('pip3 install selenium')

    
    def test(self):
        from selenium import webdriver
        from selenium.webdriver.common.keys import Keys

        # Create a new instance of the Chrome driver
        driver = webdriver.Chrome()

        # Go to the website
        driver.get("http://google.com")

        # Find the search box
        search_box = driver.find_element_by_name("q")

        # Enter search text
        search_box.send_keys("Room in New York")

        # Submit the search
        search_box.send_keys(Keys.RETURN)

        # Wait for results and process them...

        # Close the browser
        driver.quit()
