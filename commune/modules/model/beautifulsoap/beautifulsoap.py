import commune as c
import requests
from bs4 import BeautifulSoup
import random
import urllib
import os

class BeautifulSoapModule(c.Module):

    def sibling_scrap(self, url, keyword):
        # Get html content from the page
        data = requests.get(url)
        html = BeautifulSoup(data.text, 'html.parser')

        # Get a sample component which includes the keyword
        sample_text = html.find(text=lambda text: text and keyword in text)
        if not sample_text:
            return "Cannot find keyword." 
        else:
            sample_component = sample_text.find_parent()

            # Find similar level component with sample component
            tag = sample_component.name
            class_names = sample_component.get("class")

            if not class_names:
                components = html.select(tag)
            else:
                components = html.select(f"{tag}.{class_names[0]}")

            # Get text from the siblings
            result = []
            for component in components:
                result.append(component.text)
        
            return result

    def object_scrap(self, url, params):
        # Get html content from the page
        data = requests.get(url)
        html = BeautifulSoup(data.text, 'html.parser')

        # Get text from the component
        c_data = {}
        for key in params:
            components = html.select(params[key])
            c_data[key] = []

            for component in components:
                c_data[key].append(component.text)
        
        result = []
        for i in range(len(c_data[next(iter(c_data))])):
            e_data = {}
            for key in params:
                e_data[key] = c_data[key][i]
                
            result.append(e_data)

        return result

    def password_scrap(self, url, cred, params):
        payload = cred
        s = requests.session() 
        response = s.post(url, data=payload) 

        html = BeautifulSoup(response.content, 'html.parser')

        # Get text from the component
        c_data = {}
        for key in params:
            components = html.select(params[key])
            c_data[key] = []

            for component in components:
                c_data[key].append(component['value'])

        result = []
        for i in range(len(c_data[next(iter(c_data))])):
            e_data = {}
            for key in params:
                e_data[key] = c_data[key][i]
                
            result.append(e_data)

        return result

    def image_scrap(self, url):
        from commune.modules.model.beautifulsoap.image_fetcher.image_fetcher import setup_fetcher

        setup_fetcher(url)
        return "Done"

    def generate_random_url():
        while(True):
            ip0 = str(random.randint(0, 255))
            ip1 = str(random.randint(0, 255))
            ip2 = str(random.randint(0, 255))
            ip3 = str(random.randint(0, 255))
            url = 'http://' + ip0 + '.' + ip1 + '.'+ ip2 + '.'+ ip3
            print(url)
            try:
                urlContent = urllib.request.urlopen(url).read()
                print(urlContent)
                if urlContent.find('<html') > -1 or urlContent.find('<HTML') > -1:
                    print("Found URL: " + url)
                    break
            except:
                pass

    def csrf_bypass(self, username, password, login_url, scrap_url):
        login = username
        
        with requests.session() as s: 
            req = s.get(login_url).text 
            html = BeautifulSoup(req,"html.parser") 
            token = html.find("input", {"name": "authenticity_token"}).attrs["value"] 
            time = html.find("input", {"name": "timestamp"}).attrs["value"] 
            timeSecret = html.find("input", {"name": "timestamp_secret"}).attrs["value"] 
        
            payload = { 
                "authenticity_token": token, 
                "login": login, 
                "password": password, 
                "timestamp": time, 
                "timestamp_secret": timeSecret 
            } 
            res =s.post(login_url, data=payload) 
        
            r = s.get(scrap_url) 
            soup = BeautifulSoup (r.content, "html.parser") 
            
            return soup

    def get_buttons_and_inputs(self, url):
        # Make a GET request to the website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all buttons and store their HTML representation
            buttons = [str(button) for button in soup.find_all('button')]

            # Find all input elements and store their HTML representation
            inputs = [str(input_element) for input_element in soup.find_all('input')]

            # Store the results in a dictionary
            result = {'buttons': buttons, 'inputs': inputs}

            # Print or use the result as needed
            return result

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    def get_components(self, url, tags):
        # Make a GET request to the website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            result={}
            for tag in tags:
                # Find all components on their HTML representation
                components = [str(component) for component in soup.find_all(tag)]
                result[tag] = components


            # Print or use the result as needed
            return result

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
