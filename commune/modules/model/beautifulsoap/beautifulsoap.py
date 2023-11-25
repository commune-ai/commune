import commune as c
import requests
from bs4 import BeautifulSoup
import json
import random
import urllib
import os
from commune.modules.model.beautifulsoap.random_website.random_website import generate_random_ip, \
    get_domain, check_webapp, get_text_content, generate_random_website_url, is_valid_website
from commune.modules.model.openai.openai import OpenAILLM
from commune.modules.model.beautifulsoap.google_search.google_search import google_search, google_search_with_api
from googlesearch import search
from urllib.parse import urlencode, urlunparse
from urllib.request import urlopen, Request
from yahoo import search as yahoo_search

class BeautifulSoapModule(c.Module):

    def __init__(self, 
                google_api_key = None,
                **kwargs
                ):
        self.set_google_api_key(google_api_key)

    def set_google_api_key(self, api_key: str = None) -> str:
        self.google_api_key = api_key
        return {'msg': f"API Key set to {api_key}", 'success': True}

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

    def google_search(self, keyword):
        result = []
        for url in search(keyword, num_results=20):
            response = requests.get(url)
    
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text()
            
            item = {
                "url": url,
                "content": content
            }
            result.append(item)

        return result

    def yahoo_search(self, keyword):
        urls = []
        for url in yahoo_search(keyword):
            urls.append(url)

        return urls

    def bing_search(self, keyword):
        l=[]
        o={}
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
        for i in range(0,100,10):
            target_url="https://www.bing.com/search?q=" + keyword + "&rdr=1&first={}".format(i+1)
            # print(target_url)
            resp=requests.get(target_url,headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')
            completeData = soup.find_all("li",{"class":"b_algo"})
            for i in range(0, len(completeData)):
                o["Title"]=completeData[i].find("a").text
                o["link"]=completeData[i].find("a").get("href")
                o["Description"]=completeData[i].find("div",
            {"class":"b_caption"}).text
                o["Position"]=i+1
                l.append(o)
                o={}
        return l

    def scrap_title(self, title):
        while(True):
            ip_addr = generate_random_ip()
            domain = get_domain(ip_addr)
            isWebApp = False
            exist = False
            item = { 'ip_addr': ip_addr }

            # Check if the file exists
            if os.path.exists('websites.json'):
                # If the file exists, read the existing data
                with open('websites.json', 'r') as file:
                    data = json.load(file)
            else:
                # If the file does not exist, initialize an empty list
                data = []

            # Check if the IP address exists in the data
            if any(d['ip_addr'] == ip_addr for d in data):
                exist = True
                item["exist"] = exist

                print(item)
                self.scrap_title(title)

            if domain:
                isWebApp = check_webapp(domain)
                if isWebApp:
                    content = get_text_content(domain)
                    # Create an instance of the OpenAILLM class
                    model = OpenAILLM()
                    result = model.forward(content + f" I want to summarize content related {title} from here as title and content. Return result as json format.")
                    json_obj = json.loads(result)

            # If ip address doesn't have domain
            item["exist"] = exist
            item["domain"] = domain
            item["isWebApp"] = isWebApp
            item["title"] = json_obj.title
            item["content"] = json_obj.content

            data.append(item)
            
            print(item)
            
            # Write the data back to the JSON file
            with open('websites.json', 'w') as file:
                json.dump(data, file)