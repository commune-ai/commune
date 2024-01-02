import commune as c
from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp

class Web(c.Module):

    @classmethod
    async def async_request(cls, url, method, headers, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, **kwargs) as response:
                if response.status == 200:
                    return {'status_code': response.status, 'text': await response.text()}
                else:
                    return {'status_code': response.status, 'text': await response.text()}

    @classmethod
    def request(cls, url, method='GET', headers={'User-Agent': 'Mozilla/5.0'}, mode="request", **kwargs):
        if mode == "request":
            response = requests.request(method, url, headers=headers, **kwargs)
            if response.status_code == 200:
                return response.text
            else:
                return {'status_code': response.status_code, 'text': response.text}
        elif mode == "asyncio":
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(cls.async_request(url, method, headers, **kwargs))
            c.print(f'response: {response}', color='yellow')  # Consider removing the color argument, as print does not support it by default
            return response
        else:
            raise ValueError(f"Invalid mode: {mode}")



    def html(self, url:str = 'https://www.google.com/', **kwargs):
        return self.request(url, **kwargs)
    
    get_html = html

    def get_text(self, url:str, min_chars=100, **kwargs):
        text_list = [p.split('">')[-1].split('<')[0] for p in self.get_components(url, 'p')['p']]
        return [text for text in text_list if len(text) > min_chars]
    
    

    def get_components(self, url, *tags):
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
    @classmethod
    def rget(cls, url:str, **kwargs):
        return cls.request(url, 'GET', **kwargs)
    @classmethod
    def rpost(self, url:str, **kwargs):
        return cls.request(url, 'POST', **kwargs)


    def google_search(self, keyword='isreal-hamas', n=10, max_words=1000):
        from googlesearch import search
        urls = search(keyword, num_results=n)
        c.print(f'urls: {urls}', color='yellow')
        futures = []
        for url in urls:
            futures.append(c.submit(self.url2text, args=[url], return_future=True))
        results = c.wait(futures, timeout=10)
        # url2result = {url: result for url, result in zip(urls, results)}
        return results

    bing_search = google_search

    def url2text(self, url, max_words=50):
        url_text = self.request(url)
        content = self.soup(url_text)
        response = {'title': content.contents[0],
                        'url': url, 
                        'text': "\n".join([content.contents[i].text for i in range(len(content.contents)) if content.contents[i].text != '']),
                        }
    
        for i in range(10):
            response['text'] = response['text'].replace('\n\n', '')
        c.print(f'response: {url}', color='yellow')
        text_words = response['text'].split(' ')
        num_words = len(text_words)
        if num_words > max_words:
            response['text'] = ' '.join(text_words[:max_words])
        return response
    def yahoo_search(self, keyword):
        from yahoo import search as yahoo_search

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
    

    def webpage(self, url='https://www.fool.ca/recent-headlines/', **kwargs):
        from urllib.request import Request, urlopen
        webpage = self.request(url=url)
        req = Request(url , headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        return webpage
    

    def soup(self, webpage=None, url=None, **kwargs):
        if webpage is None:
            webpage = self.webpage(url)
        soup = BeautifulSoup(webpage, 'html.parser', **kwargs)
        return soup
    

    def find(self,url=None, tag='p', **kwargs):
        return self.soup(url=url).find(tag, **kwargs)

    

    def sand(self, url='https://www.fool.ca/recent-headlines/',   **kwargs):
        from bs4 import BeautifulSoup as soup

        webpage = self.webpage(url)
        page_soup = c.soup(webpage, "html.parser", **kwargs)
        title = page_soup.find("title")
        containers = page_soup.findAll("p","promo")
        for container in containers:
            print(container)



    
