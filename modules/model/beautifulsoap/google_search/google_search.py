from bs4 import BeautifulSoup
import requests

def google_search(query):
    url = 'https://www.google.com/search?q=' + query
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})
    print(result_div)
    links = []
    titles = []
    descriptions = []
    for r in result_div:
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()

            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
        except:
            continue

    return titles, links, descriptions

def google_search_with_api(search_term, api_key, cx = None):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': search_term,
        'key': api_key,
        # 'cx': cx
    }
    response = requests.get(url, params=params)
    return response.json()
