import requests
import socket
import random
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

def get_text_content(domain):
    try:
        response = requests.get("http://" + domain)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            logging.info(f"Successfully fetched and parsed text content for {domain}")
            return soup.get_text()
        else:
            logging.warning(f"Failed to fetch text content for {domain}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception occurred when fetching text content for {domain}: {e}")
        return None

def generate_random_ip():
    return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

def get_domain(ip_address):
    try:
        domain_name = socket.gethostbyaddr(ip_address)
        return domain_name[0]
    except socket.herror:
        return None
    except OSError:
        logging.error(f"Invalid IP address: {ip_address}")
        return None

def check_webapp(domain):
    try:
        response = requests.get("http://" + domain)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_html_content(domain):
    try:
        response = requests.get("http://" + domain)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def save_html_content(domain, html_content):
    with open(f"{domain}.html", "w") as file:
        file.write(html_content)

def generate_random_website_url():
    tlds = ['.com', '.org', '.net']

    while True:
        domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz.', k=random.randint(4, 15)))
        if domain.count(".") < 2 and domain[-1] != '.':
            tld = random.choice(tlds)
            url = 'https://' + domain + tld
            return url

def is_valid_website(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
