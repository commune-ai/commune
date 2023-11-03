import commune as c
import requests
from bs4 import BeautifulSoup

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