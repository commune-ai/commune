import commune as c
import requests
import base64
from PIL import Image
from io import BytesIO



# Convert image to Base64
def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class Swarms(c.Module):
    def __init__(self, 
                 url="https://api.swarms.world/v1/chat/completions"):
        self.url = url

    def forward(self, 
                text="Describe what is in the image",
                image=None, 
                max_tokens=1024, 
                temperature=0.8,
                model="cogvlm-chat-17b",
                role = "user",
                top_p=0.9):
        # Replace 'image.jpg' with the path to your image

        text_data = {"type": "text", "text": text}
        messages = [{"role": role, "content": [text_data]}]
        if image != None:
            if isinstance(image, str):
                images = [image]
            else:
                images = image

            for image in images:
                if c.exists(image):
                    image_bytes = image_to_base64(image)
                    image_data = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}}
                    messages[-1]["content"].append(image_data)
                else:
                    raise ValueError(f"Image {image} not found")

        # Construct the request data
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Specify the URL of your FastAPI application
    
        # Print the response from the server
        c.print(request_data)
        r = requests.post(self.url, json=request_data)
        return r.text