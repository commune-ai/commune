import os
import requests
from dotenv import load_dotenv
import time
import logging
import base64
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Set your KittyCAD API Key
KITTYCAD_API_TOKEN = os.getenv("KITTYCAD_API_TOKEN")
if not KITTYCAD_API_TOKEN:
    raise ValueError("Please set the KITTYCAD_API_TOKEN environment variable.")

# KittyCAD API endpoints
BASE_URL = "https://api.zoo.dev"
TEXT_TO_CAD_ENDPOINT = f"{BASE_URL}/ai/text-to-cad/{{output_format}}"
USER_TEXT_TO_CAD_STATUS_ENDPOINT = f"{BASE_URL}/user/text-to-cad/{{operation_id}}"

def text_to_cad(description: str, output_format: str):
    headers = {
        "Authorization": f"Bearer {KITTYCAD_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": description,
        "output_format": output_format
    }

    url = TEXT_TO_CAD_ENDPOINT.format(output_format=output_format)

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        if response.status_code == 201:
            operation_id = response.json().get("id")
            if operation_id:
                logging.info(f"Model generation initiated. Operation ID: {operation_id}")
                return operation_id
            else:
                logging.error("Received a 201 status but no operation ID was found in the response.")
                return None
        else:
            logging.error(f"Failed to initiate model generation. Status Code: {response.status_code}")
            logging.error(f"Error Message: {response.text}")
            return None

    except Exception as e:
        logging.exception(f"An error occurred while initiating the CAD model generation: {e}")
        return None

def save_file(file_path, base64_content):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"output_{timestamp}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = os.path.join(folder_name, f"{timestamp}_{os.path.basename(file_path)}")

    # Fix padding if necessary
    base64_content += "=" * ((4 - len(base64_content) % 4) % 4)
    try:
        file_data = base64.b64decode(base64_content)
        with open(file_name, 'wb') as file:
            file.write(file_data)
            logging.info(f"File saved: {file_name}")
    except base64.binascii.Error as e:
        logging.error(f"An error occurred while decoding the file: {e}")

def check_model_generation_status(operation_id: str):
    headers = {
        "Authorization": f"Bearer {KITTYCAD_API_TOKEN}"
    }

    url = USER_TEXT_TO_CAD_STATUS_ENDPOINT.format(operation_id=operation_id)

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        if response.status_code == 200:
            operation_status = response.json().get("status")
            logging.info(f"Model generation status: {operation_status}")
            
            # Save the file when the operation is completed
            if operation_status == 'completed':
                outputs = response.json().get('outputs')
                for file_path, base64_content in outputs.items():
                    save_file(file_path, base64_content)
            return response.json()
        else:
            logging.error(f"Failed to get model generation status. Status Code: {response.status_code}")
            logging.error(f"Error Message: {response.text}")
            return None

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP Error occurred: {http_err}")
        logging.error("Ensure the operation ID is correct and the endpoint is accessible.")
        logging.error(f"Operation ID being checked: {operation_id}")
    except Exception as e:
        logging.exception(f"An error occurred while checking the model generation status: {e}")
    finally:
        time.sleep(10)


# Example usage:
if __name__ == "__main__":
    description = "cube 2x2x2mm"
    output_format = "obj"  # Choose from fbx, glb, gltf, obj, ply, step, stl
    operation_id = text_to_cad(description, output_format)
    if operation_id:
        while True:
            result = check_model_generation_status(operation_id)
            if result and result.get("status") == "completed":
                break
            time.sleep(10)  # Poll every 10 seconds
