import os
import openai
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def get_general_schema(code:str):

    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    PROMPT=""""
    Using this as the general schema definition:
    {
        "name": "tool",
        "description": "This is a base tool that does nothing.",
        "tags": ["defi", "tool"],
        "schema": {"input": {"x": "int", "y": "int"}, "output": "int", "default": {"x": 1, "y": 1}}
    }

    Write the generalized schema for this tool:

    """
    full_prompt = PROMPT + code
    print(full_prompt)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=full_prompt,
        max_tokens=600,
        temperature=0
    )
    json_blob = response.choices[0].text.strip()
    return json.dumps(json_blob.replace("'", '"'))




def return_file_as_str(file_path:str) -> str:
    # Initialize an empty string to store the file content
    file_content = ""
    
    # Open the file in read mode ('r') and ensure it is closed properly using 'with'
    try:
        with open(file_path, 'r') as file:
            # Read the entire content of the file into the string
            file_content = file.read()
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        # Handle other possible exceptions
        print(f"Error: An error occurred while reading the file {file_path}. Details: {e}")
    
    # Return the file content as a string
    return file_content


if __name__ == "__main__":
    fc = get_general_schema("/Users/marissaposner/Autonomy-data/autonomy/tool/defillama/aave.py")
    print(fc)

