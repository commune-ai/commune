import requests
import traceback

def format_query(query):
    """
    Formats the search query into the desired form.
    This could include escaping certain characters, removing unnecessary whitespace,
    or appending additional required parameters for the search service.
    """
    # trims whitespace
    return query.strip()

def handle_error(e):
    """
    Logs the error message and stack trace.
    """
    print(f"Error: {str(e)}")
    traceback.print_exc()

def validate_data(data, required_fields):
    """
    Validates the data received from the client.
    """
    for field in required_fields:
        if field not in data:
            return False
    return True

def make_api_request(url, method, headers=None, data=None):
    """
    Makes an API request and returns the response.
    """
    response = requests.request(method, url, headers=headers, data=data)
    return response.json()