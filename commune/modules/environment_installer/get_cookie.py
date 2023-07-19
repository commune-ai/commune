import requests

url = 'https://chat.openai.com/chat'  # Replace with the URL of the website you want to request.

# Send the HTTP request to the website and capture the response.
response = requests.get(url)

# Get the cookies from the response object.
cookies = response.cookies

print("All cookies:")
for cookie in cookies:
    print(f"{cookie.name}: {cookie.value}")

# Access the specific cookie by name.
session_token_cookie = cookies.get('__Secure-next-auth.session-token')

# Check if the cookie was found and print its value.
if session_token_cookie:
    print(f"Value of __Secure-next-auth.session-token cookie: {session_token_cookie.value}")
else:
    print("Cookie __Secure-next-auth.session-token not found.")
