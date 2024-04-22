# email_processing.py
import email
from bs4 import BeautifulSoup
from email.header import decode_header

def parse_email(raw_email):
    # Parse raw email bytes to an email message
    email_message = email.message_from_bytes(raw_email)

    # Initialize email details
    details = {
        'from': '',
        'subject': '',
        'body': '',
        'attachments': 0
    }

    # Decode email header details
    details['from'] = decode_header(email_message["From"])[0][0]
    details['subject'] = decode_header(email_message["Subject"])[0][0]

    # Process each part of the email
    for part in email_message.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))

        if "attachment" in content_disposition:
            details['attachments'] += 1
        elif content_type == "text/plain" and "attachment" not in content_disposition:
            details['body'] += part.get_payload(decode=True).decode()
        elif content_type == "text/html":
            html_content = part.get_payload(decode=True).decode()
            soup = BeautifulSoup(html_content, "html.parser")
            details['body'] += soup.get_text()

    return details
