# Insurance Module README

This Python script is an Insurance module written as part of the `commune` application. It provides a user interface to handle insurance claims.

## Prerequisites

In order to run this script, you must have the `commune` Python package and `streamlit` package installed. Additionally, it uses the `json` and `typing` modules which are part of Python's standard library.

## How to Use

You can initiate the Insurance module with a specific user and password, which will then allow the program to sign them in on instantiation of the Insurance class. The script has a main function at the end which triggers the `streamlit` functionality, allowing users to interface with insurance claims and input their data.

### Functionality

This program allows users to:

- Sign in using their username and password
- Save claim information to a JSON object
- Retrieve claim
- Confirm if data is encrypted
- Retrieve all file paths associated with a claim
- Retrieve and list all claims associated with a user

### Unit Test

This script also contains a `test()` method used to test the `get()` and `put()` methods of the module.

### Streamlit Integration

The script uses `streamlit` to provide a web-based interface for users to interact with their account. The interface displays relevant information about the claims and allows for interaction.

## Warnings

The password and username are held as plaintext inside the script, which could pose a potential security risk. Furthermore, all claims are held in JSON files and are not encrypted by default. These precautions should be considered when deploying or running the script.
