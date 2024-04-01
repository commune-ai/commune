# README

This script provides a `User` class that provides functionality for managing users within the Commune (a Python blockchain networking library) ecosystem. The `User` class is a subclass of the commune's `Module` class and allows for the addition, update and removal of users.

## Usage

1. Import necessary libraries and the script.
2. You may add a new user by calling the `add_user()` method and passing the user's address, role, name and other details as arguments.
3. All users can be fetched by calling the `users()` method.
4. Individual users can be retrieved using the `get_user()` method with the user's address.
5. User details can be updated using the `update_user()` method.
6. Users may be removed by calling the `rm_user()` method with their address.
7. A user's role can be fetched using the `get_role()` method.
8. An overview of all users can be displayed using the `dashboard()` method which leverages the Streamlit library to provide an interactive interface.

## Functionality

* Adding users with roles and arbitrary details.
* Updating existing user's information.
* Fetching a list of all users and their details.
* Fetching individual users' details by their addresses.
* Deleting users.
* Visualization of the entire set of users using a built-in dashboard.

## Dependencies

* Commune - A Python library for blockchain network management.
* Streamlit - An open-source Python library that makes it easy to create and share web apps for machine learning and data science.
* JSON - A built-in Python module for working with JSON data.

These can be installed using pip.

## Run

The script can be run by executing `python <scriptname.py>` in your terminal. The built-in dashboard provided by Streamlit will start running in your local environment.

## Note

Ensure the appropriate dependencies are installed and configured. The script assumes a working installation of the Commune module, a valid user pool, and a working internet connection for the Streamlit host.