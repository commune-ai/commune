# README

The given script defines a `Combook` class which is a module in the `commune` library. The `Combook` class acts as a chatroom module, allowing users to send messages to specified chatrooms.

## Class Initialization
The `Combook` class is initialized with an optional maximum people limit set to 1000 by default. This limit is configured using the `self.set_config(kwargs=locals())` method, which stores the `max_people` argument locally in a dictionary.

```python
my_chatroom = Combook(max_people=2000)
```
## Sending Messages
The `send` method allows sending messages in chat. The method requires a text message and optionally a chatroom (set to 'lobby' by default) and a password. It generates a timestamp upon sending the message, and creates a path to store the message using the `chatroom`, `timestamp`, and `user_address`. Then, it bundles up all the chat information into a dictionary `chat_info` including username, timestamp, text of the message, and a password. Finally, it saves the chat information to the chatroom path using the `put()` method from `commune` library.

```python
my_chatroom.send("Hello everyone!", chatroom="team_chat", password="my_password")
```
Please note, the script assumes `user_address` is a predefined variable in your environment. Please ensure to set `user_address` before running this script.

Before running the script, ensure the `commune` library is available in your python environment. You can install it using pip.
```bash
pip install commune
```
