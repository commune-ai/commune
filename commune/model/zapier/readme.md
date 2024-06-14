# zapier

This module, currently at version 1.0.0, is a Discord bot crafted for reading and relaying emails from a specified IMAP server to a designated Discord channel. It includes interactive commands allowing users to check the bot's operational status, initiate manual email checks, and access detailed bot information. This bot is under active development, with plans for ongoing enhancements and feature expansions to enrich user interaction and functionality.

## Config
```sh
- mail_channel_id: # The unique identifier of the Discord channel where the bot will send email notifications and respond to user commands.

- password: # The password used for authenticating with the IMAP server to access and read emails from the specified email account. (gmail app password)

- userEmail: # The email address of the account on the IMAP server from which the bot will read and retrieve emails.

- token: # The authentication token for the Discord bot, used to log in and connect the bot to the Discord API.

- imap_host: # The hostname or IP address of the IMAP server used for connecting and accessing the email account, such as 'imap.gmail.com' for Gmail.

- imap_port: # The port number used to connect to the IMAP server, typically 993 for IMAP over SSL (secure connection)
```

## How to run?

```sh
c model.zapier run
```

## Discord bot channel commands
```sh
!info: # When a user types !info in the Discord chat, the bot responds with an embedded message providing detailed information about itself, such as its version, author, and other relevant details.

!status: # The !status command, when issued in the chat, prompts the bot to reply with its current operational status, typically indicating if it's online and functioning correctly.
```