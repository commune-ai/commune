# User Profile Module

This module provides a user profile panel that slides in from the right side of the screen.

## Features

- **Wallet Information Display**: Shows user address and crypto type
- **Message Signing**: Allows users to sign messages using their private key
- **Signature Verification**: Verify signatures with public keys
- **Quick Actions**: Convenient shortcuts for common operations

## Components

### UserProfile

The main component that renders the user profile panel.

#### Props

- `user`: User object containing address and crypto_type
- `isOpen`: Boolean to control panel visibility
- `onClose`: Callback function when panel is closed
- `keyInstance`: Key instance for cryptographic operations

## Usage

The UserProfile component is integrated into the Header component. When a user is logged in, they can click on their address in the top right to expand the profile panel.

### Sign a Message

1. Enter your message in the text area
2. Click the "$ sign" button
3. The signature will be displayed and can be copied

### Verify a Signature

1. Enter the original message
2. Paste the signature
3. Enter the public key (or use your own)
4. Click "$ verify" to check validity

## Styling

The component uses Tailwind CSS with a terminal/hacker aesthetic:
- Green text on black background
- Monospace font
- Terminal-style command prompts
- Smooth slide-in animation from the right