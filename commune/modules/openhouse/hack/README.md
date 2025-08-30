# Hackathon Platform

## Overview
This is a Streamlit-based web application for the Hackathon platform. It provides an interface for users to submit code modules, have them evaluated, and compete on a leaderboard.

## Features
- Submit code modules with a query/prompt
- View submitted modules and their code
- See scores, feedback, and improvement suggestions for each module
- Track performance on the leaderboard
- Verify project ownership for prize claims

## Setup

### Requirements
- Python 3.7+
- Streamlit
- Commune framework

### Installation
```bash
pip install streamlit pandas
```

### Running the App
```bash
streamlit run hack/app.py
```

## Usage
1. Navigate to the "Submit Module" page to create a new submission
2. Enter a unique module name, your query/prompt, and a password
3. View your submission and others on the "View Modules" page
4. Check the "Leaderboard" to see how your submission ranks
5. Use the "Verify Ownership" page to confirm you're the owner of a project and claim prizes

## Architecture
The app interfaces with the Hackathon class from hack.py, which handles:
- Module submission and storage
- Code evaluation and scoring
- Leaderboard management
- Project ownership verification