# Hackathon Dashboard

A Streamlit application for managing and scoring hackathon submissions.

## Features

- **Leaderboard**: View all submissions and their scores
- **Submit Module**: Submit new code modules for evaluation
- **Score Module**: Score existing modules with feedback
- **View Modules**: Browse through submitted modules and their details

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- Commune framework

### Installation

```bash
pip install streamlit
```

### Running the App

```bash
streamlit run hack_app.py
```

## How It Works

The application interfaces with the Hackathon class from `hack.py` to provide a user-friendly web interface for managing hackathon submissions. Users can submit new modules, view existing ones, and score them based on various criteria.

### Submission Process

1. Enter a unique module name
2. Provide a query for generating the module
3. Set a password for authentication
4. Submit and receive an automatic score with feedback

### Scoring Criteria

Modules are scored out of 100 based on:
- Readability
- Efficiency
- Style
- Correctness
- Code structure
- Documentation

## Project Structure

```
./
├── hack/
│   ├── hack.py         # Core Hackathon class
│   └── modules/        # Directory for submitted modules
├── hack_app.py         # Streamlit application
└── README.md           # Documentation
```
