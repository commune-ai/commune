
## Installation Guide

### Prerequisites
- Docker
- Python 3

### Setup Steps

1. Clone the commune repository:
   ```
   git clone https://github.com/commune-ai/commune.git
   cd commune;
   git checkout arena;
   ```

2. Install commune:


### Docker Setup:


 Make scripts executable if they arent:
```
make chmod_scripts 
```
or
```
chmod +x ./commune/scripts/*
```

Build the container
```
make build # ./commune/scripts/build.sh

```


Start the Container
```
make start 
```
or 
```
make up
```


Run tests in the container (optional):

```
make tests
```


### Local Setup (Python):

npm install -g pm2
pip install -e ./

To setup inside a virtual environment:

python3 -m venv env
source env/bin/activate
pip install -e ./

to exit the virtual environment:

deactivate


3.
4. Build and start the Docker container:
   ```
   make build
   make start
   ```

5. Run tests (optional):
   ```
   make tests
   ```

6. Start the main application:
   ```
   make app
   ```
