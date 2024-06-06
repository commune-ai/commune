<div align="center">

# **Commune Frontend AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.com/invite/DgjvQXvhqf)
[![Website Uptime](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.Commune Frontendai.org/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Commune Frontendaidotorg.svg?style=social&label=Follow)](https://twitter.com/Commune Frontendaidotorg)

</div>

Commune Frontend is a website that aims to show all features of Commune Frontend community.
Here you can know the aim of Commune Frontend community through whitepaper and several features like Commune Frontend modules, staking, huggingface modules, telemetry.

By embracing an unopinionated approach, Commune Frontend acknowledges the diverse needs and preferences of developers. It provides a flexible framework that allows developers to integrate specific tools seamlessly while avoiding imposing rigid structures or constraints. This adaptability enables developers to leverage Commune Frontend's capabilities in a manner that best aligns with their individual projects and workflows.

The overarching goal of Commune Frontend is to create a collaborative ecosystem where developers can easily share, connect, and extend their tools, ultimately fostering innovation and efficiency within the development community. By providing a network that encourages openness and accessibility, Commune Frontend empowers developers to leverage the collective knowledge and resources of the community to enhance their own projects.

# Install
### Frontend

```
    npm install
```

```
    npm run dev
```

For build
```
    npm run build
```

### Setting Up With setup.py for backend

Install setuptools:
If you haven't already installed setuptools, you can do so using pip:

```bash
pip install -r requirements; pip install -e .
```
### Setting Up Commune Frontend With Docker

Install Docker: If you don't have Docker installed on your system, download and install it from the official Docker website: [https://www.docker.com/get-started](https://www.docker.com/get-started).

Clone the Commune Frontend Repository: Open your terminal or command prompt and clone the Commune Frontend repository from GitHub:

# Get started with docker

## Frontend container: 
### Build your frontend container

```
docker build -t commune-frontend ..
```

### Run your container: 

```
docker run -p 3000:3000 commune-frontend.
```

## Backend container:

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/commune-ai/frontend

2. Navigate to the cloned repository:

   ```bash
    cd backend

3. Build the Docker image:
   ```bash
    docker build -t backend .


# Running with docker
```bash
make up 
```
or 
    
```bash
docker-compose build
```

Start Commune Frontend: Once the Docker container is built, start Commune Frontend by running the following command:

```bash
make start
```
or 
```bash
docker-compose up -d # -d for detached mode
```

To enter the docker container do, and do the following

```bash
make enter
```
or 
```bash
docker exec -it Commune Frontend bash
```

Then run the following command to sync the network

```bash
npm run dev
```

To exit the container

```bash
exit
```