## Installation

**Environment Requirements**
python 3.10>=
nodejs 14>=
npm 6>=

1. **Clone the Commune Repository**: Open your terminal or command prompt and clone the Commune repository from GitHub:

```
git clone https://github.com/commune-ai/commune.git
```
Then pip install it 

```bash
pip install -e ./
```

To make sure it is working Run the tests, and they should all pass
```
c pytest
```


## Running a Docker Container with Commune

Ensure that you have Docker installed on your machine. If you don't, you can follow the official Docker installation guide for your operating system.


3. **Build the Docker Image**: Navigate to the cloned Commune repository and build the Docker image using the provided `Dockerfile`, This can be done via the docker-compsoe file.:

```
make build
```

4. **Start Container**: Start a Docker container with the Commune image:

```
make start
```

5. **Enter the Container**: Enter the Docker container:

```bash
make enter # or docker exec -it commune bash
```
To exit the container, run the following command:
```bash
exit
```

To run commands inside the container, you can use the following command:

```bash
docker exec -it commune bash -c "c modules"
```

To Kill the container, run the following command:
```bash
make down
```

Congratulations! Commune is now set up and running inside a Docker container.

## Setting up Commune Without Docker

2. **Install Dependencies**: Navigate to the cloned Commune repository and install the required dependencies:

```
cd commune
pip install -e ./
```

3. **install npm pm2**
This is required for the webserver to run
```bash 
chmod +x ./scripts/* # make sure the scripts are executable
sudo ./scripts/install_npm_env.sh # install npm and pm2 (sudo may not be required)
```

4. Check if commune is installed correctly, try running the following command
```bash
c modules
```

That's it! Commune is now set up and ready to roll on your local machine.

Whether you choose to set up Commune with Docker or without it, you're all set to leverage the power of Commune and connect your Python objects in a collaborative ecosystem. Have fun exploring the possibilities and making the most out of Commune!

Note: Make sure to follow the official Commune documentation for detailed instructions and additional configuration options.


## Ensure the port range is open or change the port range to a range of open ports for your module to be served on

```bash

# check the port range
c port_range
```

```bash
c set_port_range 8000 9000 # set the port range to 8000-9000
```

```
