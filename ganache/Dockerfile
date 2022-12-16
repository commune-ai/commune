

# node:alpine will be our base image to create this image
FROM node:alpine

RUN apk add --no-cache --upgrade bash
# Set the /app directory as working directory
WORKDIR /app

# add scirpts
COPY ./scripts /app/scripts
RUN chmod +x /app/scripts/setup.sh

# Install ganache-cli globally
RUN npm install -g ganache-cli






