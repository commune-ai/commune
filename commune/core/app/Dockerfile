# Start from an official Node image with version >=16 (e.g., 18 or LTS)
FROM node:20
# Set the DEBIAN_FRONTEND to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Copy package.json and install dependencies
WORKDIR /app
COPY ./package.json .
RUN npm install


ENTRYPOINT ["npm", "run" , "dev"]