 # start of file
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Expose port (if needed)
EXPOSE 8545

# Default command
CMD ["npm", "run", "deploy"]
