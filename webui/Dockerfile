FROM node:20-alpine

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy dependency files first to cache install step
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install

# Copy remaining application files
COPY . .

# Expose Next.js default port
EXPOSE 3000

# Run in development mode so that NEXT_PUBLIC_ env variables can be changed at runtime
CMD ["pnpm", "dev"]
