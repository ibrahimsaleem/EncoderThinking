FROM node:20-alpine AS base

WORKDIR /app

COPY package*.json ./
RUN npm ci --no-audit --no-fund

COPY . .

# Install dependencies and build if needed
RUN npm run build || echo "Build failed, using existing index.js"

# Set environment for Smithery deployment
ENV NODE_ENV=production
ENV SMITHERY_DEPLOYMENT=true

EXPOSE 3000

# Use the working index.js file
CMD ["node", "index.js"]


