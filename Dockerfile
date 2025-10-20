FROM node:20-alpine AS base

WORKDIR /app

COPY package*.json ./
RUN npm ci --no-audit --no-fund

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["node", "dist/index.js"]


