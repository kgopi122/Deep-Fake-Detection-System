# Stage 1: Build React/Vite App
FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx Reverse Proxy
FROM nginx:alpine

# Remove default Nginx landing page and default server config
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration with API reverse proxy rules
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled Vite build output
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]