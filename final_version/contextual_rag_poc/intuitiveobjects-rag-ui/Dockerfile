# FROM node:20 AS build

# WORKDIR /app

# COPY package*.json ./
# RUN npm install

# COPY . .
# RUN npm run build

# FROM nginx:alpine
# COPY --from=build /app/build /usr/share/nginx/html

# EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"] 



FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

# Copy env first, then code (better for layer caching)
COPY .env .env
COPY . .

# Build Vite React app with env vars injected
RUN npm run build

# Serve with nginx or other static server
RUN npm install -g serve

EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]