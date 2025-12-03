# -------- Builder stage --------
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ARG VITE_API_BASE=/api               # tells the React app where the backend lives
RUN npm run build                     # creates /app/dist

# -------- Runtime stage --------
FROM nginx:alpine
# Remove the default nginx config
RUN rm /etc/nginx/conf.d/default.conf
# Copy our custom config (see §6)
COPY nginx/admin_ui.conf /etc/nginx/conf.d/
# Copy the built React app
COPY --from=builder /app/dist /usr/share/nginx/html
# Expose only the HTTP port – TLS termination will be done by the outer Nginx (or by the VPS)
EXPOSE 80
