# Stage 1: Build presentation (with DEMO_APP_URL for same-origin /app)
FROM node:20-alpine AS presentation-builder
WORKDIR /build
COPY presentation/ ./
RUN npm install && DEMO_APP_URL=/app node build.js

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Install nginx
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*

# Copy presentation build
COPY --from=presentation-builder /build/public /usr/share/nginx/html

# Copy app and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY pipeline/ pipeline/
COPY rl_agent/ rl_agent/
COPY data/ data/
COPY .streamlit/ .streamlit/

# Nginx config: remove default site, use our config
RUN rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Start script: nginx + streamlit
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.sh"]
