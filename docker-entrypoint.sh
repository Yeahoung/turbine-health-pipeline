#!/bin/sh
set -e
cd /app
# Start Streamlit in background
streamlit run /app/app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true --server.baseUrlPath=/app &
# Wait for Streamlit to be ready
sleep 5
# Start nginx in foreground
exec nginx -g "daemon off;"
