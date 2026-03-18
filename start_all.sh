#!/bin/bash
# Always work relative to this script's location (the project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "🌬️ Starting Unified Turbine Health Pipeline..."
echo "=> Project root: $SCRIPT_DIR"

# Build and serve Presentation
echo "=> Building presentation slides..."
cd "$SCRIPT_DIR/presentation"
npm install >/dev/null 2>&1
npm run build >/dev/null 2>&1

echo "=> Starting Presentation standalone server on port 8000..."
python3 -m http.server 8000 --directory "$SCRIPT_DIR/presentation/public" >/dev/null 2>&1 &
PID_PRESENTATION=$!

# Start Streamlit Pipeline App
echo "=> Starting Streamlit Unified App on port 8501..."
cd "$SCRIPT_DIR"
streamlit run "$SCRIPT_DIR/app.py" &
PID_STREAMLIT=$!

echo "============================================="
echo "✅ Everything is running successfully!"
echo "➡️ Streamlit Unified Dashboard: http://localhost:8501"
echo "➡️ Internal Presentation Host: http://localhost:8000"
echo "============================================="
echo "Press Ctrl+C to stop all."

# Wait for Ctrl+C to terminate both servers
trap "echo 'Shutting down servers...'; kill \$PID_PRESENTATION \$PID_STREAMLIT; exit" SIGINT SIGTERM
wait
