#!/bin/bash

# Start backend in a new Terminal tab or background process
echo "Starting backend..."
source backend/venv/bin/activate
uvicorn backend.app.main:app --reload &

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev
