@echo off

REM start backend
start cmd /k "backend\venv\Scripts\activate && uvicorn backend.app.main:app --reload"

REM start frontend
start cmd /k "cd frontend && npm run dev"

exit