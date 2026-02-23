@echo off
title SnailCam Visualizer
echo Starting SnailCam Visualizer...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py

pause
