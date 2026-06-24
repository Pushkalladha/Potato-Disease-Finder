# Potato Disease Finder

A deep learning web application that detects potato leaf diseases from images.

## Features

- Detects Healthy Potato leaves
- Detects Early Blight
- Detects Late Blight
- FastAPI backend
- React frontend
- TensorFlow/Keras model

## Tech Stack

- Python
- TensorFlow
- FastAPI
- React.js

## Project Structure

```
API/
frontend/
main.py
evaluate.py
```

## Installation

### Backend

```bash
pip install -r API/requirements.txt
uvicorn API.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## Dataset

PlantVillage Potato Dataset

## Author

Pushkal Ladha
IIT Bombay