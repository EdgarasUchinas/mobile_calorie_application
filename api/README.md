# Food Recognition API

This FastAPI backend provides food recognition capabilities for the Mobile Calorie Tracking App using OpenAI's Vision API.

## Features

- **Image Analysis**: Upload images of food to get AI-powered recognition with nutritional information
- **Nutrition Information**: Get detailed nutritional data for specific food items
- **OpenAI Integration**: Leverages OpenAI's powerful vision and language models for accurate food recognition

## Setup

1. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Build and run with Docker Compose:
   ```
   docker-compose up
   ```

## API Endpoints

- `GET /`: Health check endpoint
- `GET /nutrition?food=apple`: Get nutritional information for a specific food
- `POST /analyze/base64`: Analyze a base64-encoded food image
- `POST /analyze/upload`: Analyze an uploaded food image file

## Development

### Running Locally (without Docker)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
