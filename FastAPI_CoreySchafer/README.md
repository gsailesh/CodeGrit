# FastAPI Posts API

A simple FastAPI application demonstrating REST API development.

## Setup

```bash
uv sync
```

## Running

```bash
uv run uvicorn main:app --reload
```

Visit `http://localhost:8000` to access the API.

## API Endpoints

- `GET /` — Home page
- `GET /listposts` — List posts
- `GET /api/posts` — Get all posts as JSON

## Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
