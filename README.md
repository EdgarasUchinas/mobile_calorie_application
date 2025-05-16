# Mobile Calorie App

A Quasar-based mobile application for tracking calorie intake and nutrition.

## Docker Setup

This project includes Docker configuration for easy development and deployment.

### Prerequisites

- Docker
- Docker Compose

### Running with Docker

1. Start the application using Docker Compose:

```bash
docker-compose up
```

2. Access the application at http://localhost:9000

3. To stop the application:

```bash
docker-compose down
```

### Development

The Docker setup includes volume mounting, so any changes you make to the source code will be reflected in the running application without needing to rebuild the container.

## Running Without Docker

If you prefer to run the application without Docker:

1. Navigate to the quasar-project directory:

```bash
cd quasar-project
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

4. Access the application at the URL shown in the terminal output.
