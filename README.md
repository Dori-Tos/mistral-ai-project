# mistral-ai-project

## Run

Clone the repo
```bash
git clone https://github.com/Dori-Tos/mistral-ai-project.git
```

Download dependencies
```bash
pip install -r requirements.txt
```

Run the application
```bash
flask --app main run
```

Or using Docker
```bash
docker pull dsmaelen/historic-fact-checker:latest

docker-compose up -d
```

The application will be accessible at `http://localhost:5000`.

## Features

- Check historical facts in text (raw or pdf)
- Identify real or AI generated images (historical photos)
