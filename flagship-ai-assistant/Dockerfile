# Dockerfile for the Personal AI Assistant.
# Example usage:
#   docker build -t personal-ai-assistant .
#   docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY personal-ai-assistant

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
