FROM python:3.11-slim

WORKDIR /app

COPY main.py requirements.txt ./
COPY data ./data

RUN pip install --no-cache-dir -r requirements.txt

CMD ["panel", "serve", "main.py", "--address=0.0.0.0", "--port=5006", "--allow-websocket-origin=*"]
