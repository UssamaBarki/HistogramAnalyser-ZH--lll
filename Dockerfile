# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the necessary app files
COPY main.py requirements.txt ./

# Copy the local data folder into the container
COPY Data/ ./Data/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Panel/Bokeh
EXPOSE 5006

# Run the app using panel serve
CMD ["panel", "serve", "main.py", "--address=0.0.0.0", "--port=5006", "--allow-websocket-origin=*", "--autoreload"]
