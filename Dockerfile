# Base image with Python 3.9
FROM python:3.9-slim

# Install zip utility
RUN apt-get update && apt-get install -y zip && apt-get clean

# Set working directory
WORKDIR /app

# Create Lambda layer structure
RUN mkdir -p /app/python/nltk_data

# Install nltk and regex into the layer directory
RUN pip install --upgrade pip && pip install nltk regex -t /app/python

# Download VADER lexicon using the installed nltk (by modifying PYTHONPATH)
RUN PYTHONPATH=/app/python python3 -c "import nltk; nltk.data.path.append('/app/python/nltk_data'); nltk.download('vader_lexicon', download_dir='/app/python/nltk_data')"

# Zip everything into a layer file
RUN cd /app && zip -r /app/lambda-layer.zip python
