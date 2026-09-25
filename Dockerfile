# Feather v1 - CPU-native LLM, served as a Flask-style HTTP completion API
# Smallest viable offline distribution: pure Python, no GPU acceleration.
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./ 
COPY src ./src
COPY models/kaggle/feather-v1-kaggle.pt ./model.pt
COPY scripts/serve.py ./serve.py

RUN pip install --no-cache-dir --no-deps . && pip install --no-cache-dir flask

EXPOSE 8000
ENTRYPOINT ["python", "serve.py"]