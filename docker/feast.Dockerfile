FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "feast[postgres]==0.65.0" setuptools

COPY feature_repo/ ./feature_repo/

WORKDIR /app/feature_repo
RUN cp feature_store.docker.yaml feature_store.yaml

EXPOSE 6566

CMD ["feast", "serve", "--host", "0.0.0.0", "--port", "6566"]