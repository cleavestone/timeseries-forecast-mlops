FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.14.0 \
    psycopg2-binary \
    boto3

EXPOSE 5000

ENTRYPOINT ["sh", "-c", "mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri $BACKEND_STORE_URI \
    --artifacts-destination $ARTIFACTS_DESTINATION \
    --serve-artifacts"]