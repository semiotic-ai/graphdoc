#!/bin/bash
set -e

echo "Waiting for MinIO to be ready..."
until mc alias set minioserver http://minio:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} 2>&1; do
    echo "MinIO is not ready - waiting..."
    sleep 1
done

echo "MinIO is ready. Creating bucket..."

# create the mlflow bucket if it doesn't exist
if ! mc ls minioserver/mlflow 2>/dev/null; then
    mc mb minioserver/mlflow
    echo "Bucket 'mlflow' created successfully"
else
    echo "Bucket 'mlflow' already exists"
fi

# verify bucket exists
mc ls minioserver/mlflow
echo "Setup completed successfully"
