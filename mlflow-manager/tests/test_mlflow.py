# system packages 
import os
import logging
import time
from pathlib import Path

# internal packages

# external packages
import pytest
import mlflow
import requests
from dotenv import load_dotenv

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# system variables
env_path = Path(__file__).parent.parent / "docker" / ".env"
load_dotenv(env_path)

MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5001")
MLFLOW_URI = f"http://localhost:{MLFLOW_PORT}"

def wait_for_mlflow(timeout=10):
    """Wait for MLflow server to be ready"""
    start_time = time.time()
    log.info(f"Attempting to connect to MLflow at {MLFLOW_URI}")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{MLFLOW_URI}/health")
            log.info(f"MLflow health check response: {response.status_code}")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError as e:
            log.info(f"Connection failed: {str(e)}")
            log.info("Waiting for MLflow server to be ready...")
            time.sleep(2)
    raise TimeoutError(f"MLflow server at {MLFLOW_URI} did not become ready in time")

@pytest.fixture(scope="session", autouse=True)
def setup_mlflow():
    """Ensure MLflow is running before tests"""
    try:
        wait_for_mlflow()
        mlflow.set_tracking_uri(MLFLOW_URI)
        log.info(f"MLflow tracking URI set to {MLFLOW_URI}")
    except Exception as e:
        log.error(f"Failed to setup MLflow: {str(e)}")
        raise

def test_mlflow_connection():
    """Test that we can connect to MLflow"""
    try:
        response = requests.get(f"{MLFLOW_URI}/health")
        log.info(f"MLflow health check status: {response.status_code}")
        log.info(f"MLflow health check response: {response.text}")
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to connect to MLflow: {str(e)}")
        raise

def test_mlflow_experiment():
    """Test that we can create and use an experiment"""
    try:
        experiment_name = "test_experiment"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            run = mlflow.active_run()
            log.info(f"Active run ID: {run.info.run_id}")
            assert run is not None
    except Exception as e:
        log.error(f"Failed to run MLflow experiment: {str(e)}")
        raise

