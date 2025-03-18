# GraphDoc Tests

## Setup

```bash
# navigate to the graphdoc/tests directory
cd graphdoc/graphdoc/tests

# copy the .env.example file to .env and set the environment variables
cp .env.example .env

# (optional) set environment variables directly in the shell
export OPENAI_API_KEY="<your-openai-api-key>"
export HF_DATASET_KEY="<your-huggingface-dataset-key>"
export MLFLOW_TRACKING_URI="<your-mlflow-tracking-uri>"
export MLFLOW_TRACKING_USERNAME="<admin>"
export MLFLOW_TRACKING_PASSWORD="<password>"

# navigate to the graphdoc package root
cd..

# make sure the run.sh script is executable
chmod +x run.sh

# run the tests
./run.sh test

# optionally, run with linting and formatting
./run.sh commit
```

