DOCKER_DIR=docker
COMPOSE_FILE=$(DOCKER_DIR)/docker-compose.yml

install:
	docker compose \
		--env-file $(DOCKER_DIR)/.env \
		--file $(COMPOSE_FILE) \
		up --detach

test:
	poetry install
	poetry run pytest tests/test_mlflow.py

clean:
	(cd docker/; docker compose down; cd ..;)

	rm -rf .venv

all: install test clean
