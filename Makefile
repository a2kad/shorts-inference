COMPOSE ?= docker compose

.PHONY: up down build logs health shell

up:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f

health:
	curl http://127.0.0.1:8000/health

shell:
	$(COMPOSE) exec shorts-inference bash
