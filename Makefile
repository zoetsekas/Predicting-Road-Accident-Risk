#!make
include system.env

# Docker compose define targets
all-compose: up

up:
	docker-compose -f $(COMPOSE_FILE) up -d

down:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env down

clean:
	docker-compose -f $(COMPOSE_FILE) --env-file system.envdown -v

re-create:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env up -d --force-recreate

config:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env config
# Help target
help:
	@echo "Available targets:"
	@echo "  all      Run the Docker Compose service"
	@echo "  up      Run the Docker Compose service in detached mode"
	@echo "  down    Stop and remove the Docker Compose service"
	@echo "  clean   Remove all Docker Compose containers and volumes"
	@echo "  help    Show this help message"