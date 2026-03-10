.PHONY: install test lint format type-check docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length 100 --ignore E501,W503

format:
	black src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

docker-build:
	docker build -f docker/Dockerfile -t ml-platform-api:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf artifacts/ model_registry/
