SHELL := /bin/bash
export LC_ALL=C
.ONESHELL:

.PHONY: all clean install install-uv verify-python setup-dirs train-pipeline data-pipeline streaming-inference run-all help re-run-all

# Default Python interprete
# Use uv run python if .venv exists, otherwise fall back to python
PYTHON = $(shell if [ -d .venv ]; then echo "uv run python"; else echo "python"; fi)
VENV = .venv/bin/activate
MLFLOW_PORT ?= 5001
PID_DIR = runtime/pids
KAFKA_CONF = kafka/server.properties

# Default target
all: help

# Help target
help:
	@echo "🚀 Production ML System with Kafka Streaming"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install             - Install dependencies and create virtualenv"
	@echo "  make setup-dirs          - Create required directories"
	@echo "  make clean               - Cleanup artifacts and caches"
	@echo ""
	@echo "ML Pipeline Commands:"
	@echo "  make data-pipeline       - Run data pipeline"
	@echo "  make train-pipeline      - Run training pipeline"
	@echo "  make streaming-inference - Run streaming inference"
	@echo ""
	@echo "MLflow Commands:"
	@echo "  make mlflow-ui           - Start MLflow UI"
	@echo "  make stop-all            - Stop all MLflow servers"
	@echo ""
	@echo "Kafka Commands:"
	@echo "  make kafka-validate      - Validate Kafka installation"
	@echo "  make kafka-format        - Format Kafka storage"
	@echo "  make kafka-start-bg      - Start Kafka in background"
	@echo "  make kafka-stop          - Stop Kafka broker"
	@echo "  make kafka-topics        - Create topics"
	@echo "  make kafka-cleanup-topics- Cleanup topics"
	@echo "  make kafka-producer-stream - Streaming producer"
	@echo "  make kafka-producer-batch  - Batch producer"
	@echo "  make kafka-consumer        - Batch consumer"
	@echo "  make kafka-consumer-continuous - Continuous consumer"
	@echo "  make kafka-check           - Check Kafka status"
	@echo ""
	@echo "Airflow Commands:"
	@echo "  make airflow-init        - Initialize Airflow"
	@echo "  make airflow-start       - Start Airflow"
	@echo "  make airflow-kill        - Kill Airflow processes"
	@echo "  make airflow-reset       - Reset Airflow database"
	@echo ""
	@echo "Quick Start Example:"
	@echo "  make install setup-dirs"
	@echo "  make kafka-start-bg kafka-topics"
	@echo "  make kafka-producer-batch"
	@echo "  make kafka-consumer"

# Install uv package manage
install-uv:
	@echo "Installing uv package manager..."
	@if command -v uv &> /dev/null; then \
		echo "uv is already installed: $$(uv --version)"; \
	else \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed successfully! You may need to restart your shell or run: source $$HOME/.cargo/env"; \
	fi

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Checking for uv..."
	@if ! command -v uv &> /dev/null; then \
		echo "uv not found. Installing uv first..."; \
		$(MAKE) install-uv; \
		echo "Please restart your shell or run: source $$HOME/.cargo/env"; \
		echo "Then run 'make install' again."; \
		exit 1; \
	fi
	@echo "Installing Python 3.12 with uv..."
	@uv python install 3.12
	@echo "Creating virtual environment with Python 3.12..."
	@rm -rf .venv
	@uv venv --python 3.12
	@echo "Installing dependencies with uv..."
	@uv pip install -r requirements.txt
	@echo "Installation completed successfully!"
	@echo "Python version: $$(uv run python --version)"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Verify Python version
verify-python:
	@echo "Checking Python version..."
	@if [ -d .venv ]; then \
		echo "Python version in virtual environment: $$(uv run python --version)"; \
		if ! uv run python --version | grep -q "3.12"; then \
			echo "WARNING: Not using Python 3.12! Please run 'make install' to fix."; \
			exit 1; \
		else \
			echo "✓ Python 3.12 is correctly configured"; \
		fi \
	else \
		echo "Virtual environment not found. Please run 'make install' first."; \
		exit 1; \
	fi

# Create necessary directories
setup-dirs:
	@echo "Creating necessary directories..."
	@mkdir -p artifacts/data
	@mkdir -p artifacts/models
	@mkdir -p artifacts/encode
	@mkdir -p artifacts/mlflow_run_artifacts
	@mkdir -p artifacts/mlflow_training_artifacts
	@mkdir -p artifacts/inference_batches
	@mkdir -p data/processed
	@mkdir -p data/raw
	@echo "Directories created successfully!"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/*
	rm -rf mlruns
	@echo "Cleanup completed!"

# Run data pipeline
data-pipeline: setup-dirs
	@echo "Start running data pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "Data pipeline completed successfully!"

.PHONY: data-pipeline-rebuild
data-pipeline-rebuild: setup-dirs
	@source $(VENV) && $(PYTHON) -c "from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"

# Run training pipeline (PySpark MLlib by default)
train-pipeline: setup-dirs
	@echo "Running PySpark MLlib training pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py

# Run training pipeline with scikit-learn
train-sklearn: setup-dirs
	@echo "Running scikit-learn training pipeline..."
	@source $(VENV) && TRAINING_ENGINE=sklearn $(PYTHON) pipelines/training_pipeline.py

# Run training pipeline with PySpark MLlib
train-pyspark: setup-dirs
	@echo "Running PySpark MLlib training pipeline..."
	@source $(VENV) && TRAINING_ENGINE=pyspark $(PYTHON) pipelines/training_pipeline.py

# Test PySpark MLlib pipeline
test-pyspark: setup-dirs
	@echo "Testing PySpark MLlib pipeline..."
	@source $(VENV) && $(PYTHON) test_pyspark_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference: setup-dirs
	@echo "Running streaming inference pipeline with sample JSON..."
	@source $(VENV) && $(PYTHON) pipelines/model_inference_pipeline.py

# Run all pipelines in sequence
run-all: setup-dirs
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "✅ All MLflow servers have been stopped"

# ========================================================================================
# APACHE AIRFLOW ORCHESTRATION TARGETS
# ========================================================================================

airflow-init: ## Initialize Apache Airflow
	@echo "Initializing Apache Airflow..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	source $(VENV) && \
	pip install "apache-airflow>=2.10.0,<3.0.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.9.txt" && \
	pip install apache-airflow-providers-apache-spark && \
	airflow db migrate && \
	airflow users create -u admin -p admin -r Admin -e admin@example.com -f Admin -l User && \
	mkdir -p .airflow/dags && find dags -name "*.py" -exec cp {} .airflow/dags/ \;
	@echo "Airflow initialized successfully!"

airflow-webserver: ## Start Airflow webserve
	@echo "Starting Airflow webserver on http://localhost:8080..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	source $(VENV) && \
	airflow webserver --port 8080

airflow-scheduler: ## Start Airflow schedule
	@echo "Starting Airflow scheduler..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	source $(VENV) && \
	airflow schedule

airflow-start: ## Start Airflow in standalone mode (simpler for local dev)
	@echo "Checking for port conflicts..."
	@if lsof -ti:8080,8793,8794 >/dev/null 2>&1; then \
		echo "⚠️  Airflow ports are in use. Cleaning up first..."; \
		$(MAKE) airflow-kill; \
		sleep 3; \
	fi
	@echo "Ensuring DAGs are copied..."
	@find dags -name "*.py" -exec cp {} .airflow/dags/ \; 2>/dev/null || true
	@echo "Starting Airflow in standalone mode..."
	@echo "Webserver will be available at http://localhost:8080"
	@echo "Login with: admin / admin"
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	export AIRFLOW__WEBSERVER__WEB_SERVER_MASTER_TIMEOUT=300 && \
	export AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT=300 && \
	source $(VENV) && \
	airflow standalone

airflow-start-separate: ## Start Airflow webserver and scheduler separately
	@echo "Starting Airflow webserver and scheduler..."
	@echo "Webserver will be available at http://localhost:8080"
	@echo "Login with: admin / admin"
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	trap "kill 0" INT TERM EXIT && \
	airflow webserver --port 8080 & \
	airflow schedule

airflow-dags-list: ## List all available DAGs
	@echo "Listing Airflow DAGs..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	source $(VENV) && \
	airflow dags list

airflow-test-data-pipeline: ## Test data pipeline DAG
	@echo "Testing data pipeline DAG..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow tasks test data_pipeline_dag run_data_pipeline 2025-01-01

airflow-test-training-pipeline: ## Test training pipeline DAG
	@echo "Testing training pipeline DAG..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow tasks test training_pipeline_dag run_training_pipeline 2025-01-01

airflow-test-inference-pipeline: ## Test inference pipeline DAG
	@echo "Testing inference pipeline DAG..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow tasks test inference_dag run_inference_pipeline 2025-01-01

airflow-clean: ## Clean Airflow database and logs
	@echo "Cleaning Airflow database and logs..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	rm -rf .airflow/airflow.db .airflow/logs/*

airflow-delete-dags: ## Delete all DAGs from Airflow UI (removes example DAGs too)
	@echo "Stopping Airflow if running..."
	@pkill -f airflow || true
	@echo "Configuring Airflow to hide example DAGs..."
	@source .venv/bin/activate && export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	if ! grep -q "load_examples = False" .airflow/airflow.cfg; then \
		sed -i '' 's/load_examples = True/load_examples = False/g' .airflow/airflow.cfg 2>/dev/null || \
		echo "load_examples = False" >> .airflow/airflow.cfg; \
	fi
	@echo "Deleting project DAG files..."
	@if [ -d ".airflow/dags" ]; then \
		rm -rf .airflow/dags/*; \
	fi
	@echo "All DAGs deleted. Example DAGs will be hidden on next start."
	@echo "To re-add your project DAGs, run: cp dags/* .airflow/dags/"
	@echo "To start Airflow without example DAGs, run: make airflow-standalone"

airflow-kill: ## Kill all running Airflow processes and free ports
	@echo "Killing all Airflow processes..."
	@pkill -f airflow || echo "No Airflow processes found"
	@sleep 2
	@echo "Force killing any remaining Airflow processes..."
	@pkill -9 -f airflow || echo "No remaining processes"
	@sleep 1
	@echo "Freeing Airflow ports (8080, 8793, 8794)..."
	@lsof -ti:8080,8793,8794 | xargs kill -9 2>/dev/null || echo "No processes using Airflow ports"
	@sleep 1
	@echo "Cleaning up PID files..."
	@rm -f .airflow/airflow-webserver.pid .airflow/airflow-scheduler.pid .airflow/airflow-triggerer.pid
	@echo "All Airflow processes killed and ports freed successfully!"

airflow-trigger-all: ## Trigger all DAGs manually for testing
	@echo "Triggering all DAGs..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	echo "Triggering data pipeline..." && \
	airflow dags trigger data_pipeline_dag && \
	echo "Triggering training pipeline..." && \
	airflow dags trigger training_pipeline_dag && \
	echo "Triggering inference pipeline..." && \
	airflow dags trigger inference_dag
	@echo "✅ All DAGs triggered! Check the Web UI at http://localhost:8080"

airflow-health: ## Check Airflow health status
	@echo "Checking Airflow health status..."
	@curl -s http://localhost:8080/health | python -m json.tool || echo "❌ Airflow not responding"
	@echo ""
	@echo "Checking running processes..."
	@ps aux | grep airflow | grep -v grep || echo "❌ No Airflow processes found"

airflow-reset: ## Reset Airflow database and fix login issues
	@echo "Resetting Airflow database and fixing login issues..."
	@$(MAKE) airflow-kill
	@echo "Removing old database and logs..."
	@rm -rf .airflow/airflow.db .airflow/logs/*
	@find . -path "./.venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./.venv" -prune -o -name "*.pyc" -delete 2>/dev/null || true
	@echo "Reinitializing database..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow db migrate
	@echo "Creating admin user..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow users create -u admin -f Admin -l User -p admin -r Admin -e admin@example.com
	@echo "Copying DAGs..."
	@find dags -name "*.py" -exec cp {} .airflow/dags/ \;
	@echo "✓ Airflow reset complete! Login: admin/admin"
	@echo "Start with: make airflow-standalone"

	@echo "Airflow cleaned successfully!"

re-run-all: ## 🔄 Complete reset: kill processes, clean everything, restart fresh
	@echo "🔄 Starting complete system reset and restart..."
	@echo "=================================================="
	@echo "Step 1/6: Killing all Airflow processes..."
	@$(MAKE) airflow-kill
	@echo ""
	@echo "Step 2/6: Cleaning database, logs, and Python cache files..."
	@rm -rf .airflow/airflow.db .airflow/logs/* .airflow/dags/* 2>/dev/null || true
	@find . -path "./.venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./.venv" -prune -o -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Database, logs, and Python cache files cleaned"
	@echo ""
	@echo "Step 3/6: Reinitializing Airflow database..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow db migrate
	@echo "✅ Database reinitialized"
	@echo ""
	@echo "Step 4/6: Creating admin user..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	source $(VENV) && \
	airflow users create -u admin -f Admin -l User -p admin -r Admin -e admin@example.com 2>/dev/null || echo "Admin user already exists"
	@echo "✅ Admin user ready (admin/admin)"
	@echo ""
	@echo "Step 5/6: Copying fresh DAGs..."
	@find dags -name "*.py" -exec cp {} .airflow/dags/ \;
	@echo "✅ DAGs copied:"
	@ls -la .airflow/dags/*.py
	@echo ""
	@echo "Step 6/6: Starting Airflow in standalone mode..."
	@echo "🚀 Starting Airflow standalone..."
	@export AIRFLOW_HOME="$(shell pwd)/.airflow" && \
	export PYTHONWARNINGS="ignore::DeprecationWarning" && \
	export PYTHONPATH="$(shell pwd):$$PYTHONPATH" && \
	source $(VENV) && \
	echo "=== ENVIRONMENT READY ===" && \
	echo "AIRFLOW_HOME: $$AIRFLOW_HOME" && \
	echo "PYTHONPATH: $$PYTHONPATH" && \
	echo "=== STARTING AIRFLOW STANDALONE ===" && \
	echo "🌐 Web UI will be available at: http://localhost:8080" && \
	echo "🔑 Login: admin / admin" && \
	echo "📊 DAGs: data_pipeline_dag (5min), training_pipeline_dag (daily), inference_dag (1min)" && \
	echo "=== AIRFLOW STARTING... ===" && \
	airflow standalone &
	@echo ""
	@echo "=================================================="
	@echo "✅ COMPLETE RESET AND RESTART FINISHED!"
	@echo "🌐 Web UI: http://localhost:8080"
	@echo "🔑 Login: admin / admin"
	@echo "📊 Scheduling:"
	@echo "   - Data Pipeline: Every 5 minutes"
	@echo "   - Training Pipeline: Daily at 1 AM IST"
	@echo "   - Inference Pipeline: Every minute"
	@echo "=================================================="
	@echo "💡 Use 'make airflow-kill' to stop all processes"
	@echo "💡 Use 'make airflow-health' to check status"

# ==========================
# KAFKA
# ==========================
kafka-validate:
	@if [ -z "$$KAFKA_HOME" ]; then echo "❌ KAFKA_HOME not set"; exit 1; fi
	@if [ ! -d "$$KAFKA_HOME" ]; then echo "❌ KAFKA_HOME directory not found"; exit 1; fi
	@command -v java >/dev/null 2>&1 || { echo "❌ Java not found"; exit 1; }
	@echo "✅ Kafka installation is valid. KAFKA_HOME=$$KAFKA_HOME"
	@java -version | head -1

kafka-format:
	@mkdir -p runtime/kafka-logs runtime/pids
	@CLUSTER_ID=$$($(KAFKA_HOME)/bin/kafka-storage.sh random-uuid)
	@echo "Cluster ID: $$CLUSTER_ID"
	@$(KAFKA_HOME)/bin/kafka-storage.sh format -t $$CLUSTER_ID -c "$(KAFKA_CONF)"
	@echo "✅ Kafka storage formatted."

kafka-start-bg:
	@mkdir -p $(PID_DIR)
	@nohup $(KAFKA_HOME)/bin/kafka-server-start.sh "$(KAFKA_CONF)" > runtime/kafka.log 2>&1 &
	@echo $$! > $(PID_DIR)/kafka.pid
	@echo "✅ Kafka started in background (PID: $$(cat $(PID_DIR)/kafka.pid))"

kafka-stop:
	@if [ -f "$(PID_DIR)/kafka.pid" ]; then \
		PID=$$(cat $(PID_DIR)/kafka.pid); \
		kill $$PID || true; \
		rm -f $(PID_DIR)/kafka.pid; \
		echo "✅ Kafka stopped"; \
	else \
		echo "⚠️ Kafka PID not found"; \
	fi

kafka-topics:
	@$(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic churn_predictions --partitions 1 --replication-factor 1 --if-not-exists 2>&1 | grep -v "limitations in metric names" || true
	@$(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic churn_predictions_scored --partitions 1 --replication-factor 1 --if-not-exists 2>&1 | grep -v "limitations in metric names" || true
	@$(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list 2>&1 | grep -v "limitations in metric names" || true

kafka-cleanup-topics:
	@for topic in customer_events model_updates data_quality_alerts; do \
		if $(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list | grep -q "$$topic"; then \
			$(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic "$$topic"; \
		fi; \
	done

kafka-check:
	@if $(KAFKA_HOME)/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then \
		echo "✅ Kafka broker running at localhost:9092"; \
	else \
		echo "❌ Kafka broker not reachable"; \
	fi

kafka-producer-stream:
	@source $(VENV) && python pipelines/kafka_producer.py --mode streaming --rate 1 --duration 300

kafka-producer-batch:
	@source $(VENV) && python pipelines/kafka_producer.py --mode batch --num-events 100

kafka-consumer:
	@source $(VENV) && python pipelines/kafka_batch_consumer.py

kafka-consumer-continuous:
	@source $(VENV) && python pipelines/kafka_batch_consumer.py --continuous --poll-interval 5
