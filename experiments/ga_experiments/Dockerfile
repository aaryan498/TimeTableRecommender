# Dockerfile for reproducible GA experiment execution
# Based on Python 3.10 with all required dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy repository contents
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/FinalScheduler/requirements.txt

# Install experiment-specific dependencies
RUN pip install --no-cache-dir \
    pyyaml \
    pandas \
    pyarrow

# Set Python path
ENV PYTHONPATH="/app:/app/FinalScheduler:/app/experiments/ga_experiments"

# Create output directory
RUN mkdir -p /app/experiments/ga_experiments/output

# Set entrypoint to experiment runner
ENTRYPOINT ["python", "/app/experiments/ga_experiments/run_experiments.py"]

# Default command (can be overridden)
CMD ["--config", "/app/experiments/ga_experiments/config/samples/experiment_config.yaml", "--max-runs", "100"]

# Labels for documentation
LABEL maintainer="your-email@example.com"
LABEL description="GA Timetable Scheduling Experiment Runner"
LABEL version="1.0"

# Expose output volume
VOLUME ["/app/experiments/ga_experiments/output"]
