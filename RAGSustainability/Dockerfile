# =============================================================================
# RAG Sustainability Chatbot - Docker Configuration for Fly.io
# =============================================================================
# This Dockerfile creates a production-ready container for the RAG chatbot
# that includes the vector store and all dependencies needed for Fly.io deployment.
#
# Build command:    docker build -t rag-sustainability-chatbot .
# Local test:       docker run -p 7860:7860 -e OPENAI_API_KEY=your_key rag-sustainability-chatbot
# Deploy to Fly.io: flyctl deploy
# =============================================================================

# Use Python 3.11 slim image for smaller container size
FROM python:3.11-slim

# Set metadata labels
LABEL maintainer="RAG Sustainability Project"
LABEL description="RAG chatbot for sustainability topics with 3D vector visualization"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DOCKER_ENV=true

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies required for the application
# - build-essential: Required for compiling some Python packages
# - curl: For health checks and debugging
# - git: May be needed for some LangChain components
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
# This allows pip install to be cached if requirements.txt hasn't changed
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
# This includes the Python modules, knowledge base, and vector store
COPY . .

# Ensure the vector_db directory exists and has proper permissions
# The vector store (~500MB) is included in the image for simplicity
RUN mkdir -p vector_db && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose the port that Gradio runs on
# Gradio defaults to 7860, which we've configured in app.py
EXPOSE 7860

# Health check to ensure the application is running
# ML applications need more time to start up (loading vector store, etc.)
# Using a generous start period and simple health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set the default command to run the application
# This starts the RAG chatbot with Gradio interface
CMD ["python", "app.py"]

# =============================================================================
# Build and Run Instructions:
#
# 1. Build the Docker image:
#    docker build -t rag-sustainability-chatbot .
#
# 2. Test locally (replace with your OpenAI API key):
#    docker run -p 7860:7860 -e OPENAI_API_KEY=your_actual_key_here rag-sustainability-chatbot
#
# 3. Deploy to Fly.io:
#    flyctl secrets set OPENAI_API_KEY=your_actual_key_here
#    flyctl deploy
#
# 4. Access the application:
#    Local: http://localhost:7860
#    Fly.io: https://your-app-name.fly.dev
# ============================================================================= 