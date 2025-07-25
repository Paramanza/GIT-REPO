#!/bin/bash
# =============================================================================
# RAG Sustainability Chatbot - Deployment Debugging Script
# =============================================================================
# This script helps debug deployment issues step by step.
# =============================================================================

set -e

echo "🔍 RAG Sustainability Chatbot - Deployment Debugging"
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
print_status "Step 1: Checking current deployment status..."
flyctl status || print_warning "App may not be deployed yet"

echo ""
print_status "Step 2: Checking recent logs..."
flyctl logs --lines=50

echo ""
print_status "Step 3: Testing local Docker build..."
echo "Building Docker image locally to test..."

if docker build -t rag-test . ; then
    print_success "Docker build successful"
    
    echo ""
    print_status "Testing local Docker run (will stop after 30 seconds)..."
    timeout 30s docker run -p 7860:7860 -e OPENAI_API_KEY="${OPENAI_API_KEY:-test}" rag-test || true
    
else
    print_error "Docker build failed"
fi

echo ""
print_status "Step 4: Checking Fly.io configuration..."
echo "Current app name: $(grep 'app = ' fly.toml | cut -d'"' -f2)"
echo "Current region: $(grep 'region = ' fly.toml | cut -d'"' -f2)"

echo ""
print_status "Step 5: Checking secrets..."
flyctl secrets list

echo ""
print_status "Step 6: Deployment options..."
echo ""
echo "Choose a debugging approach:"
echo "1. Deploy with high memory (8GB) - RECOMMENDED for OOM issues"
echo "2. Deploy without health checks"
echo "3. Deploy with extended health check timeouts"
echo "4. View detailed logs only"
echo "5. Rebuild and deploy fresh"
echo ""

read -p "Choose option (1-5): " option

case $option in
    1)
        print_status "Deploying with high memory configuration (8GB)..."
        cp fly.toml fly.toml.backup 2>/dev/null || true
        cp fly-high-memory.toml fly.toml
        print_warning "Using high-memory config (dedicated-cpu-1x + 8GB RAM)"
        print_warning "This will cost more (~$30-50/month) but should fix OOM issues"
        flyctl deploy
        ;;
    2)
        print_status "Deploying without health checks..."
        cp fly.toml fly.toml.backup 2>/dev/null || true
        cp fly-no-healthcheck.toml fly.toml
        print_warning "Backed up original fly.toml to fly.toml.backup"
        flyctl deploy
        ;;
    3)
        print_status "Deploying with extended timeouts..."
        flyctl deploy
        ;;
    4)
        print_status "Watching logs..."
        flyctl logs -f
        ;;
    5)
        print_status "Fresh deployment..."
        flyctl deploy --force-machines
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
print_status "Debugging complete. Check the output above for issues."
echo ""
print_warning "Common fixes:"
echo "  • If 'not listening on expected address': Check Gradio server_name/port config"
echo "  • If health check timeouts: Use option 1 (no health checks) first"
echo "  • If out of memory: Increase memory in fly.toml resources section"
echo "  • If startup too slow: Check vector_db exists and is complete" 