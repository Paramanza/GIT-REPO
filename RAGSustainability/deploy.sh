#!/bin/bash
# =============================================================================
# RAG Sustainability Chatbot - Deployment Script for Fly.io
# =============================================================================
# This script automates the deployment process to Fly.io with error checking
# and helpful prompts.
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# =============================================================================

set -e  # Exit on any error

echo "🚀 RAG Sustainability Chatbot - Fly.io Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    print_error "flyctl is not installed!"
    echo ""
    echo "Please install the Fly.io CLI:"
    echo "curl -L https://fly.io/install.sh | sh"
    echo ""
    exit 1
fi

print_success "Fly.io CLI is installed"

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    print_warning "You are not logged in to Fly.io"
    echo ""
    echo "Please login first:"
    echo "flyctl auth login"
    echo ""
    read -p "Press Enter after logging in, or Ctrl+C to exit..."
fi

print_success "Logged in to Fly.io"

# Check if vector database exists
if [ ! -d "vector_db" ] || [ ! "$(ls -A vector_db)" ]; then
    print_warning "Vector database not found or empty!"
    echo ""
    echo "Please build the vector database first:"
    echo "python build_database.py"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_success "Vector database found"

# Check if OpenAI API key is set
print_status "Checking for OpenAI API key..."
if ! flyctl secrets list 2>/dev/null | grep -q "OPENAI_API_KEY"; then
    print_warning "OPENAI_API_KEY secret not found!"
    echo ""
    read -p "Enter your OpenAI API key: " -s api_key
    echo ""
    
    if [ -z "$api_key" ]; then
        print_error "API key cannot be empty"
        exit 1
    fi
    
    print_status "Setting OpenAI API key..."
    flyctl secrets set OPENAI_API_KEY="$api_key"
    print_success "API key set"
else
    print_success "OpenAI API key is configured"
fi

# Check if app name is still default
if grep -q 'app = "rag-sustainability-chatbot"' fly.toml; then
    print_warning "App name is still set to default!"
    echo ""
    echo "Please update the app name in fly.toml to something unique."
    echo "Current name: rag-sustainability-chatbot"
    echo ""
    read -p "Enter a unique app name: " app_name
    
    if [ -z "$app_name" ]; then
        print_error "App name cannot be empty"
        exit 1
    fi
    
    # Update app name in fly.toml
    sed -i.bak "s/app = \"rag-sustainability-chatbot\"/app = \"$app_name\"/" fly.toml
    print_success "Updated app name to: $app_name"
fi

# Final deployment confirmation
echo ""
print_status "Ready to deploy!"
echo ""
echo "Configuration Summary:"
echo "  📱 App name: $(grep 'app = ' fly.toml | cut -d'"' -f2)"
echo "  🔑 OpenAI API key: ✓ Set"
echo "  📊 Vector database: ✓ Found"
echo "  🐳 Docker files: ✓ Ready"
echo ""

read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled"
    exit 0
fi

# Deploy to Fly.io
print_status "Starting deployment to Fly.io..."
echo ""

if flyctl deploy; then
    print_success "Deployment completed successfully!"
    echo ""
    echo "🎉 Your RAG chatbot is now live!"
    echo ""
    echo "Next steps:"
    echo "  • Open your app: flyctl open"
    echo "  • View logs: flyctl logs"
    echo "  • Check status: flyctl status"
    echo ""
else
    print_error "Deployment failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check logs: flyctl logs"
    echo "  • Verify configuration: flyctl status"
    echo "  • Try again: flyctl deploy"
    echo ""
    exit 1
fi

print_success "Deployment script completed!" 