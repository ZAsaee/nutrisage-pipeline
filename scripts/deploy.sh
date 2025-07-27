#!/bin/bash

# NutriSage ML Pipeline Deployment Script
# This script builds and deploys the NutriSage API using Docker

set -e  # Exit on any error

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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if required files exist
check_requirements() {
    print_status "Checking requirements..."
    
    # Check if model files exist
    if [ ! -f "models/nutrition_grade_model.pkl" ]; then
        print_error "Model file not found: models/nutrition_grade_model.pkl"
        print_status "Please run the training pipeline first:"
        print_status "  python -m src.modeling.train"
        exit 1
    fi
    
    if [ ! -f "models/model_metadata.pkl" ]; then
        print_error "Model metadata not found: models/model_metadata.pkl"
        print_status "Please run the training pipeline first:"
        print_status "  python -m src.modeling.train"
        exit 1
    fi
    
    print_success "All required files found"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t nutrisage-mlops:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run container
run_container() {
    print_status "Starting NutriSage API container..."
    
    # Stop existing container if running
    if docker ps -q -f name=nutrisage-api | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop nutrisage-api
        docker rm nutrisage-api
    fi
    
    # Run the container
    docker run -d \
        --name nutrisage-api \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/reports:/app/reports" \
        --restart unless-stopped \
        nutrisage-mlops:latest
    
    if [ $? -eq 0 ]; then
        print_success "Container started successfully"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to check if API is ready
check_api_health() {
    print_status "Checking API health..."
    
    # Wait for API to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is healthy and ready!"
            return 0
        fi
        
        print_status "Waiting for API to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "API failed to start within expected time"
    return 1
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -b, --build-only    Only build the Docker image"
    echo "  -r, --run-only      Only run the container (assumes image exists)"
    echo "  -c, --compose       Use docker-compose instead of docker run"
    echo "  -p, --production    Deploy with nginx reverse proxy"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full deployment"
    echo "  $0 --build-only       # Only build image"
    echo "  $0 --compose          # Use docker-compose"
    echo "  $0 --production       # Deploy with nginx"
}

# Function to deploy with docker-compose
deploy_compose() {
    print_status "Deploying with docker-compose..."
    
    if [ "$1" = "production" ]; then
        print_status "Deploying in production mode with nginx..."
        docker-compose --profile production up -d
    else
        print_status "Deploying in development mode..."
        docker-compose up -d
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Deployed successfully with docker-compose"
    else
        print_error "Failed to deploy with docker-compose"
        exit 1
    fi
}

# Main script
main() {
    local build_only=false
    local run_only=false
    local use_compose=false
    local production_mode=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -b|--build-only)
                build_only=true
                shift
                ;;
            -r|--run-only)
                run_only=true
                shift
                ;;
            -c|--compose)
                use_compose=true
                shift
                ;;
            -p|--production)
                production_mode=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_status "Starting NutriSage ML Pipeline deployment..."
    
    # Check Docker
    check_docker
    
    # Check requirements
    check_requirements
    
    # Deploy based on options
    if [ "$use_compose" = true ]; then
        deploy_compose "$production_mode"
    else
        if [ "$run_only" = false ]; then
            build_image
        fi
        
        if [ "$build_only" = false ]; then
            run_container
            check_api_health
        fi
    fi
    
    # Show final status
    print_success "Deployment completed!"
    echo ""
    print_status "API endpoints:"
    echo "  - Health check: http://localhost:8000/health"
    echo "  - API docs: http://localhost:8000/docs"
    echo "  - Predict: http://localhost:8000/api/v1/predict"
    echo ""
    print_status "To test the API:"
    echo "  curl -X POST http://localhost:8000/api/v1/predict \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"energy_kcal_100g\": 150, \"fat_100g\": 5.2, \"carbohydrates_100g\": 25.0, \"sugars_100g\": 12.0, \"proteins_100g\": 8.0, \"sodium_100g\": 0.3}'"
    echo ""
    print_status "To view logs:"
    echo "  docker logs nutrisage-api"
    echo ""
    print_status "To stop the service:"
    echo "  docker stop nutrisage-api"
}

# Run main function with all arguments
main "$@" 