#!/bin/bash

# NutriSage AWS Deployment Script
# This script deploys the NutriSage ML API to AWS ECS with ECR

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="nutrisage-mlops"
AWS_REGION="us-east-1"
ECR_REPOSITORY="nutrisage-mlops"
ECS_CLUSTER="nutrisage-cluster"
ECS_SERVICE="nutrisage-api-service"
ECS_TASK_DEFINITION="nutrisage-api-task"
IMAGE_TAG="latest"

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

# Function to check AWS CLI
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    print_success "AWS CLI is configured"
}

# Function to check Docker
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to create ECR repository
create_ecr_repository() {
    print_status "Creating ECR repository..."
    
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
        print_warning "ECR repository already exists"
    else
        aws ecr create-repository \
            --repository-name $ECR_REPOSITORY \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        
        print_success "ECR repository created"
    fi
}

# Function to get ECR login token
get_ecr_login() {
    print_status "Getting ECR login token..."
    
    aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com
    
    print_success "Logged in to ECR"
}

# Function to build and push Docker image
build_and_push_image() {
    print_status "Building Docker image..."
    
    # Get ECR registry URL
    ECR_REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Build image
    docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
    
    print_status "Pushing image to ECR..."
    docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    print_success "Image pushed to ECR"
}

# Function to create ECS cluster
create_ecs_cluster() {
    print_status "Creating ECS cluster..."
    
    if aws ecs describe-clusters --clusters $ECS_CLUSTER --region $AWS_REGION --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
        print_warning "ECS cluster already exists"
    else
        aws ecs create-cluster \
            --cluster-name $ECS_CLUSTER \
            --region $AWS_REGION \
            --capacity-providers FARGATE \
            --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
        
        print_success "ECS cluster created"
    fi
}

# Function to create task definition
create_task_definition() {
    print_status "Creating ECS task definition..."
    
    ECR_REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com
    
    cat > task-definition.json << EOF
{
    "family": "$ECS_TASK_DEFINITION",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "nutrisage-api",
            "image": "$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "HOST",
                    "value": "0.0.0.0"
                },
                {
                    "name": "PORT",
                    "value": "8000"
                },
                {
                    "name": "DATA_SOURCE",
                    "value": "local"
                },
                {
                    "name": "MODEL_PATH",
                    "value": "/app/models/nutrition_grade_model.pkl"
                },
                {
                    "name": "METADATA_PATH",
                    "value": "/app/models/model_metadata.pkl"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/nutrisage-api",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3
            }
        }
    ]
}
EOF
    
    aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
    
    print_success "Task definition created"
}

# Function to create CloudWatch log group
create_log_group() {
    print_status "Creating CloudWatch log group..."
    
    if aws logs describe-log-groups --log-group-name-prefix "/ecs/$ECS_TASK_DEFINITION" --region $AWS_REGION --query 'logGroups[0].logGroupName' --output text 2>/dev/null | grep -q "/ecs/$ECS_TASK_DEFINITION"; then
        print_warning "Log group already exists"
    else
        aws logs create-log-group --log-group-name "/ecs/$ECS_TASK_DEFINITION" --region $AWS_REGION
        print_success "Log group created"
    fi
}

# Function to create ECS service
create_ecs_service() {
    print_status "Creating ECS service..."
    
    # Get default VPC and subnets
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0:2].SubnetId' --output text --region $AWS_REGION | tr '\t' ',' | sed 's/,$//')
    
    # Create security group
    SG_NAME="nutrisage-api-sg"
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for NutriSage API" \
        --vpc-id $VPC_ID \
        --region $AWS_REGION \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text --region $AWS_REGION)
    
    # Add inbound rule for port 8000
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION 2>/dev/null || true
    
    # Create service
    aws ecs create-service \
        --cluster $ECS_CLUSTER \
        --service-name $ECS_SERVICE \
        --task-definition $ECS_TASK_DEFINITION \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --region $AWS_REGION
    
    print_success "ECS service created"
}

# Function to check service status
check_service_status() {
    print_status "Checking service status..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        STATUS=$(aws ecs describe-services \
            --cluster $ECS_CLUSTER \
            --services $ECS_SERVICE \
            --region $AWS_REGION \
            --query 'services[0].status' --output text)
        
        if [ "$STATUS" = "ACTIVE" ]; then
            print_success "Service is active!"
            
            # Get public IP
            TASK_ARN=$(aws ecs list-tasks \
                --cluster $ECS_CLUSTER \
                --service-name $ECS_SERVICE \
                --region $AWS_REGION \
                --query 'taskArns[0]' --output text)
            
            if [ "$TASK_ARN" != "None" ]; then
                ENI_ID=$(aws ecs describe-tasks \
                    --cluster $ECS_CLUSTER \
                    --tasks $TASK_ARN \
                    --region $AWS_REGION \
                    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
                
                PUBLIC_IP=$(aws ec2 describe-network-interfaces \
                    --network-interface-ids $ENI_ID \
                    --region $AWS_REGION \
                    --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
                
                echo ""
                print_success "🎉 Deployment completed successfully!"
                echo ""
                print_status "Your NutriSage API is now running at:"
                echo "  🌐 http://$PUBLIC_IP:8000"
                echo ""
                print_status "Endpoints:"
                echo "  - Docs: http://$PUBLIC_IP:8000/docs"
                echo "  - Health: http://$PUBLIC_IP:8000/api/health"
                echo "  - Predict: http://$PUBLIC_IP:8000/api/predict"
                echo ""
                print_status "To test the API:"
                echo "  curl -X POST http://$PUBLIC_IP:8000/api/predict \\"
                echo "    -H \"Content-Type: application/json\" \\"
                echo "    -d '{\"energy_kcal_100g\": 150, \"fat_100g\": 5.2, \"carbohydrates_100g\": 25.0, \"sugars_100g\": 12.0, \"proteins_100g\": 8.0, \"sodium_100g\": 0.3}'"
                echo ""
                return 0
            fi
        fi
        
        print_status "Waiting for service to be ready... (attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    print_error "Service failed to start within expected time"
    return 1
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -r, --region        AWS region (default: us-east-1)"
    echo "  -t, --tag           Docker image tag (default: latest)"
    echo "  -c, --create-only   Only create infrastructure (don't deploy)"
    echo "  -d, --deploy-only   Only deploy (assume infrastructure exists)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full deployment"
    echo "  $0 --region us-west-2 # Deploy to different region"
    echo "  $0 --create-only      # Only create AWS resources"
    echo "  $0 --deploy-only      # Only deploy the application"
}

# Main script
main() {
    local create_only=false
    local deploy_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -c|--create-only)
                create_only=true
                shift
                ;;
            -d|--deploy-only)
                deploy_only=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_status "Starting NutriSage AWS deployment..."
    print_status "Region: $AWS_REGION"
    print_status "Image tag: $IMAGE_TAG"
    echo ""
    
    # Check prerequisites
    check_aws_cli
    check_docker
    
    if [ "$deploy_only" = false ]; then
        # Create infrastructure
        create_ecr_repository
        create_ecs_cluster
        create_log_group
        create_task_definition
    fi
    
    if [ "$create_only" = false ]; then
        # Deploy application
        get_ecr_login
        build_and_push_image
        
        if [ "$deploy_only" = false ]; then
            create_ecs_service
        fi
        
        check_service_status
    fi
    
    print_success "AWS deployment completed!"
}

# Run main function with all arguments
main "$@" 