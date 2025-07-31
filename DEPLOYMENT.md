# NutriSage Deployment Guide

This guide covers the complete setup and deployment of the NutriSage API to AWS ECS.

## Prerequisites

1. AWS CLI configured with appropriate permissions
2. Docker installed locally
3. GitHub repository with the NutriSage codebase
4. AWS account with permissions for ECR, ECS, IAM, and CloudWatch

## Step 1: Set Up AWS Infrastructure

### 1.1 Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name nutrisage-mlops \
  --region us-east-1
```

### 1.2 Create IAM Roles

#### ECS Task Execution Role
```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create the role
aws iam create-role `
  --role-name ecsTaskExecutionRole `
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "ecs-tasks.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach the required policy
aws iam attach-role-policy `
  --role-name ecsTaskExecutionRole `
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

echo "Created ECS Task Execution Role: arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole"
```

#### ECS Task Role (for application permissions)
```bash
# Create the role
aws iam create-role \
  --role-name ecsTaskRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "ecs-tasks.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Create and attach custom policy for S3 access
aws iam put-role-policy \
  --role-name ecsTaskRole \
  --policy-name NutriSageS3Access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::nutrisage",
          "arn:aws:s3:::nutrisage/*"
        ]
      }
    ]
  }'
```

### 1.3 Create CloudWatch Log Group

```bash
aws logs create-log-group `
  --log-group-name /ecs/nutrisage-api `
  --region us-east-1
```

### 1.4 Create ECS Cluster

```bash
aws ecs create-cluster `
  --cluster-name nutrisage-cluster `
  --region us-east-1
```

### 1.5 Create ECS Task Definition

The task definition will be created automatically by the GitHub Actions workflow using your AWS account ID from secrets. No manual setup required.

### 1.6 Create ECS Service

```bash
aws ecs create-service `
  --cluster nutrisage-cluster `
  --service-name nutrisage-api-service `
  --task-definition nutrisage-task-definition:1 `
  --desired-count 1 `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0777264df816d0e9b],securityGroups=[sg-0fe8d19439f8dc7a6],assignPublicIp=ENABLED}" `
  --region us-east-1
```

## Step 2: Configure GitHub Secrets

In your GitHub repository, go to Settings > Secrets and variables > Actions and add:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_ACCOUNT_ID`: Your AWS account ID (12-digit number)

### How to get your AWS Account ID:

```bash
# Method 1: Using AWS CLI
aws sts get-caller-identity --query Account --output text

# Method 2: From AWS Console
# Go to AWS Console > IAM > Account settings
# Your Account ID is displayed at the top right
```

## Step 3: Deploy via GitHub Actions

1. Push your code to the `main` branch
2. The GitHub Actions workflow will automatically:
   - Build the Docker image
   - Push to ECR
   - Update the ECS service

## Step 4: Verify Deployment

### Check ECS Service Status
```bash
aws ecs describe-services \
  --cluster nutrisage-cluster \
  --services nutrisage-api-service
```

### Test the API
```bash
# Get the public IP of your ECS task
curl http://YOUR_PUBLIC_IP:8000/health
```

## Troubleshooting

### Common Issues

1. **Task fails to start**: Check CloudWatch logs for container startup errors
2. **Image pull fails**: Verify ECR repository exists and image was pushed successfully
3. **Health check fails**: Ensure the `/health` endpoint is working in the container
4. **Permission errors**: Verify IAM roles have correct permissions
5. **Account ID errors**: Ensure `AWS_ACCOUNT_ID` secret is set correctly in GitHub

### Useful Commands

```bash
# View task logs
aws logs tail /ecs/nutrisage-api --follow

# Describe running tasks
aws ecs list-tasks --cluster nutrisage-cluster

# Get task details
aws ecs describe-tasks --cluster nutrisage-cluster --tasks TASK_ARN
```

## Security Considerations

1. Use VPC with private subnets for production
2. Configure security groups to restrict access
3. Use Application Load Balancer for HTTPS termination
4. Consider using AWS Secrets Manager for sensitive configuration
5. Enable CloudTrail for audit logging

## Cost Optimization

1. Use Fargate Spot for non-critical workloads
2. Set up auto-scaling based on CPU/memory usage
3. Monitor and optimize container resource allocation
4. Use CloudWatch alarms for cost monitoring 