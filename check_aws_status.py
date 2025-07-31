#!/usr/bin/env python
"""
Check AWS deployment status for NutriSage API.
"""

import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"AWS Account: {identity['Account']}")
        logger.info(f"User/Role: {identity['Arn']}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS CLI.")
        return False
    except Exception as e:
        logger.error(f"Error checking AWS credentials: {e}")
        return False


def check_ecs_service():
    """Check ECS service status."""
    try:
        ecs = boto3.client('ecs', region_name='us-east-1')

        # Check if cluster exists
        clusters = ecs.list_clusters()
        nutrisage_clusters = [
            c for c in clusters['clusterArns'] if 'nutrisage' in c.lower()]

        if not nutrisage_clusters:
            logger.error("No NutriSage ECS cluster found")
            return False

        cluster_name = nutrisage_clusters[0].split('/')[-1]
        logger.info(f"Found cluster: {cluster_name}")

        # Check services
        services = ecs.list_services(cluster=cluster_name)
        if not services['serviceArns']:
            logger.error("No services found in cluster")
            return False

        service_arn = services['serviceArns'][0]
        service_name = service_arn.split('/')[-1]
        logger.info(f"Found service: {service_name}")

        # Get service details
        service_details = ecs.describe_services(
            cluster=cluster_name,
            services=[service_name]
        )

        service = service_details['services'][0]
        logger.info(f"Service status: {service['status']}")
        logger.info(f"Desired count: {service['desiredCount']}")
        logger.info(f"Running count: {service['runningCount']}")
        logger.info(f"Pending count: {service['pendingCount']}")

        # Check tasks
        tasks = ecs.list_tasks(cluster=cluster_name, serviceName=service_name)
        if tasks['taskArns']:
            task_details = ecs.describe_tasks(
                cluster=cluster_name,
                tasks=tasks['taskArns']
            )

            for task in task_details['tasks']:
                logger.info(
                    f"Task {task['taskArn'].split('/')[-1]}: {task['lastStatus']}")
                if task['lastStatus'] == 'RUNNING':
                    # Get network interface
                    for attachment in task.get('attachments', []):
                        if attachment['type'] == 'ElasticNetworkInterface':
                            eni_id = None
                            for detail in attachment['details']:
                                if detail['name'] == 'networkInterfaceId':
                                    eni_id = detail['value']
                                    break

                            if eni_id:
                                ec2 = boto3.client(
                                    'ec2', region_name='us-east-1')
                                eni_details = ec2.describe_network_interfaces(
                                    NetworkInterfaceIds=[eni_id]
                                )

                                if eni_details['NetworkInterfaces']:
                                    eni = eni_details['NetworkInterfaces'][0]
                                    if 'Association' in eni:
                                        public_ip = eni['Association']['PublicIpAddress']
                                        logger.success(
                                            f"Public IP: {public_ip}")
                                        logger.info(
                                            f"API URL: http://{public_ip}:8000")
                                        return public_ip

        return False

    except ClientError as e:
        logger.error(f"AWS API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking ECS service: {e}")
        return False


def check_alb():
    """Check Application Load Balancer status."""
    try:
        elbv2 = boto3.client('elbv2', region_name='us-east-1')

        # List load balancers
        lbs = elbv2.describe_load_balancers()
        nutrisage_lbs = [lb for lb in lbs['LoadBalancers']
                         if 'nutrisage' in lb['LoadBalancerName'].lower()]

        if not nutrisage_lbs:
            logger.warning("No NutriSage ALB found")
            return False

        lb = nutrisage_lbs[0]
        logger.info(f"ALB: {lb['LoadBalancerName']}")
        logger.info(f"DNS: {lb['DNSName']}")
        logger.info(f"State: {lb['State']['Code']}")

        # Check target groups
        target_groups = elbv2.describe_target_groups()
        for tg in target_groups['TargetGroups']:
            if 'nutrisage' in tg['TargetGroupName'].lower():
                logger.info(f"Target Group: {tg['TargetGroupName']}")
                logger.info(f"Port: {tg['Port']}")
                logger.info(f"Protocol: {tg['Protocol']}")

                # Check target health
                health = elbv2.describe_target_health(
                    TargetGroupArn=tg['TargetGroupArn'])
                for target in health['TargetHealthDescriptions']:
                    logger.info(
                        f"Target {target['Target']['Id']}: {target['TargetHealth']['State']}")

        return lb['DNSName']

    except ClientError as e:
        logger.error(f"AWS API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking ALB: {e}")
        return False


def check_target_health():
    """Check ALB target health."""
    try:
        elbv2 = boto3.client('elbv2', region_name='us-east-1')

        # Get target groups
        target_groups = elbv2.describe_target_groups()
        for tg in target_groups['TargetGroups']:
            if 'nutrisage' in tg['TargetGroupName'].lower():
                logger.info(
                    f"Checking target health for: {tg['TargetGroupName']}")

                # Check target health
                health = elbv2.describe_target_health(
                    TargetGroupArn=tg['TargetGroupArn'])
                logger.info(
                    f"Found {len(health['TargetHealthDescriptions'])} targets")

                for target in health['TargetHealthDescriptions']:
                    target_id = target['Target']['Id']
                    target_health = target['TargetHealth']['State']
                    logger.info(f"Target {target_id}: {target_health}")

                    if target_health != 'healthy':
                        logger.warning(
                            f"Target {target_id} is not healthy: {target_health}")
                        if 'Description' in target['TargetHealth']:
                            logger.warning(
                                f"Reason: {target['TargetHealth']['Description']}")
                    else:
                        logger.success(f"Target {target_id} is healthy")

                return health['TargetHealthDescriptions']

        return []

    except Exception as e:
        logger.error(f"Error checking target health: {e}")
        return []


def check_ecs_task_details():
    """Get detailed ECS task information."""
    try:
        ecs = boto3.client('ecs', region_name='us-east-1')

        # Get cluster and service
        clusters = ecs.list_clusters()
        nutrisage_clusters = [
            c for c in clusters['clusterArns'] if 'nutrisage' in c.lower()]
        cluster_name = nutrisage_clusters[0].split('/')[-1]

        services = ecs.list_services(cluster=cluster_name)
        service_arn = services['serviceArns'][0]
        service_name = service_arn.split('/')[-1]

        # Get tasks
        tasks = ecs.list_tasks(cluster=cluster_name, serviceName=service_name)
        if tasks['taskArns']:
            task_details = ecs.describe_tasks(
                cluster=cluster_name,
                tasks=tasks['taskArns']
            )

            for task in task_details['tasks']:
                logger.info(f"Task ARN: {task['taskArn']}")
                logger.info(
                    f"Task Definition: {task['taskDefinitionArn'].split('/')[-1]}")
                logger.info(f"Last Status: {task['lastStatus']}")
                logger.info(f"Desired Status: {task['desiredStatus']}")

                # Check container status
                for container in task.get('containers', []):
                    logger.info(f"Container: {container['name']}")
                    logger.info(
                        f"  Status: {container.get('lastStatus', 'unknown')}")
                    logger.info(
                        f"  Exit Code: {container.get('exitCode', 'N/A')}")

                    if 'reason' in container:
                        logger.warning(f"  Reason: {container['reason']}")

                # Check network configuration
                for attachment in task.get('attachments', []):
                    if attachment['type'] == 'ElasticNetworkInterface':
                        logger.info("Network Interface Details:")
                        for detail in attachment['details']:
                            logger.info(
                                f"  {detail['name']}: {detail['value']}")

                            if detail['name'] == 'networkInterfaceId':
                                eni_id = detail['value']
                                ec2 = boto3.client(
                                    'ec2', region_name='us-east-1')
                                eni_details = ec2.describe_network_interfaces(
                                    NetworkInterfaceIds=[eni_id]
                                )

                                if eni_details['NetworkInterfaces']:
                                    eni = eni_details['NetworkInterfaces'][0]
                                    logger.info(f"  Subnet: {eni['SubnetId']}")
                                    logger.info(
                                        f"  Security Groups: {[sg['GroupId'] for sg in eni['Groups']]}")

                                    if 'Association' in eni:
                                        logger.success(
                                            f"  Public IP: {eni['Association']['PublicIpAddress']}")
                                    else:
                                        logger.warning(
                                            "  No public IP association")

    except Exception as e:
        logger.error(f"Error checking ECS task details: {e}")


def main():
    """Main function to check AWS deployment status."""
    logger.info("Checking AWS deployment status...")

    # Check credentials
    if not check_aws_credentials():
        return

    logger.info("--- ECS Service Status ---")
    public_ip = check_ecs_service()

    logger.info("--- ECS Task Details ---")
    check_ecs_task_details()

    logger.info("--- ALB Status ---")
    alb_dns = check_alb()
    logger.info(f"ALB DNS: {alb_dns}")
    logger.info(f"Target group ARN: {target_group_arn}")
    logger.info(f"API endpoint: https://{alb_dns}/api/predict")
    
    # Check application health
    health_url = f"https://{alb_dns}/api/health"

    logger.info("--- Target Health ---")
    check_target_health()

    logger.info("--- Summary ---")
    if public_ip:
        logger.success(f"ECS service is running at: http://{public_ip}:8000")
    else:
        logger.error("ECS service is not running properly")

    if alb_dns:
        logger.success(f"ALB is available at: https://{alb_dns}")
        logger.info(f"API endpoint: https://{alb_dns}/api/predict")
    else:
        logger.warning("ALB not found or not configured")


if __name__ == "__main__":
    main()
