#!/usr/bin/env python
"""
Fix AWS deployment by updating ECS service to use public subnets.
"""

import boto3
from botocore.exceptions import ClientError
from loguru import logger


def get_public_subnets():
    """Get public subnets in the default VPC."""
    try:
        ec2 = boto3.client('ec2', region_name='us-east-1')

        # Get default VPC
        vpcs = ec2.describe_vpcs(
            Filters=[{'Name': 'is-default', 'Values': ['true']}])
        if not vpcs['Vpcs']:
            logger.error("No default VPC found")
            return []

        vpc_id = vpcs['Vpcs'][0]['VpcId']
        logger.info(f"Default VPC: {vpc_id}")

        # Get public subnets
        subnets = ec2.describe_subnets(
            Filters=[
                {'Name': 'vpc-id', 'Values': [vpc_id]},
                {'Name': 'map-public-ip-on-launch', 'Values': ['true']}
            ]
        )

        public_subnets = []
        for subnet in subnets['Subnets']:
            logger.info(
                f"Public subnet: {subnet['SubnetId']} ({subnet['AvailabilityZone']})")
            public_subnets.append(subnet['SubnetId'])

        return public_subnets

    except Exception as e:
        logger.error(f"Error getting public subnets: {e}")
        return []


def update_ecs_service():
    """Update ECS service to use public subnets."""
    try:
        ecs = boto3.client('ecs', region_name='us-east-1')

        # Get cluster and service
        clusters = ecs.list_clusters()
        nutrisage_clusters = [
            c for c in clusters['clusterArns'] if 'nutrisage' in c.lower()]
        if not nutrisage_clusters:
            logger.error("No Nutrisage cluster found")
            return False

        cluster_name = nutrisage_clusters[0].split('/')[-1]

        services = ecs.list_services(cluster=cluster_name)
        if not services['serviceArns']:
            logger.error("No services found")
            return False

        service_arn = services['serviceArns'][0]
        service_name = service_arn.split('/')[-1]

        logger.info(f"Updating service: {service_name}")

        # Get public subnets
        public_subnets = get_public_subnets()
        if not public_subnets:
            logger.error("No public subnets found")
            return False

        # Get current service configuration
        service_details = ecs.describe_services(
            cluster=cluster_name,
            services=[service_name]
        )

        service = service_details['services'][0]
        current_config = service['networkConfiguration']['awsvpcConfiguration']

        logger.info(f"Current subnets: {current_config['subnets']}")
        logger.info(
            f"Current security groups: {current_config['securityGroups']}")
        logger.info(
            f"Current assign public IP: {current_config.get('assignPublicIp', 'DISABLED')}")

        # Update service with public subnets
        new_config = {
            'subnets': public_subnets[:2],  # Use first 2 public subnets
            'securityGroups': current_config['securityGroups'],
            'assignPublicIp': 'ENABLED'  # Enable public IP assignment
        }

        logger.info(f"New subnets: {new_config['subnets']}")
        logger.info(f"New assign public IP: {new_config['assignPublicIp']}")

        # Update the service
        response = ecs.update_service(
            cluster=cluster_name,
            service=service_name,
            networkConfiguration={
                'awsvpcConfiguration': new_config
            }
        )

        logger.success(f"Service updated successfully!")
        logger.info(f"New service ARN: {response['service']['serviceArn']}")

        return True

    except Exception as e:
        logger.error(f"Error updating ECS service: {e}")
        return False


def wait_for_service_stability():
    """Wait for the service to become stable."""
    try:
        ecs = boto3.client('ecs', region_name='us-east-1')

        clusters = ecs.list_clusters()
        nutrisage_clusters = [
            c for c in clusters['clusterArns'] if 'nutrisage' in c.lower()]
        cluster_name = nutrisage_clusters[0].split('/')[-1]

        services = ecs.list_services(cluster=cluster_name)
        service_arn = services['serviceArns'][0]
        service_name = service_arn.split('/')[-1]

        logger.info("Waiting for service to become stable...")

        waiter = ecs.get_waiter('services_stable')
        waiter.wait(
            cluster=cluster_name,
            services=[service_name],
            WaiterConfig={'Delay': 10, 'MaxAttempts': 30}
        )

        logger.success("Service is now stable!")
        return True

    except Exception as e:
        logger.error(f"Error waiting for service stability: {e}")
        return False


def main():
    """Main function to fix the AWS deployment."""
    logger.info("Fixing AWS deployment...")

    # Update ECS service
    if update_ecs_service():
        # Wait for stability
        wait_for_service_stability()

        logger.success("AWS deployment fixed!")
        logger.info("The service should now be accessible via the ALB.")
        logger.info("Run 'python check_aws_status.py' to verify the fix.")
    else:
        logger.error("Failed to fix AWS deployment")


if __name__ == "__main__":
    main()
