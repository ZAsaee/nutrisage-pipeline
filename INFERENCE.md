# NutriSage Inference & Health-Run Guide

This document is the **single reference** you need to:

1. Stream real records from S3 through the NutriSage API
2. Exercise automatic scaling on the ECS service
3. Observe both infrastructure and model-level metrics
4. Validate that the service stays healthy during a multi-hour run

> 📝  Everything below can be executed from your laptop **or** folded into GitHub Actions / Terraform later.  Replace placeholders (`<…>`) with your own values.

---

## 0  Prerequisites (one-time)

| Requirement | Check |
|-------------|-------|
| NutriSage image & task definition are deployed via `.github/workflows/aws.yml` | ✅ |
| ECS service exposed through an **Application Load Balancer** on port `8000` | ✅ |
| AWS CLI configured (`aws configure sso` or keys) | ✅ |
| S3 bucket with inference data, e.g. `s3://nutrisage/inference/` | ✅ |
| Security group & subnet IDs handy for one-off Fargate tasks | ✅ |

---

## 1  Enable Autoscaling (run once)

```bash
# Register the service as a scalable target
aws application-autoscaling register-scalable-target `
  --service-namespace ecs `
  --resource-id service/nutrisage-cluster/nutrisage-api-service `
  --scalable-dimension ecs:service:DesiredCount `
  --min-capacity 1 --max-capacity 5

# Target-tracking policy on ALB request rate
aws application-autoscaling put-scaling-policy \
  --policy-name alb-tt-30 \
  --service-namespace ecs \
  --resource-id service/nutrisage-cluster/nutrisage-api-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
      "TargetValue": 30,
      "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ALBRequestCountPerTarget"
      },
      "ScaleOutCooldown": 60,
      "ScaleInCooldown": 60
    }'
```

---

## 2  Hardware Metrics (Container Insights)

```bash
aws ecs update-cluster-settings \
  --cluster nutrisage-cluster \
  --settings name=containerInsights,value=enabled
```

This streams CPU, memory, network, and disk I/O for every task into **CloudWatch Metrics**.

---

## 3  Model-Level Metrics (10 LOC change)

In `src/api/utils.py` (or directly in the FastAPI endpoint) wrap the prediction logic with AWS Embedded Metrics:

```python
from aws_embedded_metrics import metric_scope

@metric_scope
def predict_nutrition_grade(metrics, nutrition_data):
    start = time.perf_counter()
    result = _infer(nutrition_data)          # existing logic
    metrics.put_metric("ModelLatencyMs", (time.perf_counter()-start)*1000, "Milliseconds")
    metrics.put_metric("Confidence", result.confidence)
    metrics.set_property("grade", result.nutrition_grade.value)
    return result
```

Metrics will appear under namespace **`NutriSage/Inference`**.

---

## 4  Publish a CloudWatch Dashboard

Create `dashboard.json` locally (or via the console) with three widgets:

* CPU & Memory utilisation
* Running vs Desired task count
* p90 Model Latency + average Confidence

```bash
aws cloudwatch put-dashboard \
  --dashboard-name NutriSage-ECS \
  --dashboard-body file://dashboard.json
```

> Tip: open the dashboard in the console → *Actions → View/edit source* to grab the JSON.

---

## 5  Optional Alarms

High CPU and high latency alarms keep on-call simple:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name NutriSage-HighCPU \
  --namespace ECS/Service \
  --metric-name CPUUtilization \
  --statistic Average --period 60 --threshold 80 \
  --comparison-operator GreaterThanThreshold --evaluation-periods 3 \
  --dimensions Name=ClusterName,Value=nutrisage-cluster Name=ServiceName,Value=nutrisage-api-service \
  --alarm-actions <SNS_TOPIC_ARN> --ok-actions <SNS_TOPIC_ARN>
```

Duplicate for `NutriSage/Inference :: ModelLatencyMs` (`p90 > 500 ms`).

---

## 6  Live Inference Load Generator

Add **`scripts/run_inference.py`** (≈50 LOC).  It:

1. Lists objects under `INPUT_PREFIX` (`s3://nutrisage/inference/`)
2. Streams each file (CSV/Parquet → dict rows)
3. Fires async POST requests to `$TARGET_URL/predict`
4. (Optional) stores responses back to S3

Environment variables consumed by the script:

| Variable | Example |
|----------|---------|
| `TARGET_URL`   | `https://alb-xyz.elb.amazonaws.com` |
| `INPUT_PREFIX` | `s3://nutrisage/inference/` |
| `WORKERS`      | `50` |

### Local run

```bash
export TARGET_URL=https://<alb-dns>
export INPUT_PREFIX=s3://nutrisage/inference/
python scripts/run_inference.py
```

### One-off Fargate task

```bash
aws ecs run-task \
  --cluster nutrisage-cluster --launch-type FARGATE \
  --task-definition nutrisage-loadtest:1 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xyz],securityGroups=[sg-abc],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides":[{"name":"run_inference","environment":[
      {"name":"TARGET_URL","value":"https://<alb-dns>"},
      {"name":"INPUT_PREFIX","value":"s3://nutrisage/inference/"},
      {"name":"WORKERS","value":"50"}]}]}'
```

> The task exits automatically once all input files are processed.  Re-launch it if you need exactly two hours of load.

---

## 7  Monitor During the Run

* **CloudWatch → Dashboards → `NutriSage-ECS`** – watch CPU, memory & task count rise.
* **CloudWatch Logs** – ensure mostly `200` responses, no stack traces.
* **CloudWatch Alarms** – should stay `OK`; investigate anything in `ALARM`.
* **ECS console → Service events** – confirm new tasks start & exit healthy.

---

## 8  Post-Run Validation

| Check | Command |
|-------|---------|
| Running tasks match desired | `aws ecs describe-services --cluster nutrisage-cluster --services nutrisage-api-service --query "services[0].{Running:runningCount,Desired:desiredCount}"` |
| Review scaling history | `aws application-autoscaling describe-scaling-activities --service-namespace ecs --resource-id service/nutrisage-cluster/nutrisage-api-service` |
| Tail last 2 h of logs | `aws logs tail /ecs/nutrisage-api --since 2h` |
| Dashboard p90 latency stable & < 500 ms | (visual) |

If everything passes, the service has **proven** it can stay live & healthy under sustained real-world inference load.

---

## 9  Portfolio Artefacts

1. Screenshot – dashboard during scale-out (multiple running tasks)
2. Screenshot – scaling activities timeline
3. Note the single-file `run_inference.py` and autoscaling policy in the repo
4. Link to the CloudWatch dashboard (if public) or embed image in README

These four artefacts show CI/CD, autoscaling, observability, and model metrics in action – a complete MLOps story in minutes.

---

**Happy inferencing 🚀**