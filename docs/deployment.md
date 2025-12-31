# Production Deployment Guide

## Docker Deployment

### Build Image

```bash
docker build -t kinetra:latest .
```

### Run Container

```bash
docker run -d \
  --name kinetra \
  -e BROKER_API_KEY=your_key \
  -e BROKER_API_SECRET=your_secret \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -p 8000:8000 \
  kinetra:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Services:
# - kinetra: Main trading system
# - prometheus: Metrics collection
# - grafana: Dashboards
```

## Cloud Deployment

### AWS ECS

```bash
# Login to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Push image
docker tag kinetra:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/kinetra:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/kinetra:latest

# Deploy
aws ecs update-service \
  --cluster kinetra-cluster \
  --service kinetra-service \
  --force-new-deployment
```

## Monitoring

### Prometheus Metrics

```
# CHS metrics
kinetra_chs_agents
kinetra_chs_risk
kinetra_chs_class
kinetra_chs_composite

# Performance metrics
kinetra_omega_ratio
kinetra_energy_captured_pct
kinetra_false_activation_rate

# Risk metrics
kinetra_risk_of_ruin
kinetra_max_drawdown
```

### Grafana Dashboards

Import `kinetra_dashboard.json` for:
- Real-time CHS tracking
- Omega ratio trends
- Energy capture efficiency
- Risk gates status

## Security

### Credential Management

**IMPORTANT**: Never commit credentials to version control.

All API keys, tokens, and secrets must be stored in environment variables:

```bash
# .env file (NEVER commit this file)
METAAPI_TOKEN=your_metaapi_token_here
METAAPI_ACCOUNT_ID=your_account_id_here
BROKER_API_KEY=your_broker_api_key
BROKER_API_SECRET=your_broker_secret
```

The `.env.example` file provides a template:

```bash
# Copy and fill in your credentials
cp .env.example .env
# Edit .env with your actual credentials
```

**Best Practices**:
1. Use `.env` files for local development (already in `.gitignore`)
2. Use cloud secret managers for production (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
3. Use GitHub OIDC for CI/CD authentication (no long-lived API keys)
4. Rotate credentials regularly
5. Use read-only credentials where possible

### Environment Variables

Required for MetaAPI integration:
```bash
METAAPI_TOKEN=<your_metaapi_token>
METAAPI_ACCOUNT_ID=<your_account_id>
```

Required for broker integration:
```bash
BROKER_API_KEY=<your_broker_api_key>
BROKER_API_SECRET=<your_broker_secret>
```

Optional monitoring:
```bash
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Secrets Management

Use cloud secret managers:
- AWS: Secrets Manager
- GCP: Secret Manager
- Azure: Key Vault

### Network Security

- Restrict API access to whitelist IPs
- Use VPC/private subnets
- Enable TLS for all connections

## Health Checks

### Readiness Probe

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "chs": 0.85,
  "risk_of_ruin": 0.05,
  "uptime": 3600
}
```

### Liveness Probe

```bash
curl http://localhost:8000/ping
```

## Rollback Procedure

If deployment fails:

```bash
# AWS ECS
aws ecs update-service \
  --cluster kinetra-cluster \
  --service kinetra-service \
  --task-definition kinetra:${PREVIOUS_VERSION}

# Docker
docker stop kinetra
docker run kinetra:${PREVIOUS_TAG}
```

## Scaling

### Horizontal Scaling

- Run multiple instances with different instruments
- Use load balancer for API endpoints
- Shared Redis for state coordination

### Vertical Scaling

- Increase CPU for faster backtesting
- Increase memory for larger lookback windows
- GPU for RL training (optional)

## Backup & Recovery

### Data Backup

```bash
# Backup results
tar -czf kinetra_backup_$(date +%Y%m%d).tar.gz results/

# Upload to S3
aws s3 cp kinetra_backup_*.tar.gz s3://your-backup-bucket/
```

### Model Checkpoints

- Save RL agent checkpoints every 1000 episodes
- Store in versioned S3 bucket
- Retain last 10 checkpoints

## Troubleshooting

### High RoR

1. Check drawdown: `tail -f logs/risk.log`
2. Verify volatility not spiking
3. Reduce position size manually if needed

### Low CHS

1. Check individual components (agents, risk, class)
2. Review recent trades for false activations
3. May need retraining if regime shift

### Circuit Breaker Triggered

1. System auto-halts trading
2. Check health dashboard
3. Manual intervention required to resume
