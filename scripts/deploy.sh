#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# RAG Agent — AWS Deployment Script
# Usage:
#   ./scripts/deploy.sh              # Full deploy (infra + build + push)
#   ./scripts/deploy.sh build        # Build & push Docker image only
#   ./scripts/deploy.sh infra        # Terraform apply only
#   ./scripts/deploy.sh ecs-update   # Force new ECS deployment
# ─────────────────────────────────────────────────────────────────

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT="rag-agent"
ENV="production"
ECR_REPO="${PROJECT}-${ENV}-api"
ECS_CLUSTER="${PROJECT}-${ENV}-cluster"
ECS_SERVICE="${PROJECT}-${ENV}-api"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[deploy]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────
check_tools() {
    for cmd in aws docker terraform; do
        command -v $cmd &>/dev/null || fail "$cmd is not installed"
    done
    aws sts get-caller-identity &>/dev/null || fail "AWS credentials not configured"
    ok "All tools available, AWS authenticated"
}

# ── Terraform ────────────────────────────────────────────────────
deploy_infra() {
    log "Deploying infrastructure with Terraform..."
    cd infrastructure

    terraform init -upgrade
    terraform validate
    terraform plan -out=tfplan
    terraform apply tfplan

    ECR_URL=$(terraform output -raw ecr_repository_url)
    ALB_DNS=$(terraform output -raw alb_dns_name)

    cd ..
    ok "Infrastructure deployed"
    log "ALB URL: http://${ALB_DNS}"
}

# ── Docker build & push ─────────────────────────────────────────
build_and_push() {
    log "Building Docker image..."

    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URL="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
    IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    docker build -t "${ECR_URL}:${IMAGE_TAG}" \
                 -t "${ECR_URL}:latest" \
                 --target production .

    log "Pushing to ECR..."
    docker push "${ECR_URL}:${IMAGE_TAG}"
    docker push "${ECR_URL}:latest"

    ok "Image pushed: ${ECR_URL}:${IMAGE_TAG}"
}

# ── ECS force deploy ─────────────────────────────────────────────
ecs_update() {
    log "Forcing new ECS deployment..."
    aws ecs update-service \
        --cluster $ECS_CLUSTER \
        --service $ECS_SERVICE \
        --force-new-deployment \
        --region $AWS_REGION

    log "Waiting for service to stabilize..."
    aws ecs wait services-stable \
        --cluster $ECS_CLUSTER \
        --services $ECS_SERVICE \
        --region $AWS_REGION

    ok "ECS deployment complete"
}

# ── Main ─────────────────────────────────────────────────────────
main() {
    local cmd="${1:-full}"
    check_tools

    case $cmd in
        infra)      deploy_infra ;;
        build)      build_and_push ;;
        ecs-update) ecs_update ;;
        full)
            deploy_infra
            build_and_push
            ecs_update
            echo ""
            ok "Full deployment complete!"
            ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
            cd infrastructure
            ALB_DNS=$(terraform output -raw alb_dns_name 2>/dev/null || echo "check terraform outputs")
            cd ..
            log "App URL: http://${ALB_DNS}"
            ;;
        *) fail "Unknown command: $cmd. Use: full, infra, build, ecs-update" ;;
    esac
}

main "$@"
