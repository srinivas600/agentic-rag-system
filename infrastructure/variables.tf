variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  default     = "rag-agent"
}

variable "environment" {
  description = "Deployment environment"
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  default     = "10.0.0.0/16"
}

# ── Database ──────────────────────────────────────────────────────────

variable "db_instance_class" {
  description = "RDS instance class"
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  default     = "ragagent"
}

variable "db_username" {
  description = "Database master username"
  default     = "ragadmin"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  sensitive   = true
}

# ── Redis ─────────────────────────────────────────────────────────────

variable "redis_node_type" {
  description = "ElastiCache node type"
  default     = "cache.t3.micro"
}

# ── ECS ───────────────────────────────────────────────────────────────

variable "api_cpu" {
  description = "Fargate CPU units (1024 = 1 vCPU)"
  default     = 512
}

variable "api_memory" {
  description = "Fargate memory (MiB)"
  default     = 1024
}

variable "api_desired_count" {
  description = "Number of API task instances"
  default     = 2
}

variable "celery_cpu" {
  description = "Celery worker CPU units"
  default     = 512
}

variable "celery_memory" {
  description = "Celery worker memory (MiB)"
  default     = 1024
}

# ── Secrets ───────────────────────────────────────────────────────────

variable "openai_api_key" {
  description = "OpenAI API key"
  sensitive   = true
}

variable "pinecone_api_key" {
  description = "Pinecone API key"
  sensitive   = true
}

variable "pinecone_index_name" {
  description = "Pinecone index name"
  default     = "rag-documents"
}
