variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "Optional explicit S3 bucket name for raw/processed/splits data. If null, one is generated from prefix + account ID + region."
  default     = null
}

variable "force_destroy" {
  type        = bool
  description = "Allow terraform destroy to delete non-empty bucket(REALLY USEFULL)"
  default     = false
}

variable "bucket_name_prefix" {
  type        = string
  description = "Prefix used when bucket_name is not set."
  default     = "raw"
}

variable "tfstate_bucket_name" {
  type        = string
  description = "Optional explicit S3 bucket name for Terraform state. If null, one is generated from prefix + account ID + region."
  default     = null
}

variable "tfstate_bucket_name_prefix" {
  type        = string
  description = "Prefix used when tfstate_bucket_name is not set."
  default     = "tfstate"
}
