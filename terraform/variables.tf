variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "Globally quantic S3 bucket name"
}

variable "force_destroy" {
  type        = bool
  description = "Allow terraform destroy to delete non-empty bucket(REALLY USEFULL)"
  default     = false
}
