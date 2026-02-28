terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "data" {
  bucket        = var.bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_ownership_controls" "own" {
  bucket = aws_s3_bucket.data.id
  rule { object_ownership = "BucketOwnerPreferred" }
}

resource "aws_s3_bucket_public_access_block" "public_block" {
  bucket             = aws_s3_bucket.data.id
  block_public_acls  = true
  ignore_public_acls = true

  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sse" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "public_read_selected_prefixes" {
  statement {
    sid     = "PublicReadRawProcessed"
    effect  = "Allow"
    actions = ["s3:GetObject"]

    resources = [
      "arn:aws:s3:::${var.bucket_name}/raw/*",
      "arn:aws:s3:::${var.bucket_name}/processed/*",
      "arn:aws:s3:::${var.bucket_name}/splits/*",
      "arn:aws:s3:::${var.bucket_name}/manifests/*"
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }
}

resource "aws_s3_bucket_policy" "public_policy" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.public_read_selected_prefixes.json

  depends_on = [aws_s3_bucket_public_access_block.public_block]
}
