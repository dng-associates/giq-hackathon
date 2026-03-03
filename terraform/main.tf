terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  effective_bucket_name = (
    var.bucket_name != null && trimspace(var.bucket_name) != ""
  ) ? var.bucket_name : "${var.bucket_name_prefix}-${data.aws_caller_identity.current.account_id}-${var.region}"
  effective_tfstate_bucket_name = (
    var.tfstate_bucket_name != null && trimspace(var.tfstate_bucket_name) != ""
  ) ? var.tfstate_bucket_name : "${var.tfstate_bucket_name_prefix}-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket" "data" {
  bucket        = local.effective_bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket" "tfstate" {
  bucket        = local.effective_tfstate_bucket_name
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_ownership_controls" "own" {
  bucket = aws_s3_bucket.data.id
  rule { object_ownership = "BucketOwnerPreferred" }
}

resource "aws_s3_bucket_ownership_controls" "tfstate_own" {
  bucket = aws_s3_bucket.tfstate.id
  rule { object_ownership = "BucketOwnerPreferred" }
}

resource "aws_s3_bucket_acl" "tfstate_acl" {
  bucket     = aws_s3_bucket.tfstate.id
  acl        = "private"
  depends_on = [aws_s3_bucket_ownership_controls.tfstate_own]
}

resource "aws_s3_bucket_public_access_block" "public_block" {
  bucket             = aws_s3_bucket.data.id
  block_public_acls  = true
  ignore_public_acls = true

  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_public_access_block" "tfstate_public_block" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_versioning" "tfstate_versioning" {
  bucket = aws_s3_bucket.tfstate.id
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

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate_sse" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "public_read_selected_prefixes" {
  statement {
    sid     = "PublicReadSelectedPrefixes"
    effect  = "Allow"
    actions = ["s3:GetObject"]

    resources = [
      "${aws_s3_bucket.data.arn}/raw/v1/*",
      "${aws_s3_bucket.data.arn}/refined/v1/*"
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }

  statement {
    sid     = "PublicListSelectedPrefixes"
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [
      aws_s3_bucket.data.arn
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        "raw/v1/*",
        "raw/v1/",
        "refined/v1/*",
        "refined/v1/"
      ]
    }
  }
}

resource "aws_s3_bucket_policy" "public_policy" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.public_read_selected_prefixes.json

  depends_on = [aws_s3_bucket_public_access_block.public_block]
}
