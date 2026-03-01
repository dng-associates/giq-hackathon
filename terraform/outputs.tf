output "bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "public_base_url" {
  value = "https://${aws_s3_bucket.data.bucket}.s3.${var.region}.amazonaws.com"
}

output "tfstate_bucket_name" {
  value = aws_s3_bucket.tfstate.bucket
}
