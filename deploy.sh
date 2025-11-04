#!/usr/bin/env bash
# deploy.sh
# Usage: ./deploy.sh my-s3-bucket-name

set -euo pipefail

BUCKET="${1:-}"

if [ -z "$BUCKET" ]; then
  echo "Usage: $0 <s3-bucket-name>"
  exit 1
fi

ART_DIR="artifacts"
MODEL="$ART_DIR/udp_ddos_pipeline.pkl"
FEATS="$ART_DIR/udp_features.json"

if [ ! -f "$MODEL" ] || [ ! -f "$FEATS" ]; then
  echo "Artifacts not found in $ART_DIR. Run train.py first."
  exit 1
fi

aws s3 mb "s3://$BUCKET" 2>/dev/null || echo "Bucket may already exist, continuing…"

aws s3 cp "$MODEL" "s3://$BUCKET/udp_ddos_pipeline.pkl"
aws s3 cp "$FEATS" "s3://$BUCKET/udp_features.json"

echo "✅ Uploaded to s3://$BUCKET/"
