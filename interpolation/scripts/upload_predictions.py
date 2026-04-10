"""Upload today's GP predictions JSON to Cloudflare R2."""
import boto3, json, os, sys
from botocore.config import Config
from pathlib import Path

date = sys.argv[1]
pred = Path(f"interpolation/data/predictions/{date}.json")
if not pred.exists():
    print(f"No predictions file for {date} — skipping upload")
    sys.exit(0)

account_id = os.environ["R2_ACCOUNT_ID"]
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
    region_name="auto",
)
s3.upload_file(
    str(pred),
    os.environ["R2_BUCKET"],
    f"data/predictions/{date}.json",
    ExtraArgs={"ContentType": "application/json", "CacheControl": "public, max-age=86400"},
)
print(f"Uploaded predictions for {date}")
