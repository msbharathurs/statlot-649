# AWS Infrastructure Setup — StatLot 649

## Step 1 — S3 Bucket

```bash
# Create bucket (replace ap-southeast-1 with your region)
aws s3api create-bucket \
  --bucket statlot-649 \
  --region ap-southeast-1 \
  --create-bucket-configuration LocationConstraint=ap-southeast-1

# Block public access (always)
aws s3api put-public-access-block \
  --bucket statlot-649 \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,\
    BlockPublicPolicy=true,RestrictPublicBuckets=true

# Upload draws.csv
aws s3 cp data/draws.csv s3://statlot-649/statlot-649/data/draws.csv
```

**Bucket structure:**
```
statlot-649/
  statlot-649/
    data/draws.csv
    models/iter1/   ← saved after Iter1 training
    models/iter2/   ← saved after Iter2 training
    models/iter3/   ← saved after Iter3 training
    results/iter1_results.json
    results/iter2_results.json
    results/iter3_results.json
    results/all_results.json
    logs/backtest_YYYYMMDD_HHMMSS.log
```

---

## Step 2 — IAM Role for EC2

Create a role named `statlot-ec2-role`:

```bash
# 1. Create trust policy
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ec2.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# 2. Create role
aws iam create-role \
  --role-name statlot-ec2-role \
  --assume-role-policy-document file://trust-policy.json

# 3. Create S3 policy (scoped to this bucket only)
cat > s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::statlot-649",
      "arn:aws:s3:::statlot-649/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name statlot-ec2-role \
  --policy-name statlot-s3-access \
  --policy-document file://s3-policy.json

# 4. Create instance profile and attach role
aws iam create-instance-profile --instance-profile-name statlot-ec2-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name statlot-ec2-profile \
  --role-name statlot-ec2-role
```

---

## Step 3 — Security Group

```bash
aws ec2 create-security-group \
  --group-name statlot-sg \
  --description "StatLot 649 ML instance"

# SSH from your IP only — replace with your actual IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-name statlot-sg \
  --protocol tcp --port 22 \
  --cidr ${MY_IP}/32
```

---

## Step 4 — Launch EC2 Instance

```bash
# Get latest Ubuntu 24.04 AMI for your region
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text)
echo "AMI: $AMI_ID"

# Launch c5a.2xlarge
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type c5a.2xlarge \
  --key-name YOUR_KEY_PAIR_NAME \
  --security-groups statlot-sg \
  --iam-instance-profile Name=statlot-ec2-profile \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 20,
      "VolumeType": "gp3",
      "Throughput": 125,
      "DeleteOnTermination": true
    }
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=statlot-649-ml}]' \
  --count 1
```

**Instance specs:**
| Item | Value |
|---|---|
| Type | c5a.2xlarge |
| vCPU | 8 (AMD EPYC) |
| RAM | 16 GB |
| EBS | 20 GB gp3 |
| Estimated run cost | ~$0.28 per full backtest |
| On-demand price | $0.308/hr (ap-southeast-1) |

> 💡 **Spot instance tip:** Use `--instance-market-options '{"MarketType":"spot"}'`
> to cut cost to ~$0.09/hr. Safe for a ~55 min batch job with no checkpointing loss
> because results auto-save to S3 after each iteration.

---

## Step 5 — Run Setup on EC2

```bash
# SSH in
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# One-line setup (takes ~8 min including pip install)
curl -fsSL https://raw.githubusercontent.com/msbharathurs/statlot-649/main/setup_ec2.sh | bash -s statlot-649
```

---

## Step 6 — Start Backtest

```bash
# Run in a screen session so it survives SSH disconnect
screen -S backtest
cd ~/statlot-649 && bash run_backtest.sh
# Ctrl+A, D to detach
# screen -r backtest to re-attach
```

---

## Cost Summary

| Item | Cost |
|---|---|
| c5a.2xlarge on-demand (~1 hr) | ~$0.31 |
| c5a.2xlarge spot (~1 hr) | ~$0.09 |
| EBS 20 GB gp3 (1 month) | ~$1.60 |
| S3 storage (<1 GB/month) | ~$0.02 |
| **Total per full backtest run** | **$0.09 – $0.31** |

> Terminate the instance after the run. EBS + S3 hold all results permanently.
