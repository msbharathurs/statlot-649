import boto3
import os
import stat
import textwrap


def restore_ssh_key(pem_path="/tmp/statlot.pem"):
    """
    Restores EC2 SSH key from Base44 vault to disk.
    Call this at the start of any script that needs EC2 access.
    Returns the pem_path on success, raises on failure.
    """
    client = boto3.client("secretsmanager", region_name="ap-southeast-1")
    secret = client.get_secret_value(SecretId="statlot-ec2-key")["SecretString"]

    # Fix newline stripping that happens during secret storage
    if "-----BEGIN RSA PRIVATE KEY-----" in secret and "\n" not in secret[30:]:
        header = "-----BEGIN RSA PRIVATE KEY-----"
        footer = "-----END RSA PRIVATE KEY-----"
        body = secret.replace(header, "").replace(footer, "").strip()
        body = "\n".join(textwrap.wrap(body, 64))
        secret = f"{header}\n{body}\n{footer}\n"

    with open(pem_path, "w") as f:
        f.write(secret)
    os.chmod(pem_path, stat.S_IRUSR)
    return pem_path
