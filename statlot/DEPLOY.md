# Deployment Status

Auto-deployed via GitHub Actions → EC2

- **Host:** 54.179.152.124
- **Service:** statlot.service (systemd)
- **Port:** 8000
- **Stack:** FastAPI + Uvicorn + Python 3.12

Every push to `main` automatically:
1. SSHs into EC2
2. `git pull origin main`
3. Reinstalls dependencies
4. Restarts the statlot service
