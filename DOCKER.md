# Docker Deployment Guide

This guide explains how to deploy the Video Labeler application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier deployment)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and start the container:**
   ```bash
   VIDEO_LABELER_PORT=8000 docker compose up -d --build
   ```

2. **View logs:**
   ```bash
   docker compose logs -f
   ```

3. **Stop the container:**
   ```bash
   docker compose down
   ```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:8000`
   - Or replace `localhost` with your server's IP address if deploying remotely

### Create the Admin User

Create the first admin user inside the running container:

```bash
docker compose exec video_labeler python manage_users.py create-admin --email you@example.com --password <password>
```

### Using Docker Directly

1. **Build the image:**
   ```bash
   docker build -t video_labeler .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name video_labeler \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     --restart unless-stopped \
     video_labeler
   ```

3. **View logs:**
   ```bash
   docker logs -f video_labeler
   ```

4. **Stop the container:**
   ```bash
   docker stop video_labeler
   docker rm video_labeler
   ```

## Production Deployment

### On a Headless Server VM

1. **SSH into your server:**
   ```bash
   ssh user@your-server-ip
   ```

2. **Clone or upload the project:**
   ```bash
   git clone <your-repo-url>
   cd video_labeler
   ```

3. **Build and start with Docker Compose:**
   ```bash
   VIDEO_LABELER_PORT=8000 docker compose up -d --build
   ```

4. **Configure firewall (if needed):**
   ```bash
   # For Ubuntu/Debian
   sudo ufw allow 8000/tcp
   ```

5. **Access the application:**
   - Navigate to `http://your-server-ip:8000` in your browser

## Data Persistence

The `data/` directory is bind-mounted into the container, so all studies, uploads, and annotations are persisted on the host filesystem. This means:

- Data is stored in `./data/` relative to where you run `docker compose up`
- You can easily access and backup the data directly from the host
- Data persists even if the container is removed

### Backing up the data:

```bash
# Create a backup
tar czf video_labeler_data_backup.tar.gz data/

# Restore from backup
tar xzf video_labeler_data_backup.tar.gz
```

## Environment Variables

You can customize the deployment using environment variables in `docker-compose.yml`:

- `VIDEO_LABELER_PORT` - Host port for the service (default: 8000)
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered (useful for logs)

## Troubleshooting

### Container won't start

1. **Check logs:**
   ```bash
   docker compose logs
   ```

2. **Verify port is available:**
   ```bash
   netstat -tuln | grep 8000
   ```

### Can't access from remote machine

1. **Check firewall:**
   ```bash
   sudo ufw status
   ```

2. **Verify container is running:**
   ```bash
   docker ps
   ```

3. **Check if port is bound correctly:**
   ```bash
   docker port video_labeler
   ```

### Data not persisting

1. **Verify bind mount:**
   ```bash
   docker inspect video_labeler | grep -A 10 Mounts
   ```

2. **Check data directory exists:**
   ```bash
   ls -la data/
   ```

3. **Check permissions:**
   ```bash
   ls -la data/
   ```

4. **Ensure directory exists before starting:**
   ```bash
   mkdir -p data/uploads data/studies data/temp
   ```

## Updating the Application

1. **Pull latest changes:**
   ```bash
   git pull
   ```

2. **Rebuild and restart:**
   ```bash
   docker compose up -d --build
   ```

**Note:** Your data directory will persist across updates, so all studies and annotations will remain intact.
