# NDVI.AI Backend - Railway Deployment

## Quick Deploy

```bash
./deploy_minimal.sh
```

## What's Included

- ✅ `api_server.py` - FastAPI backend
- ✅ `model.py` - U-Net Generator architecture
- ✅ `models/generator_final.pth` - Trained model (218MB)
- ✅ `Dockerfile` - Optimized for Railway (<4GB)
- ✅ `requirements_minimal.txt` - PyTorch CPU-only + minimal deps
- ✅ `.dockerignore` - Excludes unnecessary files
- ✅ `railway.toml` - Railway configuration

## Image Size: ~2.5GB ✅

- Base image: ~150MB
- PyTorch CPU: ~700MB
- Dependencies: ~100MB
- Model: 218MB
- Code: ~50KB

**Total: Under 4GB Railway limit!**

## Deploy Steps

1. **Deploy to Railway**
   ```bash
   ./deploy_minimal.sh
   ```

2. **Get your URL**
   ```bash
   railway domain
   ```

3. **Test the API**
   ```bash
   curl https://your-app.railway.app/
   ```

## API Endpoints

- `GET /` - Health check
- `POST /analyze` - Upload RGB image, get NDVI analysis
- `GET /health` - Detailed health check

## After Deployment

Update your Next.js frontend `.env.production`:
```env
NEXT_PUBLIC_API_URL=https://your-app.railway.app
```

## Troubleshooting

```bash
# View logs
railway logs

# Check status
railway status

# Redeploy
railway up
```
