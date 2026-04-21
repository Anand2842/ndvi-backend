# NDVI.AI Deployment Guide

## Architecture Overview

```
┌─────────────────┐
│  Hugging Face   │  ← Model Storage (218MB)
│  Anand2842001/  │     https://huggingface.co/Anand2842001/ndvi-model
│   ndvi-model    │
└────────┬────────┘
         │ downloads model
         ↓
┌─────────────────┐
│     GitHub      │  ← Code Repository
│  Anand2842/     │     https://github.com/Anand2842/ndvi-backend
│  ndvi-backend   │
└────────┬────────┘
         │ deploys from
         ↓
┌─────────────────┐
│    Railway      │  ← Backend API
│  FastAPI Server │     Downloads model from HF on startup
└────────┬────────┘
         │ API calls
         ↓
┌─────────────────┐
│     Vercel      │  ← Frontend
│  Next.js App    │     Calls Railway API
└─────────────────┘
```

## ✅ Completed Steps

### 1. Model Uploaded to Hugging Face
- **Repository**: `Anand2842001/ndvi-model`
- **File**: `generator_final.pth` (218MB)
- **URL**: https://huggingface.co/Anand2842001/ndvi-model
- **Access**: Public (no authentication needed for download)

### 2. Code Pushed to GitHub
- **Repository**: `Anand2842/ndvi-backend`
- **URL**: https://github.com/Anand2842/ndvi-backend
- **Size**: ~11KB (no model file)
- **Branch**: `main`

## 🚀 Deploy to Railway

### Option 1: Automatic Deployment (Recommended)

1. **Go to Railway Dashboard**
   ```
   https://railway.app/dashboard
   ```

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `Anand2842/ndvi-backend`

3. **Railway will automatically**:
   - Detect the Dockerfile
   - Build the image (~2.5GB)
   - Download model from Hugging Face on first startup
   - Deploy to a public URL

4. **Get your URL**:
   - Railway will provide a URL like: `https://ndvi-backend-production.up.railway.app`

### Option 2: Manual Deployment

```bash
cd ~/Desktop/ndvi-backend

# Login to Railway
railway login

# Link to project
railway link

# Deploy
railway up
```

## 📊 Expected Deployment

### Image Size Breakdown:
- Base image (python:3.10-slim): ~150MB
- PyTorch CPU-only: ~700MB
- Dependencies: ~100MB
- Code: ~50KB
- **Total**: ~2.5GB ✅ (under 4GB limit)

### First Startup:
- Downloads model from Hugging Face (~218MB)
- Takes 2-3 minutes on first run
- Subsequent restarts are instant (model cached)

## 🔧 Environment Variables (Optional)

No environment variables required! The app works out of the box.

Optional variables:
- `PORT` - Railway sets this automatically
- `HF_HOME` - Hugging Face cache directory (default: `.cache`)

## 🧪 Testing the Deployment

### 1. Health Check
```bash
curl https://your-app.railway.app/
```

Expected response:
```json
{
  "status": "online",
  "service": "NDVI.AI API",
  "version": "1.0.0",
  "device": "cpu",
  "model_source": "Hugging Face (Anand2842001/ndvi-model)"
}
```

### 2. Detailed Health Check
```bash
curl https://your-app.railway.app/health
```

### 3. Test Image Analysis
```bash
curl -X POST https://your-app.railway.app/analyze \
  -F "file=@test_image.tif"
```

## 🌐 Update Frontend (Vercel)

Once Railway is deployed, update your Next.js frontend:

### 1. Update Environment Variable
In Vercel dashboard or `.env.production`:
```env
NEXT_PUBLIC_API_URL=https://your-app.railway.app
```

### 2. Redeploy Frontend
```bash
# If using Vercel CLI
vercel --prod

# Or push to GitHub (if auto-deploy is enabled)
git push origin main
```

## 📝 Deployment Checklist

- [x] Model uploaded to Hugging Face
- [x] Code pushed to GitHub (without model)
- [ ] Deploy to Railway
- [ ] Get Railway URL
- [ ] Update Vercel environment variable
- [ ] Test end-to-end workflow

## 🔍 Monitoring

### View Logs
```bash
railway logs
```

### Check Status
```bash
railway status
```

### Redeploy
```bash
railway up
```

## 🐛 Troubleshooting

### Model Download Fails
- Check Hugging Face is accessible
- Model repo is public: https://huggingface.co/Anand2842001/ndvi-model
- Railway has internet access (it does by default)

### Image Size Too Large
Current setup is ~2.5GB, well under 4GB limit.
If you hit limits:
- Upgrade to Railway Pro ($5/month) - 8GB limit
- Or use Railway's Hobby plan - 4GB limit (current)

### Deployment Fails
```bash
# Check logs
railway logs --tail 100

# Check build logs
railway logs --deployment
```

## 💰 Cost Estimate

### Free Tier (Current Setup):
- **Hugging Face**: Free (public model)
- **GitHub**: Free (public repo)
- **Railway**: $5 credit/month (free tier)
- **Vercel**: Free (hobby plan)

**Total**: $0/month (within free tiers)

### If You Need More:
- **Railway Pro**: $5/month (8GB images, more resources)
- **Vercel Pro**: $20/month (more bandwidth)

## 🎉 Next Steps

1. Deploy to Railway (see above)
2. Get your Railway URL
3. Update Vercel with the URL
4. Test the full workflow
5. Share your project! 🚀

## �� Resources

- **Model**: https://huggingface.co/Anand2842001/ndvi-model
- **Code**: https://github.com/Anand2842/ndvi-backend
- **Railway Docs**: https://docs.railway.app
- **Vercel Docs**: https://vercel.com/docs
