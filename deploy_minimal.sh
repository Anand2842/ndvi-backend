#!/bin/bash

echo "🚀 Deploying minimal NDVI.AI backend to Railway..."
echo ""
echo "📦 Image size optimizations:"
echo "  ✓ Using PyTorch CPU-only (saves ~2GB)"
echo "  ✓ Minimal dependencies only"
echo "  ✓ Using slim Python base image"
echo ""
echo "Expected final image size: ~2.5GB (well under 4GB limit)"
echo ""

# Check if model exists
if [ ! -f "models/generator_final.pth" ]; then
    echo "❌ Error: Model file not found at models/generator_final.pth"
    echo "   Please ensure the trained model is present before deploying."
    exit 1
fi

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway whoami || railway login

# Link to project (if not already linked)
echo "🔗 Linking to Railway project..."
railway link

# Deploy
echo "🚢 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📊 Monitor your deployment:"
echo "   railway logs"
echo ""
echo "🌐 Once deployed, update your Next.js frontend with the Railway URL"
