#!/bin/bash

echo "🚀 Deploying NDVI.AI backend to Railway..."
echo ""

# Check if model exists
if [ ! -d "models" ] || [ ! -f "models/generator_final.pth" ]; then
    echo "❌ Error: Model file not found at models/generator_final.pth"
    echo ""
    echo "Copying model from main project..."
    mkdir -p models
    if [ -f "/Users/anand/Downloads/ndvi/Dataset of Sentinel-1 SAR and Sentinel-2 NDVI Imagery/models/generator_final.pth" ]; then
        cp "/Users/anand/Downloads/ndvi/Dataset of Sentinel-1 SAR and Sentinel-2 NDVI Imagery/models/generator_final.pth" models/
        echo "✅ Model copied successfully"
    else
        echo "❌ Model not found in main project either!"
        exit 1
    fi
fi

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "�� Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: NDVI.AI backend"
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
