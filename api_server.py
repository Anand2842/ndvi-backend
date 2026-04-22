"""
FastAPI backend for NDVI.AI - Serves the Next.js frontend
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from pathlib import Path

from model import UNetGenerator

# ============================================================================
# LIFESPAN (replaces deprecated @app.on_event)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup if available."""
    print("🚀 Starting NDVI.AI API Server...")
    print(f"📱 Device: {get_device()}")
    try:
        load_model()
        print("✅ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️  Model file not found: {e}")
        print("   Server will run in fallback mode (no AI generation)")
        print("   Upload models/generator_final.pth to enable AI features")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   Server will run in fallback mode")
    yield

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(title="NDVI.AI API", version="1.0.0", lifespan=lifespan)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ndvi-frontend-iota.vercel.app",
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Automatically detect the best device (MPS for M1 Mac)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================================
# LOAD MODEL
# ============================================================================

device = get_device()
generator = None

def load_model():
    """Load the trained generator model."""
    global generator
    if generator is None:
        model_path = Path("models/generator_final.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
        
        # Load checkpoint - handle both direct state_dict and checkpoint formats
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'generator_state' in checkpoint:
            # Checkpoint format with metadata
            generator.load_state_dict(checkpoint['generator_state'])
        else:
            # Direct state_dict format
            generator.load_state_dict(checkpoint)
        
        generator.eval()
    return generator

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor for model input."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def tensor_to_base64(tensor: torch.Tensor) -> str:
    import cv2
    import matplotlib.cm as cm
    from scipy.ndimage import median_filter

    # ── Step 1: Extract model output ──────────────────
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip((img * 0.5 + 0.5), 0, 1)

    # ── Step 2: Weighted vegetation signal ────────────
    ndvi = (0.6 * img[:,:,1] +
            0.2 * img[:,:,0] +
            0.2 * img[:,:,2])

    # ── Step 3: Adaptive Masking ──────────────────────
    # We want to heavily smooth water (to kill grid artifacts)
    # but barely touch vegetation (to preserve fine texture/dots).
    # Create a soft mask: 0 = water, 1 = vegetation
    # Water in this model is typically < 0.4, land is > 0.5
    mask = np.clip((ndvi - 0.35) / 0.15, 0, 1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)

    # ── Step 4: Water Base (Heavy Smooth) ─────────────
    # Aggressive median to kill checkerboard, then Gaussian
    water_base = median_filter(ndvi, size=7)
    water_base = cv2.GaussianBlur(water_base.astype(np.float32), (21, 21), 0)

    # ── Step 5: Land Base (Light Smooth) ──────────────
    # Only a small median to kill the worst noise, preserve texture
    land_base = median_filter(ndvi, size=3).astype(np.float32)

    # ── Step 6: Blend ─────────────────────────────────
    ndvi_blended = water_base * (1 - mask) + land_base * mask

    # ── Step 7: Upscale to 1024×1024 ─────────────────
    ndvi_big = cv2.resize(ndvi_blended, (1024, 1024),
                          interpolation=cv2.INTER_CUBIC)

    # ── Step 8: Bilateral filter ──────────────────────
    # Sharpens boundaries between zones
    ndvi_bilateral = cv2.bilateralFilter(
        ndvi_big.astype(np.float32),
        d=9,
        sigmaColor=0.04,
        sigmaSpace=15
    )

    # ── Step 9: Percentile-clipped normalization ─────
    p2, p98 = np.percentile(ndvi_bilateral, [2, 98])
    ndvi_norm = np.clip(
        (ndvi_bilateral - p2) / (p98 - p2 + 1e-6),
        0, 1
    )

    # ── Step 10: Apply RdYlGn colormap ────────────────
    cmap = cm.get_cmap('RdYlGn')
    colored = (cmap(ndvi_norm)[:,:,:3] * 255).astype(np.uint8)

    # ── Step 11: Saturation + brightness boost ────────
    hsv = cv2.cvtColor(colored, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.8, 0, 255)   # +80% saturation
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.15, 0, 255)   # +15% brightness
    colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # ── Step 12: Light final smooth ──────────────────
    colored = cv2.GaussianBlur(colored, (3, 3), 0)

    buf = io.BytesIO()
    Image.fromarray(colored).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def calculate_health_metrics(ndvi_image: Image.Image) -> dict:
    """Calculate crop health metrics from NDVI image."""
    # Convert to numpy array
    img_array = np.array(ndvi_image)
    
    # Calculate average greenness (simple heuristic)
    # Green channel intensity as proxy for vegetation health
    green_channel = img_array[:, :, 1].astype(float)
    avg_green = np.mean(green_channel)
    
    # Normalize to 0-100 scale
    health_score = int((avg_green / 255) * 100)
    
    # Classify zones based on pixel intensity
    total_pixels = green_channel.size
    healthy = np.sum(green_channel > 170) / total_pixels * 100
    moderate = np.sum((green_channel > 85) & (green_channel <= 170)) / total_pixels * 100
    stressed = np.sum(green_channel <= 85) / total_pixels * 100
    
    # Determine status
    if health_score >= 70:
        status = "Healthy"
        color = "green"
        message = "Excellent crop health detected. Vegetation is thriving with strong photosynthetic activity."
    elif health_score >= 50:
        status = "Moderate"
        color = "yellow"
        message = "Moderate vegetation health. Some areas may benefit from additional care or monitoring."
    else:
        status = "Stressed"
        color = "red"
        message = "Vegetation stress detected. Immediate attention recommended to assess water, nutrients, or pest issues."
    
    return {
        "score": health_score,
        "status": status,
        "message": message,
        "color": color,
        "zones": {
            "healthy": round(healthy, 1),
            "moderate": round(moderate, 1),
            "stressed": round(stressed, 1)
        }
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "NDVI.AI API",
        "version": "1.0.0",
        "device": str(device)
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an RGB satellite image and generate NDVI health map.
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG, TIF)
    
    Returns:
        JSON with original image, NDVI image, and health metrics
    """
    try:
        # Read and validate image first
        contents = await file.read()
        try:
            rgb_image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Resize original to match output
        rgb_image_resized = rgb_image.resize((256, 256))
        original_b64 = image_to_base64(rgb_image_resized)
        
        # Try to load model and generate NDVI
        try:
            model = load_model()
            
            # Process image
            rgb_tensor = preprocess_image(rgb_image).to(device)
            
            # Generate NDVI
            with torch.no_grad():
                ndvi_tensor = model(rgb_tensor)
            
            # Convert NDVI tensor to smooth base64 image
            ndvi_b64 = tensor_to_base64(ndvi_tensor)
            
            # For health metrics, we need to decode the NDVI image
            ndvi_image_bytes = base64.b64decode(ndvi_b64)
            ndvi_image = Image.open(io.BytesIO(ndvi_image_bytes))
            health_metrics = calculate_health_metrics(ndvi_image)
            
            return JSONResponse({
                "success": True,
                "original_image": original_b64,
                "ndvi_image": ndvi_b64,
                "health": health_metrics,
                "model_info": {
                    "name": "Pix2Pix U-Net Generator",
                    "accuracy": "~86.6% (approximate NDVI)",
                    "trained_on": "2,200 Sentinel-2 image pairs"
                }
            })
            
        except FileNotFoundError:
            # Graceful fallback: model not available
            return JSONResponse({
                "success": True,
                "original_image": original_b64,
                "ndvi_image": original_b64,  # Return original as fallback
                "health": {
                    "score": 0,
                    "status": "Model Not Available",
                    "message": "Backend is running but AI model is not loaded. Displaying original image. To enable AI-powered NDVI analysis, upload the model file to the server.",
                    "color": "gray",
                    "zones": {
                        "healthy": 0,
                        "moderate": 0,
                        "stressed": 0
                    }
                },
                "model_info": {
                    "name": "Fallback Mode (No AI)",
                    "accuracy": "N/A - Model file not found",
                    "trained_on": "Model not loaded"
                }
            })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    try:
        model = load_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": str(device),
        "device_available": device.type in ["mps", "cuda", "cpu"]
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
