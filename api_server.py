"""
FastAPI backend for NDVI.AI - Serves the Next.js frontend
"""

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
# FASTAPI SETUP
# ============================================================================

app = FastAPI(title="NDVI.AI API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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
        checkpoint = torch.load(model_path, map_location=device)
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
    
    # Convert tensor to image
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip((img * 0.5 + 0.5), 0, 1)
    
    # Single channel — weighted green emphasis
    ndvi = (0.6 * img[:,:,1] + 
            0.2 * img[:,:,0] + 
            0.2 * img[:,:,2])
    
    # Heavy smooth FIRST — kills the noise before coloring
    ndvi_smooth = cv2.GaussianBlur(ndvi, (15, 15), 0)
    
    # Gentle contrast stretch — DO NOT use percentile clipping
    # Just normalize to 0-1 range directly
    mn = ndvi_smooth.min()
    mx = ndvi_smooth.max()
    if mx > mn:
        ndvi_norm = (ndvi_smooth - mn) / (mx - mn)
    else:
        ndvi_norm = ndvi_smooth
    
    # Apply RdYlGn colormap
    colormap  = cm.get_cmap('RdYlGn')
    colored   = colormap(ndvi_norm)
    colored   = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Light final smooth to blend edges
    colored = cv2.GaussianBlur(colored, (5, 5), 0)
    
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
        # Load model
        model = load_model()
        
        # Read and validate image
        contents = await file.read()
        try:
            rgb_image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Process image
        rgb_tensor = preprocess_image(rgb_image).to(device)
        
        # Generate NDVI
        with torch.no_grad():
            ndvi_tensor = model(rgb_tensor)
        
        # Convert NDVI tensor to smooth base64 image
        ndvi_b64 = tensor_to_base64(ndvi_tensor)
        
        # Resize original to match output and convert to base64
        rgb_image_resized = rgb_image.resize((256, 256))
        original_b64 = image_to_base64(rgb_image_resized)
        
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
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
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

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("🚀 Starting NDVI.AI API Server...")
    print(f"📱 Device: {device}")
    try:
        load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   Model will be loaded on first request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
