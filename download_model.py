"""
Download model from Hugging Face on startup
"""
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

def download_model():
    """Download model from Hugging Face if not present locally."""
    model_dir = Path("models")
    model_path = model_dir / "generator_final.pth"
    
    # Create models directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return str(model_path)
    
    print("📥 Downloading model from Hugging Face...")
    print("   This may take a few minutes (218MB)...")
    
    try:
        # Download from Hugging Face
        downloaded_path = hf_hub_download(
            repo_id="Anand2842001/ndvi-model",
            filename="generator_final.pth",
            cache_dir=".cache",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded successfully to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model()
