"""Debug test to see if Stable Diffusion loads"""
from app import image_generator

print("Attempting to load Stable Diffusion...")
try:
    pipeline = image_generator.load_stable_diffusion_model()
    print(f"Success! Pipeline loaded: {pipeline is not None}")
except Exception as e:
    print(f"Failed to load: {e}")
    import traceback
    traceback.print_exc()
