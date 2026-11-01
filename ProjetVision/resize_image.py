import cv2

def resize(image):
    max_width = 1300  # Leave margin for window chrome on 1440p
    max_height = 1024
    
    h, w = image.shape[0], image.shape[1]
    
    # Calculate scaling factor to fit within max dimensions while keeping aspect ratio
    scale = min(max_width / w, max_height / h, 1.0)  # Don't upscale tiny images beyond 1x
    
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    return cv2.resize(image, (new_width, new_height))