import cv2
import numpy as np
import os

def heuristic_qr_detect(image_path):
    image = cv2.imread(image_path)
    if image is None: return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Gradient-based density
    # QR codes have lots of vertical and horizontal edges
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Blur to merge the high-frequency transitions into a single blob
    blurred = cv2.blur(grad, (25, 25))
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the dense regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = w * h
        
        # QR codes are square (aspect ratio ~1.0) and relatively large
        if 0.7 < aspect_ratio < 1.3 and area > (image.shape[0] * image.shape[1] * 0.01):
            candidates.append((x, y, w, h, area))
            
    if not candidates:
        print("No QR candidates found")
        return
        
    # Pick the one that is most "square" or has highest density?
    # Usually the QR on Aadhar is on the right side if it's the front, or left if back.
    # In Ronaldo Aadhar, there's a big square on the bottom right usually.
    candidates.sort(key=lambda c: c[4], reverse=True) # Largest one first
    
    for c in candidates[:2]:
        print(f"Candidate: x={c[0]}, y={c[1]}, w={c[2]}, h={c[3]} (Area: {c[4]})")
        # Just to verify, let's see if it's not a face (faces are already detected)
        
if __name__ == "__main__":
    heuristic_qr_detect("samples/image/ronaldo_aadhar.jpg")
