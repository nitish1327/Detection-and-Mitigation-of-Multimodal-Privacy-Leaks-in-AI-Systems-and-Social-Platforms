import cv2
import os

def test_qr():
    image_path = "samples/image/ronaldo_aadhar.jpg"
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return
        
    print(f"Image loaded: {image.shape}")
    
    # Try WeChat QR
    try:
        # If no models are found, this might fail or do nothing
        detector = cv2.wechat_qrcode_WeChatQRCode()
        res, points = detector.detectAndDecode(image)
        if res:
            print(f"WeChat QR detected: {res}")
            return
        else:
            print("WeChat QR failed (no detections)")
    except Exception as e:
        print(f"WeChat QR error: {e}")

    # Try default detector with different scales
    qr_detector = cv2.QRCodeDetector()
    for scale in [0.5, 1.0, 1.5, 2.0]:
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        ok, res, points, _ = qr_detector.detectAndDecodeMulti(resized)
        if ok and any(res):
            print(f"Default QR detected at scale {scale}: {res}")
            return
            
    print("All automated QR detection failed")

if __name__ == "__main__":
    test_qr()
