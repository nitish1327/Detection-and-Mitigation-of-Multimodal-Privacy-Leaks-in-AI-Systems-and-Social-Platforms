import requests
import json
import time

def test_text_scan():
    url = "http://127.0.0.1:8000/scan/text"
    payload = {
        "text": "Hello, my name is John Doe. My email is john.doe@example.com and my phone number is +1-555-010-9988. Please keep this confidential.",
        "policy": {
            "mask_on": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
        }
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Wait a bit for service to ensure it's up if run immediately after start
        time.sleep(2) 
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\n--- Response ---")
        print(f"Decision: {result.get('decision')}")
        print(f"Entities Detected: {len(result.get('entities', []))}")
        for entity in result.get('entities', []):
            print(f" - {entity['label']}: {entity['text']} (Confidence: {entity['confidence']})")
            
        print("\n--- Mitigated Output ---")
        print(result.get('mitigated_output'))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_text_scan()
