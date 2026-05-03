import requests
import json
from pathlib import Path

url = "http://127.0.0.1:8000/api/cases"

# Construct the JSON part matching what the frontend sends
case_data = {
    "firstName": "Jane",
    "lastName": "Doe",
    "mrn": "MRN-12345",
    "sex": "F",
    "age": 62,
    "troponin": 0.08,
    "bnp": 400.0,
    "wbc": 12.5,
    "rhythm": "Sinus Tachycardia"
}

image_path = Path(r"b:\symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0\frontend\public\mock-data\dicoms\case_1.png")

with open(image_path, "rb") as f:
    files = {
        "image": ("case_1.png", f, "image/png")
    }
    data = {
        "case_data": json.dumps(case_data)
    }
    
    print("Sending POST request to create case...")
    response = requests.post(url, data=data, files=files)
    
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
