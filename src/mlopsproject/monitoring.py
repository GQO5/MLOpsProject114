# src/mlopsproject/monitoring.py
import os
import uuid
import json
from datetime import datetime
from google.cloud import storage

# Feature Flag: Only collect data if this is set to "True"
ENABLE_DATA_COLLECTION = os.getenv("ENABLE_DATA_COLLECTION", "False").lower() == "true"
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "group114-bucket")

def save_to_gcs(image_bytes: bytes, pred_result: dict):
    """
    Uploads data to GCS. This function is meant to run in the background.
    """
    # 1. Check the flag. If OFF, stop immediately.
    if not ENABLE_DATA_COLLECTION:
        return

    try:
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Upload Image
        blob_img = bucket.blob(f"data_collection/images/{unique_id}.jpg")
        blob_img.upload_from_string(image_bytes, content_type="image/jpeg")

        # Upload Metadata
        metadata = {"id": unique_id, "timestamp": timestamp, "prediction": pred_result}
        blob_json = bucket.blob(f"data_collection/metadata/{unique_id}.json")
        blob_json.upload_from_string(json.dumps(metadata), content_type="application/json")
        
        print(f"Background Task: Data saved {unique_id}")
        
    except Exception as e:
        print(f"Background Task Failed: {e}")