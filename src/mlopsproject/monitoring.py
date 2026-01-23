# src/mlopsproject/monitoring.py
import os
import uuid
import json
from datetime import datetime, timedelta
from google.cloud import storage
import pandas as pd

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



def load_recent_data_from_gcs(bucket_name: str, days: int = 7) -> pd.DataFrame:
    """
    Connects to GCS, finds all JSONs from the last 'days', 
    and returns them as a Pandas DataFrame.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Calculate the cutoff time (e.g., 7 days ago)
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # List all files in the metadata folder
        blobs = bucket.list_blobs(prefix="data_collection/metadata/")
        
        data_list = []
        
        for blob in blobs:
            # Check if the file is new enough
            # Note: timezone handling can be tricky, we simplify here
            if blob.time_created.replace(tzinfo=None) > cutoff_time:
                # Download and parse the JSON
                content = blob.download_as_text()
                record = json.loads(content)
                
                # Extract the prediction values
                row = {
                    "total_calories": record["prediction"]["total_calories"],
                    "total_fat": record["prediction"]["total_fat"],
                    "total_carb": record["prediction"]["total_carb"],
                    "total_protein": record["prediction"]["total_protein"],
                    "timestamp": record["timestamp"]
                }
                data_list.append(row)

        if not data_list:
            return pd.DataFrame()
            
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"Error loading data from GCS: {e}")
        return pd.DataFrame()