import os
from datetime import datetime
import pandas as pd

LOG_FILE = "logs.csv"


def save_result(image_path, result):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "status": result.get("status"),
        "predicted_class": result.get("predicted_class"),
        "confidence": result.get("confidence"),
        "risk": result.get("risk"),
        "alert": result.get("alert"),
        "model_name": result.get("model_name"),
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            df_old = pd.read_csv(LOG_FILE)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new

    df.to_csv(LOG_FILE, index=False)