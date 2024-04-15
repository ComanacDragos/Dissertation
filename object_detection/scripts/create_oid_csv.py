import os

import pandas as pd
from pathlib import Path

data = {
    "image": [],
    "label": [],
    "stage": [],
}

root = Path(r"C:\Users\Dragos\datasets\OID\OID\PublicTransportFilteredProcessed")

for stage in ["train", "val", "test", "test_filtered"]:
    for image in os.listdir(root / stage):
        if 'jpg' not in image:
            continue
        data['image'].append(f"{stage}/{image}")
        data['stage'].append(stage)
        label = image.split(".")[0] + ".txt"
        data['label'].append(f"{stage}/label/{label}")


pd.DataFrame(data).to_csv("PublicTransportFilteredProcessed.csv", index=False)
