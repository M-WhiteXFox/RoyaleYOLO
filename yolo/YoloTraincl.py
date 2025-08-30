import os
import subprocess
import labelme

json_dir = "datasets/label/train"
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        subprocess.run(["labelme_json_to_dataset", os.path.join(json_dir, file)])
