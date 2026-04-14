import os
import json
from pathlib import Path

dataset_path = Path('dataset')

label_map = {}

people = sorted(
    [path.name for path in dataset_path.iterdir() if path.is_dir()],
    key=str.lower,
)

for i, person in enumerate(people):
    label_map[i] = person
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("Label map generated and saved to label_map.json:")
print(json.dumps(label_map, indent=4))