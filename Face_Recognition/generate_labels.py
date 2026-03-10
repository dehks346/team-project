import os 
import json

dataset_path = 'dataset'

label_map = {}

people = os.listdir(dataset_path)

for i, person in enumerate(people):
    label_map[i] = person
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("Label map generated and saved to label_map.json:")
print(json.dumps(label_map, indent=4))