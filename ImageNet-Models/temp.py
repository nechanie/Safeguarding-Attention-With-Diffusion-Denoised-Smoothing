import json
import os

dir_path = "/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ImageNet-Models/heyo/"

class_map = json.load(open('imagenet_class_index.json', 'r'))

print(class_map[str(0)][0])

for idx in class_map.keys():
    print(idx)
    id_name = class_map[str(idx)][0]

    new_folder = dir_path + id_name
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    break