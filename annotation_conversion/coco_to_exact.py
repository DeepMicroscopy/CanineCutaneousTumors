import sys
sys.path.append("../../SlideRunner/")
import numpy as np
import json
from tqdm import tqdm
import pandas as pd



def convert(annotation_path):
    rows = []
    with open(annotation_path) as f:
        data = json.load(f)
        categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        for row in data["images"]:
            file_name = row["file_name"]
            image_id = row["id"]
            width = row["width"]
            height = row["height"]
            for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
                polygon = annotation["segmentation"]
                cat = categories[annotation["category_id"]]
                rows.append([file_name, image_id, width, height, polygon, cat])

    df = pd.DataFrame(rows, columns=["file_name", "image_id", "width", "height", "polygon", "cat"])
    result = []
    for slide in tqdm(df["file_name"].unique()):
        imageDf = df[df["file_name"] == slide]
        for label, vector in zip(imageDf['cat'], imageDf['polygon']):
            result_dict = {}
            index = 1
            vector = np.array(vector).reshape((-1,2))
            for x, y in vector:
                result_dict['x{}'.format(index)] = int(x)
                result_dict['y{}'.format(index)] = int(y)
                index += 1
            poly_dict = json.dumps(result_dict)
            row = "{0}|{1}|".format(slide, label) + poly_dict + "\n"
            result.append(row)
    if len(result) > 0:
        with open('CATCH.txt', 'w') as f:
            f.writelines(result)


if __name__ == '__main__':
    # Define annotation path
    annotation_file = "CATCH.json"

    # Conversion
    convert(annotation_file)






