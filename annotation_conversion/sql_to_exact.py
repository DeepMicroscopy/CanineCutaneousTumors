import sys
sys.path.append("../../SlideRunner/")
from pathlib import Path
import json
from SlideRunner.dataAccess.database import Database
from tqdm import tqdm



def convert(annotation_path):
    database = Database()
    database.open(annotation_path)
    result = []
    classes = database.getAllClasses()
    classes = {id:name for name, id, colour in  classes}


    getslides = """SELECT uid, filename FROM Slides"""
    for currslide, filename in tqdm(database.execute(getslides).fetchall()):
        database.loadIntoMemory(currslide)

        for id, annotation in database.annotations.items():
            if len(annotation.labels) != 0 and annotation.deleted != 1:
                if annotation.annotationType == 3:
                    duplicates = {}
                    result_dict = {}
                    index = 1
                    for x, y in annotation.coordinates:

                        result_dict['x{}'.format(index)] = int(x)
                        result_dict['y{}'.format(index)] = int(y)
                        index += 1

                    poly_dict = json.dumps(result_dict)
                    label = classes[annotation.labels[0].classId]

                    row = "{0}|{1}|".format(filename, label) + poly_dict + "\n"
                    result.append(row)

    if len(result) > 0:
        with open('canine_cutaneous_tumors.txt', 'w') as f:
            f.writelines(result)

if __name__ == '__main__':
    # Define annotation path
    annotation_file = "canine_cutaneous_tumors.sqlite"

    # Conversion
    convert(annotation_file)






