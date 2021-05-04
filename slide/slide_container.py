import openslide
import cv2
from fastai.vision import *
from shapely import geometry

class SlideContainer:

    def __init__(self, file: Path,
                 annotation_file,
                 level: int = 0,
                 width: int = 256, height: int = 256,
                 sample_func = None,dataset_type=None, label_dict=None):
        self.file = file
        with open(annotation_file) as f:
            data = json.load(f)
            self.tissue_classes = dict(zip([cat["name"] for cat in data["categories"]],[cat["id"] for cat in data["categories"]]))
            image_id = [i["id"] for i in data["images"] if i["file_name"] == file.name][0]
            polygons = [anno for anno in data['annotations'] if anno["image_id"] == image_id]
        self.polygons = sorted(polygons, key=lambda p: p['hierarchy'])
        if dataset_type == "classification":
            self.polygons = [poly for poly in self.polygons if poly["category_id"] >= 7]
        self.labels = list(set([poly["category_id"] for poly in self.polygons]))
        self.slide = openslide.open_slide(str(file))
        thumbnail = cv2.cvtColor(
            np.array(self.slide.read_region((0, 0), self.slide.level_count - 1, self.slide.level_dimensions[-1]))[:, :,
            :3], cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(thumbnail,(5,5),0)
        self.white,_ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]

        if level is None:
            level = self.slide.level_count - 1
        self._level = level
        self.sample_func = sample_func
        self.dataset_type = dataset_type
        self.label_dict = label_dict


    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_new_level(self):
        return self._level

    def get_patch(self, x: int = 0, y: int = 0):
        rgb = np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.width, self.height)))[:, :, :3]
        return rgb

    def get_y_patch(self, x: int = 0, y: int = 0):
        y_patch = -1*np.ones(shape=(self.height, self.width), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}

        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ self.down_factor
            coordinates = coordinates - (x, y)
            label = self.label_dict[inv_map[poly["category_id"]]]
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        if self.dataset_type == 'segmentation':
            white_mask = cv2.cvtColor(self.get_patch(x,y),cv2.COLOR_RGB2GRAY) > self.white
            excluded = (y_patch == -1)
            y_patch[np.logical_and(white_mask, excluded)] = 0
        return y_patch

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.polygons, **{"classes":self.labels ,"size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self.level})
        # default sampling method
        xmin, ymin = 0,0
        found = False
        while not found:
            iter = 0
            label = random.choice(self.labels)
            polygon = random.choice([poly for poly in self.polygons if poly["category_id"] == label])
            coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
            minx, miny, xrange, yrange = polygon["bbox"]
            while iter < 25 and not found:
                iter += 1
                pnt = geometry.Point(random.uniform(minx, minx + xrange), random.uniform(miny, miny + yrange))
                if geometry.Polygon(coordinates).contains(pnt):
                    xmin = pnt.x // self.down_factor - self.width / 2
                    ymin = pnt.y // self.down_factor - self.height / 2
                    found = True
        return xmin, ymin


    def __str__(self):
        return str(self.path)
