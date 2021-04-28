from fastai.vision import *
from fastai.data_block import *
from fastai.vision.data import SegmentationProcessor
from slide.slide_container import SlideContainer

PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'

class SlideLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]
                slide_container_y = self.y.items[idxs]

                xmin, ymin = slide_container.get_new_train_coordinates()

                x = self.x.get(idxs, xmin, ymin)
                try:
                    y = self.y.get(idxs, xmin, ymin)
                except:
                    y = self.y.get(idxs)
            else:
                x, y = self.item, -1
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = -1
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])

class SlideItemList(ItemList):

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        self._label_list,self._split = SlideLabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self,idxs: int, x: int=0, y: int=0)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)

class SlideImageItemList(SlideItemList):
    pass


class SlideSegmentationItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        patch = fn.get_patch(x, y) / 255.

        return Image(pil2tensor(patch, np.float32))


class SlideSegmentationLabelList(ImageList, SlideImageItemList):
    "`ItemList` for segmentation masks."
    _processor=SegmentationProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes = classes


    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        patch = fn.get_y_patch(x, y)
        return ImageSegment(pil2tensor(patch, np.float32))

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]
    
    def reconstruct(self, t:Tensor):
        return ImageSegment(t)