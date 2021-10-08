from fastai.vision import *
from fastai.callbacks import TrackerCallback
from fastai.callbacks.mixup import MixUpLoss

class UpdateProbabilitiesCallback(TrackerCallback):
    def __init__(self, learn:Learner, trainslides):
        self.iou_dict = dict.fromkeys(["background_iou" , "dermis_iou", "epidermis_iou", "subcutis_iou", "infl_nec_iou", "tumor_iou"],0)
        self.tissue_to_iou = {0: "background_iou", 3: "dermis_iou", 4: "epidermis_iou", 5: "subcutis_iou", 6:"infl_nec_iou",
                              7: "tumor_iou", 8: "tumor_iou",9: "tumor_iou",10: "tumor_iou",11: "tumor_iou",12: "tumor_iou",13: "tumor_iou"}
        self.trainslides = trainslides
        super().__init__(learn)

    def on_epoch_end(self, epoch, **kwargs:Any):
        for iou in (self.iou_dict.keys()):
            position = self.learn.recorder.metrics_names.index(iou)
            value = self.learn.recorder.metrics[0][position]
            self.iou_dict[iou] = 1 - float(value)
        for slide in self.trainslides:
            slide.probabilities.update((k, self.iou_dict[self.tissue_to_iou[k]]) for k in set(self.tissue_to_iou).intersection(slide.probabilities))