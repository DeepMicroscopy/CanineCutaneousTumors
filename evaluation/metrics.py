from fastai.vision import *
from sklearn.metrics import jaccard_score

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None)))

def background_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[0])))

def dermis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[1])))

def epidermis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[2])))

def subcutis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[3])))

def infl_nec_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[4])))

def tumor_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return tensor(np.mean(jaccard_score(to_np(outputs_max.view(-1)),to_np(labels_squeezed.view(-1)), average=None, labels=[5])))
