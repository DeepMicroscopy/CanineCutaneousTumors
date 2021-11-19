import sys
sys.path.insert(0, '../')
from torchvision import transforms
from fastai.vision import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.nn.functional import fold

def segmentation_inference(slide,store, patch_size, level, batch_size, learner, overlap_factor, indices = None):
    shape = slide.level_dimensions[level]
    classification_indices = torch.zeros((0,2)).to(learner.data.device)
    if indices != None:
        x_indices = indices[:,0]
        y_indices = indices[:,1]
    else:
        x_indices = np.arange(0,int((shape[0] // (patch_size*overlap_factor)) + 1))* int(patch_size * overlap_factor)
        y_indices = np.arange(0,int((shape[1] // (patch_size*overlap_factor)) + 1))* int(patch_size * overlap_factor)
    segmentation_results = store.create_dataset("segmentation", (shape[1], shape[0]), compression="gzip")
    temp = torch.zeros(learner.data.c, int(2*patch_size-overlap_factor*patch_size), shape[0]).to(learner.data.device)
    with torch.no_grad():
        # segmentation inference
        learner.model.eval()
        for y in tqdm(y_indices,desc='Processing %s' % Path(slide._filename).stem):
            x_loader = DataLoader(x_indices, batch_size=batch_size)
            row_temp = []
            for xs in x_loader:
                input_batch = torch.stack([transforms.Normalize(*learner.data.stats)(pil2tensor(np.array(
                    slide.read_region(location=(int(x * slide.level_downsamples[level]),
                                                int(y * slide.level_downsamples[level])),
                                      level=level, size=(patch_size, patch_size)))[:, :, :3] / 255., np.float32)) for x
                                           in xs])
                seg_pred = torch.softmax(learner.model(input_batch.to(device=learner.data.device)),dim=1)
                row_temp += [s.view(s.shape[0],s.shape[1]*s.shape[2],1) for s in seg_pred]
            row_output = fold(torch.cat(row_temp, dim=2), (patch_size,int((len(x_indices) + 1) * patch_size * overlap_factor)),
                 kernel_size=(patch_size, patch_size), stride=int(patch_size * overlap_factor)).squeeze(1)[:,:,:shape[0]]
            temp[:, int(patch_size * overlap_factor):, :] += row_output
            temp = temp.roll(-int(patch_size * overlap_factor),dims=1)
            temp[:, -int(patch_size * overlap_factor):, :] = 0
            width = segmentation_results[y:int(y + patch_size * overlap_factor),:].shape[0]
            for x in range(0, int(shape[0] // patch_size + 1)):
                height = segmentation_results[:, int(x * patch_size):int((x + 1) * patch_size)].shape[1]
                segmentation_results[
                y:int(y + patch_size * overlap_factor),
                int(x * patch_size):int((x + 1) * patch_size)] = temp[:,:width,int(x*patch_size):int(x*patch_size) + height].argmax(dim=0).cpu()
                classification_indices = torch.cat((classification_indices, (torch.nonzero(
                    temp[:, :width, int(x * patch_size):int(x * patch_size) + height].argmax(
                        dim=0) == learner.data.classes.index("Tumor"))+torch.Tensor([y,x*patch_size]).to(learner.data.device))*slide.level_downsamples[level]), dim=0)
            torch.cuda.empty_cache()
    return classification_indices


def classification_inference(slide,store, patch_size,level,batch_size, learner, indices = None):
    shape = slide.level_dimensions[level]
    if indices != None:
        indices = torch.unique(indices//slide.level_downsamples[level]//patch_size, dim=0).cpu().flip(dims=[1])*patch_size
    else:
        indices = np.indices((int(shape[0] // patch_size),int(shape[1]//patch_size))).reshape(2,-1).T*patch_size
    classification_results = store.create_dataset("classification", (int(shape[1] // patch_size) , int(shape[0] // patch_size)), compression="gzip")

    with torch.no_grad():
        index_loader = DataLoader(indices, batch_size=batch_size)
        # classification inference
        learner.model.eval()
        for ind in tqdm(index_loader,desc='Processing %s' % Path(slide._filename).stem):
            input_batch = torch.stack([transforms.Normalize(*learner.data.stats)(pil2tensor(np.array(
                slide.read_region(location=(int(i[0] * slide.level_downsamples[level]),
                                            int(i[1] * slide.level_downsamples[level])),
                                  level=level, size=(patch_size, patch_size)))[:, :, :3] / 255., np.float32)) for i
                                       in ind])
            clas_pred = learner.model(input_batch.to(device=learner.data.device))
            if is_tuple(clas_pred):
                clas_pred = clas_pred[-1]
            clas_pred = torch.softmax(clas_pred,dim=1)
            for j, i in enumerate(ind):
                try:
                    classification_results[int(i[1]//patch_size), int(i[0]//patch_size)] = torch.argmax(clas_pred, dim=1)[j].cpu()
                except:
                    continue

def segmentation_cm_matrix(slide_container, prediction, classes):
    cm = np.zeros((classes,classes))
    x_length = np.arange(int((slide_container.slide_shape[0] // slide_container.width) + 1))*slide_container.width
    y_length = np.arange(int((slide_container.slide_shape[1] // slide_container.height) + 1))*slide_container.height
    for x in tqdm(x_length):
        for y in y_length:
            pred = prediction["segmentation"][y:y+slide_container.height,x:x+slide_container.width]
            gt = slide_container.get_y_patch(x, y)[:pred.shape[0],:pred.shape[1]]
            cm += confusion_matrix(gt.flatten(), pred.flatten(), labels=range(classes))
    return cm

def classification_cm_matrix(slide_container, prediction, classes):
    cm = np.zeros((classes,classes))
    x_length = int(slide_container.slide_shape[0] // slide_container.width)
    y_length = int(slide_container.slide_shape[1] // slide_container.height)
    for x in tqdm(range(0,x_length)):
        for y in range(0,y_length):
            pred = prediction["classification"][y,x]
            gt = slide_container.get_y_patch(int(x*slide_container.width), int(y*slide_container.height))
            gt =  np.unique(gt)[np.argmax(np.unique(gt, return_counts=True)[1])]
            if (pred != -1 and gt != -1):
                cm[int(gt),int(pred)] += 1
    return cm

def slide_jaccard_score(cm_matrix, labels):
    ious = np.zeros((len(labels)))
    ious[:] = np.NAN
    total = cm_matrix.sum()
    tp = np.diagonal(cm_matrix)
    posPred = cm_matrix.sum(axis=0)
    posGt = cm_matrix.sum(axis=1)

    # Check which classes have elements
    valid = posGt > 0
    iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

    # Compute per-class results
    ious[iousValid] = np.divide(tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid])
    freqs = np.divide(posGt, total)

    # Compute evaluation metrics
    miou = np.mean(ious[iousValid])
    fwiou = np.sum(np.multiply(ious[iousValid], freqs[iousValid]))

    print("IoUs: ", dict(zip(np.array(labels)[iousValid], np.round(ious[iousValid],4))), "Mean: ", miou)
    print("Frequency-Weighted IoU: ", np.round(fwiou,4))

def slide_tumor_recall(cm_matrix, labels):
    slide_label = np.argmax(np.sum(cm_matrix[1:,:], axis=1)) + 1
    print("Slide Label: ", labels[slide_label])
    print("Slide Classification: ", labels[np.argmax(cm_matrix[slide_label,1:]) + 1])
    print("Tumor Recall: ", np.round(cm_matrix[slide_label, slide_label] / np.sum(cm_matrix[1:, 1:]), 4))






