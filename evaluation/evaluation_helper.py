import sys
sys.path.insert(0, '../')
from slide.slide_container import SlideContainer
import h5py
from torchvision import transforms
from fastai.vision import *
import cv2
from sklearn.metrics import jaccard_score, confusion_matrix


def wsi_inference(slide_container, device, patch_size_s, patch_size_c,level_s,level_c, batch_size, segmentation_learner, classification_learner):
    factor_s = slide_container.slide.level_downsamples[level_s]
    factor_c = slide_container.slide.level_downsamples[level_c]
    shape_s = slide_container.slide.level_dimensions[level_s]
    shape_c = slide_container.slide.level_dimensions[level_c]
    segmentation_store = h5py.File("{}_segmentation.hdf5".format(slide_container.file.stem), "w")
    classification_store = h5py.File("{}_classification.hdf5".format(slide_container.file.stem),"w")
    segmentation_results = segmentation_store.create_dataset("results",(shape_s[1], shape_s[0]), compression="gzip")
    classification_results = classification_store.create_dataset("results", (int(np.round(shape_c[1]//patch_size_c)), int(np.round(shape_c[0]//patch_size_c))),compression="gzip")

    x_length = int((shape_s[0] // patch_size_s) + 1)
    y_length = int((shape_s[1] // patch_size_s) + 1)
    columns, rows = np.indices((x_length, y_length)).reshape(2, -1) * patch_size_s
    tumor_class = segmentation_learner.data.classes.index('Tumor')

    with torch.no_grad():
        # segmentation inference
        segmentation_learner.model.eval()
        indices = np.vstack((columns, rows)).transpose()
        index_loader = DataLoader(indices, batch_size=batch_size)
        for index in index_loader:
            columns, rows = index[:, 0].numpy(), index[:, 1].numpy()
            batch = torch.stack([transforms.Normalize(*segmentation_learner.data.stats)(
                pil2tensor(slide_container.get_patch(column, row) / 255., np.float32)) for column, row in
                                 zip(columns, rows)])
            seg_preds = segmentation_learner.model(batch.to(device=device))
            seg_preds = seg_preds.argmax(dim=1)[None]
            seg_preds = seg_preds.cpu().numpy().squeeze(0)
            for column, row, seg_pred in zip(columns,rows,seg_preds):
                stop_y_s = int(shape_s[1] % patch_size_s)
                stop_x_s = int(shape_s[0] % patch_size_s)
                if row // patch_size_s != y_length - 1:
                    stop_y_s = patch_size_s
                if column // patch_size_s != x_length - 1:
                    stop_x_s = patch_size_s
                segmentation_results[int(row):int(row + stop_y_s), int(column):int(column + stop_x_s)] = seg_pred[:stop_y_s,:stop_x_s]
                # classification inference
                classification_learner.model.eval()
                stop_y_c = int((patch_size_s*factor_s)//(patch_size_c*factor_c))
                stop_x_c = int((patch_size_s*factor_s)//(patch_size_c*factor_c))
                if column // (patch_size_s) == int(shape_s[0] // (patch_size_s)):
                    stop_x_c = int(((shape_s[0] % (patch_size_s))*factor_s)//(patch_size_c*factor_c))
                if row // (patch_size_s) == int(shape_s[1] // (patch_size_s)):
                    stop_y_c = int(((shape_s[1] % (patch_size_s))*factor_s)//(patch_size_c*factor_c))
                idxs = np.mgrid[0:stop_x_c,0:stop_y_c:].reshape(2,-1).T * int((patch_size_c*factor_c) // factor_s)
                seg_outputs = [seg_pred[i[1]:i[1] + int((patch_size_c*factor_c) // factor_s),
                                i[0]:i[0] + int((patch_size_c*factor_c) // factor_s)] for i in idxs]
                # only pass patches to classification network, if more than 90% was segmented as tumor
                idxs = [idxs[i] for i in range(idxs.shape[0]) if
                        seg_outputs[i].size == int((patch_size_c * factor_c) // factor_s) * int(
                            (patch_size_c * factor_c) // factor_s) and (
                                    np.count_nonzero(seg_outputs[i] == tumor_class) / seg_outputs[i].size) > 0.9]
                classification_loader = DataLoader(idxs, batch_size)
                for i in classification_loader:
                    cs,rs = i[:, 0].numpy(), i[:, 1].numpy()
                    classification_batch = [pil2tensor(np.array(slide_container.slide.read_region(location=(
                    int((column + c) * factor_s), int((row + r) * factor_s)),level=level_c, size=(
                        patch_size_c, patch_size_c)))[:, :, :3] / 255., np.float32) for c, r in zip(cs, rs)]
                    class_batch = torch.stack([transforms.Normalize(*classification_learner.data.stats)(c) for c in classification_batch])
                    class_pred = torch.softmax(classification_learner.model(class_batch.to(device=device)),1)
                    classifications = (class_pred).argmax(dim=1)
                    for c,r,cl in zip(cs,rs,classifications):
                        classification_results[int(round(((r + row)*factor_s)/factor_c/patch_size_c)),int(round(((c + column)*factor_s)/factor_c/patch_size_c))] = cl.cpu() + 1
    segmentation_store.close()
    classification_store.close()

def jaccard_score(cm_matrix, labels):
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


def slide_cm_matrix(slide_container, classes):
    try:
        prediction = h5py.File("{}_segmentation.hdf5".format(slide_container.file.stem), "r")
    except:
        print("Please run slide inference first to generate a segmentation output")
        return
    gt = -1*np.ones(shape=(slide_container.slide_shape[1], slide_container.slide_shape[0]), dtype=np.int8)
    inv_map = {v: k for k, v in slide_container.tissue_classes.items()}
    for poly in slide_container.polygons:
        coordinates = np.array(poly['segmentation']).reshape((-1,2))/ slide_container.down_factor
        label = slide_container.label_dict[inv_map[poly["category_id"]]]
        cv2.drawContours(gt, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)
    white_mask = cv2.cvtColor(np.array(slide_container.slide.read_region(location=(0, 0), level=slide_container._level,
                                                                         size=(slide_container.slide_shape[0],
                                                                               slide_container.slide_shape[1])))[:, :,
                              :3], cv2.COLOR_RGB2GRAY) > slide_container.white
    excluded = (gt == -1)
    gt[np.logical_and(white_mask, excluded)] = 0
    cm = confusion_matrix(gt.flatten(),np.array(prediction["results"]).flatten(),labels=range(classes))
    return cm

def slide_accuracy(slide_container, labels):
    try:
        prediction = h5py.File("{}_classification.hdf5".format(slide_container.file.stem), "r")
    except:
        print("Please run slide inference first to generate a classifiation output")
        return
    tumors, counts = np.unique(prediction["results"], return_counts=True)
    counts = counts[tumors != 0]
    tumors = tumors[tumors != 0]
    counts = np.round(counts / np.sum(counts), 4)
    print("Classification Accuracies: ", dict(zip(np.array(labels)[np.array(tumors-1, dtype=int)], counts)))
    print("Slide Classification: ", np.array(labels)[int(tumors[np.argmax(counts)] - 1)] )




