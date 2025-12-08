import numpy as np

def calculate_miou(mask_list1, mask_list2):
    """
 `   Calculate the mean Intersection over Union (mIoU) for two lists of masks.

    Parameters:
        mask_list1 (list of numpy.ndarray): The first list of binary masks.
        mask_list2 (list of numpy.ndarray): The second list of binary masks.

    Returns:
        float: The mean IoU value.
    """
    assert len(mask_list1) == len(mask_list2), "The two mask lists must have the same length."
    
    iou_list = []
    
    for mask1, mask2 in zip(mask_list1, mask_list2):
        # Compute intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        # Avoid division by zero
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_list.append(iou)
    
    # Calculate mean IoU
    miou = np.mean(iou_list)
    return miou, iou_list



import numpy as np

def calculate_map(mask_list_pred, mask_list_gt, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculate mean Average Precision (mAP) over multiple IoU thresholds for binary segmentation masks.

    Parameters:
        mask_list_pred (list of numpy.ndarray): List of predicted binary masks.
        mask_list_gt (list of numpy.ndarray): List of ground truth binary masks.
        thresholds (iterable of float): IoU thresholds to compute AP at (default: 0.5 to 0.95 with step 0.05)

    Returns:
        float: mean Average Precision (mAP)
        list of float: AP at each IoU threshold
    """
    assert len(mask_list_pred) == len(mask_list_gt), "Mask list lengths do not match."

    ap_list = []

    for t in thresholds:
        correct = 0
        total = len(mask_list_pred)

        for pred, gt in zip(mask_list_pred, mask_list_gt):
            # Ensure masks are binary
            pred = (pred > 0).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)

            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()

            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union

            if iou >= t:
                correct += 1

        ap = correct / total
        ap_list.append(ap)

    map_value = np.mean(ap_list)
    return map_value, ap_list



def calculate_dice(mask_list1, mask_list2):
    dice_list = []
    for m1, m2 in zip(mask_list1, mask_list2):
        m1 = (m1 > 0).astype(np.uint8)
        m2 = (m2 > 0).astype(np.uint8)
        intersection = np.logical_and(m1, m2).sum()
        total = m1.sum() + m2.sum()
        if total == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / total
        dice_list.append(dice)
    return np.mean(dice_list), dice_list