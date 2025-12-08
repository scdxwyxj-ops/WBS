import numpy as np
import cv2

def preprocess_and_extract_largest_component(mask, kernel_size=5):
    """
    先腐蚀，再提取二值mask中最大连通分支，最后进行膨胀操作。
    参数:
        mask: numpy array, 二值化后的mask (0和1组成，或bool)
        kernel_size: int, 形态学操作的卷积核大小
    返回:
        final_mask: numpy array, 经过腐蚀、提取最大连通分支、膨胀后的mask (0和1组成)
    """
    # 确保mask是uint8类型，且值为0或255
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() == 1:
        mask = mask * 255
    else:
        mask = mask.astype(np.uint8)

    # 创建形态学操作的核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 步骤 1: 腐蚀操作
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # 步骤 3: 膨胀操作
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

    # 步骤 2: 提取最大连通分支
    binary_mask = (dilated_mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)

    # 如果没有连通域，返回空mask
    if num_labels <= 1:
        return np.zeros_like(mask // 255, dtype=np.uint8)

    # 计算每个连通域的面积
    areas = [np.sum(labels == i) for i in range(num_labels)]
    max_label = np.argmax(areas[1:]) + 1

    # 提取最大连通分支，并将其转换为 0 和 1
    final_mask = (labels == max_label).astype(np.uint8)

    return final_mask