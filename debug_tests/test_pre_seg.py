import os
import sys
import numpy as np
from PIL import Image

# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # debug_tests 目录
project_root = os.path.dirname(current_dir)  # WBS 目录
sys.path.append(project_root)

from image_processings.image_pre_seg import image_i_segment, change_image_type


image_path = "./GrTh/dataset_v0/images/11804ccni4tvlhqgtkuh15er8d.png"
pil_img = Image.open(image_path)
image = np.array(pil_img)

# 创建 image_i_segment 类的实例
A = image_i_segment(
    name=None,
    label=None,
    image=image,
    new_size_of_image=512,
    num_node_for_graph=100,
    compactness_in_SLIC=12,
    sigma_in_SLIC=1.0,
    min_size_factor_in_SLIC=0.6,
    max_size_factor_in_SLIC=1.2
)

print("图像分割完成!")
print(f"分割后的图像尺寸: {A.image_resized_padding.shape}")
print(f"分割数量: {A.num_of_different_segments_in_SLIC}")