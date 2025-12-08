import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image

def visualize_with_seg_id(image, seg_id, segment, save_path = None):
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for idx in seg_id:
        mask += (segment == idx).astype(np.uint8)
    visualize_by_mask(image, mask, save_path = save_path)


def _to_numpy_image_array(image):
    """Convert common image representations to an HxWxC numpy array."""
    if isinstance(image, np.ndarray):
        return image

    try:
        import torch
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))
            return array
    except ImportError:
        pass

    if isinstance(image, Image.Image):
        return np.array(image)

    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_numpy_mask_array(mask):
    """Convert segmentation labels or masks to a 2D numpy array."""
    if isinstance(mask, np.ndarray):
        return mask

    try:
        import torch
        if isinstance(mask, torch.Tensor):
            return mask.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(mask, Image.Image):
        return np.array(mask)

    raise TypeError(f"Unsupported mask type: {type(mask)}")


def _compute_segment_centroids(segments):
    centroids = {}
    unique_labels = np.unique(segments)
    for label in unique_labels:
        coords = np.column_stack(np.nonzero(segments == label))
        if coords.size == 0:
            continue
        y, x = coords.mean(axis=0)
        centroids[int(label)] = (float(y), float(x))
    return centroids


def visualize_by_mask(image, mask, color=(0, 0, 255), save_path = None, need_show = True):
## 输入image和mask，输出带有掩膜的图像

    # 检查图像和掩膜的形状是否匹配
    if image.shape[:2] != mask.shape:
        raise ValueError("图像和掩膜的尺寸不匹配")
    
    # 创建一个与图像相同大小的副本
    image_with_mask = image.copy()
    
    # 将掩膜区域替换为指定的颜色
    for c in range(3):  # 对每个颜色通道应用掩膜
        image_with_mask[:, :, c] = np.where(mask == 1, color[c], image[:, :, c])
    
    # 显示图像和掩膜
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_mask)
    plt.axis('off')  # 不显示坐标轴
    if need_show:
        plt.show()

    # 保存图像
    if save_path is not None:
        # 如果路径是图片文件（如 PNG、JPEG），使用 PIL 保存
        save_image = Image.fromarray(image_with_mask.astype(np.uint8))
        save_image.save(save_path)
        print(f"图像已保存到 {save_path}")


def show_original_and_slic(
    image,
    segments,
    save_path=None,
    need_show=True,
    titles=("Original", "SLIC Boundaries"),
    show_indices=True,
    index_text_kwargs=None,
):
    """Display the original image and its SLIC boundary visualisation side by side."""
    image_np = _to_numpy_image_array(image)
    segments_np = _to_numpy_mask_array(segments)

    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.ndim != 3:
        raise ValueError(f"`image` must be HxWxC after conversion, got shape {image_np.shape}")

    if segments_np.ndim == 3 and segments_np.shape[-1] == 1:
        segments_np = segments_np[..., 0]
    if segments_np.ndim != 2:
        raise ValueError(f"`segments` must be HxW after conversion, got shape {segments_np.shape}")

    if segments_np.shape != image_np.shape[:2]:
        segments_np = cv2.resize(
            segments_np.astype(np.int32),
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    boundary_img = mark_boundaries(img_as_float(image_np), segments_np)

    label_min = int(np.min(segments_np))
    label_max = int(np.max(segments_np))
    num_labels = len(np.unique(segments_np))
    print(
        f"SLIC 标签统计 -> min: {label_min}, max: {label_max}, unique_count: {num_labels}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(boundary_img)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    if show_indices:
        centroids = _compute_segment_centroids(segments_np)
        text_kwargs = {
            "fontsize": 8,
            "color": "yellow",
            "ha": "center",
            "va": "center",
            "bbox": {
                "facecolor": "black",
                "alpha": 0.5,
                "pad": 1.5,
                "edgecolor": "none",
            },
        }
        if index_text_kwargs:
            text_kwargs.update(index_text_kwargs)

        for label, (y, x) in centroids.items():
            axes[1].text(x, y, str(label), **text_kwargs)

    plt.tight_layout()
    if need_show:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if not need_show:
        plt.close(fig)

    return fig, axes, {
        "label_min": label_min,
        "label_max": label_max,
        "unique_count": num_labels,
    }


def visualize_box(image, box, save_path = None, need_show = True):
##输入box和image，可视化

    # 解包box坐标 (left_bottom_x, left_bottom_y) 和 (right_top_x, right_top_y)
    left_bottom_x, left_bottom_y, right_top_x, right_top_y = box
    
    # 在图像上绘制边界框，注意OpenCV的矩形是 (左上角, 右下角)
    # OpenCV需要以 (x, y) 形式输入
    image_with_box = image.copy()
    
    # 绘制矩形 (left_bottom_x, left_bottom_y) 是左下角，(right_top_x, right_top_y) 是右上角
    cv2.rectangle(image_with_box, (left_bottom_x, left_bottom_y), (right_top_x, right_top_y), (0, 255, 0), 2)
    
    
    # 显示图像
    plt.imshow(image_with_box)
    plt.axis('off')  # 不显示坐标轴
    if need_show:
        plt.show()
    if save_path is not None:
        # 如果路径是图片文件（如 PNG、JPEG），使用 PIL 保存
        save_image = Image.fromarray(image_with_box.astype(np.uint8))
        save_image.save(save_path)
        print(f"图像已保存到 {save_path}")


def get_bounding_box(mask):
## 输入mask，输出box的坐标
    # 获取非零像素的坐标
    non_zero_coords = np.argwhere(mask > 0)
    
    # 如果没有非零像素，返回None
    if non_zero_coords.size == 0:
        return None
    
    # 计算左上角和右下角
    top_left = non_zero_coords.min(axis=0)  # 最小的y和x
    bottom_right = non_zero_coords.max(axis=0)  # 最大的y和x
    
    return [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]


def get_mask_by_label(picture, label, segment, seg_node_list):
##在图像的聚类中，输入label，返回list中全部label相同的mask

    nodes_id = [seg.i for seg in seg_node_list if seg.cluster_label == label]
    mask = np.zeros(picture.shape[:-1])

    for id in nodes_id:
        mask += (segment == id)
    return mask


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 


def show_slic_and_points(image, seg_by_slic, point_list, point_label_list):
    # Convert the segmentation boundaries to an overlay on the original image
    boundary_img = mark_boundaries(image, seg_by_slic)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the image with boundaries
    ax.imshow(boundary_img)

    # Overlay the points on the same axis
    show_points(point_list, point_label_list, ax)

    # Display the combined plot
    plt.show()


def show_combined_plots(image, seg_by_slic, point_list, point_label_list, mask, color=(0, 0, 255), save_path=None, need_show=True):
    """
    在一个图中显示:
    - 左子图: SLIC 分割结果及标注点
    - 右子图: 带掩膜的图像
    """
    # 检查颜色参数
    if not (isinstance(color, (tuple, list)) and len(color) == 3 and all(isinstance(c, int) for c in color)):
        raise ValueError("`color` 参数必须是一个长度为 3 的整数元组，例如 (0, 0, 255)")
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左子图: SLIC 分割结果及标注点
    print(f"调试: SLIC分割结果形状: {seg_by_slic.shape}")
    print(f"调试: SLIC分割结果类型: {seg_by_slic.dtype}")
    print(f"调试: SLIC分割结果范围: {seg_by_slic.min()} - {seg_by_slic.max()}")
    print(f"调试: SLIC分割结果唯一值数量: {len(np.unique(seg_by_slic))}")
    
    try:
        boundary_img = mark_boundaries(image, seg_by_slic)
        print(f"调试: 边界图像形状: {boundary_img.shape}")
        print(f"调试: 边界图像类型: {boundary_img.dtype}")
        print(f"调试: 边界图像范围: {boundary_img.min()} - {boundary_img.max()}")
        axes[0].imshow(boundary_img)
    except Exception as e:
        print(f"警告: mark_boundaries失败: {e}")
        # 如果mark_boundaries失败，直接显示原图
        axes[0].imshow(image)
        print("使用原图替代SLIC边界图")
    
    axes[0].set_title("SLIC and Points")
    axes[0].axis("off")

    # 绘制标注点
    if len(point_list) > 0:
        # 确保point_list是numpy数组
        if isinstance(point_list, np.ndarray):
            if len(point_list.shape) == 2 and point_list.shape[1] == 2:
                # 格式: [[x1, y1], [x2, y2], ...]
                for i, (x, y) in enumerate(point_list):
                    # 检查坐标是否在图像边界内
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        point_color = "green" if point_label_list[i] == 1 else "red"
                        axes[0].scatter(x, y, color=point_color, s=50, edgecolor="white", alpha=0.8)
                    else:
                        print(f"警告: 提示点坐标超出图像边界: ({x}, {y}), 图像尺寸: {image.shape[1]}x{image.shape[0]}")
            else:
                print(f"警告: point_list格式异常: shape={point_list.shape}")
        else:
            # 如果是列表格式
            for i, point in enumerate(point_list):
                if len(point) == 2:
                    x, y = point
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        point_color = "green" if point_label_list[i] == 1 else "red"
                        axes[0].scatter(x, y, color=point_color, s=50, edgecolor="white", alpha=0.8)
                    else:
                        print(f"警告: 提示点坐标超出图像边界: ({x}, {y}), 图像尺寸: {image.shape[1]}x{image.shape[0]}")
                else:
                    print(f"警告: 提示点格式异常: {point}")
    
    # 添加调试信息
    if len(point_list) > 0:
        print(f"调试: 图像尺寸: {image.shape[1]}x{image.shape[0]}")
        print(f"调试: 提示点数量: {len(point_list)}")
        print(f"调试: 提示点坐标: {point_list}")
        print(f"调试: 提示点标签: {point_label_list}")

    # 右子图: 带掩膜的图像
    image_with_mask = image.copy()
    for c in range(3):  # 对每个颜色通道应用掩膜
        image_with_mask[:, :, c] = np.where(mask > 0, color[c], image[:, :, c])
    axes[1].imshow(image_with_mask)
    axes[1].set_title("Mask Visualization")
    axes[1].axis("off")

    # 显示图像
    plt.tight_layout()
    if need_show:
        plt.show()
    
    # 保存图像
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        except Exception as e:
            print(f"保存图像时出错: {e}")
    
    # 只有在需要显示时才关闭图像
    if not need_show:
        plt.close(fig)


import os
import numpy as np
import cv2

def visualize_superpixel_indices(image, segments, save_path=None):
    # 如果输入是 float 类型 (0~1)，先转为 uint8
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    # 如果是灰度图，转成3通道
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_with_labels = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_labels = image.copy()

    # 添加边界（用 skimage 的 mark_boundaries）
    # mark_boundaries 返回 RGB float 图，范围 0~1，需要乘 255 转回 uint8
    boundary_image = mark_boundaries(image_with_labels, segments, color=(0, 255, 0))  # green boundaries
    boundary_image = (boundary_image * 255).astype(np.uint8)

    # 画每个 superpixel 的编号
    superpixel_ids = np.unique(segments)
    for superpixel_id in superpixel_ids:
        mask = segments == superpixel_id
        coords = np.column_stack(np.nonzero(mask))
        center = coords.mean(axis=0).astype(int)
        cv2.putText(boundary_image, str(superpixel_id),
                    (center[1], center[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),  # blue text
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, boundary_image)

    return boundary_image
