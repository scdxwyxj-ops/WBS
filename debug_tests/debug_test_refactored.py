#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBC图像分割调试脚本 - 重构版本

该脚本用于调试和测试WBC（白细胞）图像分割算法，包括：
1. SLIC超像素分割
2. SAM2模型预测
3. 迭代优化
4. 结果评估和保存

作者: [Your Name]
日期: [Current Date]
版本: 2.0
"""

import os
import cv2
import numpy as np
import sys
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional

# 导入日志管理器
try:
    from logger import create_logger
except ImportError:
    # 如果直接导入失败，尝试从image_processings目录导入
    sys.path.insert(0, '/app/SAM2_proj/WBC/image_processings')
    from logger import create_logger

# 添加外部依赖路径
sys.path.append('/app/SAM2_proj/sam2-main')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 添加WBC项目路径
sys.path.append('/app/SAM2_proj/WBC')
sys.path.append('/app/SAM2_proj/WBC/image_processings')

# 导入WBC项目模块
try:
    from metrics.metric import calculate_miou
    from metrics.visualize import show_combined_plots
    from image_pre_seg import image_i_segment, change_image_type, get_resize_shape
    from info import Info
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试从image_processings目录导入...")
    # 如果直接导入失败，尝试从image_processings目录导入
    sys.path.insert(0, '/app/SAM2_proj/WBC/image_processings')
    try:
        from metrics.metric import calculate_miou
        from metrics.visualize import show_combined_plots
        from image_pre_seg import image_i_segment, change_image_type, get_resize_shape
        from info import Info
    except ImportError as e2:
        print(f"仍然无法导入: {e2}")
        print("请检查模块路径和依赖关系")
        sys.exit(1)


class WBCSegmentationDebugger:
    """
    WBC图像分割调试器类
    
    负责管理整个WBC分割流程，包括图像加载、预处理、分割、优化和结果保存
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 model_config: str,
                 root_dir: str = './GrTh/cropped',
                 new_size: int = 260,
                 num_nodes: int = 100,
                 compactness: int = 12,
                 sigma: float = 1.0,
                 min_size_factor: float = 0.6,
                 max_size_factor: float = 1.2,
                 negative_pct: float = 0.1,
                 max_epochs: int = 10,
                 max_iterations: int = 20,
                 refinement_iterations: int = 5,
                 output_dir: str = './assets/new_test_000',
                 mask_color: Tuple[int, int, int] = (0, 0, 255),
                 iou_threshold: float = 0.75,
                 max_images: int = -1,
                 subset_size: int = 20,
                 debug_mode: bool = False,
                 save_intermediate: bool = True,
                 verbose: bool = True):
        """
        初始化WBC分割调试器
        
        Args:
            checkpoint_path: SAM2模型checkpoint路径
            model_config: SAM2模型配置文件路径
            root_dir: 数据集根目录
            new_size: 图像调整后的目标尺寸
            num_nodes: 图节点数量
            compactness: SLIC分割的紧凑度参数
            sigma: SLIC分割的高斯模糊参数
            min_size_factor: SLIC分割的最小尺寸因子
            max_size_factor: SLIC分割的最大尺寸因子
            negative_pct: 负样本比例
            max_epochs: 最大训练轮数
            max_iterations: 每轮最大迭代次数
            refinement_iterations: 精细化迭代次数
        """
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.root_dir = root_dir
        self.new_size = new_size
        self.num_nodes = num_nodes
        self.compactness = compactness
        self.sigma = sigma
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.negative_pct = negative_pct
        self.max_epochs = max_epochs
        self.max_iterations = max_iterations
        self.refinement_iterations = refinement_iterations
        self.output_dir = output_dir
        self.mask_color = mask_color
        self.iou_threshold = iou_threshold
        self.max_images = max_images
        self.subset_size = subset_size
        self.debug_mode = debug_mode
        self.save_intermediate = save_intermediate
        self.verbose = verbose
        
        # 初始化路径
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        
        # 初始化结果存储
        self.pred_mask_list = []
        self.gt_mask_list = []
        
        # 初始化日志管理器（延迟初始化，因为需要配置信息）
        self.logger = None
        
        # 初始化SAM2模型
        self.predictor = self._initialize_sam2_model()
        
        # 获取图像列表
        self.image_names, self.mask_names = self._get_dataset_files()
        
        # 应用子集限制
        self._apply_subset_limitation()
    
    def _initialize_logger(self, config: dict) -> None:
        """
        初始化日志管理器
        
        Args:
            config: 配置字典
        """
        if self.logger is None:
            self.logger = create_logger(config)
    
    def _apply_subset_limitation(self) -> None:
        """
        应用子集限制，根据配置限制处理的图像数量
        """
        total_images = len(self.image_names)
        
        if self.debug_mode and self.subset_size > 0:
            # 调试模式：只处理指定数量的图像
            limit = min(self.subset_size, total_images)
            self.image_names = self.image_names[:limit]
            self.mask_names = self.mask_names[:limit]
            if self.verbose:
                print(f"调试模式：限制处理图像数量为 {limit}/{total_images}")
        elif self.max_images > 0:
            # 正常模式：限制最大图像数量
            limit = min(self.max_images, total_images)
            self.image_names = self.image_names[:limit]
            self.mask_names = self.mask_names[:limit]
            if self.verbose:
                print(f"限制处理图像数量为 {limit}/{total_images}")
        else:
            # 处理所有图像
            if self.verbose:
                print(f"处理所有图像：{total_images} 张")
    
    def _initialize_sam2_model(self) -> SAM2ImagePredictor:
        """
        初始化SAM2模型
        
        Returns:
            SAM2ImagePredictor: 初始化后的SAM2预测器
            
        Raises:
            FileNotFoundError: 当checkpoint文件不存在时
            RuntimeError: 当模型初始化失败时
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint文件不存在: {self.checkpoint_path}")
        
        try:
            print("Initializing SAM2 model...")
            model = build_sam2(self.model_config, self.checkpoint_path)
            predictor = SAM2ImagePredictor(model)
            print("SAM2 model initialized successfully")
            return predictor
        except Exception as e:
            raise RuntimeError(f"SAM2 model initialization failed: {e}")
    
    def _get_dataset_files(self) -> Tuple[List[str], List[str]]:
        """
        获取数据集文件列表
        
        Returns:
            Tuple[List[str], List[str]]: (图像文件名列表, 掩码文件名列表)
        """
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
        
        # 获取所有图像文件
        all_image_names = sorted(os.listdir(self.image_dir))
        all_mask_names = sorted(os.listdir(self.mask_dir))
        
        if len(all_image_names) == 0:
            raise ValueError("图像目录为空")
        if len(all_mask_names) == 0:
            raise ValueError("掩码目录为空")
        
        # Record found file count
        if self.verbose:
            print(f"Found {len(all_image_names)} image files")
        
        if self.logger:
            self.logger.info(f"Dataset info: image_dir={self.image_dir}, mask_dir={self.mask_dir}")
            self.logger.info(f"Found {len(all_image_names)} image files")
        
        return all_image_names, all_mask_names
    
    def load_image_with_color_fix(self, image_path: str) -> np.ndarray:
        """
        加载图像并修复颜色空间问题
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            np.ndarray: RGB格式的图像数组
            
        Raises:
            FileNotFoundError: 当图像文件不存在时
            ValueError: 当图像格式不支持时
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 使用PIL加载图像，确保RGB格式一致性
            pil_img = Image.open(image_path)
            image = np.array(pil_img)
            
            # 确保图像是RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                    image = np.array(pil_img)
            else:
                raise ValueError(f"不支持的图像格式: 形状={image.shape}, 模式={pil_img.mode}")
            
            if self.verbose:
                print(f"Image loaded successfully: {image_path}, shape: {image.shape}, mode: {pil_img.mode}")
            
            if self.logger:
                self.logger.debug(f"Image loaded successfully: {image_path}")
                self.logger.debug(f"Image info: shape={image.shape}, mode={pil_img.mode}, value_range={image.min()}-{image.max()}")
            
            return image
            
        except Exception as e:
            error_msg = f"Image loading failed {image_path}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_ground_truth_mask(self, mask_path: str) -> np.ndarray:
        """
        加载地面真值掩码
        
        Args:
            mask_path: 掩码文件路径
            
        Returns:
            np.ndarray: 二值掩码数组
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
        
        gt_mask_pil = Image.open(mask_path).convert("L")
        gt_mask_np = np.array(gt_mask_pil) > 0
        return gt_mask_np
    
    def resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        调整掩码尺寸
        
        Args:
            mask: 输入掩码
            target_shape: 目标尺寸 (height, width)
            
        Returns:
            np.ndarray: 调整后的掩码
        """
        mask_resized = cv2.resize(mask.astype(np.uint8), target_shape, 
                                 interpolation=cv2.INTER_NEAREST)
        return mask_resized > 0
    
    def perform_slic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, object]:
        """
        执行SLIC超像素分割
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, np.ndarray, object]: (调整后的图像, 分割结果, 图结构)
        """
        if self.verbose:
            print("Executing SLIC superpixel segmentation...")
        
        if self.logger:
            self.logger.info("Starting SLIC superpixel segmentation")
            self.logger.debug(f"SLIC parameters: image_size={self.new_size}, node_count={self.num_nodes}, compactness={self.compactness}")
        
        A = image_i_segment(
            name=None,
            label=None,
            image=image,
            new_size_of_image=self.new_size,
            num_node_for_graph=self.num_nodes,
            compactness_in_SLIC=self.compactness,
            sigma_in_SLIC=self.sigma,
            min_size_factor_in_SLIC=self.min_size_factor,
            max_size_factor_in_SLIC=self.max_size_factor
        )
        
        img = change_image_type(A.image_resized, 'np.array')
        seg_by_slic = np.array(A.segment_without_padding)
        graph = A.graph
        
        slic_info = {
            'image_shape': img.shape,
            'segment_count': len(np.unique(seg_by_slic)),
            'compactness': self.compactness,
            'sigma': self.sigma
        }
        
        if self.verbose:
            print(f"SLIC segmentation completed: image_shape={img.shape}, segment_count={len(np.unique(seg_by_slic))}")
        
        if self.logger:
            self.logger.log_slic_results(slic_info)
        
        return img, seg_by_slic, graph
    
    def run_sam2_prediction(self, img: np.ndarray, seg_by_slic: np.ndarray, 
                           graph: object, epoch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        运行SAM2模型预测
        
        Args:
            img: 输入图像
            seg_by_slic: SLIC分割结果
            graph: 图结构
            epoch: 当前图像索引
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, int]: (最佳掩码, 最佳点, 最佳标签, 迭代次数)
        """
        print("Running SAM2 model prediction...")
        
        # Create info object
        info_obj = Info(
            segment=seg_by_slic,
            logits=None,
            image=img,
            graph=graph,
            negative_pct=self.negative_pct,
            debug_mode=False
        )
        
        # Initial prediction
        self.predictor.set_image(img)
        initial_prompts = info_obj.get_prompts()
        
        print(f"Initial prompt points count: {len(initial_prompts['points'])}")
        print(f"Initial prompt points: {initial_prompts['points']}")
        print(f"Initial labels: {initial_prompts['labels']}")
        
        logits, _, low_res_mask = self.predictor.predict(
            point_coords=initial_prompts['points'],
            point_labels=initial_prompts['labels'],
            box=None,
            mask_input=None,
            multimask_output=False,
            return_logits=True
        )
        logits = logits[0]
        current_mask = logits > 0
        
        # 初始化最佳结果变量
        best_points = initial_prompts['points']
        best_labels = initial_prompts['labels']
        best_mask = current_mask
        best_score = -1
        
        # 保存初始预测结果
        if self.save_intermediate:
            self.save_debug_intermediate(
                img, seg_by_slic, initial_prompts['points'], 
                initial_prompts['labels'], current_mask,
                'initial_prediction', epoch, 0,
                {'score': 'N/A', 'prompt_count': len(initial_prompts['points'])}
            )
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*50}")
            
            # Check if iteration should end
            is_end = info_obj.update_nodes(logits, need_lower_bound=False)
            if is_end:
                print(f"Iteration end condition met, stopping at iteration {iteration + 1}")
                break
            
            # Get new prompt points
            prompts = info_obj.get_prompts(need_aug=True, use_subset=True)
            
            # Check if there are new points
            if 'new_points' not in prompts or len(prompts['new_points']) == 0:
                print("No new prompt points, iteration ends")
                break
            
            new_points = prompts['new_points']
            print(f"Got {len(new_points)} new prompt points:")
            for i, point in enumerate(new_points):
                print(f"  New point {i + 1}: {point}")
            
            # 尝试每个新点，选择最佳结果
            best_score = -1
            best_mask = None
            best_points = None
            best_labels = None
            best_low_res_mask = None
            best_point_index = -1  # 记录最佳点的索引
            all_scores = []  # 记录所有点的分数
            
            for i, new_point in enumerate(new_points):
                print(f"\n--- Trying point {i + 1}: {new_point} ---")
                
                # Build current prompt point set - using accumulated points
                # Get current accumulated points from info_obj
                current_prompts = info_obj.get_prompts(need_aug=False, use_subset=False)
                current_points = current_prompts['points'].copy()
                current_labels = current_prompts['labels'].copy()
                
                # Add new point as positive prompt
                current_points = np.vstack([current_points, new_point])
                current_labels = np.append(current_labels, 1)  # 1 means positive
                
                print(f"Current prompt points count: {len(current_points)}")
                print(f"Current labels: {current_labels}")
                
                # 运行预测
                logits, score, current_low_res_mask = self.predictor.predict(
                    point_coords=current_points,
                    point_labels=current_labels,
                    box=None,
                    mask_input=low_res_mask,  # 使用上一轮的low_res_mask
                    multimask_output=False,
                    return_logits=True
                )
                
                # 确保score是标量值
                if hasattr(score, 'item'):
                    score = score.item()
                
                all_scores.append(score)
                print(f"Point {i + 1} score: {score:.4f}")
                
                # Save prediction result for each point
                if self.save_intermediate:
                    self.save_debug_intermediate(
                        img, seg_by_slic, current_points, current_labels, 
                        logits[0] > 0,
                        f'iteration_{iteration+1}_point_{i+1}', epoch, iteration + 1,
                        {'score': f'{score:.4f}', 'point': str(new_point), 'iteration': iteration + 1}
                    )
                
                # Update best result
                if best_score < score:
                    best_score = score
                    best_mask = logits[0] > 0
                    best_points = current_points
                    best_labels = current_labels
                    best_low_res_mask = current_low_res_mask
                    best_point_index = i  # 记录最佳点的索引
                    print(f"  -> New best result!")
            
            print(f"\nBest score this round: {best_score:.4f}")
            print(f"Best point index: {best_point_index + 1}")
            
            # 现在基于分数选择最佳点，并更新info_obj
            if best_point_index >= 0:
                # 获取最佳点（基于上一轮的候选点）
                best_selected_point = new_points[best_point_index]
                print(f"选择最佳点: {best_selected_point}, score: {best_score:.4f}")
                
                # 调用get_prompts，传递最佳点
                final_prompts = info_obj.get_prompts(
                    need_aug=True, 
                    use_subset=True, 
                    best_point=best_selected_point
                )
                print(f"基于分数选择后的提示点数量: {len(final_prompts['points'])}")
            
            # Save best result for this round
            if self.save_intermediate:
                self.save_debug_intermediate(
                    img, seg_by_slic, best_points, best_labels, best_mask,
                    f'iteration_{iteration+1}_best', epoch, iteration + 1,
                    {'best_score': f'{best_score:.4f}', 'iteration': iteration + 1, 'best_point_index': best_point_index + 1}
                )
            
            # 使用最佳结果继续下一轮迭代
            if best_mask is not None:
                current_mask = best_mask
                # 保存最佳的低分辨率掩码用于下一轮迭代
                low_res_mask = best_low_res_mask
                
                # 重要：不再手动更新info_obj的positive_point_coords
                # 因为get_prompts方法已经基于分数选择了最佳点并添加了
                print(f"使用最佳结果继续下一轮迭代，最佳点索引: {best_point_index + 1}")
            else:
                print("Warning: No valid mask obtained")
                break
        
        # Refinement processing
        print(f"\n{'='*50}")
        print("Starting refinement processing...")
        print(f"{'='*50}")
        
        for i in range(self.refinement_iterations):
            print(f"Refinement iteration {i + 1}/{self.refinement_iterations}")
            
            logits, score, low_res_mask = self.predictor.predict(
                point_coords=best_points,
                point_labels=best_labels,
                box=None,
                mask_input=low_res_mask,  # Directly use SAM2 returned low resolution mask
                multimask_output=False,
                return_logits=True
            )
            logits = logits[0]
            current_mask = logits > 0
            
            # Ensure score is scalar value
            if hasattr(score, 'item'):
                score = score.item()
            
            print(f"Refinement iteration {i + 1}: score = {score:.4f}")
            
            # Save refinement results
            if self.save_intermediate:
                self.save_debug_intermediate(
                    img, seg_by_slic, best_points, best_labels, current_mask,
                    f'refinement_{i+1}', epoch, iteration + 1,
                    {'score': f'{score:.4f}', 'refinement_step': i + 1}
                )
        
        # Save final result
        if self.save_intermediate:
            self.save_debug_intermediate(
                img, seg_by_slic, best_points, best_labels, current_mask,
                'final_result', epoch, iteration + 1,
                {'final_score': f'{score:.4f}', 'total_iterations': iteration + 1}
            )
        
        print(f"\nSAM2 prediction completed: final score={score:.4f}, total iterations={iteration + 1}")
        return current_mask, best_points, best_labels, iteration + 1
    
    def save_results(self, img: np.ndarray, seg_by_slic: np.ndarray, 
                    points: np.ndarray, labels: np.ndarray, mask: np.ndarray, 
                    epoch: int, end_iter: int) -> str:
        """
        保存分割结果
        
        Args:
            img: 处理后的图像
            seg_by_slic: SLIC分割结果
            points: 标注点
            labels: 标注标签
            mask: 预测掩码
            epoch: 当前轮数
            end_iter: 迭代次数
            
        Returns:
            str: 保存路径
        """
        try:
            # 确保保存前图像格式正确
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 确保图像是RGB格式
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            # 创建保存目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 保存结果
            save_path = os.path.join(self.output_dir, f'{epoch}_{end_iter}.png')
            
            # 调用可视化函数保存图像
            show_combined_plots(img, seg_by_slic, points, labels, mask, 
                               color=self.mask_color, save_path=save_path, need_show=False)
            
            # 验证文件是否成功保存
            if os.path.exists(save_path):
                print(f"结果已保存: {save_path}")
                return save_path
            else:
                print(f"警告: 文件保存失败: {save_path}")
                return ""
                
        except Exception as e:
            print(f"保存结果时出错: {e}")
            return ""
    
    def save_debug_intermediate(self, img: np.ndarray, seg_by_slic: np.ndarray, 
                               points: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                               step_name: str, epoch: int, iteration: int, 
                               additional_info: dict = None) -> str:
        """
        保存调试中间过程
        
        Args:
            img: 处理后的图像
            seg_by_slic: SLIC分割结果
            points: 标注点
            labels: 标注标签
            mask: 预测掩码
            step_name: 步骤名称
            epoch: 当前轮数
            iteration: 当前迭代次数
            additional_info: 额外信息
            
        Returns:
            str: 保存路径
        """
        try:
            # 确保保存前图像格式正确
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 确保图像是RGB格式
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            # 创建debug目录
            debug_dir = os.path.join(self.output_dir, 'debug_intermediate')
            os.makedirs(debug_dir, exist_ok=True)
            
            # 生成清晰的文件名
            filename = self._generate_debug_filename(step_name, epoch, iteration)
            save_path = os.path.join(debug_dir, filename)
            
            # 调用可视化函数保存图像
            show_combined_plots(img, seg_by_slic, points, labels, mask, 
                               color=self.mask_color, save_path=save_path, need_show=False)
            
            # 保存额外信息到文本文件
            if additional_info:
                info_filename = filename.replace('.png', '_info.txt')
                info_path = os.path.join(debug_dir, info_filename)
                with open(info_path, 'w') as f:
                    for key, value in additional_info.items():
                        f.write(f"{key}: {value}\n")
            
            if os.path.exists(save_path):
                print(f"调试中间结果已保存: {filename}")
                return save_path
            else:
                print(f"警告: 调试文件保存失败: {save_path}")
                return ""
                
        except Exception as e:
            print(f"保存调试中间结果时出错: {e}")
            return ""
    
    def _generate_debug_filename(self, step_name: str, epoch: int, iteration: int) -> str:
        """
        生成清晰的debug文件名
        
        Args:
            step_name: 步骤名称
            epoch: 图像索引
            iteration: 迭代次数
            
        Returns:
            str: 文件名
        """
        # 获取图像名称（不含扩展名）
        if epoch < len(self.image_names):
            image_name = os.path.splitext(self.image_names[epoch])[0]
        else:
            image_name = f"img_{epoch:03d}"
        
        # 定义步骤顺序和描述 - 统一使用英文
        step_order = {
            'slic_result': (0, '01_SLIC_Segmentation'),
            'initial_prediction': (1, '02_Initial_Prediction'),
            'iteration_1_point_1': (2, '03_Iter1_Point1'),
            'iteration_1_point_2': (3, '04_Iter1_Point2'), 
            'iteration_1_point_3': (4, '05_Iter1_Point3'),
            'iteration_1_best': (5, '06_Iter1_Best'),
            'iteration_2_point_1': (6, '07_Iter2_Point1'),
            'iteration_2_point_2': (7, '08_Iter2_Point2'),
            'iteration_2_point_3': (8, '09_Iter2_Point3'),
            'iteration_2_best': (9, '10_Iter2_Best'),
            'iteration_3_point_1': (10, '11_Iter3_Point1'),
            'iteration_3_point_2': (11, '12_Iter3_Point2'),
            'iteration_3_point_3': (12, '13_Iter3_Point3'),
            'iteration_3_best': (13, '14_Iter3_Best'),
            'iteration_4_point_1': (14, '15_Iter4_Point1'),
            'iteration_4_point_2': (15, '16_Iter4_Point2'),
            'iteration_4_point_3': (16, '17_Iter4_Point3'),
            'iteration_4_best': (17, '18_Iter4_Best'),
            'iteration_5_point_1': (18, '19_Iter5_Point1'),
            'iteration_5_point_2': (19, '20_Iter5_Point2'),
            'iteration_5_point_3': (20, '21_Iter5_Point3'),
            'iteration_5_best': (21, '22_Iter5_Best'),
            'iteration_6_point_1': (22, '23_Iter6_Point1'),
            'iteration_6_point_2': (23, '24_Iter6_Point2'),
            'iteration_6_point_3': (24, '25_Iter6_Point3'),
            'iteration_6_best': (25, '26_Iter6_Best'),
            'refinement_1': (26, '27_Refinement1'),
            'refinement_2': (27, '28_Refinement2'),
            'refinement_3': (28, '29_Refinement3'),
            'refinement_4': (29, '30_Refinement4'),
            'refinement_5': (30, '31_Refinement5'),
            'final_result': (31, '32_Final_Result')
        }
        
        if step_name in step_order:
            order, description = step_order[step_name]
            filename = f"{image_name}_{order:02d}_{description}.png"
        else:
            # 对于未知步骤，使用原始命名
            filename = f"{image_name}_{iteration:02d}_{step_name}.png"
        
        return filename
    
    def process_single_image(self, image_name: str, mask_name: str, epoch: int) -> None:
        """
        处理单张图像
        
        Args:
            image_name: 图像文件名
            mask_name: 掩码文件名
            epoch: 当前轮数
        """
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info(f"处理图像 {epoch + 1}/{len(self.image_names)}: {image_name}")
            self.logger.info("=" * 50)
        
        try:
            # 构建文件路径
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            # 加载图像和掩码
            image = self.load_image_with_color_fix(image_path)
            gt_mask = self.load_ground_truth_mask(mask_path)
            
            if self.logger:
                self.logger.info(f"图像加载成功: {image_path}, 形状: {image.shape}, 模式: RGB")
                self.logger.info(f"真实掩码加载成功: {mask_path}, 形状: {gt_mask.shape}")
            
            # 调整掩码尺寸
            shape = get_resize_shape(self.new_size, image)
            new_h, new_w = shape[1][0], shape[1][1]
            gt_mask_resized = self.resize_mask(gt_mask, (new_w, new_h))
            
            if self.logger:
                self.logger.info(f"调整后尺寸: 图像={image.shape}, 掩码={gt_mask_resized.shape}")
            
            # 执行SLIC分割
            img, seg_by_slic, graph = self.perform_slic_segmentation(image)
            
            # 保存SLIC分割结果
            if self.save_intermediate:
                self.save_debug_intermediate(
                    img, seg_by_slic, np.array([]), np.array([]), 
                    np.zeros_like(seg_by_slic, dtype=bool),
                    'slic_result', epoch, 0,
                    {'segment_count': len(np.unique(seg_by_slic)), 'image_shape': str(img.shape)}
                )
            
            # 运行SAM2预测
            mask, points, labels, end_iter = self.run_sam2_prediction(img, seg_by_slic, graph, epoch)
            
            # 保存最终结果
            self.pred_mask_list.append(mask)
            self.gt_mask_list.append(gt_mask_resized)
            
            # 计算当前图像的IoU
            current_iou_result = calculate_miou([gt_mask_resized], [mask])
            if isinstance(current_iou_result, tuple):
                current_iou = current_iou_result[0]  # 取第一个值（单个图像的IoU）
            else:
                current_iou = current_iou_result
            
            if hasattr(current_iou, 'item'):
                current_iou = current_iou.item()
            
            if self.logger:
                self.logger.info(f"当前图像IoU: {current_iou:.4f}")
            
            save_path = self.save_results(img, seg_by_slic, points, labels, mask, epoch, end_iter)
            
            if self.logger:
                self.logger.info(f"图像 {image_name} 处理完成")
            
        except Exception as e:
            error_msg = f"处理图像 {image_name} 时出错: {e}"
            if self.logger:
                self.logger.error(error_msg)
                import traceback
                self.logger.error(traceback.format_exc())
            # 添加空结果以保持列表长度一致
            self.pred_mask_list.append(np.zeros((self.new_size, self.new_size), dtype=bool))
            self.gt_mask_list.append(np.zeros((self.new_size, self.new_size), dtype=bool))
    
    def run(self, config: dict = None) -> None:
        """
        运行完整的WBC分割调试流程
        
        Args:
            config: 配置字典，用于初始化日志管理器
        """
        # 初始化日志管理器
        if config and self.logger is None:
            self._initialize_logger(config)
        
        # 记录配置参数到日志
        if self.logger:
            self.logger.log_config({
                'new_size': self.new_size,
                'num_nodes': self.num_nodes,
                'compactness': self.compactness,
                'max_epochs': self.max_epochs,
                'max_iterations': self.max_iterations,
                'refinement_iterations': self.refinement_iterations,
                'output_dir': self.output_dir,
                'mask_color': self.mask_color,
                'iou_threshold': self.iou_threshold,
                'debug_mode': self.debug_mode,
                'subset_size': self.subset_size,
                'max_images': self.max_images
            })
        
        # 终端只显示进度信息
        print("🚀 WBC图像分割调试开始...")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 调试模式: {'开启' if self.debug_mode else '关闭'}")
        if self.debug_mode:
            print(f"🔍 子集大小: {self.subset_size}")
        print("📝 详细日志请查看 ./log/ 目录下的日志文件")
        print("-" * 50)
        
        # 处理所有图像
        total_images = len(self.image_names)
        for epoch, (image_name, mask_name) in enumerate(
            tqdm(zip(self.image_names, self.image_names), 
                  total=total_images, 
                  desc="🔄 处理图像",
                  unit="张")
        ):
            if self.logger:
                self.logger.log_progress(epoch + 1, total_images, f"处理图像 {image_name}")
            
            self.process_single_image(image_name, mask_name, epoch)
        
        # 计算评估指标
        self._calculate_metrics()
        
        # 终端显示完成信息
        print("-" * 50)
        print("✅ WBC图像分割调试完成！")
        print(f"📊 处理图像总数: {total_images}")
        print(f"📝 详细结果请查看: {self.output_dir}")
        print(f"📋 详细日志请查看: ./log/ 目录")
    
    def _calculate_metrics(self) -> None:
        """
        计算评估指标
        """
        if self.logger:
            self.logger.info("开始计算评估指标")
        
        try:
            miou, iou_list = calculate_miou(self.pred_mask_list, self.gt_mask_list)
            fail_cases = [idx for idx, iou in enumerate(iou_list) if iou < self.iou_threshold]
            
            metrics = {
                'Mean IoU': miou,
                '失败案例数量': len(fail_cases),
                '失败案例索引': fail_cases,
                'IoU阈值': self.iou_threshold,
                '处理图像总数': len(self.pred_mask_list)
            }
            
            # 记录到日志
            if self.logger:
                self.logger.info("=" * 50)
                self.logger.info("评估结果")
                self.logger.info("=" * 50)
                for key, value in metrics.items():
                    self.logger.info(f"{key}: {value}")
                self.logger.info("=" * 50)
                self.logger.info(f"详细IoU值: {[f'{iou:.4f}' for iou in iou_list]}")
                self.logger.info("=" * 50)
            
            # 终端只显示关键结果
            print(f"📊 Mean IoU: {miou:.4f}")
            print(f"❌ 失败案例: {len(fail_cases)}/{len(self.pred_mask_list)}")
            if fail_cases:
                print(f"⚠️  失败案例索引: {fail_cases}")
            
        except Exception as e:
            error_msg = f"计算评估指标时出错: {e}"
            if self.logger:
                self.logger.error(error_msg)
            print(f"❌ {error_msg}")


def main():
    """
    主函数
    """
    try:
        # 导入配置
        from config import get_full_config, validate_config
        
        print("正在加载配置...")
        config = get_full_config()
        
        print("正在验证配置...")
        validate_config(config)
        
        print("正在创建调试器...")
        # 创建调试器实例
        debugger = WBCSegmentationDebugger(**config)
        
        print("正在运行调试流程...")
        # 运行调试流程，传递配置用于日志初始化
        debugger.run(config)
        
        print("\n🎉 WBC分割调试完成！")
        
    except ImportError as e:
        print(f"\n❌ 导入配置失败: {e}")
        print("请确保 config.py 文件存在且可访问")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
