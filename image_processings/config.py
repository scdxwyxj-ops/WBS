#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBC图像分割配置文件

该文件包含所有可配置的参数，便于调整和实验
"""

import os

# ============ 模型配置 ============
MODEL_CONFIG = {
    'checkpoint_path': "/app/SAM2_proj/sam2-main/checkpoints/sam2.1_hiera_large.pt",
    'model_config': "configs/sam2.1/sam2.1_hiera_l.yaml"
}

# ============ 数据集配置 ============
DATASET_CONFIG = {
    'root_dir': './GrTh/dataset_v0'  # 从WBC目录开始的路径
}

# ============ 图像处理配置 ============
IMAGE_CONFIG = {
    'new_size': 512  # 图像调整后的目标尺寸
}

# ============ SLIC分割配置 ============
SLIC_CONFIG = {
    'num_nodes': 100,  # 图节点数量
    'compactness': 20,  # SLIC分割的紧凑度参数
    'sigma': 1.0,  # SLIC分割的高斯模糊参数
    'min_size_factor': 0.6,  # SLIC分割的最小尺寸因子
    'max_size_factor': 1.2   # SLIC分割的最大尺寸因子
}

# ============ SAM2预测配置 ============
SAM2_CONFIG = {
    'negative_pct': 0.1,  # 负样本比例
    'max_epochs': 10,  # 最大训练轮数
    'max_iterations': 20,  # 每轮最大迭代次数
    'refinement_iterations': 5  # 精细化迭代次数
}

# ============ 评估配置 ============
EVALUATION_CONFIG = {
    'iou_threshold': 0.75  # IoU阈值，低于此值认为是失败案例
}

# ============ 输出配置 ============
OUTPUT_CONFIG = {
    'output_dir': './assets/recontructed_code_014',  # 结果保存目录
    'mask_color': (0, 0, 255)  # 掩码显示颜色 (BGR格式)
}

# ============ 可视化配置 ============
# 这些参数在WBCSegmentationDebugger中未使用，已移除

# =========== 日志配置 ===========
LOGGING_CONFIG = {
    'log_dir': './log',  # 日志文件目录
    'log_level': 'INFO',  # 日志级别: DEBUG, INFO, WARNING, ERROR
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    'max_log_size': 50 * 1024 * 1024,  # 最大日志文件大小 (50MB)
    'backup_count': 5  # 保留的备份文件数量
}

# ============ 调试配置 ============
DEBUG_CONFIG = {
    'max_images': -1,  # 最大处理图像数量 (设为-1表示处理所有图像)
    'subset_size': 1,  # 调试时处理的图像子集大小
    'debug_mode': True,  # 是否启用调试模式
    'save_intermediate': True,  # 是否保存中间结果
    'verbose': True  # 是否详细输出
}

# ============ 获取完整配置 ============
def get_full_config():
    """
    获取完整的配置字典
    
    Returns:
        dict: 包含所有配置的字典
    """
    config = {}
    config.update(MODEL_CONFIG)
    config.update(DATASET_CONFIG)
    config.update(IMAGE_CONFIG)
    config.update(SLIC_CONFIG)
    config.update(SAM2_CONFIG)
    config.update(EVALUATION_CONFIG)
    config.update(OUTPUT_CONFIG)
    config.update(DEBUG_CONFIG)
    
    return config

# ============ 配置验证 ============
def validate_config(config):
    """
    验证配置参数的有效性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 当配置参数无效时
    """
    # 检查必要的路径
    if not os.path.exists(config['checkpoint_path']):
        raise ValueError(f"SAM2 checkpoint文件不存在: {config['checkpoint_path']}")
    
    if not os.path.exists(config['root_dir']):
        raise ValueError(f"数据集根目录不存在: {config['root_dir']}")
    
    # 检查数值参数
    if config['new_size'] <= 0:
        raise ValueError("图像尺寸必须大于0")
    
    if config['num_nodes'] <= 0:
        raise ValueError("图节点数量必须大于0")
    
    if config['compactness'] <= 0:
        raise ValueError("SLIC紧凑度必须大于0")
    
    if config['max_epochs'] <= 0:
        raise ValueError("最大训练轮数必须大于0")
    
    if config['max_iterations'] <= 0:
        raise ValueError("最大迭代次数必须大于0")
    
    if not (0 <= config['negative_pct'] <= 1):
        raise ValueError("负样本比例必须在0到1之间")
    
    if not (0 <= config['iou_threshold'] <= 1):
        raise ValueError("IoU阈值必须在0到1之间")
    
    print("✅ 配置验证通过")

if __name__ == "__main__":
    import os
    
    # 测试配置
    try:
        config = get_full_config()
        validate_config(config)
        print("配置测试成功")
    except Exception as e:
        print(f"配置测试失败: {e}")
