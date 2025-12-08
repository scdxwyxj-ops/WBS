#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBC分割调试器日志管理模块

提供统一的日志记录功能，支持文件和控制台输出
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

class WBCLogger:
    """WBC调试专用日志管理器"""
    
    def __init__(self, log_dir: str = './log', log_level: str = 'INFO'):
        self.log_dir = log_dir
        self.log_level = log_level
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 生成日志文件名（按时间）
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_file = os.path.join(log_dir, f'{timestamp}_wbc_debug.log')
        
        # 设置日志格式
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 初始化日志记录器
        self._setup_logger()
        
        # 重定向print到日志
        self._redirect_print()
        
        # 记录启动信息
        self.logger.info("=" * 80)
        self.logger.info("WBC调试系统启动")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("=" * 80)
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('WBCDebugger')
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置格式
        formatter = logging.Formatter(self.log_format)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
    
    def _redirect_print(self):
        """重定向print到日志文件"""
        class PrintRedirector:
            def __init__(self, logger):
                self.logger = logger
            
            def write(self, text):
                if text.strip():  # 忽略空行
                    self.logger.info(f"[PRINT] {text.strip()}")
            
            def flush(self):
                pass
        
        # 重定向stdout和stderr
        sys.stdout = PrintRedirector(self.logger)
        sys.stderr = PrintRedirector(self.logger)
    
    def log_config(self, config: dict):
        """记录配置参数"""
        self.logger.info("=" * 80)
        self.logger.info("配置参数")
        self.logger.info("=" * 80)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 80)
    
    def log_progress(self, current: int, total: int, message: str):
        """记录进度信息"""
        percentage = (current / total) * 100
        self.logger.info(f"进度: {current}/{total} ({percentage:.1f}%) - {message}")
    
    def log_slic_results(self, info: dict):
        """记录SLIC分割结果"""
        self.logger.info("SLIC分割完成")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """记录严重错误"""
        self.logger.critical(message)

def create_logger(config: dict) -> WBCLogger:
    """创建日志管理器"""
    log_dir = config.get('log_dir', './log')
    log_level = config.get('log_level', 'INFO')
    return WBCLogger(log_dir=log_dir, log_level=log_level)


