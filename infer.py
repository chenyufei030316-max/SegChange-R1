# -*- coding: utf-8 -*-
import torch
from engines import predict
from utils import get_args_config
# 必须导入 Config 类以便将其加入安全名单
from utils.misc import Config 

def main():
    # --- [核心修复] 允许加载自定义 Config 类 ---
    # PyTorch 2.6+ 默认 weights_only=True，需要手动放行安全类
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Config])
    
    cfg = get_args_config()
    
    names = ['change']
    for name in names:
        cfg.infer.name = name
        cfg.infer.input_dir = '/data/change/infer'
        print(f"Starting inference for: {name}")
        predict(cfg)

if __name__ == "__main__":
    main()