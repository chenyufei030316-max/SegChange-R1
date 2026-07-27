import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import imageio.v2 as imageio
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
from utils import load_config
from models import build_model, PostProcessor

def save_comparison(a_path, b_path, pred_mask, save_path):
    a = imageio.imread(a_path)
    b = imageio.imread(b_path)
    mask = (pred_mask * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_overlay = a.copy()
    mask_overlay[pred_mask > 0] = [255, 50, 50]
    comparison = np.concatenate([a, b, mask_rgb, mask_overlay], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

def run(args):
    cfg = load_config(args.config)
    cfg.device = 'cuda:0'

    model = build_model(cfg, training=False)
    model.to(cfg.device)

    # 加载最佳权重
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    save_dirs = {
        'TP_correct': os.path.join(args.save_dir, 'TP_correct'),
        'FP_wrong': os.path.join(args.save_dir, 'FP_wrong'),
        'FN_wrong': os.path.join(args.save_dir, 'FN_wrong'),
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    all_preds = []
    all_labels = []
    saved_counts = {'TP_correct': 0, 'FP_wrong': 0, 'FN_wrong': 0}

    for split, label, data_path, list_path in [
        ('TP', 1, args.tp_path, args.tp_list),
        ('FP', 0, args.fp_path, args.fp_list)
    ]:
        with open(list_path, 'r') as f:
            data_list = [line.strip() for line in f]

        print(f'推理 {split} ({len(data_list)} 张)...')
        for fname in tqdm(data_list):
            a_path = os.path.join(data_path, 'A', fname)
            b_fname = fname.replace('s2_2019_', 's2_2024_')
            b_path = os.path.join(data_path, 'B', b_fname)

            if not os.path.exists(a_path) or not os.path.exists(b_path):
                continue

            a_img = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
            b_img = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)

            a_tensor = transform(a_img).unsqueeze(0).to(cfg.device)
            b_tensor = transform(b_img).unsqueeze(0).to(cfg.device)

            # 获取文本嵌入
            from models import build_embs
            prompt = args.prompt if args.prompt else (cfg.prompt if cfg.prompt else '')
            embs = build_embs(
                prompts=[prompt],
                text_encoder_name=cfg.model.text_encoder_name,
                freeze_text_encoder=cfg.model.freeze_text_encoder,
                device=cfg.device,
                batch_size=1
            )

            with torch.no_grad():
                output = model(a_tensor, b_tensor, embs)
                pred_mask = (torch.sigmoid(output) > cfg.training.threshold).float().squeeze().cpu().numpy()

            ratio = pred_mask.sum() / pred_mask.size
            pred = 1 if ratio > args.threshold else 0

            all_preds.append(pred)
            all_labels.append(label)

            if label == 1 and pred == 1 and saved_counts['TP_correct'] < args.max_save:
                save_comparison(a_path, b_path, pred_mask, os.path.join(save_dirs['TP_correct'], fname))
                saved_counts['TP_correct'] += 1
            elif label == 0 and pred == 1 and saved_counts['FP_wrong'] < args.max_save:
                save_comparison(a_path, b_path, pred_mask, os.path.join(save_dirs['FP_wrong'], fname))
                saved_counts['FP_wrong'] += 1
            elif label == 1 and pred == 0 and saved_counts['FN_wrong'] < args.max_save:
                save_comparison(a_path, b_path, pred_mask, os.path.join(save_dirs['FN_wrong'], fname))
                saved_counts['FN_wrong'] += 1

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    print(f'\n=== SegChange-R1 泛化实验结果（阈值={args.threshold}）===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1:        {f1:.4f}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'总样本: {len(all_labels)}, 预测有滑塌: {sum(all_preds)}, 真实有滑塌: {sum(all_labels)}')
    print(f'已保存: TP_correct={saved_counts["TP_correct"]}, FP_wrong={saved_counts["FP_wrong"]}, FN_wrong={saved_counts["FN_wrong"]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--tp_path', type=str, required=True)
    parser.add_argument('--tp_list', type=str, required=True)
    parser.add_argument('--fp_path', type=str, required=True)
    parser.add_argument('--fp_list', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='/hdd10Ta/chenyf/vis/segchange_generalization')
    parser.add_argument('--max_save', type=int, default=99999)
    parser.add_argument('--prompt', type=str, default='')
    args = parser.parse_args()
    run(args)
