import argparse
import csv
import os
import time

import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from models import build_model, build_embs
from utils import load_config

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(cfg, weights_dir, device):
    model = build_model(cfg, training=False)
    model.to(device)
    checkpoint = torch.load(weights_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def preprocess_pair(a_path, b_path, img_size, device):
    transform = Compose([ToTensor(), Normalize(mean=MEAN, std=STD)])
    img_a = cv2.imread(a_path)
    img_b = cv2.imread(b_path)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    img_a = cv2.resize(img_a, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_b = cv2.resize(img_b, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_a = transform(img_a).unsqueeze(0).to(device)
    img_b = transform(img_b).unsqueeze(0).to(device)
    return img_a, img_b


def run_inference(model, embs, a_path, b_path, img_size, threshold, device):
    img_a, img_b = preprocess_pair(a_path, b_path, img_size, device)
    with torch.no_grad():
        outputs = model(img_a, img_b, embs)
        preds = (torch.sigmoid(outputs) > threshold).float()
    changed_pixels = int(preds.sum().item())
    total_pixels = int(preds.numel())
    return changed_pixels, total_pixels


def list_pairs(root, label):
    a_dir = os.path.join(root, 'A')
    b_dir = os.path.join(root, 'B')
    pairs = []
    for fn in sorted(os.listdir(a_dir)):
        a_path = os.path.join(a_dir, fn)
        b_fn = fn.replace('_2019_', '_2024_')
        b_path = os.path.join(b_dir, b_fn)
        if os.path.isfile(a_path) and os.path.isfile(b_path):
            pairs.append((fn, a_path, b_path, label))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--prompt', default='', help='fixed prompt text for this checkpoint (empty for no-text)')
    parser.add_argument('--test-root', default='/ssd4Tb/chenyf/Test/RGB')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.device = args.device
    model = load_model(cfg, args.checkpoint, args.device)
    img_size = cfg.model.img_size

    prompts = [args.prompt] if args.prompt else None
    embs = build_embs(prompts=prompts, text_encoder_name=cfg.model.text_encoder_name,
                      freeze_text_encoder=cfg.model.freeze_text_encoder, device=args.device, batch_size=1)
    print(f'prompt={args.prompt!r}, embs shape={embs.shape}, img_size={img_size}')

    tp_pairs = list_pairs(os.path.join(args.test_root, 'positive'), 1)
    tn_pairs = list_pairs(os.path.join(args.test_root, 'negative'), 0)
    if args.limit:
        half = args.limit // 2
        tp_pairs = tp_pairs[:half]
        tn_pairs = tn_pairs[:half]
    pairs = tp_pairs + tn_pairs
    print(f'total pairs: {len(pairs)} (TP={len(tp_pairs)}, TN={len(tn_pairs)})')

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    t0 = time.time()
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_label', 'changed_pixels', 'total_pixels', 'pred_label'])
        for i, (fn, a_path, b_path, true_label) in enumerate(pairs):
            changed_pixels, total_pixels = run_inference(model, embs, a_path, b_path, img_size, args.threshold, args.device)
            pred_label = 1 if changed_pixels > 0 else 0
            writer.writerow([fn, true_label, changed_pixels, total_pixels, pred_label])
            if (i + 1) % 200 == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pairs) - i - 1) / rate
                print(f'{i+1}/{len(pairs)} done, {rate:.2f} img/s, eta {eta/60:.1f} min')
    print('DONE', args.out_csv)


if __name__ == '__main__':
    main()
