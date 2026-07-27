import sys
import torch
from utils import load_config
from dataloader.loading_data import loading_data
from models import build_embs
from models.segchange import ChangeModel

cfg_path = sys.argv[1]
cfg = load_config(cfg_path)
device = torch.device('cpu')
cfg.device = device

train_ds, val_ds = loading_data(cfg)
img_a, img_b, prompt, label = train_ds[0]
print('原始prompt:', repr(prompt)[:80])
img_a = img_a.unsqueeze(0)
img_b = img_b.unsqueeze(0)

model = ChangeModel(cfg).to(device)
model.eval()

texts_to_compare = [
    "",  # 无文本，零向量
    "No retrogressive thaw slumps / No change in the retrogressive thaw slumps area.",
    "A large retrogressive thaw slump with pronounced headwall retreat and extensive lateral expansion into surrounding tundra terrain.",
]

outputs = []
embs_list = []
with torch.no_grad():
    for t in texts_to_compare:
        embs = build_embs(
            prompts=[t],
            text_encoder_name=cfg.model.text_encoder_name,
            freeze_text_encoder=cfg.model.freeze_text_encoder,
            device=device,
            batch_size=1,
        )
        embs_list.append(embs)
        out = model(img_a, img_b, embs)
        outputs.append(out)

print()
for i, t in enumerate(texts_to_compare):
    print(f'[{i}] text={t[:50]!r} -> embs norm={embs_list[i].norm().item():.4f}, output mean={outputs[i].mean().item():.6f}, output std={outputs[i].std().item():.6f}')

print()
print('=== 两两对比输出差异 ===')
for i in range(len(outputs)):
    for j in range(i+1, len(outputs)):
        diff = (outputs[i] - outputs[j]).abs()
        pred_i = (torch.sigmoid(outputs[i]) > 0.5).float()
        pred_j = (torch.sigmoid(outputs[j]) > 0.5).float()
        pixel_diff_ratio = (pred_i != pred_j).float().mean().item()
        print(f'[{i}] vs [{j}]: mean_abs_diff={diff.mean().item():.6f}, max_abs_diff={diff.max().item():.6f}, 预测像素不同比例={pixel_diff_ratio*100:.2f}%')

print()
print('DONE')
