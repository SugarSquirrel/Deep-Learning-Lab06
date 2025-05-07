import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
from diffusers import UNet2DModel
from evaluator import evaluation_model
from tqdm import tqdm
import wandb
import math
import json
import random

VAL_CONDS = json.load(open('./test.json'))
evaluator = evaluation_model()
evaluator.load_state_dict(torch.load('./checkpoint.pth', map_location='cuda'))
evaluator.cuda().eval()

# 評估模型的準確率
def validate_model(model, dataset, args, betas, alphas, alphas_cum, device):
    model.eval()
    imgs = []
    for cond in VAL_CONDS:
        # one-hot
        y = torch.zeros(1, 24, device=device)
        for lbl in cond:
            y[0, dataset.obj2id[lbl]] = 1
        # 反向擴散生成
        x = torch.randn(1, 3, 64, 64, device=device)
        for i in reversed(range(args.timesteps)):
            t = torch.tensor([i], device=device)
            eps = model(x, t, y)
            beta, alpha, cum = betas[i], alphas[i], alphas_cum[i]
            x = (x - beta/torch.sqrt(1-cum)*eps)/torch.sqrt(alpha)
            if i > 0:
                x += torch.sqrt(beta)*torch.randn_like(x)
        imgs.append(x.cpu())
    batch = torch.cat(imgs, 0)   # shape=(len(VAL_CONDS),3,64,64)

    # one-hot labels for evaluator
    labels = torch.zeros(len(VAL_CONDS), 24)
    for i, cond in enumerate(VAL_CONDS):
        for lbl in cond:
            labels[i, dataset.obj2id[lbl]] = 1

    # 評估
    acc = evaluator.eval(batch, labels)
    return acc

# ========= DataLoader ==========
class CLEVRDataset(Dataset):
    def __init__(self, json_path, obj_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        with open(obj_path, 'r') as f:
            obj2id = json.load(f)
        self.image_dir = image_dir
        self.obj2id = obj2id
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isinstance(data, dict):
            self.samples = list(data.items())
        else:
            self.samples = [(None, labels) for labels in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, labels = self.samples[idx]
        if fname:
            img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
            img = self.transform(img)
        else:
            img = None
        y = torch.zeros(len(self.obj2id))
        for lbl in labels:
            y[self.obj2id[lbl]] = 1
        return img, y

# ===== Noise schedule =====
def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# ===== Conditional UNet =====
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim=num_classes,
        )

    def forward(self, x, t, y):
        return self.unet(x, timestep=t, encoder_hidden_states=y).sample

# ===== Training =====
def train(args):
    # Initialize W&B
    wandb.init(project="Lab6-DDPM", config=vars(args))
    config = wandb.config

    # Dataset and DataLoader
    dataset = CLEVRDataset(args.train_json, args.obj_json, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Device and multi-GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()

    # Model, optimizer, scheduler
    model = ConditionalUNet(num_classes=len(dataset.obj2id))
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    betas = get_beta_schedule(args.timesteps).to(device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    # Optional EMA
    if args.use_ema:
        ema_model = ConditionalUNet(num_classes=len(dataset.obj2id)).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_decay = args.ema_decay

    # Evaluator for classifier guidance (optional)
    # if args.classifier_guidance:
    #     evaluator = evaluation_model()

    best_eval_metric = -math.inf

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        running_loss = 0.0
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            B = imgs.size(0)
            t = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(imgs)
            alpha_t = alphas_cum[t].view(-1, 1, 1, 1)
            noisy = torch.sqrt(alpha_t) * imgs + torch.sqrt(1 - alpha_t) * noise

            pred_noise = model(noisy, t, labels)
            loss = nn.MSELoss()(pred_noise, noise)

            # Classifier guidance loss
            if args.classifier_guidance:
                with torch.no_grad():
                    # denormalize and resize if needed
                    pred_imgs = (noisy - noisy.min()) / (noisy.max() - noisy.min())
                    cls_out = evaluator.resnet18(pred_imgs)
                cls_loss = nn.BCELoss()(cls_out, labels)
                loss = loss + args.guidance_weight * cls_loss
                wandb.log({'cls_loss': cls_loss.item()}, step=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            if args.use_ema:
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data = ema_p.data * ema_decay + p.data * (1 - ema_decay)

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Scheduler step
        if args.use_scheduler:
            scheduler.step()

        # Log metrics
        epoch_loss = running_loss / len(loader)
        wandb.log({'train_loss': epoch_loss, 'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        # --- 評估並儲存最佳模型 ---
        if (epoch + 1) % args.eval_freq == 0:
            model_to_eval = ema_model if args.use_ema else model
            val_acc = validate_model(
                model_to_eval,
                dataset.obj2id,  # 只傳 obj2id
                args, betas, alphas, alphas_cum, device
            )
            wandb.log({'val_accuracy': val_acc}, step=epoch)
            print(f"[Validation] Epoch {epoch+1}  Accuracy = {val_acc:.4f}")

            if val_acc > best_eval_metric:
                best_eval_metric = val_acc
                # 儲存 EMA 權重與 optimizer state
                ema_state = (
                    model_to_eval.module.state_dict()
                    if isinstance(model_to_eval, nn.DataParallel)
                    else model_to_eval.state_dict()
                )
                torch.save({
                    'epoch': epoch + 1,
                    'ema_model_state_dict': ema_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_eval_metric
                }, os.path.join(args.out_dir, 'best_model.pth'))
                print(f"→ Saved best_model.pth at epoch {epoch+1}, val_acc={val_acc:.4f}")

        # --- 每個 epoch 儲存完整 checkpoint ---
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch{epoch+1}.pth')
        model_state = (
            model.module.state_dict() if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_eval_metric
        }
        if args.use_ema:
            ckpt['ema_model_state_dict'] = (
                ema_model.module.state_dict() if isinstance(ema_model, nn.DataParallel)
                else ema_model.state_dict()
            )
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        wandb.save(ckpt_path)  # 可選：上傳到 W&B

        # Save checkpoint
        # torch.save({
        #     'model': model.state_dict(),
        #     'opt': optimizer.state_dict()
        # }, (args.ckpt + f'{str(epoch)}.pth'))

# ===== Sampling & Evaluation =====
def sample(args):
    # 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CLEVRDataset(args.train_json, args.obj_json, args.image_dir)
    model = ConditionalUNet(num_classes=len(dataset.obj2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    betas = get_beta_schedule(args.timesteps).to(device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    with open(args.test_json, 'r') as f:
        tests = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    imgs_list = []
    for cond in tests:
        y = torch.zeros(1, len(dataset.obj2id), device=device)
        for lbl in cond:
            y[0, dataset.obj2id[lbl]] = 1
        x = torch.randn(1, 3, 64, 64, device=device)
        for i in reversed(range(args.timesteps)):
            t = torch.tensor([i], device=device)
            eps = model(x, t, y)
            beta, alpha = betas[i], alphas[i]
            cum = alphas_cum[i]
            x = (1/torch.sqrt(alpha)) * (x - (beta/torch.sqrt(1-cum)) * eps)
            if i > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        imgs_list.append(x.cpu())

    grid = make_grid(torch.cat(imgs_list, 0), nrow=8)
    save_image((grid * 0.5 + 0.5), os.path.join(args.out_dir, 'samples.png'))

# ===== Main =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'sample'], required=True)
    parser.add_argument('--train_json', default='./train.json')
    parser.add_argument('--test_json', default='./test.json')
    parser.add_argument('--obj_json', default='./objects.json')
    parser.add_argument('--image_dir', default='./iclevr')
    parser.add_argument('--out_dir', default='./results')
    parser.add_argument('--ckpt', default='./checkpoint_ddpm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--classifier_guidance', action='store_true')
    parser.add_argument('--guidance_weight', type=float, default=0.1)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)