import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Model.ISAN import Model
from Data.dataset import get_dataloaders
from torch.optim import AdamW
from utils.training_logger import TrainingLogger
from losses.grl import GradientReverseLayer
from losses.FD_Loss_SAM import SAM, compute_focal_domain_loss
from losses.ortho_loss import orthogonality_loss
import argparse
import os

# ---------------------------
# Parameter Settings (using argparse)
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Domain Adaptation Model Training Parameters")

    # Basic training parameters
    parser.add_argument('--feature-dim', type=int, default=320, help='Feature dimension output by feature extractor')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of label categories')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--epochs', type=int, default=3500, help='Total training epochs')

    # Loss weighting parameters
    parser.add_argument('--lambda-ortho-s', type=float, default=0.01, help='Initial source orthogonality loss weight')
    parser.add_argument('--lambda-dom', type=float, default=0.5, help='Domain discriminator loss weight')
    parser.add_argument('--lambda-li', type=float, default=0.01, help='Label-independent discriminator loss weight')
    parser.add_argument('--lambda-recon', type=float, default=0.01, help='Initial reconstruction loss weight')
    parser.add_argument('--lambda-dom-local', type=float, default=0.1, help='Local domain loss weight')

    # Experiment settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save-prefix', type=str, default='model', help='Model save prefix')

    return parser.parse_args()

# Initialize parameters
args = parse_args()

# Set device using parameters
device = torch.device(args.device)

# Dynamic loss weighting function
def dynamic_weight_lambda(loss_val, ref_loss_val, base_lambda, max_ratio=1e1):
    if loss_val > max_ratio * ref_loss_val:
        return base_lambda * (max_ratio * ref_loss_val / (loss_val + 1e-18))
    return base_lambda

# Initialize subdomain attention
subdomain_attention = SAM(args.num_classes, device)


def main():
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = Model(args.feature_dim, args.num_classes)
    model.to(device)
    train_loader, target_loader = get_dataloaders()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    criterion_cls = nn.CrossEntropyLoss()

    # SWA configuration
    swa_start = int(0.95 * args.epochs)  # SWA start epoch
    swa_n = 10
    swa_state_dict = None
    best_train_loss = float('inf')
    best_state_dict = None

    # Initialize training logger (using output directory)
    train_logger = TrainingLogger(os.path.join(args.output_dir, "training_log.csv"))

    # Learning rate scheduler (using parameters)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Set modules to trainable
    model.FRFE.requires_grad_(True)
    model.domain_discriminator.requires_grad_(True)
    model.label_invariant_discriminator.requires_grad_(True)
    model.FIFE.requires_grad_(True)
    model.main_classifier.requires_grad_(True)
    model.decoder.requires_grad_(True)
    for ld in model.local_discriminators:
        ld.requires_grad_(True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        epoch_weight_sum = torch.zeros(args.num_classes)
        min_steps = min(len(train_loader), len(target_loader))
        src_iter = iter(train_loader)
        tgt_iter = iter(target_loader)

        for _ in tqdm(range(min_steps), desc=f"Epoch {epoch + 1}/{args.epochs}"):
            src_imgs, src_labels = next(src_iter)
            tgt_imgs = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            optimizer.zero_grad()

            # Feature extraction
            shared_feats_src = model.FRFE(src_imgs)
            shared_feats_tgt = model.FRFE(tgt_imgs)
            source_private_feats = model.FIFE(src_imgs)

            out_main_src = model.main_classifier(shared_feats_src)
            loss_main = criterion_cls(out_main_src, src_labels)

            # Label-independent adversarial on source domain
            reversed_feats_src = GradientReverseLayer.apply(source_private_feats, 1.0)
            out_label_inv_src = model.label_invariant_discriminator(reversed_feats_src)
            loss_label_inv = criterion_cls(out_label_inv_src, src_labels)

            # Orthogonality loss
            shared_feats_src_1d = shared_feats_src.mean(dim=-1)
            source_private_feats_1d = source_private_feats.mean(dim=-1)
            loss_ortho_src = orthogonality_loss(shared_feats_src_1d, source_private_feats_1d)

            # Global domain discriminator
            dom_labels_src = torch.zeros(src_imgs.size(0), dtype=torch.long, device=device)
            dom_labels_tgt = torch.ones(tgt_imgs.size(0), dtype=torch.long, device=device)
            feats_dom = torch.cat([shared_feats_src, shared_feats_tgt], dim=0)
            labels_dom = torch.cat([dom_labels_src, dom_labels_tgt], dim=0)
            reversed_feats_dom = GradientReverseLayer.apply(feats_dom, 1.0)
            global_dom_pred = model.domain_discriminator(reversed_feats_dom)
            loss_global_dom = criterion_cls(global_dom_pred, labels_dom)

            # Local domain discriminators
            with torch.no_grad():
                logits = model.main_classifier(shared_feats_tgt)
                probs = F.softmax(logits, dim=1)
                max_probs, tgt_pseudo_labels = probs.max(dim=1)

                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

                confidence_threshold = max(0.7, 0.9 - epoch * 0.002)
                entropy_threshold = 0.9
                confident_mask = (max_probs > confidence_threshold) & (entropy < entropy_threshold)

            shared_feats_tgt_filtered = shared_feats_tgt[confident_mask]
            tgt_pseudo_labels = tgt_pseudo_labels[confident_mask]

            if shared_feats_tgt_filtered.numel() == 0:
                weighted_local_loss = torch.tensor(0.0, device=device)
                weight_vec = torch.zeros(args.num_classes)
            else:
                weighted_local_loss, weight_vec, _ = compute_focal_domain_loss(
                    shared_feats_src, shared_feats_tgt_filtered,
                    src_labels, tgt_pseudo_labels, model, subdomain_attention)

            epoch_weight_sum += weight_vec.cpu()

            # Reconstruction
            combined_feats_src = torch.cat([shared_feats_src, source_private_feats], dim=1)
            reconstructed_src = model.decoder(combined_feats_src)
            loss_recon_src = F.mse_loss(reconstructed_src, src_imgs)
            loss_recon = loss_recon_src

            # Dynamic weight calculation
            lambda_label_inv_dynamic = dynamic_weight_lambda(loss_label_inv.item(), loss_main.item(), args.lambda_li)
            lambda_ortho_src_dynamic = dynamic_weight_lambda(loss_ortho_src.item(), loss_main.item(), args.lambda_ortho_s)
            lambda_dom_dynamic = dynamic_weight_lambda(loss_global_dom.item(), loss_main.item(), args.lambda_dom)
            lambda_recon_dynamic = dynamic_weight_lambda(loss_recon.item(), loss_main.item(), args.lambda_recon)
            lambda_local_loss_dynamic = dynamic_weight_lambda(weighted_local_loss.item(), loss_main.item(), args.lambda_dom_local)

            # Total loss (with dynamic weights)
            loss = (loss_main
                    + lambda_ortho_src_dynamic * loss_ortho_src
                    + lambda_dom_dynamic * loss_global_dom
                    + lambda_label_inv_dynamic * loss_label_inv
                    + lambda_recon_dynamic * loss_recon
                    + lambda_local_loss_dynamic * weighted_local_loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Log losses
            loss_dict = {
                'loss_main': loss_main.item(),
                'loss_diff_source': loss_ortho_src.item(),
                'loss_label_inv': loss_label_inv.item(),
                'loss_global_dom': loss_global_dom.item(),
                'loss_recon': loss_recon.item(),
                'loss_local_dom': weighted_local_loss.item(),
                'loss_total': loss.item()
            }
            train_logger.log_losses(loss_dict)

            # Memory cleanup
            del src_imgs, src_labels, tgt_imgs, shared_feats_src, source_private_feats
            del shared_feats_tgt, feats_dom, global_dom_pred
            del tgt_pseudo_labels, shared_feats_tgt_filtered
            torch.cuda.empty_cache()

        # Update learning rate
        lr_scheduler.step()

        # SWA weight averaging
        if epoch + 3490 >= swa_start:
            if swa_state_dict is None:
                swa_state_dict = copy.deepcopy(model.state_dict())
            else:
                for k in swa_state_dict.keys():
                    swa_state_dict[k] = (swa_state_dict[k].cpu() * swa_n + model.state_dict()[k].cpu()) / (swa_n + 1)
            swa_n += 10

        # Save best weights
        if loss_main.item() < best_train_loss:
            best_train_loss = loss_main.item()
            best_state_dict = copy.deepcopy(model.state_dict())

    # Save final model (using output directory and save prefix)
    best_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_best.pth")
    torch.save(best_state_dict, best_model_path)

    if swa_state_dict is not None:
        swa_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_swa.pth")
        torch.save(swa_state_dict, swa_model_path)

    train_logger.save_to_file()
    print(f"Training complete! Best training loss: {best_train_loss:.4f}")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()