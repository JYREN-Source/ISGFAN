import os, itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from Model.ISAN import Model
from Data.dataset import get_dataloaders, SourceDataset, data_manager

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

# Define global visualization parameters
CLASS_NAMES = [
    "ball_07", "ball_14", "ball_21",
    "inner_07", "inner_14", "inner_21",
    "normal",
    "outer_07", "outer_14", "outer_21"
]
DISPLAY_NAMES = [
    "B1", "B2", "B3",
    "I1", "I2", "I3",
    "N",
    "O1", "O2", "O3"
]
SOURCE_DIR = os.path.join("Data", "1HP")  # Source domain data for t-SNE
VISUALIZATIONS_DIR = "visualizations"  # Directory for all visualization outputs


# ------------------------------------------------------------

def test_target_domain(model_path, batch_size=32, num_workers=4):
    """Test model performance on target domain and generate diagnostic plots"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure visualizations directory exists
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Initialize model
    model = Model(feature_dim=320, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get test dataloader
    get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    test_loader = data_manager.test_loader  # Labeled test set

    # Run predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seg, label in test_loader:
            seg, label = seg.to(device), label.to(device)
            logits = model.main_classifier(model.FRFE(seg))
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate and print metrics
    acc = (all_preds == all_labels).mean()
    print("\n[Test Results]")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, zero_division=0))

    # Generate confusion matrix with path in visualizations directory
    conf_matrix_path = os.path.join(VISUALIZATIONS_DIR, "test_confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_preds),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=DISPLAY_NAMES,
                yticklabels=DISPLAY_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted");
    plt.ylabel("True")
    plt.xticks(rotation=45);
    plt.yticks(rotation=0)
    plt.tight_layout();
    plt.savefig(conf_matrix_path);
    plt.close()
    print(f"Confusion matrix saved to {conf_matrix_path}")

    # Generate t-SNE plot
    plot_tsne(model, device, test_loader, max_samples=2100)


# --------------------------------------------------------------------
def plot_tsne(model, device, test_loader, max_samples=2100):
    """Generate t-SNE visualization of feature distribution"""
    # Prepare source dataset
    source_ds = SourceDataset(SOURCE_DIR)
    if len(source_ds) > max_samples:
        idx = np.random.choice(len(source_ds), max_samples, replace=False)
        source_ds = torch.utils.data.Subset(source_ds, idx.tolist())

    source_loader = torch.utils.data.DataLoader(
        source_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features from both domains
    feats, labels, domains = [], [], []  # domains: 0=source, 1=target
    model.eval()
    with torch.no_grad():
        # Process source domain
        for seg, lab in source_loader:
            seg = seg.to(device)
            f = model.FRFE(seg).flatten(1)
            feats.append(f.cpu().numpy())
            labels.append(lab.numpy())
            domains.append(np.zeros_like(lab.numpy()))
        # Process target domain
        for seg, lab in itertools.islice(test_loader, None):
            seg = seg.to(device)
            f = model.FRFE(seg).flatten(1)
            feats.append(f.cpu().numpy())
            labels.append(lab.numpy())
            domains.append(np.ones_like(lab.numpy()))

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    domains = np.concatenate(domains)

    # Compute t-SNE embeddings
    tsne = TSNE(n_components=2, perplexity=30, init='pca',
                n_iter=1000, random_state=42, verbose=1)
    feats_2d = tsne.fit_transform(feats)

    # Create color palette
    ukiyoe_autumn = [
        "#d5c2de", "#4ab09e", "#d07b90", "#e2c38c", "#9b3766",
        "#a94889", "#c46a9a", "#8cc776", "#e44d49", "#43369c"
    ]
    color_set = sns.color_palette(ukiyoe_autumn)

    # Plot t-SNE results with path in visualizations directory
    tsne_path = os.path.join(VISUALIZATIONS_DIR, "tsne_source_target.png")
    plt.figure(figsize=(8, 6))
    for cid in range(len(CLASS_NAMES)):
        # Source domain points (circles)
        m = (labels == cid) & (domains == 0)
        if m.any():
            plt.scatter(feats_2d[m, 0], feats_2d[m, 1],
                        color=color_set[cid], marker='o',
                        s=18, alpha=1.0, edgecolors='none')

        # Target domain points (triangles)
        m = (labels == cid) & (domains == 1)
        if m.any():
            plt.scatter(feats_2d[m, 0], feats_2d[m, 1],
                        color=color_set[cid], marker='^',
                        s=18, alpha=1.0, edgecolors='none')

    plt.tight_layout()
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {tsne_path}")

    # Generate separate legend in visualizations directory
    legend_path = os.path.join(VISUALIZATIONS_DIR, "tsne_legend.png")
    save_tsne_legend(color_set, DISPLAY_NAMES, legend_path)


# --------------------------------------------------------------------
def save_tsne_legend(color_set, display_names, save_path):
    """Create a standalone legend for t-SNE plot"""
    n_cls = len(display_names)

    fig, ax = plt.subplots(figsize=(0.9 * n_cls, 1.6))
    ax.axis('off')

    # Position markers and labels
    y_src, y_tgt = 0.30, 0
    for i, (c, name) in enumerate(zip(color_set, display_names)):
        ax.scatter(i, y_src, color=c, marker='o', s=90)
        ax.text(i, y_src + 0.15, name, ha='center', va='bottom', fontsize=18)
        ax.scatter(i, y_tgt, color=c, marker='^', s=90)

    # Add domain labels
    ax.text(-0.7, y_src, "Source", ha='right', va='center', fontsize=18)
    ax.text(-0.7, y_tgt, "Target", ha='right', va='center', fontsize=18)

    ax.set_xlim(-1, n_cls)
    ax.set_ylim(-0.4, 0.9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Legend figure saved to {save_path}")


# --------------------------------------------------------------------

if __name__ == "__main__":
    # Update model path to match new training output directory
    test_target_domain(
        model_path=os.path.join("results", "model_best.pth"),
        batch_size=32
    )