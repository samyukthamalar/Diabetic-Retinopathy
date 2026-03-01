import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

from config_classification import Config
from train_classification import get_model, get_transforms

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, auc, save_path):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, dataset, device, save_path, num_samples=8):
    """Visualize sample predictions"""
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            output = model(image_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = prob[0][pred].item()
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            ax.imshow(img_display)
            true_label = dataset.classes[label]
            pred_label = dataset.classes[pred]
            color = 'green' if pred == label else 'red'
            ax.set_title(f'True: {true_label}\\nPred: {pred_label} ({confidence:.2f})', color=color)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test_model():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(
        config.TEST_DIR,
        transform=get_transforms(config, train=False)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load model
    model = get_model(config)
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model_classification.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Test
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS - DR CLASSIFICATION")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*50)
    
    # Save metrics
    with open(os.path.join(config.RESULTS_DIR, 'test_metrics_classification.txt'), 'w') as f:
        f.write("TEST RESULTS - DR CLASSIFICATION\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, config.CLASS_NAMES,
                         os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, auc,
                  os.path.join(config.RESULTS_DIR, 'roc_curve.png'))
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, device,
                         os.path.join(config.RESULTS_DIR, 'sample_predictions.png'))
    
    print(f"\\n✓ Results saved to {config.RESULTS_DIR}")

if __name__ == '__main__':
    test_model()
