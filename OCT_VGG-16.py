from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# ---------- Extra metrics (sklearn) ----------
try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_score,
        recall_score,
        f1_score,
        log_loss,
        roc_curve,
        auc,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as e:
    raise ImportError(
        "This script needs scikit-learn for the requested metrics/ROC/AUC. "
        "Please install it first, e.g. `pip install scikit-learn`, then rerun."
    ) from e

plt.ion()

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

device = torch.device("cuda" if use_gpu else "cpu")

data_dir = 'OCT2017_'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

# NOTE: keeping your original settings; if you want deterministic evaluation,
# you can set shuffle=False for VAL/TEST.
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=32,
        shuffle=True, num_workers=0
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(class_names)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


# Get a batch of training data
inputs, classes = next(iter(dataloaders[TRAIN]))
show_databatch(inputs, classes)


def visualize_model(vgg, num_images=6):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[TEST]):
        inputs, labels = data
        size = inputs.size()[0]

        if use_gpu:
            with torch.no_grad():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            with torch.no_grad():
                inputs, labels = Variable(inputs), Variable(labels)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]

        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)

        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


def _safe_div(n, d):
    return float(n) / float(d) if d != 0 else 0.0


def _specificity_from_cm(cm):
    """Compute per-class specificity for multi-class confusion matrix."""
    n_classes = cm.shape[0]
    total = cm.sum()
    specs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specs.append(_safe_div(tn, tn + fp))
    return np.array(specs, dtype=float)


def get_predictions(vgg, dataloader):
    """Collect y_true, y_pred, y_prob (softmax) for a given dataloader."""
    vgg.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = vgg(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            del inputs, labels, outputs, probs, preds
            torch.cuda.empty_cache()

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])

    return y_true, y_pred, y_prob


def plot_roc_curves(y_true, y_prob, class_names, title_prefix="Test"):
    """Plot one-vs-rest ROC curves for multi-class classification."""
    n_classes = len(class_names)
    if n_classes <= 1:
        print("ROC/AUC requires at least 2 classes. Skipping plot.")
        return

    # Binarize labels for OvR ROC
    classes = list(range(n_classes))
    y_true_bin = label_binarize(y_true, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        # Guard against missing positives for a class
        if y_true_bin[:, i].sum() == 0:
            fpr[i], tpr[i], roc_auc[i] = None, None, None
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        roc_auc["micro"] = None

    plt.figure()
    for i, name in enumerate(class_names):
        if fpr.get(i) is None:
            continue
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC = {roc_auc[i]:.4f})")

    if roc_auc.get("micro") is not None:
        plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC = {roc_auc['micro']:.4f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title_prefix} ROC Curves")
    plt.legend(loc="lower right")
    plt.show()


def eval_model(vgg, criterion, split=TEST, plot_roc=True):
    """Evaluate model with extended metrics on the given split."""
    since = time.time()

    dataloader = dataloaders[split]
    dataset_size = len(dataloader.dataset)

    print(f"Evaluating model on {split}")
    print('-' * 10)

    vgg.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = vgg(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            del inputs, labels, outputs
            torch.cuda.empty_cache()

    avg_loss = running_loss / dataset_size if dataset_size else 0.0

    # Collect predictions/probabilities for metrics
    y_true, y_pred, y_prob = get_predictions(vgg, dataloader)

    # Basic metrics
    accuracy = (y_pred == y_true).mean() if y_true.size else 0.0
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Sensitivity == Recall per class
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    sensitivity_per_class = recall_per_class

    # Specificity per class
    specificity_per_class = _specificity_from_cm(cm)

    # Log loss (multi-class)
    try:
        ll = log_loss(y_true, y_prob, labels=list(range(len(class_names))))
    except Exception:
        ll = float('nan')

    # AUC (multi-class OvR)
    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except Exception:
        auc_macro, auc_weighted = float('nan'), float('nan')

    elapsed_time = time.time() - since

    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print(f"Avg loss ({split}): {avg_loss:.4f}")
    print(f"Classification accuracy ({split}): {accuracy:.4f}")
    print()
    print("Confusion matrix:")
    print(cm)
    print()

    print("Precision (macro): {:.4f}".format(precision_macro))
    print("Recall (macro): {:.4f}".format(recall_macro))
    print("F1 score (macro): {:.4f}".format(f1_macro))
    print("Log loss: {}".format(f"{ll:.4f}" if not np.isnan(ll) else "nan"))
    print()

    # Per-class sensitivity/specificity
    print("Per-class Sensitivity (Recall):")
    for name, val in zip(class_names, sensitivity_per_class):
        print(f"  {name}: {val:.4f}")

    print("Per-class Specificity:")
    for name, val in zip(class_names, specificity_per_class):
        print(f"  {name}: {val:.4f}")

    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"AUC (macro, OvR): {auc_macro if not np.isnan(auc_macro) else 'nan'}")
    print(f"AUC (weighted, OvR): {auc_weighted if not np.isnan(auc_weighted) else 'nan'}")
    print('-' * 10)

    # Plot ROC curves
    if plot_roc and y_true.size and len(class_names) > 1:
        plot_roc_curves(y_true, y_prob, class_names, title_prefix=split.capitalize())

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "log_loss": ll,
        "confusion_matrix": cm,
        "sensitivity_per_class": sensitivity_per_class,
        "specificity_per_class": specificity_per_class,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
    }


# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn()
# Your original path is kept
vgg16.load_state_dict(torch.load("vgg16bn/vgg16_bn-6c64b313.pth", map_location=device))
print(vgg16.classifier[6].out_features)  # 1000


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with outputs = num classes
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
print(vgg16)

# If you want to train the model for more than 2 epochs, set this to True after the first run
resume_training = False

if resume_training:
    print("Loading pretrained model..")
    vgg16.load_state_dict(torch.load('vgg16-transfer-learning-pytorch/VGG16_v2-OCT_Retina.pt', map_location=device))
    print("Loaded!")

vgg16 = vgg16.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.1)

print("Test before training")
eval_model(vgg16, criterion, split=TEST, plot_roc=True)

visualize_model(vgg16)  # test before training


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        train_dataset_length = len(dataloaders[TRAIN].dataset)
        print(f"Length of training dataset: {train_dataset_length}")
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        loss_train = 0.0
        loss_val = 0.0
        acc_train = 0
        acc_val = 0

        # ---- Train ----
        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rtraining batch {}/{}".format(i, int(train_batches)), end='', flush=True)

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = vgg(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item() * inputs.size(0)
            acc_train += torch.sum(preds == labels.data).item()

            del inputs, labels, outputs, probs, preds
            torch.cuda.empty_cache()

        print()
        avg_loss_train = loss_train / len(dataloaders[TRAIN].dataset)
        avg_acc_train = acc_train / len(dataloaders[TRAIN].dataset)

        # ---- Val ----
        vgg.train(False)
        vgg.eval()

        with torch.no_grad():
            for i, data in enumerate(dataloaders[VAL]):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = vgg(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.item() * inputs.size(0)
                acc_val += torch.sum(preds == labels.data).item()

                del inputs, labels, outputs, probs, preds
                torch.cuda.empty_cache()

        avg_loss_val = loss_val / len(dataloaders[VAL].dataset)
        avg_acc_val = acc_val / len(dataloaders[VAL].dataset)

        # scheduler step per epoch
        if scheduler is not None:
            scheduler.step()

        print()
        print("Epoch {} result: ".format(epoch + 1))
        print("Avg loss (train): {:.4f}".format(avg_loss_train))
        print("Avg acc (train): {:.4f}".format(avg_acc_train))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc (val): {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg


vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

# Save trained weights
torch.save(vgg16.state_dict(), 'VGG16_OCT_Retina_trained_model.pt')

# Evaluate after training with full metrics + ROC/AUC
metrics_out = eval_model(vgg16, criterion, split=TEST, plot_roc=True)

# Optional: visualize more predictions
visualize_model(vgg16, num_images=32)

