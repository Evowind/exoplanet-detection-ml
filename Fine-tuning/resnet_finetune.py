import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from astropy.io import fits
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit

SAVE_DIR = "resnet_finetune"
os.makedirs(SAVE_DIR, exist_ok=True)
CLASS_NAMES = ['FP', 'CP'] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExoplanetFitsDataset(Dataset):
    def __init__(self, root_dir, classes=CLASS_NAMES, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder): continue
            for f in os.listdir(cls_folder):
                if f.endswith(('.fits', '.fit')):
                    self.samples.append((os.path.join(cls_folder, f), idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with fits.open(path) as hdul:
                table_data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
                names = table_data.columns.names
                flux = table_data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in names else (table_data['SAP_FLUX'] if 'SAP_FLUX' in names else table_data[names[0]])
                flux = np.nan_to_num(flux)
                side = int(np.sqrt(len(flux)))
                if side == 0: data_2d = np.zeros((224, 224), dtype=np.float32)
                else:
                    data_2d = flux[:side*side].reshape((side, side))
                    v_min, v_max = np.percentile(data_2d, [1, 99])
                    data_2d = np.clip((data_2d - v_min) / (v_max - v_min + 1e-8), 0, 1)
        except: data_2d = np.zeros((224, 224), dtype=np.float32)

        img_uint8 = (data_2d * 255).astype(np.uint8)
        from PIL import Image
        img = Image.fromarray(img_uint8).convert('RGB').resize((224, 224))
        if self.transform: img = self.transform(img)
        else: img = transforms.ToTensor()(img)
        return img, label

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset_raw = ExoplanetFitsDataset('dataset_fits/train')
labels = [s[1] for s in full_dataset_raw.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

train_ds = Subset(ExoplanetFitsDataset('dataset_fits/train', transform=train_transform), train_idx)
val_ds = Subset(ExoplanetFitsDataset('dataset_fits/train', transform=val_transform), val_idx)
test_ds = ExoplanetFitsDataset('dataset_fits/test', transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': [], 't_auc': [], 'v_auc': []}
best_v_loss = float('inf')

for epoch in range(50):
    model.train()
    t_loss, t_correct, t_total, t_probs, t_y = 0, 0, 0, [], []
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad(); out = model(imgs); loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        t_loss += loss.item(); t_total += lbls.size(0)
        t_correct += out.max(1)[1].eq(lbls).sum().item()
        t_probs.extend(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()); t_y.extend(lbls.cpu().numpy())

    model.eval()
    v_loss, v_correct, v_total, v_probs, v_y = 0, 0, 0, [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs); v_loss += criterion(out, lbls).item(); v_total += lbls.size(0)
            v_correct += out.max(1)[1].eq(lbls).sum().item()
            v_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); v_y.extend(lbls.cpu().numpy())

    v_loss_avg = v_loss/len(val_loader)
    history['t_loss'].append(t_loss/len(train_loader)); history['v_loss'].append(v_loss_avg)
    history['t_acc'].append(t_correct/t_total); history['v_acc'].append(v_correct/v_total)
    history['t_auc'].append(auc(*roc_curve(t_y, t_probs)[:2]))
    history['v_auc'].append(auc(*roc_curve(v_y, v_probs)[:2]))
    
    scheduler.step(v_loss_avg)
    if v_loss_avg < best_v_loss:
        best_v_loss = v_loss_avg
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
    print(f"Epoch {epoch+1:02d} | Val Loss: {v_loss_avg:.3f} | Val AUC: {history['v_auc'][-1]:.3f}")

model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
model.eval()
final_probs, final_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = model(imgs.to(device))
        final_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        final_labels.extend(lbls.numpy())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

for i, m in enumerate(['loss', 'acc', 'auc']):
    axes[0, i].plot(history[f't_{m}'], label='Train'); axes[0, i].plot(history[f'v_{m}'], label='Val')
    axes[0, i].set_title(f"{m.upper()} History"); axes[0, i].legend()

cm = confusion_matrix(final_labels, [1 if p > 0.5 else 0 for p in final_probs])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1, 0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(final_labels, final_probs)
axes[1, 1].plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.4f}'); axes[1, 1].set_title("ROC Curve"); axes[1, 1].legend()

axes[1, 2].hist([final_probs[i] for i,l in enumerate(final_labels) if l==1], alpha=0.5, label='CP', color='orange')
axes[1, 2].hist([final_probs[i] for i,l in enumerate(final_labels) if l==0], alpha=0.5, label='FP', color='steelblue')
axes[1, 2].set_title("Prob. Distribution"); axes[1, 2].legend()

plt.savefig(os.path.join(SAVE_DIR, "final_plots.png"))
plt.show()