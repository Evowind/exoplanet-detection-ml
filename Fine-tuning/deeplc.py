import os
import numpy as np
from astropy.io import fits
import torch
from torch.utils.data import Dataset, DataLoader
from deep_lc import DeepLC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import warnings
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TESSLightCurveDataset(Dataset):
    """Dataset pour les courbes de lumière TESS au format FITS"""
    
    def __init__(self, root_dir, classes=['CP', 'FP'], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        self.labels = []
        
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Attention: {class_dir} n'existe pas")
                continue
                
            fits_files = [f for f in os.listdir(class_dir) if f.endswith('.fits')]
            print(f"Trouvé {len(fits_files)} fichiers dans {class_name}")
            
            for fits_file in fits_files:
                self.samples.append(os.path.join(class_dir, fits_file))
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fits_path = self.samples[idx]
        label = self.labels[idx]
        
        with fits.open(fits_path) as hdul:

            data = hdul[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['SAP_FLUX']
            
            mask = ~np.isnan(time) & ~np.isnan(flux)
            time = time[mask]
            flux = flux[mask]
            
            flux = (flux - np.median(flux)) / np.std(flux)
        
        lightcurve = np.column_stack([time, flux])
        
        if self.transform:
            lightcurve = self.transform(lightcurve)
        
        return {
            'lightcurve': torch.FloatTensor(lightcurve),
            'label': torch.LongTensor([label]),
            'filename': os.path.basename(fits_path)
        }


def collate_fn(batch):
    """Fonction pour gérer les courbes de lumière de longueurs variables"""
    lightcurves = [item['lightcurve'] for item in batch]
    labels = torch.cat([item['label'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    return {
        'lightcurves': lightcurves,
        'labels': labels,
        'filenames': filenames
    }


def plot_evaluation_metrics(metrics_zeroshot, metrics_finetuned, history=None, save_dir='plots'):
    """
    Créer tous les plots de métriques pour comparaison
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
        if 'val_acc' in history and history['val_acc']:
            ax2.plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    zeroshot_values = [
        metrics_zeroshot['accuracy'],
        metrics_zeroshot['precision'],
        metrics_zeroshot['recall'],
        metrics_zeroshot['f1_score']
    ]
    finetuned_values = [
        metrics_finetuned['accuracy'],
        metrics_finetuned['precision'],
        metrics_finetuned['recall'],
        metrics_finetuned['f1_score']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, zeroshot_values, width, label='Zero-Shot', alpha=0.8)
    bars2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-Tuned', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(metrics_zeroshot['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['CP', 'FP'], yticklabels=['CP', 'FP'])
    ax1.set_title('Confusion Matrix - Zero-Shot')
    
    sns.heatmap(metrics_finetuned['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['CP', 'FP'], yticklabels=['CP', 'FP'])
    ax2.set_title('Confusion Matrix - Fine-Tuned')
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    
    for m, label, color in [(metrics_zeroshot, 'Zero-Shot', 'blue'), (metrics_finetuned, 'Fine-Tuned', 'green')]:
        if len(m['probabilities']) > 0:
            y_true = m['mapped_labels']
            y_probs = m['probabilities'][:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', color=color)
            
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for m, ax, title in [(metrics_zeroshot, ax1, 'Zero-Shot'), (metrics_finetuned, ax2, 'Fine-Tuned')]:
        if len(m['probabilities']) > 0:
            probs = m['probabilities'][:, 1]
            labels = m['mapped_labels']
            ax.hist(probs[labels == 0], bins=30, alpha=0.5, label='CP (True)', color='blue')
            ax.hist(probs[labels == 1], bins=30, alpha=0.5, label='FP (True)', color='red')
            ax.set_title(f'Probability Distribution - {title}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_distributions.png'), dpi=300)
    plt.close()
    
    print(f"\n✓ Tous les plots sauvegardés dans: {save_dir}/")


def evaluate_model(model, dataloader, device, class_names, exclude_pc=True, use_finetuned=False):
    """Évaluer le modèle et générer les prédictions avec filtrage cohérent des probabilités"""
    
    all_labels = []
    all_confidences = []
    all_filenames = []
    all_raw_predictions = [] 
    all_probabilities = [] 
    
    has_classifier = hasattr(model, 'classifier') and hasattr(model, 'feature_extractor')
    if use_finetuned and not has_classifier:
        print("ATTENTION: Modèle fine-tuné demandé mais aucun classificateur trouvé. Utilisation du modèle de base.")
        use_finetuned = False
    
    if use_finetuned:
        print("Utilisation du classificateur fine-tuné")
        model.classifier.eval()
    
    for batch in tqdm(dataloader, desc="Évaluation"):
        lightcurves = batch['lightcurves']
        labels = batch['labels'].numpy()
        filenames = batch['filenames']
        
        for i, lc in enumerate(lightcurves):
            lc_data = lc.numpy()
            
            if len(lc_data) == 0:
                continue
            
            try:
                result = model.predict(lc_data)
                
                if use_finetuned:
                    time = lc_data[:, 0]
                    flux = lc_data[:, 1]
                    feature_vec = np.array([
                        np.mean(flux), np.std(flux), np.median(flux),
                        np.max(flux) - np.min(flux),
                        np.percentile(flux, 75) - np.percentile(flux, 25),
                        len(flux), np.sum(np.diff(flux) ** 2),
                    ])
                    
                    if hasattr(model, 'scaler'):
                        feature_vec = model.scaler.transform(feature_vec.reshape(1, -1))[0]
                    
                    with torch.no_grad():
                        feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0).to(device)
                        logits = model.classifier(feature_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_class = int(np.argmax(probs))
                        
                        all_probabilities.append(probs) 
                        all_raw_predictions.append(f"FT_Class_{pred_class}")
                        all_confidences.append(float(np.max(probs)))
                else:
                    prediction = result[0] if isinstance(result, tuple) else result
                    
                    if isinstance(prediction, dict):
                        label_str = prediction.get('label', 'Unknown')
                        conf = prediction.get('probability', 0.5)
                    else:
                        label_str = str(prediction)
                        conf = 1.0
                    
                    all_raw_predictions.append(label_str)
                    all_confidences.append(conf)
                    all_probabilities.append([1.0 - conf, conf])
                
                all_labels.append(labels[i])
                all_filenames.append(filenames[i])
                
            except Exception as e:
                print(f"Erreur sur {filenames[i]}: {e}")
                continue

    def map_deeplc_to_class(deeplc_label):
        label_lower = str(deeplc_label).lower()
        if 'ft_class' in label_lower:
            return int(label_lower.split('_')[-1])
        if any(kw in label_lower for kw in ['transit', 'eb', 'eclipse']):
            return 0 
        return 1 

    initial_predictions = np.array([map_deeplc_to_class(p) for p in all_raw_predictions])
    initial_labels = np.array(all_labels)
    initial_probs = np.array(all_probabilities)

    if exclude_pc:
        cp_idx = class_names.index('CP') if 'CP' in class_names else 0
        fp_idx = class_names.index('FP') if 'FP' in class_names else 1
        
        mask = np.array([(l == cp_idx or l == fp_idx) for l in initial_labels])
        
        mapped_labels = initial_labels[mask]
        mapped_predictions = initial_predictions[mask]
        final_probs = initial_probs[mask]
        
        label_remap = {cp_idx: 0, fp_idx: 1}
        mapped_labels = np.array([label_remap.get(l, l) for l in mapped_labels])
        mapped_predictions = np.array([label_remap.get(p, p) for p in mapped_predictions])
        metric_class_names = ['CP', 'FP']
    else:
        mapped_labels = initial_labels
        mapped_predictions = initial_predictions
        final_probs = initial_probs
        metric_class_names = class_names

    accuracy = accuracy_score(mapped_labels, mapped_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        mapped_labels, mapped_predictions, average='weighted', zero_division=0
    )
    
    unique_labels = sorted(set(mapped_labels) | set(mapped_predictions))
    report = classification_report(
        mapped_labels, mapped_predictions,
        target_names=[metric_class_names[i] for i in unique_labels],
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(mapped_labels, mapped_predictions),
        'probabilities': final_probs,
        'mapped_labels': mapped_labels,
        'mapped_predictions': mapped_predictions
    }
    
    results_df = pd.DataFrame({
        'Filename': all_filenames,
        'True_Label': [class_names[l] for l in all_labels],
        'DeepLC_Prediction': all_raw_predictions,
        'Confidence_Score': all_confidences
    })
    
    return results_df, metrics

def fine_tune_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3):
    """
    Fine-tuner le modèle Deep-LC
    """
    print("\n=== Configuration du fine-tuning ===")
    print("APPROCHE: Entraînement d'un classificateur sur les features extraites par Deep-LC")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n=== Extraction des features avec Deep-LC ===")
    
    def extract_features(dataloader, desc="Extraction"):
        """Extraire les features et labels depuis le dataloader"""
        features_list = []
        labels_list = []
        
        for batch in tqdm(dataloader, desc=desc):
            lightcurves = batch['lightcurves']
            labels = batch['labels'].numpy()
            
            for i, lc in enumerate(lightcurves):
                lc_data = lc.numpy()
                
                if len(lc_data) == 0:
                    continue
                
                try:
                    result = model.predict(lc_data)
                    
                    if isinstance(result, tuple) and len(result) > 1:
                        features = result[1] if isinstance(result[1], np.ndarray) else result[0]
                    elif isinstance(result, dict):
                        features = result.get('features', result.get('embeddings', result.get('probabilities')))
                    else:
                        features = result
                    
                    if isinstance(features, str):
                        time = lc_data[:, 0]
                        flux = lc_data[:, 1]
                        feature_vec = np.array([
                            np.mean(flux),
                            np.std(flux),
                            np.median(flux),
                            np.max(flux) - np.min(flux), 
                            np.percentile(flux, 75) - np.percentile(flux, 25),  
                            len(flux), 
                            np.sum(np.diff(flux) ** 2), 
                        ])
                    elif isinstance(features, (list, np.ndarray)):
                        feature_vec = np.array(features).flatten()
                    else:
                        time = lc_data[:, 0]
                        flux = lc_data[:, 1]
                        feature_vec = np.array([
                            np.mean(flux), np.std(flux), np.median(flux),
                            np.max(flux) - np.min(flux),
                            np.percentile(flux, 75) - np.percentile(flux, 25),
                            len(flux), np.sum(np.diff(flux) ** 2),
                        ])
                    
                    features_list.append(feature_vec)
                    labels_list.append(labels[i])
                    
                except Exception as e:
                    time = lc_data[:, 0]
                    flux = lc_data[:, 1]
                    feature_vec = np.array([
                        np.mean(flux), np.std(flux), np.median(flux),
                        np.max(flux) - np.min(flux),
                        np.percentile(flux, 75) - np.percentile(flux, 25),
                        len(flux), np.sum(np.diff(flux) ** 2),
                    ])
                    features_list.append(feature_vec)
                    labels_list.append(labels[i])
        
        return np.array(features_list), np.array(labels_list)
    
    X_train, y_train = extract_features(train_loader, "Extraction train")
    print(f"Features train: {X_train.shape}, Labels train: {y_train.shape}")
    print(f"Distribution des labels train: {np.bincount(y_train)}")
    
    if val_loader is not None:
        X_val, y_val = extract_features(val_loader, "Extraction val")
        print(f"Features val: {X_val.shape}, Labels val: {y_val.shape}")
        print(f"Distribution des labels val: {np.bincount(y_val)}")
    
    print("\n=== Normalisation des features ===")
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if val_loader is not None:
        X_val_scaled = scaler.transform(X_val)
    
    print(f"Features normalisées - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
    
    import pickle
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler sauvegardé: feature_scaler.pkl")
    
    print("\n=== Entraînement du classificateur ===")
    
    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))
    
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, num_classes)
    ).to(device)
    
    print(f"Classificateur: {input_dim} → 128 → 64 → {num_classes}")
    print(f"Paramètres entraînables: {sum(p.numel() for p in classifier.parameters()):,}")
    
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts)).to(device)
    print(f"Poids des classes: {class_weights.cpu().numpy()}")
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    if val_loader is not None:
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    batch_size = 32
    num_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size else 0)
    
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        indices = torch.randperm(len(X_train))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            outputs = classifier(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        avg_train_loss = train_loss / num_batches
        train_acc = 100 * correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        if val_loader is not None:
            classifier.eval()
            with torch.no_grad():
                outputs = classifier(X_val_tensor)
                val_loss = criterion(outputs, y_val_tensor).item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc = 100 * (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(classifier.state_dict(), 'best_finetuned_classifier.pth')
                print("✓ Meilleur classificateur sauvegardé!")
        
        print("-" * 50)
    
    torch.save(classifier.state_dict(), 'final_classifier.pth')
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Entraînement terminé")
    print("✓ Fichiers sauvegardés:")
    print("  - best_finetuned_classifier.pth")
    print("  - final_classifier.pth")
    print("  - training_history.json")
    
    model.classifier = classifier
    model.scaler = scaler
    model.feature_extractor = True
    
    return model


def main():
    DATASET_ROOT = "dataset_fits"
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Utilisation de: {DEVICE}")
    
    train_classes = ['CP', 'FP']
    test_classes = ['CP', 'PC', 'FP']
    
    print("\n=== Chargement des données ===")
    train_dataset = TESSLightCurveDataset(
        os.path.join(DATASET_ROOT, 'train'),
        classes=train_classes
    )
    
    test_dataset = TESSLightCurveDataset(
        os.path.join(DATASET_ROOT, 'test'),
        classes=test_classes
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    MODEL_PATH = "combined_7_conformal_calibrated.ckpt"  
    
    print("\n=== ÉVALUATION ZERO-SHOT ===")
    print(f"Chargement du modèle depuis: {MODEL_PATH}")
    
    model_zeroshot = DeepLC(combined_model=MODEL_PATH)
    
    results_zeroshot, metrics_zeroshot = evaluate_model(
        model_zeroshot,
        test_loader,
        DEVICE,
        test_classes,
        exclude_pc=True
    )
    
    print("\n--- Résultats Zero-Shot ---")
    print(f"Accuracy: {metrics_zeroshot['accuracy']:.4f}")
    print(f"Precision: {metrics_zeroshot['precision']:.4f}")
    print(f"Recall: {metrics_zeroshot['recall']:.4f}")
    print(f"F1-Score: {metrics_zeroshot['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics_zeroshot['classification_report'])
    
    results_zeroshot.to_csv('predictions_zeroshot.csv', index=False)
    print("\nPrédictions sauvegardées: predictions_zeroshot.csv")
    
    print("\n=== FINE-TUNING ===")
    model_finetuned = DeepLC(combined_model=MODEL_PATH)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader_ft = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader_ft = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train subset: {len(train_subset)} samples")
    print(f"Val subset: {len(val_subset)} samples")
    
    model_finetuned = fine_tune_model(
        model_finetuned,
        train_loader_ft,
        val_loader=val_loader_ft,
        epochs=50,
        lr=1e-3 
    )
    
    print("\n=== ÉVALUATION FINE-TUNED ===")
    results_finetuned, metrics_finetuned = evaluate_model(
        model_finetuned,
        test_loader,
        DEVICE,
        test_classes,
        exclude_pc=True,
        use_finetuned=True  
    )
    
    print("\n--- Résultats Fine-Tuned ---")
    print(f"Accuracy: {metrics_finetuned['accuracy']:.4f}")
    print(f"Precision: {metrics_finetuned['precision']:.4f}")
    print(f"Recall: {metrics_finetuned['recall']:.4f}")
    print(f"F1-Score: {metrics_finetuned['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics_finetuned['classification_report'])
    
    results_finetuned.to_csv('predictions_finetuned.csv', index=False)
    print("\nPrédictions sauvegardées: predictions_finetuned.csv")
    
    print("\n=== GÉNÉRATION DES VISUALISATIONS ===")
    
    history = None
    if os.path.exists('training_history.json'):
        with open('training_history.json', 'r') as f:
            history = json.load(f)
    
    plot_evaluation_metrics(
        metrics_zeroshot=metrics_zeroshot,
        metrics_finetuned=metrics_finetuned,
        history=history,
        save_dir='plots'
    )
    
    print("\n=== COMPARAISON ZERO-SHOT vs FINE-TUNED ===")
    comparison = pd.DataFrame({
        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Zero-Shot': [
            metrics_zeroshot['accuracy'],
            metrics_zeroshot['precision'],
            metrics_zeroshot['recall'],
            metrics_zeroshot['f1_score']
        ],
        'Fine-Tuned': [
            metrics_finetuned['accuracy'],
            metrics_finetuned['precision'],
            metrics_finetuned['recall'],
            metrics_finetuned['f1_score']
        ]
    })
    
    print(comparison.to_string(index=False))
    comparison.to_csv('comparison_results.csv', index=False)
    
    print("\n=== TERMINÉ ===")
    print("Fichiers générés:")
    print("- predictions_zeroshot.csv: Prédictions zero-shot avec scores de confiance")
    print("- predictions_finetuned.csv: Prédictions fine-tuned avec scores de confiance")
    print("- comparison_results.csv: Comparaison des métriques")
    print("- best_model.pth: Meilleur modèle fine-tuné")


if __name__ == "__main__":
    main()