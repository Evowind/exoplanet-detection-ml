"""
Multi-Architecture pour la Détection d'Exoplanètes
Supporte: ResNet, LSTM, TCN, Transformer
Analyse des courbes de luminosité stellaire avec gestion du déséquilibre
Chargement depuis fichiers FITS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
import argparse
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# Configuration
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# CHARGEMENT DES DONNÉES FITS
# ============================================================================

def load_fits_file(filepath):
    """
    Charge un fichier FITS et extrait la courbe de lumière (flux)
    
    Args:
        filepath: Chemin vers le fichier FITS
        
    Returns:
        numpy array contenant les valeurs de flux
    """
    try:
        with fits.open(filepath) as hdul:
            # Essayer différentes extensions communes
            for ext in range(len(hdul)):
                data = hdul[ext].data
                if data is not None:
                    # Chercher les colonnes de flux possibles
                    flux_columns = ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX', 'flux']
                    
                    for col in flux_columns:
                        if hasattr(data, col):
                            flux = data[col]
                            # Nettoyer les NaN et Inf
                            flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
                            return flux
                    
                    # Si les colonnes ne sont pas trouvées, essayer d'utiliser les données directement
                    if isinstance(data, np.ndarray):
                        if data.ndim == 1:
                            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                        elif data.ndim == 2:
                            # Prendre la première colonne
                            flux = data[:, 0] if data.shape[1] > 0 else data[0, :]
                            return np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Attention: Aucune donnée de flux trouvée dans {filepath}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None


from sklearn.model_selection import train_test_split

def load_fits_dataset(base_path, split='train', target_length=3197, test_size=0.2):
    """
    Charge les fichiers FITS depuis:
    dataset_fits/
      ├── CP/
      └── FP/
    et effectue un split train/test
    """

    base_path = Path(base_path)
    cp_path = base_path / 'CP'
    fp_path = base_path / 'FP'

    if not cp_path.exists() or not fp_path.exists():
        raise ValueError("Les dossiers CP/ et FP/ sont introuvables")

    X = []
    y = []

    # CP = 1
    for f in tqdm(list(cp_path.glob("*.fits")), desc="Chargement CP"):
        flux = load_fits_file(f)
        if flux is not None and len(flux) > 50:
            X.append(normalize_length(flux, target_length))
            y.append(1)

    # FP = 0
    for f in tqdm(list(fp_path.glob("*.fits")), desc="Chargement FP"):
        flux = load_fits_file(f)
        if flux is not None and len(flux) > 50:
            X.append(normalize_length(flux, target_length))
            y.append(0)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Aucune donnée FITS valide chargée")

    # SPLIT ICI
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    if split == 'train':
        return X_train, y_train
    elif split == 'test':
        return X_test, y_test
    else:
        raise ValueError("split doit être 'train' ou 'test'")


def normalize_length(flux, target_length):
    """
    Normalise la longueur d'une série temporelle
    
    Args:
        flux: Série temporelle originale
        target_length: Longueur cible
        
    Returns:
        Série normalisée
    """
    current_length = len(flux)
    
    if current_length == target_length:
        return flux
    elif current_length > target_length:
        # Sous-échantillonnage
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return flux[indices]
    else:
        # Sur-échantillonnage par interpolation
        old_indices = np.arange(current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        return np.interp(new_indices, old_indices, flux)

def robust_normalize(flux):
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad == 0: mad = 1e-6
    return (flux - median) / mad


# ============================================================================
# ARCHITECTURES DE MODÈLES
# ============================================================================

def residual_block(x, filters, kernel_size=3, stride=1, name=''):
    """Bloc résiduel avec connexion skip"""
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    x = layers.Dropout(0.3, name=f'{name}_dropout1')(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same',
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                  name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)
    return x


def build_resnet(input_shape):
    """Architecture ResNet optimisée pour déséquilibre de classes"""
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Convolution initiale
    x = layers.Conv1D(32, 7, strides=2, padding='same', 
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)
    
    # Blocs résiduels
    x = residual_block(x, 32, name='stage1_block1')
    x = residual_block(x, 32, name='stage1_block2')
    
    x = residual_block(x, 64, stride=2, name='stage2_block1')
    x = residual_block(x, 64, name='stage2_block2')
    
    x = residual_block(x, 128, stride=2, name='stage3_block1')
    x = residual_block(x, 128, name='stage3_block2')
    
    # Classification head
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout_final2')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs, outputs, name='ResNet_Exoplanet')
    return model


def build_lstm(input_shape):
    """Architecture LSTM pour séries temporelles"""
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Couches LSTM empilées
    x = layers.LSTM(128, return_sequences=True, 
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    name='lstm1')(inputs)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    
    x = layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    name='lstm2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    x = layers.LSTM(32, return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(1e-4),
                    name='lstm3')(x)
    x = layers.Dropout(0.4, name='dropout3')(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs, outputs, name='LSTM_Exoplanet')
    return model


def build_tcn(input_shape, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16]):
    """Architecture Temporal Convolutional Network (TCN)"""
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # Blocs TCN avec convolutions dilatées
    for i, dilation in enumerate(dilations):
        # Convolution causale dilatée
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f'tcn_conv_{i}'
        )(x)
        conv = layers.BatchNormalization(name=f'tcn_bn_{i}')(conv)
        conv = layers.Dropout(0.3, name=f'tcn_dropout_{i}')(conv)
        
        # Connexion résiduelle si les dimensions correspondent
        if x.shape[-1] != num_filters:
            x = layers.Conv1D(num_filters, 1, padding='same', 
                             name=f'tcn_residual_{i}')(x)
        x = layers.Add(name=f'tcn_add_{i}')([x, conv])
    
    # Global pooling et classification
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout_final2')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs, outputs, name='TCN_Exoplanet')
    return model


def build_transformer(input_shape, num_heads=4, ff_dim=128, num_blocks=3):
    """Architecture Transformer pour séries temporelles"""
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Projection initiale
    x = layers.Dense(64, name='initial_projection')(inputs)
    
    # Blocs Transformer
    for i in range(num_blocks):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=64 // num_heads,
            dropout=0.1,
            name=f'mha_{i}'
        )(x, x)
        attn_output = layers.Dropout(0.3, name=f'dropout_attn_{i}')(attn_output)
        x = layers.Add(name=f'add_attn_{i}')([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln_attn_{i}')(x)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(1e-4),
                          name=f'ffn_dense1_{i}')(x)
        ffn = layers.Dropout(0.3, name=f'dropout_ffn_{i}')(ffn)
        ffn = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4),
                          name=f'ffn_dense2_{i}')(ffn)
        x = layers.Add(name=f'add_ffn_{i}')([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln_ffn_{i}')(x)
    
    # Global pooling et classification
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout_fc1')(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4),
                     name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout_fc2')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs, outputs, name='Transformer_Exoplanet')
    return model


# ============================================================================
# FOCAL LOSS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        cross_entropy = -y_true * keras.backend.log(y_pred)
        weight = alpha * y_true * keras.backend.pow(1 - y_pred, gamma)
        
        cross_entropy_neg = -(1 - y_true) * keras.backend.log(1 - y_pred)
        weight_neg = (1 - alpha) * (1 - y_true) * keras.backend.pow(y_pred, gamma)
        
        loss = weight * cross_entropy + weight_neg * cross_entropy_neg
        return keras.backend.mean(loss)
    return focal_loss_fixed


# ============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================================

def train_model(model_type='resnet', data_path='D:/Projet/ResNet/dataset_fits', target_length=3197):
    """
    Entraîne le modèle spécifié
    
    Args:
        model_type: 'resnet', 'lstm', 'tcn', ou 'transformer'
        data_path: Chemin vers le répertoire dataset_fits
        target_length: Longueur cible pour normaliser les séries
    """
    
    print("="*60)
    print(f"{model_type.upper()} - Détection d'Exoplanètes (FITS)")
    print("="*60)
    
    # ============================================================================
    # 1. CHARGEMENT DES DONNÉES FITS
    # ============================================================================
    print("\n[1] Chargement des données FITS...")
    print(f"Chemin de base: {data_path}")
    
    try:
        X_train, y_train = load_fits_dataset(data_path, split='train', target_length=target_length)
        X_test, y_test = load_fits_dataset(data_path, split='test', target_length=target_length)
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        return
    
    # ============================================================================
    # 2. PRÉPARATION DES DONNÉES
    # ============================================================================
    print("\n[2] Préparation des données...")

    print(f"✓ Shape X_train: {X_train.shape}")
    print(f"✓ Shape X_test: {X_test.shape}")
    print(f"✓ Distribution des classes (train):")
    print(f"  - FP (Non-exoplanète): {np.sum(y_train == 0)} ({100*np.mean(y_train==0):.2f}%)")
    print(f"  - CP (Exoplanète): {np.sum(y_train == 1)} ({100*np.mean(y_train==1):.2f}%)")

    # Vérifier et remplacer les NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisation robuste : centrer sur la médiane et diviser par MAD
    X_train = np.array([robust_normalize(f) for f in X_train])
    X_test = np.array([robust_normalize(f) for f in X_test])

    # Clip pour éviter les valeurs extrêmes
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)
    print("✓ Normalisation robuste effectuée avec clipping")

    # Reshape pour Conv1D / LSTM / Transformer
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print(f"✓ Reshape final: {X_train.shape}")
    
    # ============================================================================
    # 3. CONSTRUCTION DU MODÈLE
    # ============================================================================
    print(f"\n[3] Construction de l'architecture {model_type.upper()}...")
    
    input_shape = (X_train.shape[1], 1)
    
    if model_type.lower() == 'resnet':
        model = build_resnet(input_shape)
    elif model_type.lower() == 'lstm':
        model = build_lstm(input_shape)
    elif model_type.lower() == 'tcn':
        model = build_tcn(input_shape)
    elif model_type.lower() == 'transformer':
        model = build_transformer(input_shape)
    else:
        print(f"✗ Erreur: modèle '{model_type}' non reconnu. Utilisez: resnet, lstm, tcn, ou transformer")
        return
    
    print("✓ Modèle créé")
    print(f"✓ Nombre total de paramètres: {model.count_params():,}")
    
    # ============================================================================
    # 4. COMPILATION
    # ============================================================================
    print("\n[4] Compilation du modèle...")
    
    # Calculer le poids des classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"✓ Poids des classes: {class_weight_dict}")
    
    # Metrics personnalisées
    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=METRICS
    )
    print("✓ Modèle compilé avec Focal Loss")
    
    # ============================================================================
    # 5. CALLBACKS
    # ============================================================================
    print("\n[5] Configuration des callbacks...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_type}_exoplanet.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    print("✓ Callbacks configurés")
    
    # ============================================================================
    # 6. ENTRAÎNEMENT
    # ============================================================================
    print(f"\n[6] Démarrage de l'entraînement {model_type.upper()}...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("\n✓ Entraînement terminé!")
    
    # ============================================================================
    # 7. ÉVALUATION
    # ============================================================================
    print("\n[7] Évaluation du modèle...")
    
    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0)
    
    # Tester différents seuils
    thresholds = np.arange(0.55, 0.66, 0.01)
    best_f1 = 0
    best_threshold = 0.6
    
    print("\n" + "="*60)
    print("TEST DE DIFFÉRENTS SEUILS")
    print("="*60)
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba > threshold).astype(int)
        precision = np.sum((y_pred_temp==1) & (y_test==1)) / max(np.sum(y_pred_temp==1),1)
        recall = np.sum((y_pred_temp==1) & (y_test==1)) / max(np.sum(y_test==1),1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✓ Meilleur seuil: {best_threshold} (F1={best_f1:.4f})")
    
    # Utiliser le meilleur seuil
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION (Seuil optimal)")
    print("="*60)
    print(classification_report(
        y_test, y_pred,
        target_names=['FP (Non-exoplanète)', 'CP (Exoplanète)'],
        digits=4,
        zero_division=0
    ))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMATRICE DE CONFUSION")
    print("="*60)
    print(cm)
    print(f"\nVrais Négatifs:  {cm[0,0]}")
    print(f"Faux Positifs:   {cm[0,1]}")
    print(f"Faux Négatifs:   {cm[1,0]}")
    print(f"Vrais Positifs:  {cm[1,1]}")
    
    # ============================================================================
    # 8. VISUALISATIONS
    # ============================================================================
    print("\n[8] Génération des visualisations...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    # Courbes de loss
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'{model_type.upper()} - Evolution de la Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Courbes d'accuracy
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Evolution de l\'Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AUC
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC', linewidth=2)
    plt.plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    plt.title('Evolution de l\'AUC', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Matrice de confusion
    ax4 = plt.subplot(2, 3, 4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['FP', 'CP'],
                yticklabels=['FP', 'CP'])
    plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    # Courbe ROC
    ax5 = plt.subplot(2, 3, 5)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Hasard')
    plt.title('Courbe ROC', fontsize=14, fontweight='bold')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution des prédictions
    ax6 = plt.subplot(2, 3, 6)
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='FP (Non-exoplanète)', color='blue')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='CP (Exoplanète)', color='red')
    plt.axvline(x=best_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Seuil optimal ({best_threshold})')
    plt.title('Distribution des Probabilités', fontsize=14, fontweight='bold')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_exoplanet_fits_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphiques sauvegardés: {model_type}_exoplanet_fits_results.png")
    plt.show()
    
    # ============================================================================
    # 9. SAUVEGARDE
    # ============================================================================
    print("\n[9] Sauvegarde du modèle...")
    model.save(f'{model_type}_exoplanet_fits_final.h5')
    print(f"✓ Modèle sauvegardé: {model_type}_exoplanet_fits_final.h5")
    
    import pickle
    with open(f'{model_type}_scaler_fits.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{model_type}_best_threshold_fits.pkl', 'wb') as f:
        pickle.dump(best_threshold, f)
    print("✓ Scaler et seuil sauvegardés")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"\nPerformances finales (seuil={best_threshold}):")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  F1-Score: {best_f1:.4f}")
    print("\nFichiers générés:")
    print(f"  - best_{model_type}_exoplanet.h5")
    print(f"  - {model_type}_exoplanet_fits_final.h5")
    print(f"  - {model_type}_scaler_fits.pkl")
    print(f"  - {model_type}_best_threshold_fits.pkl")
    print(f"  - {model_type}_exoplanet_fits_results.png")
    print("\n" + "="*60)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraîner un modèle de classification d\'exoplanètes à partir de fichiers FITS')
    parser.add_argument('--model', type=str, default='resnet',
                       choices=['resnet', 'lstm', 'tcn', 'transformer'],
                       help='Type de modèle à entraîner: resnet, lstm, tcn, ou transformer')
    parser.add_argument('--data-path', type=str, default=r'D:\Projet\ResNet\dataset_fits',
                       help='Chemin vers le répertoire dataset_fits')
    parser.add_argument('--target-length', type=int, default=3197,
                       help='Longueur cible pour normaliser les séries temporelles')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SYSTÈME DE DÉTECTION D'EXOPLANÈTES - MULTI-ARCHITECTURE")
    print("Chargement depuis fichiers FITS")
    print("="*60)
    print(f"\nModèle sélectionné: {args.model.upper()}")
    print(f"Chemin des données: {args.data_path}")
    print(f"Longueur cible: {args.target_length}")
    print("\nArchitectures disponibles:")
    print("  - ResNet: Réseau résiduel avec connexions skip")
    print("  - LSTM: Réseau récurrent avec mémoire long-terme")
    print("  - TCN: Réseau convolutif temporel avec dilatations")
    print("  - Transformer: Architecture basée sur l'attention")
    print("\nClasses:")
    print("  - CP (Confirmed Planet): Exoplanètes confirmées (label=1)")
    print("  - FP (False Positive): Faux positifs (label=0)")
    print("  - PC (Potential Candidate): Non utilisé")
    print("="*60 + "\n")
    
    train_model(model_type=args.model, data_path=args.data_path, target_length=args.target_length)