# ML Détection Amas de Galaxies - Image + Catalogue MAST
# Combine des données FITS (image + catalogue) pour un meilleur ML

# ============================================================================
# PARTIE 1: Installation
# ============================================================================

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installation des dépendances...")
packages = ['astropy', 'scikit-learn', 'numpy', 'pandas', 'matplotlib', 
            'scipy', 'photutils', 'opencv-python']
for pkg in packages:
    install(pkg)
print("✓ Dépendances installées\n")

# ============================================================================
# PARTIE 2: Imports
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional: photutils pour analyse morphologique
try:
    from photutils.segmentation import SegmentationImage
    from photutils.source_detection import detect_sources
    PHOTUTILS_AVAILABLE = True
except:
    PHOTUTILS_AVAILABLE = False
    print("⚠ photutils non disponible - quelques features morphologiques seront limitées\n")

# ============================================================================
# PARTIE 3: Scanning et appairage des fichiers
# ============================================================================

print("="*70)
print("SCANNING DES FICHIERS FITS (IMAGE + CATALOGUE)")
print("="*70)

# Chercher tous les répertoires I1, I2, I3, ...
base_dir = './galaxy_data_mast'
i_dirs = sorted(glob.glob(f'{base_dir}/I*'))

if not i_dirs:
    print(f"\n⚠ Pas de répertoires trouvés dans {base_dir}/")
    print("Créez une structure: galaxy_data_mast/I1/, I2/, etc. avec les fichiers FITS\n")
    i_dirs = []

print(f"\nRépertoires trouvés: {len(i_dirs)}")

# Fonction pour appairer image et catalogue
def find_fits_pairs(directory):
    """Trouve les paires image/catalogue dans un répertoire"""
    fits_files = sorted(glob.glob(f'{directory}/*.fits'))
    
    pairs = {}
    
    # Extraire timestamp commun (avant le dernier suffixe numérique)
    for f in fits_files:
        basename = os.path.basename(f)
        
        # Chercher le pattern: extraire tout avant le dernier nombre
        # Exemple: ADP.2023-01-24T10_30_25.107.fits -> ADP.2023-01-24T10_30_25
        name_no_ext = basename.replace('.fits', '')
        parts = name_no_ext.rsplit('.', 1)  # Split par le dernier point
        
        if len(parts) == 2:
            base_name = parts[0]  # ADP.2023-01-24T10_30_25
            suffix = parts[1]     # 107 ou 405
            
            if base_name not in pairs:
                pairs[base_name] = {'image': None, 'catalogue': None}
            
            # Déterminer image ou catalogue par le suffixe
            try:
                suffix_num = int(suffix)
                if suffix_num > 400:  # .405 = catalogue
                    pairs[base_name]['catalogue'] = f
                else:  # .107 = image
                    pairs[base_name]['image'] = f
            except ValueError:
                pass
    
    # Garder seulement les paires complètes
    complete_pairs = {k: v for k, v in pairs.items() 
                      if v['image'] is not None and v['catalogue'] is not None}
    
    return complete_pairs

# Scanner tous les répertoires
all_pairs = {}
for i_dir in i_dirs:
    dir_name = os.path.basename(i_dir)
    pairs = find_fits_pairs(i_dir)
    all_pairs[dir_name] = pairs
    print(f"  {dir_name}: {len(pairs)} paire(s) trouvée(s)")

# ============================================================================
# PARTIE 4: Extraction de features (Image + Catalogue)
# ============================================================================

print("\n[1] Extraction des features des images et catalogues...\n")

def extract_image_features(image_file):
    """Extrait les features visuelles d'une image FITS"""
    try:
        with fits.open(image_file) as hdul:
            # Debug: afficher les extensions
            print(f"    Debug IMAGE - Extensions: {[type(ext).__name__ for ext in hdul]}", end=' ')
            
            # Trouver l'extension SCI ou ImageHDU
            sci_data = None
            for ext in hdul:
                if isinstance(ext, fits.ImageHDU) or isinstance(ext, fits.PrimaryHDU):
                    if ext.data is not None and len(ext.data.shape) >= 2:
                        sci_data = ext.data
                        print(f"Shape: {sci_data.shape}", end=' ')
                        break
            
            if sci_data is None:
                print("| ✗ Pas de données image trouvées")
                return None
            
            # Assurez-vous que les données sont 2D
            if len(sci_data.shape) != 2:
                return None
            
            # Normaliser et limiter la taille pour performance
            sci_data = np.float32(sci_data)
            if sci_data.shape[0] > 1000 or sci_data.shape[1] > 1000:
                sci_data = sci_data[::2, ::2]  # Downsampling
            
            # Features statistiques
            features = {
                'IMAGE_MEAN': np.nanmean(sci_data),
                'IMAGE_STD': np.nanstd(sci_data),
                'IMAGE_MEDIAN': np.nanmedian(sci_data),
                'IMAGE_MAX': np.nanmax(sci_data),
                'IMAGE_MIN': np.nanmin(sci_data),
                'IMAGE_SKEWNESS': np.nanmean((sci_data - np.nanmean(sci_data))**3) / (np.nanstd(sci_data)**3) if np.nanstd(sci_data) > 0 else 0,
            }
            
            # Gradient (texture)
            try:
                grad_y = np.nanmean(np.abs(np.diff(sci_data, axis=0)))
                grad_x = np.nanmean(np.abs(np.diff(sci_data, axis=1)))
                features['IMAGE_GRADIENT'] = (grad_x + grad_y) / 2
            except:
                features['IMAGE_GRADIENT'] = 0
            
            return features
    
    except Exception as e:
        print(f"  ⚠ Erreur image {image_file}: {e}")
        return None

def extract_catalogue_features(catalogue_file):
    """Extrait les features du catalogue (source list)"""
    try:
        with fits.open(catalogue_file) as hdul:
            # Trouver la table de données
            data = None
            for ext in hdul:
                if isinstance(ext, fits.BinTableHDU):
                    data = ext.data
                    break
            
            if data is None or len(data) == 0:
                return None, []
            
            # Colonnes disponibles
            cols = data.names if hasattr(data, 'names') else []
            
            # Mapper les noms de colonnes courants
            ra_col = next((c for c in cols if 'RA' in c.upper()), None)
            dec_col = next((c for c in cols if 'DEC' in c.upper()), None)
            mag_col = next((c for c in cols if 'MAG' in c.upper()), None)
            flux_col = next((c for c in cols if 'FLUX' in c.upper() or 'FLUX_AUTO' in c.upper()), None)
            
            if not (ra_col and dec_col):
                return None, []
            
            ra = np.array(data[ra_col], dtype=float)
            dec = np.array(data[dec_col], dtype=float)
            
            # Features spatiales et photométriques
            features = {
                'N_OBJECTS': len(data),
                'RA_MEAN': np.nanmean(ra),
                'DEC_MEAN': np.nanmean(dec),
                'RA_STD': np.nanstd(ra),
                'DEC_STD': np.nanstd(dec),
                'SPATIAL_EXTENT': np.sqrt(np.nanstd(ra)**2 + np.nanstd(dec)**2),
            }
            
            # Features photométriques si disponibles
            if mag_col:
                mag = np.array(data[mag_col], dtype=float)
                features['MAG_MEAN'] = np.nanmean(mag)
                features['MAG_STD'] = np.nanstd(mag)
            
            if flux_col:
                flux = np.array(data[flux_col], dtype=float)
                flux = flux[flux > 0]
                if len(flux) > 0:
                    features['FLUX_MEDIAN'] = np.nanmedian(flux)
                    features['FLUX_STD'] = np.nanstd(np.log10(flux + 1))
            
            # Densité locale (clustering)
            distances = []
            for i in range(min(100, len(ra))):
                dists = np.sqrt((ra - ra[i])**2 + (dec - dec[i])**2)
                dists[i] = np.inf
                nearest = np.sort(dists)[:5]
                distances.extend(nearest)
            
            if distances:
                features['NEIGHBOR_DIST_MEAN'] = np.nanmean(distances)
                features['CLUSTERING_INDEX'] = 1.0 / (1.0 + np.nanmean(distances))
            
            objects_data = {
                'RA': ra,
                'DEC': dec,
            }
            
            return features, objects_data
    
    except Exception as e:
        print(f"  ⚠ Erreur catalogue {catalogue_file}: {e}")
        return None, []

# Extraire toutes les features
all_features = []
count = 0

for dir_name, pairs in all_pairs.items():
    print(f"\n{dir_name}:")
    
    for timestamp, files in pairs.items():
        image_file = files['image']
        catalogue_file = files['catalogue']
        
        if not image_file or not catalogue_file:
            print(f"  ⚠ Paire incomplète pour {timestamp}")
            continue
        
        print(f"  Traitement: {timestamp}...", end=' ')
        
        # Extraire features
        img_features = extract_image_features(image_file)
        cat_features, objects_data = extract_catalogue_features(catalogue_file)
        
        if img_features and cat_features:
            # Combiner les features
            combined = {**img_features, **cat_features}
            combined['DIRECTORY'] = dir_name
            all_features.append(combined)
            count += 1
            print("✓")
        else:
            print("✗")

print(f"\n✓ {count} paires traitées avec succès")

if len(all_features) == 0:
    print("⚠ Aucune donnée exploitable trouvée!")
    exit()

# ============================================================================
# PARTIE 5: Créer les labels et préparer les données
# ============================================================================

print("\n[2] Préparation des données pour ML...\n")

df = pd.DataFrame(all_features)

# Créer les labels (simpliste pour la démo: répertoires pairs = amas, impairs = champ)
def get_label(directory):
    if 'I1' in directory or 'I3' in directory:
        return 1  # Amas
    else:
        return 0  # Champ

df['LABEL'] = df['DIRECTORY'].apply(get_label)

print(f"Total objets: {len(df)}")
print(f"\nDistribution des classes:")
print(df['LABEL'].value_counts())
print(f"\nFeatures disponibles ({len(df.columns)-2}):")
print(df.columns.tolist())
print(f"\nAperçu des données:")
print(df.head())
print(f"\nStatistiques:")
print(df.describe())

# ============================================================================
# PARTIE 6: Entraînement du modèle
# ============================================================================

if len(df) > 5:
    print("\n[3] Entraînement du modèle ML...\n")
    
    # Sélectionner les features (exclure les colonnes non-numériques)
    feature_cols = [c for c in df.columns if c not in ['DIRECTORY', 'LABEL'] and df[c].dtype in [np.float64, np.float32, np.int64]]
    
    X = df[feature_cols].fillna(0)
    y = df['LABEL']
    
    print(f"Features utilisées ({len(feature_cols)}):")
    print(f"  {', '.join(feature_cols)}\n")
    
    # Normaliser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Entraîner
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Évaluer
    y_pred = model.predict(X_test)
    
    print("Résultats de classification:")
    print(classification_report(y_test, y_pred, 
          target_names=['Champ', 'Amas de Galaxies']))
    
    # ============================================================================
    # PARTIE 7: Visualisations
    # ============================================================================
    
    print("\nGénération des visualisations...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    axes[0, 0].imshow(cm, cmap='Blues')
    axes[0, 0].set_title('Matrice de Confusion', fontweight='bold')
    axes[0, 0].set_xlabel('Prédiction')
    axes[0, 0].set_ylabel('Réalité')
    axes[0, 0].set_xticklabels(['Champ', 'Amas'])
    axes[0, 0].set_yticklabels(['Champ', 'Amas'])
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                          color='white', fontweight='bold')
    
    # Importances des features
    importances = model.feature_importances_
    idx = np.argsort(importances)[-8:]
    top_features = [feature_cols[i] for i in idx]
    axes[0, 1].barh(top_features, importances[idx], color='steelblue')
    axes[0, 1].set_title('Top 8 Features Importantes', fontweight='bold')
    axes[0, 1].set_xlabel('Importance')
    
    # Distribution des probabilités
    probs = model.predict_proba(X_test)
    axes[1, 0].hist(probs[y_test==0, 1], bins=20, alpha=0.6, label='Champ (vrai)', color='blue')
    axes[1, 0].hist(probs[y_test==1, 1], bins=20, alpha=0.6, label='Amas (vrai)', color='red')
    axes[1, 0].set_xlabel('Probabilité Amas')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution des Probabilités Prédites', fontweight='bold')
    axes[1, 0].legend()
    
    # Features image vs catalogue
    if 'IMAGE_MEAN' in df.columns and 'SPATIAL_EXTENT' in df.columns:
        scatter = axes[1, 1].scatter(df.loc[y_test.index, 'IMAGE_MEAN'],
                                     df.loc[y_test.index, 'SPATIAL_EXTENT'],
                                     c=y_pred, cmap='coolwarm', s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Intensité Moyenne Image')
        axes[1, 1].set_ylabel('Étendue Spatiale du Catalogue')
        axes[1, 1].set_title('Features Image vs Catalogue', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1], label='Classe')
    
    plt.tight_layout()
    output_file = f'{base_dir}/results_combined.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Résultats sauvegardés: {output_file}\n")
    
    print("="*70)
    print("✓ PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"\nRésumé:")
    print(f"  • Répertoires traités: {len(all_pairs)}")
    print(f"  • Paires image/catalogue: {count}")
    print(f"  • Features combinées: {len(feature_cols)}")
    print(f"  • Objets ML: {len(df)}")
    print(f"  • Précision modèle: {(y_pred == y_test).mean():.1%}")

else:
    print("⚠ Pas assez de données pour entraîner le modèle (minimum 5)")