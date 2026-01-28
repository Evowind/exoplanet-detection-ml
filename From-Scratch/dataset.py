import os
import shutil
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = 'exofop_tess_kois.csv'
BASE_DIR_FITS = "dataset_fits"
BASE_DIR_PNG = "dataset_fits/png"
TRAIN_COUNT = 1260 
TEST_COUNT = 50   

# ==========================================
# FONCTIONS DE TRAITEMENT
# ==========================================

def create_png_from_fits(fits_path, save_path):
    """Lit un fichier FITS local et génère le PNG associé."""
    try:
        # Chargement de la courbe depuis le fichier FITS local
        lc = lk.read(fits_path).remove_nans().flatten()
        
        # Prétraitement (identique à ton script initial)
        lc.flux = savgol_filter(lc.flux, window_length=21, polyorder=3)
        pg = lc.to_periodogram(method='bls')
        lc_fold = lc.fold(period=pg.period_at_max_power)
        
        # Sauvegarde de l'image 224x224
        plt.figure(figsize=(2.24, 2.24), dpi=100)
        plt.scatter(lc_fold.time.value, lc_fold.flux.value, s=0.5, c='black')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        print(f"      > [ERREUR PNG] Impossible de traiter {fits_path}: {e}")
        return False

def process_target(tic_id, sub_path):
    """Gère le cycle : Recherche -> Téléchargement FITS -> Création PNG."""
    fits_folder = os.path.join(BASE_DIR_FITS, sub_path)
    png_folder = os.path.join(BASE_DIR_PNG, sub_path)
    
    os.makedirs(fits_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)
    
    png_path = os.path.join(png_folder, f"{tic_id}.png")
    
    # 1. Vérification si le PNG existe déjà
    if os.path.exists(png_path):
        return True

    # 2. Recherche et téléchargement du FITS
    pipelines = ["SPOC", "TESS-SPOC", "QLP"]
    for author in pipelines:
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", author=author)
            if len(search) > 0:
                lc_file = search[0].download()
                original_fits = lc_file.filename
                
                # Sauvegarde locale du FITS
                dest_fits = os.path.join(fits_folder, os.path.basename(original_fits))
                if not os.path.exists(dest_fits):
                    shutil.copy2(original_fits, dest_fits)
                
                # 3. Génération immédiate du PNG à partir du FITS local
                if create_png_from_fits(dest_fits, png_path):
                    print(f"  > [OK] TIC {tic_id} : FITS et PNG créés ({author})")
                    return True
        except:
            continue
            
    print(f"  > [ERREUR] TIC {tic_id} introuvable sur MAST.")
    return False

# ==========================================
# MAIN
# ==========================================

def main():
    try:
        df_csv = pd.read_csv(CSV_FILE, skiprows=2)
    except:
        print(f"Fichier {CSV_FILE} introuvable.")
        return

    cp_pool = df_csv[df_csv['Disposition'] == 'CP']['TIC ID'].unique().tolist()
    fp_pool = df_csv[df_csv['Disposition'] == 'FP']['TIC ID'].unique().tolist()
    pc_pool = df_csv[df_csv['Disposition'] == 'PC']['TIC ID'].unique().tolist()

    tasks = [
        ("train/CP", cp_pool, TRAIN_COUNT),
        ("train/FP", fp_pool, TRAIN_COUNT),
        ("test/CP", cp_pool[TRAIN_COUNT:], TEST_COUNT),
        ("test/FP", fp_pool[TRAIN_COUNT:], TEST_COUNT),
        ("test/PC", pc_pool, TEST_COUNT)
    ]

    print("--- DÉBUT DU TRAITEMENT COMPLET (FITS + PNG) ---")
    for sub, pool, target in tasks:
        print(f"\n📁 Catégorie : {sub}")
        count = 0
        for tid in pool:
            if count >= target: break
            if process_target(tid, sub):
                count += 1
    
    print("\n--- OPÉRATION TERMINÉE ---")

if __name__ == "__main__":
    main()