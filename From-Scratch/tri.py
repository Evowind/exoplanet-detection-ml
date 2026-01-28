import shutil
from pathlib import Path
from tqdm import tqdm
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

CACHE_DIR = Path(r"C:\Users\Zack\.lightkurve\cache\mastDownload\Kepler")
OUTPUT_DIR = Path(r"D:\Projet\ResNet\dataset_fits")

CP_DIR = OUTPUT_DIR / "CP"
FP_DIR = OUTPUT_DIR / "FP"

CP_DIR.mkdir(parents=True, exist_ok=True)
FP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# LOAD OFFICIAL KOI LABELS (DR25)
# ------------------------------------------------------------------

print("Chargement table KOI DR25...")
koi = NasaExoplanetArchive.query_criteria(
    table="q1_q17_dr25_koi",
    select="kepid,koi_disposition"
).to_pandas()

koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
label_map = dict(zip(koi.kepid.astype(str), koi.koi_disposition))

print(f"{len(label_map)} KOI labellisés chargés")

# ------------------------------------------------------------------
# SCAN CACHE AND SORT
# ------------------------------------------------------------------

fits_files = list(CACHE_DIR.rglob("*.fits"))
print(f"{len(fits_files)} fichiers FITS trouvés")

copied_cp = 0
copied_fp = 0

for fits_file in tqdm(fits_files):
    name = fits_file.name.lower()

    # Extraction KIC depuis le nom de fichier
    # ex: kplr011446443-2010174085026_llc.fits
    if not name.startswith("kplr"):
        continue

    kepid = name[4:13].lstrip("0")

    if kepid not in label_map:
        continue

    label = label_map[kepid]

    if label == "CONFIRMED":
        dest = CP_DIR / fits_file.name
        copied_cp += 1
    else:
        dest = FP_DIR / fits_file.name
        copied_fp += 1

    if not dest.exists():
        shutil.copy2(fits_file, dest)

print("\nTRI TERMINÉ")
print(f"CP copiés : {copied_cp}")
print(f"FP copiés : {copied_fp}")
