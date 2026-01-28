# Détection d’Exoplanètes

Projet de détection d’exoplanètes à partir de courbes de lumière au format **FITS**, en utilisant plusieurs architectures de deep learning.

## Objectif

Classifier des signaux astronomiques en :

* **CP** : Candidate Planet
* **FP** : False Positive

Les modèles travaillent sur des séries temporelles de flux.

## Modèles testés

* CNN / ResNet
* LSTM
* TCN
* Transformer

## Structure du projet

```
README.md

Fine-Tune/
From-Scratch/
    CNN.ipynb
    LSTM.ipynb
    RESNET.ipynb
    TCN.ipynb
    dataset.py
    script.py
    tri.py

Visualisation/
    cnn_exoplanet_fits_results.png
    lstm_exoplanet_fits_results.png
    resnet_exoplanet_fits_results.png
    tcn_exoplanet_fits_results.png
```

## Exemples de résultats

### CNN

![CNN Results](Visualisation/cnn_exoplanet_fits_results.png)

### LSTM

![LSTM Results](Visualisation/lstm_exoplanet_fits_results.png)

### ResNet

![ResNet Results](Visualisation/resnet_exoplanet_fits_results.png)

### TCN

![TCN Results](Visualisation/tcn_exoplanet_fits_results.png)

## Données

Les données doivent être au format **.fits** et organisées comme suit :

```
dataset_fits/
 ├── CP/
 └── FP/
```

## Entraînement

* Exécution prévue sur **Google Colab**
* Le modèle est sélectionné via :

```python
model_type = 'tcn'  # resnet, lstm, tcn, transformer
```

## Sorties générées

* Modèle entraîné (`.h5`)
* Seuil optimal (`.pkl`)
* Graphiques de performance (loss, accuracy, AUC, ROC, matrice de confusion)

## Dépendances

* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Astropy
* Matplotlib, Seaborn

## Note

Projet à but expérimental pour comparer différentes architectures sur des données astronomiques.