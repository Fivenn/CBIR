# Projet traitement et Analyse d'images
## Préparation des données
Pour tout ajout d'un répertoire contenant des données, faites attention à bien le placer à la racine de ce projet avec comme nom __CorelDB__.

Pour séparer les données de votre base de données en 3 sous-ensembles en fonction d'un ratio, vous aurez besoin d'utiliser le script ```split.dataset.sh``` présent à la racine du projet. 

Attention, pour faire fonctionner ce script, il est nécessaire d'avoir installé ```splitfolders``` sur votre machine. Pour cela, vous pouvez exécuter la commande ```pip install splitfolders```.

Voici à quoi doit ressembler l'arborescence après ajout de la base de données 

    ├── src/            # Fichiers sources
    ├── result/         # Résultats
    ├── README.md       # Notice de fonctionnement du projet
    ├── CorelDB/        # Répertoire de vos images sans séparation
    └── CorelDBDataSet/ # Répertoire de vos images après séparation
__Toutes vos images doivent être présentes dans le dossier CorelDBDataSet__

## Méthode old school

### Pour utiliser l'extrateur d'attributs color
```python
python3 src/color.py
```

### Pour utiliser l'extracteur d'attributs daisy
```python
python3 src/daisy.py
```

### Pour utiliser l'extrateur d'attributs fusion
```python
python3 src/fusion.py
```

## Méthode new school

### Pour utiliser la prédiction à l'aide d'un réseau de neurones
```python
python3 src/evaluation_CNN.py
```
*Remarque : la prédiction des images se lance après fermeture de la fenêtre des graphes de précision et de perte du modèle.*
