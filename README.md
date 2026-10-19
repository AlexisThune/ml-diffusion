# Downscaling océanique : U-Net vs diffusion conditionnelle

> Un benchmark reproductible pour comparer une reconstruction déterministe et une génération probabiliste de champs océaniques à haute résolution.

Le projet étudie une question simple : **que gagne-t-on, et que perd-on, en remplaçant un U-Net de régression par un modèle de diffusion conditionnel pour le downscaling ?** Les deux approches reçoivent les mêmes champs d'entrée, sont entraînées sur le même jeu synthétique et sont évaluées sur un jeu de test tenu à l'écart.

Les données sont synthétiques et inspirées de structures océaniques (fronts, oscillations et tourbillons). Elles permettent de vérifier le protocole et le comportement des modèles; elles ne constituent pas une validation sur des observations océaniques réelles.

## Résumé

- **Régression U-Net** : prédit directement un champ fin avec une perte MSE. Une entrée conduit à une seule prédiction.
- **Diffusion conditionnelle** : apprend à retirer du bruit d'un champ cible bruité, conditionné par les mêmes variables. Le sampling DDIM part de plusieurs bruits initiaux et génère plusieurs champs plausibles pour une même entrée.
- **Références** : la carte basse résolution réinterpolée par bicubique, le U-Net et la moyenne des tirages diffusion sont comparés sur les mêmes exemples.
- **Incertitude** : la dispersion spatiale des tirages, la couverture de l'intervalle empirique à 95 % et sa largeur moyenne complètent les métriques d'erreur.

## Lancer le projet

Python 3.10 ou plus récent est recommandé. Installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Lancer les tests :

```bash
python -m unittest discover -s tests -v
```

Lancer le benchmark avec les paramètres par défaut :

```bash
python train.py
```

Un essai rapide de bout en bout :

```bash
python train.py \
  --epochs 2 \
  --train-samples 32 --val-samples 8 --test-samples 8 \
  --size 32 --batch-size 8 \
  --diffusion-steps 20 --sampling-steps 8 \
  --uncertainty-samples 4 \
  --output-dir outputs/quick-run
```

Par défaut, le script choisit CUDA lorsqu'elle est disponible, sinon le CPU. Pour sélectionner explicitement le matériel, utiliser `--device cpu` ou `--device cuda`. Le nombre de canaux du réseau doit être divisible par 8; la taille spatiale doit être au moins 16 et divisible par 4.

Les checkpoints et résultats sont enregistrés dans `outputs/` :

| Fichier | Contenu |
| --- | --- |
| `unet.pt`, `diffusion.pt` | Poids des meilleurs modèles selon la perte de validation propre à chaque objectif |
| `metrics.csv` | Métriques de test pour le bicubique, le U-Net et la diffusion |
| `comparison.png` | Exemples, réalisations diffusion, erreurs et écart-type spatial |
| `training_history.json` | Pertes d'entraînement et de validation par époque |
| `experiment.json` | Paramètres, graines, meilleures époques et résultats de test |

Réévaluer un checkpoint sur le split de test reproductible :

```bash
python inference.py --checkpoint-dir outputs --output-dir outputs/re-evaluation
```

Le nombre de réalisations est configurable, par exemple `--uncertainty-samples 32`. Plus il est élevé, plus l'estimation de la distribution et des quantiles est stable, mais plus l'évaluation est coûteuse.

## Protocole expérimental

Chaque échantillon contient trois cartes d'entrée (`Coarse SSS`, `SST`, `SSH`) de taille `64 x 64` et une cible `SSS` fine. Le champ fin combine des fronts, des motifs ondulatoires, des tourbillons et un faible bruit. La cible grossière est créée en faisant une moyenne spatiale par blocs (`downscale_factor=4`, donc `16 x 16`), puis en la réinterpolant en bicubique à `64 x 64`. Les deux variables auxiliaires sont générées à partir de structures corrélées mais distinctes.

Les graines et plages d'indices isolent les ensembles d'entraînement, de validation et de test. Les modèles voient les mêmes batches d'entraînement. Le meilleur U-Net est sélectionné par MSE de validation; le meilleur débruiteur est sélectionné par MSE de bruit de validation. Ces pertes ne sont pas directement comparables entre elles : la comparaison finale repose sur les métriques calculées sur les champs reconstruits du jeu de test.

## Métriques et incertitude

Les métriques communes sont calculées contre la cible fine :

| Mesure | Interprétation | Sens favorable |
| --- | --- | --- |
| RMSE | Erreur quadratique, sensible aux grands écarts | Plus bas |
| MAE | Erreur absolue moyenne | Plus bas |
| Biais | Moyenne de `prédiction - cible` | Proche de zéro |
| PSNR | Rapport signal/bruit dérivé du RMSE | Plus haut |
| SSIM | Similarité locale de structure | Plus haut |
| Corrélation | Corrélation linéaire spatiale | Plus proche de 1 |

Pour la diffusion, les prédictions ponctuelles sont la **moyenne Monte-Carlo** des réalisations DDIM. Les réalisations sont également conservées pour calculer, pixel par pixel, les quantiles empiriques 2,5 % et 97,5 %. Le rapport ajoute la couverture observée de cet intervalle (`coverage_95`), l'écart-type moyen (`mean_std`) et sa largeur moyenne (`interval_width_95`). Une bonne couverture seule ne garantit pas une bonne calibration : elle doit être lue avec la largeur de l'intervalle et l'erreur de la moyenne.

Cette dispersion décrit la variabilité conditionnelle des générations du modèle, **pas une incertitude bayésienne garantie ni une incertitude observationnelle**. Avec peu de réalisations, les quantiles sont bruyants. Les résultats doivent être accompagnés du nombre de tirages, du nombre de pas DDIM et de la graine.

## Architecture

```text
Coarse SSS ─┐
SST ────────┼──> U-Net de régression ──> champ SSS fin (unique)
SSH ────────┘

SSS cible bruitée + [Coarse SSS, SST, SSH] + pas t
                 └──> U-Net de débruitage ──> bruit prédit
                       schedule cosinus / objectif DDPM
                       sampling DDIM répété ──> champs plausibles
```

Les deux réseaux ont un encodeur-décodeur avec connexions de saut. Le débruiteur reçoit le champ cible bruité concaténé aux conditions et un embedding temporel sinusoidal. L'entraînement optimise la prédiction du bruit ajouté selon un calendrier cosinus; l'évaluation utilise un échantillonnage DDIM déterministe par trajectoire, avec des états initiaux distincts pour obtenir plusieurs réalisations.

## Structure

```text
data/dataset.py          Générateur déterministe des champs synthétiques
models/unet.py           Baseline U-Net de régression
models/diffusion.py      U-Net temporel, objectif DDPM et sampler DDIM
utils/metrics.py         Métriques de reconstruction
utils/benchmark.py       Évaluation, incertitude, CSV et figures
train.py                 Entraînement, validation, test et sauvegardes
inference.py             Réévaluation des checkpoints
tests/test_project.py    Tests des données, modèles et résultats
```

## Limites et prochaines étapes

- Le générateur n'est pas un modèle physique et les données ne représentent pas une distribution océanographique mesurée.
- Les splits sont séparés par graine mais partagent la même famille de générateurs; cela mesure la généralisation à de nouveaux motifs synthétiques, pas le transfert à une autre région ou période.
- Les résultats dépendent du budget d'entraînement, du calendrier de bruit, du nombre de pas d'échantillonnage et du nombre de réalisations. La diffusion coûte davantage à l'inférence qu'une passe U-Net.
- La couverture empirique doit être vérifiée sur plusieurs graines et sur des données réelles avant toute conclusion de calibration.
- Une suite naturelle consiste à remplacer le générateur par des champs observés co-localisés, définir les unités et normalisations, puis évaluer sur des régions ou périodes géographiques jamais vues.

## Présentation en entretien

> « J'ai construit un benchmark contrôlé de downscaling pour comparer une baseline U-Net déterministe à une diffusion conditionnelle. Le point important est que la diffusion ne renvoie pas seulement une carte : en répétant le sampling pour une même observation, elle fournit une distribution de sorties. J'évalue donc à la fois l'erreur de la moyenne et l'incertitude empirique, avec couverture et largeur d'intervalle, face à un baseline bicubique. Le protocole est reproductible et les données sont synthétiques; la suite logique est une validation géographique sur des observations réelles avant d'interpréter les scores comme des résultats océanographiques. »