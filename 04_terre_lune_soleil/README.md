# Chapitre 4 – Le mouvement du système Terre – Lune – Soleil

Simulation du problème à trois corps Soleil – Terre – Lune (Jupiter en option) sur 4000 ans,
par la méthode d'Euler symplectique, avec un pas de 500 s.

## Programmes

| Fichier | Rôle |
|---|---|
| `simulation_core.c` | Noyau de calcul en C : boucle d'intégration, détection des périgées, nœuds, années |
| `simulation_TLS_Jupiter.py` | **Programme principal** : compile le C, lance la simulation, FFT, éphémérides, figures |
| `refaire_figures_livre.py` | Régénère toutes les figures du chapitre avec les réglages du livre |
| `figures_jupiter.py` | Figures comparatives avec / sans Jupiter (exercice du chapitre) |
| `parametres_orbitaux.py` | Trace les paramètres orbitaux à partir d'un fichier `.npz` déjà calculé |
| `doc_parametres_orbitaux.py` | Fiche de documentation des paramètres orbitaux (PDF) |
| `make_angles2.py`, `make_angles3.py` | Schémas des angles (coordonnées lunaires, paramètres de l'orbite) |
| `figure_energie.py` + `energie_core.c` | Conservation de l'énergie selon l'intégrateur (Euler explicite, Runge-Kutta 4, Euler symplectique), sur 4000 ans (~1 min 30) |

## Utilisation

```bash
python3 simulation_TLS_Jupiter.py                 # 4000 ans, avec Jupiter (~35 s)
python3 simulation_TLS_Jupiter.py --no-jupiter    # modèle à trois corps
python3 simulation_TLS_Jupiter.py --years 100 --dt 250   # 100 ans, pas de 250 s
python3 refaire_figures_livre.py                  # toutes les figures du livre
```

| Option | Défaut | Rôle |
|---|---|---|
| `--no-jupiter` | (Jupiter inclus) | supprime Jupiter (masse mise à zéro) |
| `--dt` | 500 | pas de temps en secondes |
| `--years` | 4000 | durée simulée en années |
| `--store-every` | 400 | on stocke un point tous les N pas (100 pour une FFT plus fine) |
| `--tag` | auto | suffixe des fichiers produits |

Résultats : figures et fichier `simulation_data_avecJupiter.npz` (ou `_sansJupiter`) dans
`resultats/`. Ce fichier `.npz` est relu par les programmes du chapitre 5.

## Python, C et bibliothèque `.so`

Le premier lancement compile `simulation_core.c` en une bibliothèque partagée
`libsimulation.so` (il faut un compilateur `gcc`), que Python charge ensuite avec le module
`ctypes`. Python réserve les tableaux NumPy, le C les remplit directement, puis Python fait
les analyses et les figures. Voir la section 4 du [README principal](../README.md) pour le
détail. Si vous modifiez `simulation_core.c`, la bibliothèque est recompilée automatiquement
au lancement suivant.

Un exercice instructif : relancer avec `--dt 1000`, `--dt 250` et `--dt 125`, et comparer les
mois lunaires obtenus. L'erreur est divisée par deux quand le pas est divisé par deux :
c'est la signature d'une méthode d'ordre 1.
