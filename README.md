# Projets célestes

Programmes Python (et C) accompagnant le livre de Thierry Alhalel consacré à des projets de
mécanique céleste et d'astronomie :
simulations numériques en mécanique céleste et en astronomie, à destination des étudiants
de classes préparatoires scientifiques (CPGE), de leurs enseignants et des passionnés
d'astronomie.

Chaque chapitre du livre a son dossier. Tous les programmes produisent les figures du livre
et peuvent être modifiés librement pour explorer d'autres situations : c'est leur but.

| Dossier | Chapitre du livre | Contenu |
|---|---|---|
| [`01_voilier_solaire`](01_voilier_solaire) | Une course de voiliers solaires | Satellite à voile solaire autour de la Terre : intégration Euler explicite / implicite |
| [`02_equation_du_temps_analemme`](02_equation_du_temps_analemme) | Équation du temps, analemme et cadrans solaires | Équation du temps, équation du centre, analemme (Terre et Mars), saisons |
| [`03_ascenseur_spatial`](03_ascenseur_spatial) | L'ascenseur spatial | Toutes les figures du chapitre |
| [`04_terre_lune_soleil`](04_terre_lune_soleil) | Le mouvement du système Terre – Lune – Soleil | Simulation à 3 (ou 4) corps sur 4000 ans, noyau de calcul en C, analyses de Fourier, éphémérides |
| [`05_eclipses`](05_eclipses) | Éclipses luni-solaires | Détection et classement des éclipses, cycle de Saros, comparaison aux catalogues |
| [`06_annexe_methodes_numeriques`](06_annexe_methodes_numeriques) | Annexe : méthodes de résolution numérique | Euler explicite, implicite, symplectique, Runge-Kutta sur des exemples simples |

---

## 1. Accéder aux programmes

Vous n'avez **pas besoin de compte GitHub** pour consulter ou télécharger les programmes.

### Méthode 1 – Lire un programme en ligne (rien à installer)

Ouvrez la page du dépôt : **https://github.com/thierryalhalel-dotcom/projets-celestes**,
cliquez sur un dossier, puis sur un fichier `.py` : son contenu s'affiche directement dans le
navigateur.

### Méthode 2 – Tout télécharger d'un coup (recommandé)

1. Sur la page du dépôt, cliquez sur le bouton vert **`<> Code`** (en haut à droite de la liste
   des fichiers).
2. Choisissez **`Download ZIP`**.
3. Décompressez le fichier `projets-celestes-main.zip` où vous voulez (double-clic sous
   Windows ou macOS).

Vous obtenez un dossier `projets-celestes-main` contenant les six sous-dossiers ci-dessus.

### Méthode 3 – Avec Git (pour les habitués)

```bash
git clone https://github.com/thierryalhalel-dotcom/projets-celestes.git
```

Un simple `git pull` dans le dossier récupère ensuite les mises à jour éventuelles.

---

## 2. Installer Python et les bibliothèques

Les programmes demandent **Python 3.9 ou plus récent** et quelques bibliothèques scientifiques.

**Installer Python**
- *Windows* : télécharger l'installateur sur <https://www.python.org/downloads/> et, au premier
  écran, **cocher la case « Add python.exe to PATH »**.
  Une alternative simple est la distribution [Anaconda](https://www.anaconda.com/download),
  qui contient déjà toutes les bibliothèques nécessaires.
- *macOS* : installateur sur <https://www.python.org/downloads/>, ou Anaconda.
- *Linux* : Python 3 est en général déjà présent (`sudo apt install python3 python3-pip` sinon).

**Installer les bibliothèques** : ouvrir un terminal (sous Windows : « Invite de commandes »),
se placer dans le dossier téléchargé, puis taper :

```bash
pip install -r requirements.txt
```

(sous Linux ou macOS, utiliser au besoin `pip3` au lieu de `pip`). Cela installe `numpy`,
`scipy`, `matplotlib` et `astropy`.

**Pour le chapitre 4 (et donc le chapitre 5) seulement : un compilateur C**
- *Linux* : `sudo apt install build-essential` (fournit `gcc`).
- *macOS* : `xcode-select --install` (fournit le compilateur, appelé lui aussi par `gcc`).
- *Windows* : le plus simple est d'utiliser le sous-système Linux de Windows
  ([WSL](https://learn.microsoft.com/fr-fr/windows/wsl/install)) et de suivre les instructions
  Linux. Les programmes ont été développés et testés sous Linux.

---

## 3. Lancer un programme

Dans un terminal, placez-vous dans le dossier du chapitre, puis lancez le programme :

```bash
cd 04_terre_lune_soleil
python3 simulation_TLS_Jupiter.py --no-jupiter
```

(sous Windows : `python` au lieu de `python3`). Les figures s'enregistrent dans un
sous-dossier `figures/` ou `resultats/` du chapitre, ou s'affichent à l'écran.
Le fichier `README.md` de chaque dossier détaille les programmes et leurs options.

On peut aussi ouvrir les fichiers `.py` dans un environnement comme **Spyder**, **Thonny**,
**IDLE**, **VS Code** ou **Jupyter**, et les exécuter depuis celui-ci.

---

## 4. Comment Python, le C et la bibliothèque `.so` travaillent ensemble (chapitres 4 et 5)

La simulation Terre – Lune – Soleil effectue environ **250 millions de pas de calcul** pour
couvrir 4000 ans. En Python pur, cela prendrait des heures ; en C, une trentaine de secondes.
Le programme combine donc les deux langages, chacun dans son rôle :

```
   simulation_TLS_Jupiter.py  (Python : pilotage, analyse, figures)
        │
        │ 1. au premier lancement, compile le fichier C :
        │       gcc -O2 -shared -fPIC -o libsimulation.so simulation_core.c -lm
        ▼
   libsimulation.so  (bibliothèque partagée = code machine compilé)
        ▲
        │ 2. Python la charge avec le module standard  ctypes,
        │    réserve des tableaux NumPy vides et passe leurs adresses au C
        │
        │ 3. le C calcule les 250 millions de pas et écrit directement
        │    dans ces tableaux NumPy (aucune copie)
        ▼
   tableaux NumPy remplis  →  Python : FFT, éphémérides, figures, fichier .npz
```

- **`simulation_core.c`** : le *noyau de calcul*. Il contient la boucle d'intégration
  (méthode d'Euler symplectique) et la détection des événements (périgées, nœuds, années).
  C'est un fichier source, lisible, mais que Python ne peut pas exécuter tel quel.
- **`libsimulation.so`** : la *bibliothèque partagée* (*shared object*), c'est-à-dire le
  noyau C traduit en code machine par le compilateur `gcc`. C'est l'équivalent d'un fichier
  `.dll` sous Windows. **Vous n'avez pas à la créer vous-même** : le programme Python la
  compile automatiquement au premier lancement (fonction `ensure_library()`), puis la
  recompile seulement si vous modifiez `simulation_core.c`. Elle n'est pas fournie dans le
  dépôt, car elle dépend de votre système.
- **`simulation_TLS_Jupiter.py`** : le *chef d'orchestre*. Il lit les options de la ligne de
  commande, calcule les conditions initiales, appelle le noyau C via `ctypes`, puis réalise
  toutes les analyses (transformées de Fourier, comparaison aux éphémérides) et les figures.
  Les résultats bruts sont sauvegardés dans `resultats/simulation_data_*.npz` (format NumPy
  compressé), que relisent les programmes du chapitre 5.

Pour **modifier la physique** (ajouter un corps, changer une force), c'est
`simulation_core.c` qu'il faut éditer ; pour **modifier les analyses ou les figures**, c'est le
Python. Le chapitre 4 du livre présente cette architecture en détail.

---

## 5. Signaler une erreur, poser une question

Le dépôt est **en lecture pour tous** : chacun peut consulter, télécharger et modifier les
programmes *sur son propre ordinateur*. Seul l'auteur peut modifier les fichiers du dépôt
lui-même.

Pour signaler une erreur ou poser une question, utilisez l'onglet **`Issues`** de la page
GitHub (bouton **`New issue`** ; un compte GitHub gratuit est alors nécessaire).

---

## 6. Licence

Les programmes sont distribués sous **licence MIT** (voir le fichier [`LICENSE`](LICENSE)) :
vous pouvez les utiliser, les copier, les modifier et les redistribuer librement, y compris
dans vos cours ou vos TIPE, à condition de conserver la mention de l'auteur et de la licence.

© 2026 Thierry Alhalel
