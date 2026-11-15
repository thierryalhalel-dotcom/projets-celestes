# Annexe – Méthodes de résolution numérique

Comparaison des méthodes d'Euler explicite, d'Euler implicite, d'Euler symplectique et de
Runge-Kutta sur des équations dont on connaît la solution exacte.

| Programme | Équation | Ce qu'on observe |
|---|---|---|
| `ordre1.py` | y' + 3y = 0 | précision et stabilité selon le pas |
| `ordre2.py` | y'' + 3y = 0 (oscillateur) | conservation de l'énergie, portrait de phase ; figures dans `figures/` |
| `oscillateur_amorti.py`, `oscillateur_amorti2.py` | y'' + 5y' + 100y = 0 | oscillateur amorti |

```bash
python3 ordre2.py
```
