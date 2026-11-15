# Chapitre 1 – Une course de voiliers solaires

Un satellite muni d'une voile solaire de 2,13 km² part d'une orbite circulaire autour de la
Terre ; la pression de radiation solaire le fait spiraler jusqu'à la libération.

| Programme | Rôle |
|---|---|
| `voilier_solaire.py` | Programme du livre (méthode d'Euler explicite) : trajectoire, énergie, temps de libération |
| `voilier_euler_implicite.py` | Même problème résolu par la méthode d'Euler implicite (prédicteur-correcteur) |

```bash
python3 voilier_solaire.py
```

Paramètres à modifier en tête de fichier : durée simulée (`jour`), pas de temps (`dt`),
masse et surface de la voile, orbite de départ (`r0`).
