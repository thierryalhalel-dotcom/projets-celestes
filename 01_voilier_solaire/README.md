# Chapitre 1 – Une course de voiliers solaires

Un satellite muni d'une voile solaire de 2,13 km² part d'une orbite circulaire autour de la
Terre ; la pression de radiation solaire le fait spiraler jusqu'à la libération.

| Programme | Dans le livre | Rôle |
|---|---|---|
| `voilier_solaire.py` | cours, §1.7 et 1.8 | Méthode d'Euler explicite (dt = 1 s) : temps de libération, trajectoire, énergie, distance (fig. 1.4, 1.9, 1.12) |
| `voilier_euler_implicite.py` | exercice 1 | Même problème par Euler implicite (itérations de point fixe), dt = 10 s |
| `voilier_rk4.py` | exercice 2 | Même problème par Runge-Kutta 4, dt = 10 s ; détecte les collisions avec la Terre (fig. 1.15 et 1.16) |
| `voilier_ilots_collision.py` | « altitudes interdites » | Balayage de r0 et mesure de la largeur des îlots de collision (fig. 1.17 et 1.18) |

Les noms de variables et de fonctions sont ceux du livre ; les extraits imprimés dans le
chapitre sont repris tels quels dans ces fichiers.

## Lancer les programmes

```bash
python3 voilier_solaire.py              # cas du cours : libération en ~116 h
python3 voilier_euler_implicite.py
python3 voilier_rk4.py                  # cas du cours
python3 voilier_rk4.py 6.01 0 9.3       # orbite 1 de collision (r0 = 6,01 RT, phi0 = 0°, 9,3 jours)
python3 voilier_rk4.py 6.98 270 9.3     # orbite 2 de collision
python3 voilier_ilots_collision.py 2    # îlot de l'orbite 2 (environ 6 minutes)
```

Paramètres à modifier en tête de fichier : durée simulée (`jour`), pas de temps (`dt`),
masse `m` et surface `S_voile` de la voile, orbite de départ (`r0`, `phi0`).
