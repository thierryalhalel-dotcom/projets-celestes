# Chapitre 5 – Éclipses luni-solaires

Ces programmes **ne refont pas la simulation** : ils relisent les trajectoires calculées au
chapitre 4. Il faut donc d'abord lancer, une fois :

```bash
cd ../04_terre_lune_soleil
python3 simulation_TLS_Jupiter.py            # produit resultats/simulation_data_avecJupiter.npz
cd ../05_eclipses
```

| Programme | Rôle |
|---|---|
| `eclipses_saros.py` | Détecte et classe toutes les éclipses sur 4000 ans, met en évidence le Saros et les autres cycles, étudie la divergence par rapport aux éphémérides du JPL (via `astropy`) |
| `comparaison_eclipses_premier_saros.py` | Exercice du chapitre : compare les éclipses simulées au catalogue NASA/JPL sur le premier Saros (2005 – 2023) et produit un PDF de synthèse |

```bash
python3 eclipses_saros.py
python3 eclipses_saros.py ../04_terre_lune_soleil/resultats/simulation_data_sansJupiter.npz
python3 comparaison_eclipses_premier_saros.py
```

Les figures et le PDF sont enregistrés dans `resultats/`. Les programmes retrouvent tout seuls
le dossier voisin `04_terre_lune_soleil` (simulateur et fichiers `.npz`) : gardez les deux
dossiers côte à côte.
