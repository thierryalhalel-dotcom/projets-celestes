#!/usr/bin/env python3
"""
Regenere toutes les figures du chapitre "Mouvement Terre / Lune / Soleil"
=========================================================================
Un seul point d'entree pour reproduire a l'identique les PNG utilises par
TLS2.tex, avec les bons parametres de simulation pour chaque figure :

  * 4000 ans, store_every=400, sans Jupiter  -> fig1 a fig5, fig8 (ephemerides)
  * 25 ans,   store_every=100, sans Jupiter  -> fig6, fig7 (FFT des inegalites :
    il faut dt_ech = 0,58 j pour descendre a 9,6 j, et 19 ans suffisent)
  * 4000 ans avec ET sans Jupiter            -> fig1c, fig1d, fig1e, fig5b

Usage :
    python3 refaire_figures_livre.py            # tout
    python3 refaire_figures_livre.py --sans-jupiter-seul   # sans les comparaisons
"""

import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulation_TLS_Jupiter as S
import figures_jupiter as FJ

OUT = os.path.join(HERE, "resultats")


def _args(no_jupiter, years, store_every, dt=500.0):
    return argparse.Namespace(no_jupiter=no_jupiter, dt=dt, years=years,
                              store_every=store_every, tag=None)


def main():
    p = argparse.ArgumentParser(description="Figures du chapitre TLS")
    p.add_argument("--sans-jupiter-seul", action="store_true",
                   help="ne regenere pas les figures comparatives avec Jupiter")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("== Simulation 4000 ans, sans Jupiter (fig. 4.3 a 4.7, 4.10, 4.11) ==")
    res = S.run_simulation(_args(True, 4000.0, 400))
    S.plot_fig1_terre(res, "sansJupiter", OUT);        print("  -> fig1 (4.3)")
    S.plot_fig2_distance_TL(res, "sansJupiter", OUT);  print("  -> fig2 (4.4)")
    S.plot_fig3_lune(res, "sansJupiter", OUT);         print("  -> fig3 (4.5)")
    S.plot_fig4_seculaires(res, "sansJupiter", OUT);   print("  -> fig4 (4.6)")
    S.plot_fig5_ensemble(res, "sansJupiter", OUT);     print("  -> fig5 (4.7)")
    FJ.fig_fft_params_lunaires_3corps(res, OUT)     # fig5b (4.14), sans Jupiter
    S.generate_ephemeris_plots(res, "sansJupiter", OUT)  # fig8 (4.10 et 4.11)

    print("== Simulation 25 ans, store_every=100, sans Jupiter (fig. 4.12, 4.13) ==")
    res_fin = S.run_simulation(_args(True, 25.0, 100))
    S.plot_fig6_fft_longitude(res_fin, "sansJupiter", OUT); print("  -> fig6 (4.12)")
    S.plot_fig7_fft_latitude(res_fin, "sansJupiter", OUT);  print("  -> fig7 (4.13)")

    if not a.sans_jupiter_seul:
        print("== Simulations avec / sans Jupiter (fig. 4.14 a 4.17, exo 4.18) ==")
        avec = S.run_simulation(_args(False, 4000.0, 400))
        FJ.fig_jupiter_terre(avec, res, OUT)         # 4.15
        FJ.fig_jupiter_lune(avec, res, OUT)          # 4.16
        FJ.fig_excentricite_fft(avec, res, OUT)      # 4.17
        FJ.fig_derive_eT_jupiter(avec, res, OUT)     # 4.18 (exercice, question 5)

    print("\nTermine : figures dans", OUT)


if __name__ == "__main__":
    main()
