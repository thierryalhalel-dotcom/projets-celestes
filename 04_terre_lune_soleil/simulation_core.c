/*
 * Noyau de calcul : simulation du systeme Soleil-Terre-Lune-Jupiter
 * Methode d'Euler symplectique (semi-implicite) : positions mises a jour
 * avec les anciennes vitesses, puis vitesses avec les accelerations aux
 * nouvelles positions. Schema explicite, d'ordre 1, symplectique.
 * Coordonnees heliocentriques ecliptiques
 *
 * Compile : gcc -O2 -shared -fPIC -o libsimulation.so simulation_core.c -lm
 */

#include <math.h>
#include <stdio.h>

#define YEAR_STRIDE  4   /* t, dur_days, rST, e_earth */
#define PERI_STRIDE  6   /* t, rTL, e_moon, v_km_s, cos_xi, sin_xi */
#define NODE_STRIDE  4   /* t, incl_deg, sin_omega, cos_omega */
#define SID_STRIDE   2   /* t, dur_days */
#define PERIH_STRIDE 3   /* t, rST, e_earth */

void simulate(
    double xT, double yT, double zT, double vxT, double vyT, double vzT,
    double xL, double yL, double zL, double vxL, double vyL, double vzL,
    double xJ, double yJ, double zJ, double vxJ, double vyJ, double vzJ,
    double G_val, double ms, double mt, double ml, double mj,
    double dt, long n_steps, int store_every,
    double *t_samp, double *rST_samp, double *rTL_samp,
    double *cos_lambda_samp, double *sin_beta_samp,
    double *xT_samp, double *yT_samp, double *zT_samp,
    double *xTL_samp, double *yTL_samp, double *zTL_samp,
    int *n_samp_out, int max_samp,
    double *year_data, int *n_year_out, int max_years,
    double *peri_data, int *n_peri_out, int max_peri,
    double *node_data, int *n_node_out, int max_nodes,
    double *sid_data,  int *n_sid_out,  int max_sid,
    double *perih_data, int *n_perih_out, int max_perih)
{
    int n_samp = 0, n_year = 0, n_peri = 0, n_node = 0, n_sid = 0, n_perih = 0;

    double Gml = G_val * ml;
    double Gmj = G_val * mj;
    double Gmt = G_val * mt;
    double G_ms_mt = G_val * (ms + mt);
    double G_ms_ml = G_val * (ms + ml);
    double G_ms_mj = G_val * (ms + mj);
    double mu_TL = G_val * (mt + ml);
    double mu_ST = G_val * (ms + mt);

    double rTL_h0 = 0.0, rTL_h1 = 0.0, rTL_h2 = 0.0;
    double rST_h0 = 0.0, rST_h1 = 0.0, rST_h2 = 0.0;
    int ring = 0;

    double zTL_prev = zL - zT;
    double yT_prev  = yT;
    double yTL_prev = yL - yT;

    double sv_xT, sv_yT, sv_zT, sv_vxT, sv_vyT, sv_vzT;
    double sv_xL, sv_yL, sv_zL, sv_vxL, sv_vyL, sv_vzL;

    double t_last_year = -1.0;
    double t_last_sid  = -1.0;

    printf("Simulation : %ld pas, dt = %.0f s, duree ~ %.0f ans\n",
           n_steps, dt, n_steps * dt / (365.25 * 86400.0));
    fflush(stdout);

    for (long step = 0; step < n_steps; step++) {
        double t = (step + 1) * dt;

        sv_xT = xT; sv_yT = yT; sv_zT = zT;
        sv_vxT = vxT; sv_vyT = vyT; sv_vzT = vzT;
        sv_xL = xL; sv_yL = yL; sv_zL = zL;
        sv_vxL = vxL; sv_vyL = vyL; sv_vzL = vzL;

        /* === Etape 1 : positions === */
        double xxT = xT + vxT * dt;
        double yyT = yT + vyT * dt;
        double zzT = zT + vzT * dt;
        double xxL = xL + vxL * dt;
        double yyL = yL + vyL * dt;
        double zzL = zL + vzL * dt;
        double xxJ = xJ + vxJ * dt;
        double yyJ = yJ + vyJ * dt;
        double zzJ = zJ + vzJ * dt;

        /* distances */
        double rST = sqrt(xxT*xxT + yyT*yyT + zzT*zzT);
        double rSL = sqrt(xxL*xxL + yyL*yyL + zzL*zzL);
        double rSJ = sqrt(xxJ*xxJ + yyJ*yyJ + zzJ*zzJ);

        double dxTL = xxL - xxT, dyTL = yyL - yyT, dzTL = zzL - zzT;
        double rTL = sqrt(dxTL*dxTL + dyTL*dyTL + dzTL*dzTL);

        double dxTJ = xxJ - xxT, dyTJ = yyJ - yyT, dzTJ = zzJ - zzT;
        double rTJ = sqrt(dxTJ*dxTJ + dyTJ*dyTJ + dzTJ*dzTJ);

        double dxLJ = xxJ - xxL, dyLJ = yyJ - yyL, dzLJ = zzJ - zzL;
        double rLJ = sqrt(dxLJ*dxLJ + dyLJ*dyLJ + dzLJ*dzLJ);

        double rST3 = rST * rST * rST;
        double rSL3 = rSL * rSL * rSL;
        double rSJ3 = rSJ * rSJ * rSJ;
        double rTL3 = rTL * rTL * rTL;
        double rTJ3 = rTJ * rTJ * rTJ;
        double rLJ3 = rLJ * rLJ * rLJ;

        /* === Etape 2 : accelerations === */
        double fxT = -G_ms_mt / rST3 * xxT
                   + Gml * (dxTL / rTL3 - xxL / rSL3)
                   + Gmj * (dxTJ / rTJ3 - xxJ / rSJ3);
        double fyT = -G_ms_mt / rST3 * yyT
                   + Gml * (dyTL / rTL3 - yyL / rSL3)
                   + Gmj * (dyTJ / rTJ3 - yyJ / rSJ3);
        double fzT = -G_ms_mt / rST3 * zzT
                   + Gml * (dzTL / rTL3 - zzL / rSL3)
                   + Gmj * (dzTJ / rTJ3 - zzJ / rSJ3);

        double fxL = -G_ms_ml / rSL3 * xxL
                   + Gmt * (-dxTL / rTL3 - xxT / rST3)
                   + Gmj * (dxLJ / rLJ3 - xxJ / rSJ3);
        double fyL = -G_ms_ml / rSL3 * yyL
                   + Gmt * (-dyTL / rTL3 - yyT / rST3)
                   + Gmj * (dyLJ / rLJ3 - yyJ / rSJ3);
        double fzL = -G_ms_ml / rSL3 * zzL
                   + Gmt * (-dzTL / rTL3 - zzT / rST3)
                   + Gmj * (dzLJ / rLJ3 - zzJ / rSJ3);

        double fxJ = -G_ms_mj / rSJ3 * xxJ
                   + Gmt * (-dxTJ / rTJ3 - xxT / rST3)
                   + Gml * (-dxLJ / rLJ3 - xxL / rSL3);
        double fyJ = -G_ms_mj / rSJ3 * yyJ
                   + Gmt * (-dyTJ / rTJ3 - yyT / rST3)
                   + Gml * (-dyLJ / rLJ3 - yyL / rSL3);
        double fzJ = -G_ms_mj / rSJ3 * zzJ
                   + Gmt * (-dzTJ / rTJ3 - zzT / rST3)
                   + Gml * (-dzLJ / rLJ3 - zzL / rSL3);

        /* === Etape 3 : vitesses === */
        vxT += fxT * dt; vyT += fyT * dt; vzT += fzT * dt;
        vxL += fxL * dt; vyL += fyL * dt; vzL += fzL * dt;
        vxJ += fxJ * dt; vyJ += fyJ * dt; vzJ += fzJ * dt;

        xT = xxT; yT = yyT; zT = zzT;
        xL = xxL; yL = yyL; zL = zzL;
        xJ = xxJ; yJ = yyJ; zJ = zzJ;

        /* === Echantillonnage === */
        if (step % store_every == 0 && n_samp < max_samp) {
            t_samp[n_samp]   = t;
            rST_samp[n_samp] = rST;
            rTL_samp[n_samp] = rTL;
            double rTL_xy = sqrt(dxTL*dxTL + dyTL*dyTL);
            cos_lambda_samp[n_samp] = (rTL_xy > 0) ? dxTL / rTL_xy : 1.0;
            sin_beta_samp[n_samp]   = dzTL / rTL;
            xT_samp[n_samp] = xxT;  yT_samp[n_samp] = yyT;  zT_samp[n_samp] = zzT;
            xTL_samp[n_samp] = dxTL; yTL_samp[n_samp] = dyTL; zTL_samp[n_samp] = dzTL;
            n_samp++;
        }

        /* === Tampon circulaire des distances === */
        switch (ring % 3) {
            case 0: rTL_h0 = rTL; rST_h0 = rST; break;
            case 1: rTL_h1 = rTL; rST_h1 = rST; break;
            case 2: rTL_h2 = rTL; rST_h2 = rST; break;
        }
        ring++;

        double rTL_curr, rTL_prev_val, rTL_prev2;
        double rST_curr, rST_prev_val, rST_prev2;
        {
            int ic = (ring - 1) % 3, ip = (ring - 2 + 3) % 3, ip2 = (ring - 3 + 6) % 3;
            double rh[3] = {rTL_h0, rTL_h1, rTL_h2};
            double sh[3] = {rST_h0, rST_h1, rST_h2};
            rTL_curr = rh[ic]; rTL_prev_val = rh[ip]; rTL_prev2 = rh[ip2];
            rST_curr = sh[ic]; rST_prev_val = sh[ip]; rST_prev2 = sh[ip2];
        }

        double yT_curr  = yyT;
        double zTL_curr = dzTL;
        double yTL_curr = dyTL;
        double xTL_curr = dxTL;

        /* === Detection d'evenements === */
        if (step >= 3) {

            /* --- Perigee lunaire --- */
            if (rTL_prev2 > rTL_prev_val && rTL_prev_val < rTL_curr
                && n_peri < max_peri) {
                double dx = sv_xL - sv_xT, dy = sv_yL - sv_yT, dz = sv_zL - sv_zT;
                double dvx = sv_vxL - sv_vxT, dvy = sv_vyL - sv_vyT, dvz = sv_vzL - sv_vzT;
                double r_g = sqrt(dx*dx + dy*dy + dz*dz);
                double v2  = dvx*dvx + dvy*dvy + dvz*dvz;
                double v_g = sqrt(v2);

                double hx = dy*dvz - dz*dvy;
                double hy = dz*dvx - dx*dvz;
                double hz = dx*dvy - dy*dvx;
                double h2 = hx*hx + hy*hy + hz*hz;

                double eps = v2 / 2.0 - mu_TL / r_g;
                double a_o = -mu_TL / (2.0 * eps);
                double ecc2 = 1.0 - h2 / (mu_TL * a_o);
                double ecc = (ecc2 > 0) ? sqrt(ecc2) : 0.0;

                double rp_ecl = sqrt(dx*dx + dy*dy);
                double cxi = (rp_ecl > 0) ? dx / rp_ecl : 1.0;
                double sxi = (rp_ecl > 0) ? dy / rp_ecl : 0.0;

                int k = n_peri * PERI_STRIDE;
                peri_data[k+0] = t - dt;
                peri_data[k+1] = rTL_prev_val;
                peri_data[k+2] = ecc;
                peri_data[k+3] = v_g / 1000.0;
                peri_data[k+4] = cxi;
                peri_data[k+5] = sxi;
                n_peri++;
            }

            /* --- Perihelie terrestre --- */
            if (rST_prev2 > rST_prev_val && rST_prev_val < rST_curr
                && n_perih < max_perih) {
                double v2e = sv_vxT*sv_vxT + sv_vyT*sv_vyT + sv_vzT*sv_vzT;
                double re  = sqrt(sv_xT*sv_xT + sv_yT*sv_yT + sv_zT*sv_zT);
                double hxe = sv_yT*sv_vzT - sv_zT*sv_vyT;
                double hye = sv_zT*sv_vxT - sv_xT*sv_vzT;
                double hze = sv_xT*sv_vyT - sv_yT*sv_vxT;
                double h2e = hxe*hxe + hye*hye + hze*hze;
                double epse = v2e / 2.0 - mu_ST / re;
                double ae = -mu_ST / (2.0 * epse);
                double ecc2e = 1.0 - h2e / (mu_ST * ae);
                double ecce = (ecc2e > 0) ? sqrt(ecc2e) : 0.0;

                int k = n_perih * PERIH_STRIDE;
                perih_data[k+0] = t - dt;
                perih_data[k+1] = rST_prev_val;
                perih_data[k+2] = ecce;
                n_perih++;
            }

            /* --- Noeud ascendant lunaire --- */
            if (zTL_prev < 0 && zTL_curr >= 0 && n_node < max_nodes) {
                double frac = -zTL_prev / (zTL_curr - zTL_prev);
                double t_ev = t - (1.0 - frac) * dt;

                double dx = dxTL, dy = dyTL, dz = dzTL;
                double dvx = vxL - vxT, dvy = vyL - vyT, dvz = vzL - vzT;
                double hhx = dy*dvz - dz*dvy;
                double hhy = dz*dvx - dx*dvz;
                double hhz = dx*dvy - dy*dvx;
                double hh  = sqrt(hhx*hhx + hhy*hhy + hhz*hhz);
                double incl = acos(hhz / hh) * (180.0 / M_PI);

                double re2d = sqrt(dx*dx + dy*dy);
                double som = (re2d > 0) ? dy / re2d : 0.0;
                double com = (re2d > 0) ? dx / re2d : 1.0;

                int k = n_node * NODE_STRIDE;
                node_data[k+0] = t_ev;
                node_data[k+1] = incl;
                node_data[k+2] = som;
                node_data[k+3] = com;
                n_node++;
            }

            /* --- Annee siderale terrestre --- */
            if (yT_prev < 0 && yT_curr >= 0 && xxT > 0 && n_year < max_years) {
                double frac = -yT_prev / (yT_curr - yT_prev);
                double t_ev = t - (1.0 - frac) * dt;

                if (t_last_year > 0) {
                    double dur = (t_ev - t_last_year) / 86400.0;

                    double v2e = vxT*vxT + vyT*vyT + vzT*vzT;
                    double re  = rST;
                    double hxe = yyT*vzT - zzT*vyT;
                    double hye = zzT*vxT - xxT*vzT;
                    double hze = xxT*vyT - yyT*vxT;
                    double h2e = hxe*hxe + hye*hye + hze*hze;
                    double epse = v2e / 2.0 - mu_ST / re;
                    double ae = -mu_ST / (2.0 * epse);
                    double ecc2e = 1.0 - h2e / (mu_ST * ae);
                    double ecce = (ecc2e > 0) ? sqrt(ecc2e) : 0.0;

                    int k = n_year * YEAR_STRIDE;
                    year_data[k+0] = t_ev;
                    year_data[k+1] = dur;
                    year_data[k+2] = rST;
                    year_data[k+3] = ecce;
                    n_year++;
                }
                t_last_year = t_ev;
            }

            /* --- Mois sideral lunaire --- */
            if (yTL_prev < 0 && yTL_curr >= 0 && xTL_curr > 0 && n_sid < max_sid) {
                double frac = -yTL_prev / (yTL_curr - yTL_prev);
                double t_ev = t - (1.0 - frac) * dt;

                if (t_last_sid > 0) {
                    double dur = (t_ev - t_last_sid) / 86400.0;
                    int k = n_sid * SID_STRIDE;
                    sid_data[k+0] = t_ev;
                    sid_data[k+1] = dur;
                    n_sid++;
                }
                t_last_sid = t_ev;
            }
        }

        yT_prev  = yT_curr;
        zTL_prev = zTL_curr;
        yTL_prev = yTL_curr;

        if (step % 50000000 == 0 && step > 0) {
            printf("  %ld / %ld  (%.1f%%)\n", step, n_steps,
                   100.0 * step / n_steps);
            fflush(stdout);
        }
    }

    *n_samp_out  = n_samp;
    *n_year_out  = n_year;
    *n_peri_out  = n_peri;
    *n_node_out  = n_node;
    *n_sid_out   = n_sid;
    *n_perih_out = n_perih;

    printf("Termine. Echantillons=%d, Annees=%d, Perigees=%d, "
           "Noeuds=%d, MoisSid=%d, Perihelies=%d\n",
           n_samp, n_year, n_peri, n_node, n_sid, n_perih);
    fflush(stdout);
}
