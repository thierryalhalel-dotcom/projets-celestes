/*
 * energie_core.c : comparaison de trois integrateurs sur le probleme a trois
 * corps Soleil - Terre - Lune (memes equations (I) heliocentriques que
 * simulation_core.c), avec suivi de l'energie totale du systeme.
 *
 * Methodes : 0 = Euler explicite, 1 = Runge-Kutta 4, 2 = Euler symplectique
 * (position puis acceleration puis vitesse, comme dans simulation_core.c).
 *
 * L'energie est l'energie mecanique totale, calculee dans le referentiel du
 * barycentre (seul referentiel ou elle est conservee) a partir des positions
 * et vitesses heliocentriques.
 *
 * Compilation : gcc -O2 -shared -fPIC -o libenergie.so energie_core.c -lm
 */
#include <math.h>

static double G, MS, MT, ML;

/* etat : s[0..2] = r_T, s[3..5] = r_L, s[6..8] = v_T, s[9..11] = v_L */
static void accel(const double *s, double *aT, double *aL)
{
    double xT = s[0], yT = s[1], zT = s[2];
    double xL = s[3], yL = s[4], zL = s[5];
    double rST = sqrt(xT*xT + yT*yT + zT*zT);
    double rSL = sqrt(xL*xL + yL*yL + zL*zL);
    double dx = xL - xT, dy = yL - yT, dz = zL - zT;
    double rTL = sqrt(dx*dx + dy*dy + dz*dz);
    double rST3 = rST*rST*rST, rSL3 = rSL*rSL*rSL, rTL3 = rTL*rTL*rTL;
    aT[0] = -G*(MS+MT)/rST3*xT + G*ML*(dx/rTL3 - xL/rSL3);
    aT[1] = -G*(MS+MT)/rST3*yT + G*ML*(dy/rTL3 - yL/rSL3);
    aT[2] = -G*(MS+MT)/rST3*zT + G*ML*(dz/rTL3 - zL/rSL3);
    aL[0] = -G*(MS+ML)/rSL3*xL + G*MT*(-dx/rTL3 - xT/rST3);
    aL[1] = -G*(MS+ML)/rSL3*yL + G*MT*(-dy/rTL3 - yT/rST3);
    aL[2] = -G*(MS+ML)/rSL3*zL + G*MT*(-dz/rTL3 - zT/rST3);
}

static void deriv(const double *s, double *d)
{
    double aT[3], aL[3];
    accel(s, aT, aL);
    for (int k = 0; k < 6; k++) d[k] = s[6+k];
    for (int k = 0; k < 3; k++) { d[6+k] = aT[k]; d[9+k] = aL[k]; }
}

static double energie(const double *s)
{
    double M = MS + MT + ML, vS[3], ke, pe;
    for (int k = 0; k < 3; k++) vS[k] = -(MT*s[6+k] + ML*s[9+k]) / M;   /* vitesse du Soleil / barycentre */
    ke = 0.0;
    for (int k = 0; k < 3; k++) {
        double vT = s[6+k] + vS[k], vL = s[9+k] + vS[k];
        ke += 0.5*(MS*vS[k]*vS[k] + MT*vT*vT + ML*vL*vL);
    }
    double rST = sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    double rSL = sqrt(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]);
    double dx = s[3]-s[0], dy = s[4]-s[1], dz = s[5]-s[2];
    double rTL = sqrt(dx*dx + dy*dy + dz*dz);
    pe = -G*(MS*MT/rST + MS*ML/rSL + MT*ML/rTL);
    return ke + pe;
}

/* Integre n_steps pas. Pour chaque intervalle de store_every pas, stocke le
   MAXIMUM de |E/E0 - 1| atteint dans l'intervalle (enveloppe de l'erreur) :
   un simple echantillonnage tomberait sur des phases arbitraires des
   oscillations mensuelles et annuelles de l'energie. */
long run(int methode, double dt, long n_steps, int store_every,
         const double *etat0, double g, double ms, double mt, double ml,
         double *E_out)
{
    double s[12], d1[12], d2[12], d3[12], d4[12], tmp[12], aT[3], aL[3];
    G = g; MS = ms; MT = mt; ML = ml;
    for (int k = 0; k < 12; k++) s[k] = etat0[k];
    long ns = 0;
    double E0 = energie(s), emax = 0.0;
    E_out[ns++] = 0.0;
    for (long i = 1; i <= n_steps; i++) {
        if (methode == 0) {                       /* Euler explicite */
            deriv(s, d1);
            for (int k = 0; k < 12; k++) s[k] += dt*d1[k];
        } else if (methode == 1) {                /* Runge-Kutta 4 */
            deriv(s, d1);
            for (int k = 0; k < 12; k++) tmp[k] = s[k] + 0.5*dt*d1[k];
            deriv(tmp, d2);
            for (int k = 0; k < 12; k++) tmp[k] = s[k] + 0.5*dt*d2[k];
            deriv(tmp, d3);
            for (int k = 0; k < 12; k++) tmp[k] = s[k] + dt*d3[k];
            deriv(tmp, d4);
            for (int k = 0; k < 12; k++) s[k] += dt/6.0*(d1[k] + 2*d2[k] + 2*d3[k] + d4[k]);
        } else {                                  /* Euler symplectique */
            for (int k = 0; k < 6; k++) s[k] += dt*s[6+k];   /* 1. positions */
            accel(s, aT, aL);                                 /* 2. accelerations */
            for (int k = 0; k < 3; k++) { s[6+k] += dt*aT[k]; s[9+k] += dt*aL[k]; }  /* 3. vitesses */
        }
        double e = fabs(energie(s)/E0 - 1.0);
        if (e > emax) emax = e;
        if (i % store_every == 0) { E_out[ns++] = emax; emax = 0.0; }
    }
    return ns;
}

/*
 * Variante : Euler symplectique en coordonnees BARYCENTRIQUES (referentiel
 * galileen), les trois corps etant integres avec leurs attractions mutuelles.
 * Dans ces variables, le systeme est hamiltonien a hamiltonien separable
 * H = T(v) + V(r) et le schema est rigoureusement symplectique.
 * etat0 : meme format heliocentrique que run() ; conversion interne.
 */
long run_bary(double dt, long n_steps, int store_every,
              const double *etat0, double g, double ms, double mt, double ml,
              double *E_out)
{
    double m[3] = {ms, mt, ml}, r[3][3], v[3][3], a[3][3];
    double M = ms + mt + ml;
    G = g; MS = ms; MT = mt; ML = ml;
    /* heliocentrique -> barycentrique */
    for (int k = 0; k < 3; k++) {
        double Rc = (mt*etat0[k] + ml*etat0[3+k]) / M;
        double Vc = (mt*etat0[6+k] + ml*etat0[9+k]) / M;
        r[0][k] = -Rc;               v[0][k] = -Vc;
        r[1][k] = etat0[k] - Rc;     v[1][k] = etat0[6+k] - Vc;
        r[2][k] = etat0[3+k] - Rc;   v[2][k] = etat0[9+k] - Vc;
    }
    long ns = 0;
    for (long i = 0; i <= n_steps; i++) {
        if (i > 0) {
            for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) r[b][k] += dt*v[b][k];
            for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) a[b][k] = 0.0;
            for (int b = 0; b < 3; b++) for (int c = b+1; c < 3; c++) {
                double d[3], d2 = 0.0;
                for (int k = 0; k < 3; k++) { d[k] = r[c][k] - r[b][k]; d2 += d[k]*d[k]; }
                double inv3 = 1.0/(d2*sqrt(d2));
                for (int k = 0; k < 3; k++) { a[b][k] += G*m[c]*d[k]*inv3; a[c][k] -= G*m[b]*d[k]*inv3; }
            }
            for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) v[b][k] += dt*a[b][k];
        }
        if (i % store_every == 0) {
            double ke = 0.0, pe = 0.0;
            for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) ke += 0.5*m[b]*v[b][k]*v[b][k];
            for (int b = 0; b < 3; b++) for (int c = b+1; c < 3; c++) {
                double d2 = 0.0;
                for (int k = 0; k < 3; k++) { double d = r[c][k] - r[b][k]; d2 += d*d; }
                pe -= G*m[b]*m[c]/sqrt(d2);
            }
            E_out[ns++] = ke + pe;
        }
    }
    return ns;
}
