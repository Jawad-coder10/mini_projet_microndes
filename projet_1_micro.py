#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse transitoire d'une ligne de transmission à deux conducteurs couplés
avec effet de peau (Mini-Projet 1, Micro-ondes, Pr. A. Maaouni).

Méthode
-------
1. Les 8 grandeurs V1(0), V2(0), I1(0), I2(0), V1(L), V2(L), I1(L), I2(L)
   sont calculées dans le domaine de Laplace (découplage modal pair/impair,
   matrice chaîne de Paul, conditions de Thévenin) : fonction `reponse_lmc`.
2. Elles sont inversées numériquement par l'algorithme NILT de L. Brancik
   (FFT + algorithme epsilon de Wynn) : fonction `nilt`, portage Python de
   la version « vectorielle » `niltv` du programme MATLAB de l'article
   L. Brancik, "Programs for Fast Numerical Inversion of Laplace Transforms
   in Matlab Language Environment", MATLAB 99, Prague, 1999, pp. 27-39.

Utilisation
-----------
    python nilt_lmc.py            # validations + 8 courbes (dossier resultats/)
    python nilt_lmc.py --rapide   # 8 courbes seulement (sans les validations)
"""
import argparse
import os
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# 1. Paramètres de la ligne et des terminaisons (énoncé, Fig. 3)
# ----------------------------------------------------------------------------
@dataclass
class Ligne:
    ell: float = 0.3048                                   # longueur (m)
    L: np.ndarray = field(default_factory=lambda: np.array(
        [[494.6, 63.3], [63.3, 494.6]]) * 1e-9)           # H/m
    C: np.ndarray = field(default_factory=lambda: np.array(
        [[62.8, -4.9], [-4.9, 62.8]]) * 1e-12)            # F/m
    Rdc: float = 0.3691                                   # ohm/m
    b: float = 0.01 * np.sqrt(2 * np.pi)                  # coefficient B (effet de peau)
    RS: tuple = (50.0, 100.0)                             # source : conducteurs 1 et 2
    RL: tuple = (100.0, 100.0)                            # charge : conducteurs 1 et 2


# Excitation : trapèze de 1 V, montée = descente = 1,5 ns, plateau jusqu'à 6 ns
TR, TT, TD = 1.5e-9, 6.0e-9, 1.5e-9


def vs_trapeze(s, tr=TR, T=TT, td=TD):
    """Transformée de Laplace du trapèze : somme de 4 rampes retardées."""
    return (1 - np.exp(-s * tr) - np.exp(-s * T) + np.exp(-s * (T + td))) / (tr * s**2)


def vs_echelon(s):
    return 1.0 / s


def vs_temps(t, tr=TR, T=TT, td=TD):
    """Trapèze dans le domaine temporel (pour la simulation de référence)."""
    r = lambda x: np.maximum(x, 0.0)
    return (r(t) - r(t - tr) - r(t - T) + r(t - T - td)) / tr


NOMS = ["V1(0)", "V2(0)", "I1(0)", "I2(0)", "V1(L)", "V2(L)", "I1(L)", "I2(L)"]


# ----------------------------------------------------------------------------
# 2. Les 8 expressions fréquentielles (tâche 1)
# ----------------------------------------------------------------------------
def reponse_lmc(s, lg, VS=vs_trapeze):
    """
    Retourne un tableau (8, len(s)) : [V1(0), V2(0), I1(0), I2(0),
                                       V1(L), V2(L), I1(L), I2(L)].
    Formules fermées : modes pair/impair (T = [[1,1],[1,-1]]/sqrt(2)),
    matrice chaîne Phi (p,q,u,w,x,y), système D I(0) = N.
    """
    s = np.asarray(s, dtype=complex)
    a = lg.Rdc + lg.b * np.sqrt(s)                         # a(s) = Rdc + B sqrt(s)
    Le, Lo = lg.L[0, 0] + lg.L[0, 1], lg.L[0, 0] - lg.L[0, 1]
    Ce, Co = lg.C[0, 0] + lg.C[0, 1], lg.C[0, 0] - lg.C[0, 1]

    def mode(Lm, Cm):
        gam = np.sqrt(s * Cm * (a + s * Lm))               # racine principale: Re(gam) >= 0
        Zc = (a + s * Lm) / gam
        return gam, Zc

    ge, Zce = mode(Le, Ce)
    go, Zco = mode(Lo, Co)
    che, she = np.cosh(ge * lg.ell), np.sinh(ge * lg.ell)
    cho, sho = np.cosh(go * lg.ell), np.sinh(go * lg.ell)

    p, q = (che + cho) / 2, (che - cho) / 2
    u, w = (Zce * she + Zco * sho) / 2, (Zce * she - Zco * sho) / 2
    x, y = (she / Zce + sho / Zco) / 2, (she / Zce - sho / Zco) / 2

    RS1, RS2 = lg.RS
    RL1, RL2 = lg.RL
    D11 = p * (RS1 + RL1) + u + RL1 * RS1 * x
    D12 = q * (RS2 + RL1) + w + RL1 * RS2 * y
    D21 = q * (RS1 + RL2) + w + RL2 * RS1 * y
    D22 = p * (RS2 + RL2) + u + RL2 * RS2 * x

    V = VS(s)
    N1 = (p + RL1 * x) * V
    N2 = (q + RL2 * y) * V
    Delta = D11 * D22 - D12 * D21

    I10 = (D22 * N1 - D12 * N2) / Delta
    I20 = (D11 * N2 - D21 * N1) / Delta
    V10 = V - RS1 * I10
    V20 = -RS2 * I20
    I1L = -x * V + (p + x * RS1) * I10 + (q + y * RS2) * I20
    I2L = -y * V + (q + y * RS1) * I10 + (p + x * RS2) * I20
    V1L = RL1 * I1L
    V2L = RL2 * I2L
    return np.vstack([V10, V20, I10, I20, V1L, V2L, I1L, I2L])


def reponse_expm(s, lg, VS=vs_trapeze):
    """Même calcul par la méthode directe de Paul : Phi(L) = expm(A L) (éq. 7.112),
    sans aucun découplage. Sert uniquement de référence de validation."""
    from scipy.linalg import expm
    s = np.atleast_1d(np.asarray(s, dtype=complex))
    ZS, ZL = np.diag(lg.RS), np.diag(lg.RL)
    out = np.zeros((8, s.size), dtype=complex)
    for k, sk in enumerate(s):
        Z = (lg.Rdc + lg.b * np.sqrt(sk)) * np.eye(2) + sk * lg.L
        Y = sk * lg.C
        A = np.block([[np.zeros((2, 2)), -Z], [-Y, np.zeros((2, 2))]])
        Ph = expm(A * lg.ell)
        P11, P12, P21, P22 = Ph[:2, :2], Ph[:2, 2:], Ph[2:, :2], Ph[2:, 2:]
        VSv = np.array([VS(sk), 0.0])
        D = P11 @ ZS - P12 + ZL @ P22 - ZL @ P21 @ ZS
        I0 = np.linalg.solve(D, (P11 - ZL @ P21) @ VSv)
        V0 = VSv - ZS @ I0
        IL = P21 @ VSv + (P22 - P21 @ ZS) @ I0
        VL = ZL @ IL
        out[:, k] = np.concatenate([V0, I0, VL, IL])
    return out


# ----------------------------------------------------------------------------
# 3. NILT de Brancik : FFT + algorithme epsilon (portage de `niltv`)
# ----------------------------------------------------------------------------
def nilt(F, tm, M=256, P=2, alfa=0.0):
    """
    Inversion numérique rapide de la transformée de Laplace.

    F    : fonction s -> tableau (J, len(s)) (ou (len(s),)) : J transformées à inverser
    tm   : instant maximal (s) ; la réponse est calculée sur [0, tm]
    M    : nombre de points temporels (puissance de 2)
    P    : paramètre de l'algorithme epsilon (2 ou 3 d'après Brancik)
    alfa : ordre exponentiel de la fonction originale (0 pour un système stable)

    Retourne (t, f) avec t de taille (M,) et f de taille (J, M).
    Correspondance avec le code MATLAB : N=2M, wyn=2P+1, NT=2*tm*N/(N-2),
    omega=2*pi/NT, c=alfa+25/NT, s=c-i*omega*(0:N+wyn-2).
    """
    N = 2 * M
    wyn = 2 * P + 1
    t = np.linspace(0.0, tm, M)
    NT = 2.0 * tm * N / (N - 2)
    omega = 2.0 * np.pi / NT
    c = alfa + 25.0 / NT

    s = c - 1j * omega * np.arange(N + wyn - 1)
    Fsc = np.atleast_2d(F(s))                              # (J, N+wyn-1)
    J = Fsc.shape[0]

    # (i) somme des N premiers termes par FFT, tronquée à M points
    ft0 = np.fft.fft(Fsc[:, :N], axis=1)[:, :M]

    # (ii) 2P termes supplémentaires -> 2P+1 sommes partielles
    ft = np.empty((wyn, J, M), dtype=complex)
    ft[0] = ft0
    for n in range(N, N + wyn - 1):
        ft[n - N + 1] = Fsc[:, n][:, None] * np.exp(-1j * n * omega * t)[None, :]
    ft1 = np.cumsum(ft, axis=0)                            # sommes partielles S_0..S_2P

    # (iii) algorithme epsilon de Wynn (losange de la Fig. 1 de l'article)
    ft2 = np.zeros((wyn - 1, J, M), dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        for I in range(1, wyn - 1):
            nouv = ft2 + 1.0 / np.diff(ft1, axis=0)
            ft2 = ft1[1:wyn - I]
            ft1 = nouv
        res = (ft2 + 1.0 / np.diff(ft1, axis=0))[0]        # epsilon_{2P}^{(0)}

    f = 2.0 * np.real(res) - np.real(Fsc[:, 0])[:, None]
    f = np.exp(c * t)[None, :] / NT * f
    f[:, 0] *= 2.0                                         # point t = 0 (moyenne des limites)
    return t, f


# ----------------------------------------------------------------------------
# 4. Validations
# ----------------------------------------------------------------------------
def val_nilt_scalaire():
    print("\n[1] NILT seule, sur des transformées dont l'original est connu")
    print("    (l'origine et les discontinuités sont exclues, comme le précise Brancik)")
    tests = [
        ("1/(s+1)            -> exp(-t)", lambda s: 1 / (s + 1), 10.0,
         lambda t: np.exp(-t), None),
        ("(1-e^-s)/s^2       -> rampe saturée", lambda s: (1 - np.exp(-s)) / s**2, 6.0,
         lambda t: np.minimum(t, 1.0), None),
        ("1/(s^2+1)          -> sin(t)", lambda s: 1 / (s**2 + 1), 12.0,
         lambda t: np.sin(t), None),
        ("e^-2s/(s+1)        -> exp(-(t-2)) u(t-2)", lambda s: np.exp(-2 * s) / (s + 1), 10.0,
         lambda t: np.where(t >= 2, np.exp(-(t - 2)), 0.0), 0.1),
    ]
    for nom, F, tm, ex, excl in tests:
        t, f = nilt(F, tm, M=256)
        err = np.abs(f[0] - ex(t))
        masque = t > 0.05 * tm                             # on exclut le voisinage de l'origine
        if excl is not None:                               # ... et celui du saut
            masque &= np.abs(t - 2.0) > excl
        print(f"    {nom:48s} erreur max (t > {0.05 * tm:.2f}) = {err[masque].max():.1e}")


def val_formules_vs_expm(lg):
    print("\n[2] Formules fermées (tâche 1) contre Phi = expm(A L), 3 valeurs de s")
    for s in (1e6 + 2e8j, 5e7 + 1e9j, 3e8 + 5e9j):
        a = reponse_lmc(s, lg)[:, 0]
        r = reponse_expm(s, lg)[:, 0]
        print(f"    s = {s:.2e}: écart relatif max = {np.max(np.abs(a - r) / np.abs(r)):.1e}")


def reference_echelle(lg, t, N=200):
    """
    Simulation temporelle indépendante (SANS effet de peau, b = 0) : la ligne
    est découpée en N cellules LC(R) couplées, intégrée exactement en
    variables d'état (scipy.signal.lsim, entrée affine par morceaux).
    Retourne un tableau (8, len(t)).
    """
    from scipy.signal import StateSpace, lsim
    dz = lg.ell / N
    Cn, Cend = lg.C * dz, lg.C * dz / 2
    Ld = lg.L * dz
    Rd = lg.Rdc * dz * np.eye(2)
    Ldinv, Cninv, Cendinv = (np.linalg.inv(m) for m in (Ld, Cn, Cend))
    RSi, RLi = np.diag(1 / np.array(lg.RS)), np.diag(1 / np.array(lg.RL))
    n = 2 * (N + (N + 1))
    iI = lambda k: slice(2 * (k - 1), 2 * k)                # courant de la cellule k (1..N)
    iV = lambda k: slice(2 * N + 2 * k, 2 * N + 2 * k + 2)  # tension du noeud k (0..N)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for k in range(1, N + 1):                               # L di_k/dt = v_{k-1} - v_k - R i_k
        A[iI(k), iV(k - 1)] += Ldinv
        A[iI(k), iV(k)] -= Ldinv
        A[iI(k), iI(k)] -= Ldinv @ Rd
    A[iV(0), iV(0)] -= Cendinv @ RSi                        # noeud 0 : source + premier courant
    A[iV(0), iI(1)] -= Cendinv
    B[iV(0), 0] = Cendinv @ (RSi @ np.array([1.0, 0.0]))
    for k in range(1, N):                                   # noeuds internes
        A[iV(k), iI(k)] += Cninv
        A[iV(k), iI(k + 1)] -= Cninv
    A[iV(N), iI(N)] += Cendinv                              # noeud N : dernier courant + charge
    A[iV(N), iV(N)] -= Cendinv @ RLi
    Cmat = np.zeros((8, n))
    Dmat = np.zeros((8, 1))
    Cmat[0, 2 * N + 0] = 1.0                                # V1(0)
    Cmat[1, 2 * N + 1] = 1.0                                # V2(0)
    Cmat[2, 2 * N + 0] = -1.0 / lg.RS[0]; Dmat[2, 0] = 1.0 / lg.RS[0]   # I1(0) = (Vs - v0)/RS1
    Cmat[3, 2 * N + 1] = -1.0 / lg.RS[1]                    # I2(0) = -v0/RS2
    Cmat[4, 2 * N + 2 * N + 0] = 1.0                        # V1(L)
    Cmat[5, 2 * N + 2 * N + 1] = 1.0                        # V2(L)
    Cmat[6, 2 * N + 2 * N + 0] = 1.0 / lg.RL[0]             # I1(L)
    Cmat[7, 2 * N + 2 * N + 1] = 1.0 / lg.RL[1]             # I2(L)
    sys = StateSpace(A, B, Cmat, Dmat)
    _, y, _ = lsim(sys, U=vs_temps(t), T=t, interp=True)
    return y.T


def val_echelle(lg):
    print("\n[3] NILT + formules contre une simulation temporelle indépendante")
    print("    (échelle de 200 cellules LC(R) couplées, cas sans effet de peau b = 0)")
    lg0 = Ligne(b=0.0)
    tm = 40e-9
    t, f = nilt(lambda s: reponse_lmc(s, lg0), tm, M=1024)
    tr = np.linspace(0, tm, 4001)
    y = reference_echelle(lg0, tr)
    yi = np.array([np.interp(t, tr, y[k]) for k in range(8)])
    pic = np.max(np.abs(yi), axis=1)
    e = np.abs(f - yi)
    rms = np.sqrt(np.mean(e**2, axis=1))
    for k in range(8):
        print(f"    {NOMS[k]}: pic = {pic[k]:9.3e}  écart RMS = {100 * rms[k] / pic[k]:5.2f} % du pic  "
              f"écart max = {100 * e[k].max() / pic[k]:5.2f} % du pic")
    print("    (les écarts max de V2(L), I2(L) sont aux fronts raides : sans pertes, la FEXT est")
    print("     discontinue, cas où la NILT est la moins précise ; l'effet de peau les lisse.)")


def val_regime_etabli(lg):
    print("\n[4] Régime établi : échelon de 1 V, fenêtre de 100 ns")
    R = lg.Rdc * lg.ell
    I1 = 1 / (lg.RS[0] + lg.RL[0] + R)
    att = {"I1(0)": I1, "V1(0)": 1 - lg.RS[0] * I1, "V1(L)": lg.RL[0] * I1,
           "I1(L)": I1, "V2(0)": 0.0, "V2(L)": 0.0, "I2(0)": 0.0, "I2(L)": 0.0}
    t, f0 = nilt(lambda s: reponse_lmc(s, Ligne(b=0.0), vs_echelon), 100e-9, M=1024)
    t, f1 = nilt(lambda s: reponse_lmc(s, lg, vs_echelon), 100e-9, M=1024)
    print("    valeur continue attendue = pont diviseur avec Rdc*L seule")
    print(f"    {'':6s}{'attendu':>13s}{'b = 0':>13s}{'avec peau':>13s}")
    for nom, val in att.items():
        k = NOMS.index(nom)
        print(f"    {nom:6s}{val:13.6e}{f0[k, -5:].mean():13.6e}{f1[k, -5:].mean():13.6e}")
    fc = lg.Rdc**2 / (np.pi * lg.b**2)
    print(f"    Sans effet de peau, on retrouve la valeur continue. Avec B donné, B*sqrt(pi f) = Rdc")
    print(f"    dès f = {fc:.0f} Hz : la résistance ne revient à Rdc qu'après des millisecondes,")
    print(f"    donc à 100 ns le courant est encore inférieur à la valeur continue (attendu).")


def val_convergence(lg):
    print("\n[5] Sensibilité aux paramètres de la NILT (trapèze, tm = 40 ns)")
    t, ref = nilt(lambda s: reponse_lmc(s, lg), 40e-9, M=1024, P=2)
    pic = np.max(np.abs(ref), axis=1)
    for M, P in ((512, 2), (2048, 2), (1024, 3)):
        t2, f2 = nilt(lambda s: reponse_lmc(s, lg), 40e-9, M=M, P=P)
        f2i = np.array([np.interp(t, t2, f2[k]) for k in range(8)])
        e = np.max((np.abs(f2i - ref) / pic[:, None])[:, t > 0.3e-9])
        print(f"    (M={M:4d}, P={P}) contre (M=1024, P=2) : écart max / pic (t > 0,3 ns) = {e:.1e}")


# ----------------------------------------------------------------------------
# 5. Simulation et tracé des 8 courbes
# ----------------------------------------------------------------------------
def simuler(lg, tm=40e-9, M=1024, P=2):
    return nilt(lambda s: reponse_lmc(s, lg), tm, M=M, P=P)


def tracer(t, f, f_sans=None, chemin="huit_courbes.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordre = [0, 1, 4, 5, 2, 3, 6, 7]                       # tensions puis courants
    titres = {0: "$V_1(0)$ (entrée, conducteur actif)", 1: "$V_2(0)$ (NEXT)",
              4: "$V_1(L)$ (sortie, conducteur actif)", 5: "$V_2(L)$ (FEXT)",
              2: "$I_1(0)$", 3: "$I_2(0)$", 6: "$I_1(L)$", 7: "$I_2(L)$"}
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
    for a, k in zip(ax.ravel(), ordre):
        echelle = 1.0 if k in (0, 1, 4, 5) else 1e3
        unite = "V" if k in (0, 1, 4, 5) else "mA"
        a.plot(t * 1e9, f[k] * echelle, lw=1.6, label="avec effet de peau")
        if f_sans is not None:
            a.plot(t * 1e9, f_sans[k] * echelle, "--", lw=1.2, label="sans effet de peau")
        if k == 0:
            a.plot(t * 1e9, vs_temps(t), ":", color="gray", lw=1.2, label="$V_i(t)$")
        a.set_title(titres[k], fontsize=10)
        a.set_ylabel(unite)
        a.grid(alpha=0.3)
        if k in (6, 7, 2, 3):
            a.set_xlabel("t (ns)")
        if k in (0, 4, 2) and f_sans is not None:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160)
    plt.close(fig)


def resume(t, f):
    print("\nRepères pour l'interprétation :")
    for k in (1, 5):
        i = np.argmax(np.abs(f[k]))
        print(f"    {NOMS[k]}: extremum = {f[k, i] * 1e3:+7.2f} mV à t = {t[i] * 1e9:5.2f} ns")
    for k in (0, 4):
        i = np.argmax(f[k])
        print(f"    {NOMS[k]}: maximum  = {f[k, i]:.3f} V à t = {t[i] * 1e9:5.2f} ns")
    i1 = int(np.argmax(np.abs(f[4]) > 0.02 * np.max(np.abs(f[4]))))
    print(f"    V1(L) dépasse 2 % de son maximum à t = {t[i1] * 1e9:.2f} ns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true", help="pas de validations")
    ap.add_argument("--sortie", default="resultats", help="dossier de sortie")
    args = ap.parse_args()
    os.makedirs(args.sortie, exist_ok=True)
    lg = Ligne()

    if not args.rapide:
        val_nilt_scalaire()
        val_formules_vs_expm(lg)
        val_echelle(lg)
        val_regime_etabli(lg)
        val_convergence(lg)

    t, f = simuler(lg)
    _, f0 = simuler(Ligne(b=0.0))
    tracer(t, f, None, os.path.join(args.sortie, "huit_courbes.png"))
    tracer(t, f, f0, os.path.join(args.sortie, "comparaison_effet_peau.png"))
    np.savetxt(os.path.join(args.sortie, "resultats.csv"),
               np.column_stack([t * 1e9, f.T]), delimiter=",", header="t_ns," + ",".join(NOMS), comments="")
    resume(t, f)
    print(f"\nFichiers écrits dans '{args.sortie}/'.")


if __name__ == "__main__":
    main()
