#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass, field
import numpy as np
# ----------------------------------------------------------------------------
# 1. Paramètres de la ligne et des terminaisons (énoncé, Fig. 3)
# ----------------------------------------------------------------------------
@dataclass
class Ligne:
    ell: float = 0.3048
    L: np.ndarray = field(default_factory=lambda: np.array(
        [[494.6, 63.3], [63.3, 494.6]]) * 1e-9)
    C: np.ndarray = field(default_factory=lambda: np.array(
        [[62.8, -4.9], [-4.9, 62.8]]) * 1e-12)
    Rdc: float = 0.3691
    b: float = 0.01 * np.sqrt(2 * np.pi)
    RS: tuple = (50.0, 100.0)
    RL: tuple = (100.0, 100.0)

TR, TT, TD = 1.5e-9, 6.0e-9, 1.5e-9

def vs_trapeze(s, tr=TR, T=TT, td=TD):
    return (1 - np.exp(-s * tr) - np.exp(-s * T) + np.exp(-s * (T + td))) / (tr * s**2)

def vs_echelon(s):
    return 1.0 / s

def vs_temps(t, tr=TR, T=TT, td=TD):
    r = lambda x: np.maximum(x, 0.0)
    return (r(t) - r(t - tr) - r(t - T) + r(t - T - td)) / tr

NOMS = ["V1(0)", "V2(0)", "I1(0)", "I2(0)", "V1(L)", "V2(L)", "I1(L)", "I2(L)"]
# ----------------------------------------------------------------------------
# 2. Les 8 expressions fréquentielles (tâche 1)
# ----------------------------------------------------------------------------
def reponse_lmc(s, lg, VS=vs_trapeze):
    s = np.asarray(s, dtype=complex)
    a = lg.Rdc + lg.b * np.sqrt(s)
    Le, Lo = lg.L[0, 0] + lg.L[0, 1], lg.L[0, 0] - lg.L[0, 1]
    Ce, Co = lg.C[0, 0] + lg.C[0, 1], lg.C[0, 0] - lg.C[0, 1]

    def mode(Lm, Cm):
        gam = np.sqrt(s * Cm * (a + s * Lm))
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
# ----------------------------------------------------------------------------
# 3. NILT de Brancik : FFT + algorithme epsilon (portage de `niltv`)
# ----------------------------------------------------------------------------
def nilt(F, tm, M=256, P=2, alfa=0.0):

    N = 2 * M
    wyn = 2 * P + 1
    t = np.linspace(0.0, tm, M)
    NT = 2.0 * tm * N / (N - 2)
    omega = 2.0 * np.pi / NT
    c = alfa + 25.0 / NT

    s = c - 1j * omega * np.arange(N + wyn - 1)
    Fsc = np.atleast_2d(F(s))
    J = Fsc.shape[0]

    ft0 = np.fft.fft(Fsc[:, :N], axis=1)[:, :M]

    ft = np.empty((wyn, J, M), dtype=complex)
    ft[0] = ft0
    for n in range(N, N + wyn - 1):
        ft[n - N + 1] = Fsc[:, n][:, None] * np.exp(-1j * n * omega * t)[None, :]
    ft1 = np.cumsum(ft, axis=0)

    ft2 = np.zeros((wyn - 1, J, M), dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        for I in range(1, wyn - 1):
            nouv = ft2 + 1.0 / np.diff(ft1, axis=0)
            ft2 = ft1[1:wyn - I]
            ft1 = nouv
        res = (ft2 + 1.0 / np.diff(ft1, axis=0))[0]

    f = 2.0 * np.real(res) - np.real(Fsc[:, 0])[:, None]
    f = np.exp(c * t)[None, :] / NT * f
    f[:, 0] *= 2.0
    return t, f
# ----------------------------------------------------------------------------
# 4. Validations légères
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
        masque = t > 0.05 * tm
        if excl is not None:
            masque &= np.abs(t - 2.0) > excl
        print(f"    {nom:48s} erreur max (t > {0.05 * tm:.2f}) = {err[masque].max():.1e}")

def val_regime_etabli(lg):
    print("\n[2] Régime établi : échelon de 1 V, fenêtre de 100 ns")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true", help="pas de validations")
    ap.add_argument("--sortie", default="resultats", help="dossier de sortie")
    args = ap.parse_args()
    os.makedirs(args.sortie, exist_ok=True)
    lg = Ligne()

    if not args.rapide:
        val_nilt_scalaire()
        val_regime_etabli(lg)

    t, f = simuler(lg)
    _, f0 = simuler(Ligne(b=0.0))
    tracer(t, f, None, os.path.join(args.sortie, "huit_courbes.png"))
    tracer(t, f, f0, os.path.join(args.sortie, "comparaison_effet_peau.png"))
    np.savetxt(os.path.join(args.sortie, "resultats.csv"),
               np.column_stack([t * 1e9, f.T]), delimiter=",", header="t_ns," + ",".join(NOMS), comments="")
    print(f"\nFichiers écrits dans '{args.sortie}/'.")

if __name__ == "__main__":
    main()
