"""Vier Orte, ein Schritt exakt (Handbeispiel aus PLAN-02, handbeispiele_02.py Abschnitte A bis D): Längen, Wahrscheinlichkeiten, Score und Gradient von Hand, endliche Differenz als zweiter Weg,
Erwartung und Streuung aller Kombinationen in Brüchen und in Gleitkomma, Mittel = (B-1)/B mal Leave-one-out, POMO mit N = 4 (Orakel O3)."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import bsl_constants as C
import bsl_tabular as TB

ZERO = np.zeros(4)


def _fd(f, theta, h=1e-6):
    g = np.zeros(4)
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        g[j] = (f(np.asarray(theta) + e) - f(np.asarray(theta) - e)) / (2 * h)
    return g


def test_touren_und_laengen_von_hand():
    assert TB.touren_ab(0) == [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)]
    assert TB.laengen_ab(0) == [14, 16, 18, 16, 18, 14]
    assert TB.ABSTAND[0][1] == 3 and TB.ABSTAND[1][2] == 4 and TB.ABSTAND[0][2] == 5 and TB.ABSTAND[1][3] == 5
    assert min(TB.laengen_ab(0)) == 14 and sum(TB.laengen_ab(0)) / 6 == 16
    assert all(sorted(TB.laengen_ab(s)) == [14, 14, 16, 16, 18, 18] for s in range(4))


def test_held_karp_der_vier_orte_gleich_aufzaehlung():
    """Teilmengen-DP über die vier Orte (unabhängig von bsl_exact) gegen das Minimum der Aufzählung: 14."""
    d = TB.ABSTAND
    dp = {(1 << j, j): d[0][j] for j in (1, 2, 3)}
    for size in (2, 3):
        for sub in itertools.combinations((1, 2, 3), size):
            mask = sum(1 << j for j in sub)
            for j in sub:
                dp[(mask, j)] = min(dp[(mask ^ (1 << j), i)] + d[i][j] for i in sub if i != j)
    assert min(dp[(0b1110, j)] + d[j][0] for j in (1, 2, 3)) == min(TB.laengen_ab(0)) == 14


def test_bei_theta_null_hat_jede_tour_ein_sechstel_und_der_score_der_ersten_tour_ist_von_hand():
    assert np.allclose(TB.wahrscheinlichkeiten(ZERO), 1 / 6, atol=1e-15)
    assert TB._scores_gleich(0)[0] == [0, Fr(4, 6), Fr(1, 6), Fr(-5, 6)]                      # (0; 4; 1; -5)/6
    S = TB.scores_ab(ZERO)
    assert np.allclose(S[0], [0, 4 / 6, 1 / 6, -5 / 6]) and np.allclose(S.sum(axis=1), 0) and np.allclose(S.sum(axis=0), 0)


@pytest.mark.parametrize("politik", C.POLITIKEN)
def test_score_gleich_endlicher_differenz_von_log_p(politik):
    theta = np.array(C.POLITIK_THETA[politik])
    for t in TB.touren_ab(0):
        assert np.allclose(TB.score(theta, t), _fd(lambda th, t=t: TB.log_wahrscheinlichkeit(th, t), theta), atol=1e-8)


@pytest.mark.parametrize("politik", C.POLITIKEN)
def test_wahrer_gradient_gleich_endlicher_differenz_von_j(politik):
    theta = np.array(C.POLITIK_THETA[politik])
    assert abs(TB.wahrscheinlichkeiten(theta).sum() - 1) < 1e-14
    assert np.allclose(TB.wahrer_gradient(theta), _fd(TB.erwartete_laenge, theta), atol=1e-7)
    alle = lambda th: sum(TB.erwartete_laenge(th, s) for s in range(4)) / 4  # noqa: E731
    assert np.allclose(TB.wahrer_gradient_alle_starts(theta), _fd(alle, theta), atol=1e-7)


def test_gradient_bei_theta_null_ist_null_ein_sechstel_minus_ein_sechstel_null():
    assert np.allclose(TB.wahrer_gradient(ZERO), [0, 1 / 6, -1 / 6, 0])


def test_greedy_laenge_bei_gleichstand_baut_die_tour_mit_dem_kleinsten_index():
    assert TB.greedy_laenge(ZERO) == 14 and TB.greedy_laenge(C.POLITIK_THETA["kurz"]) == 14 and TB.greedy_laenge(C.POLITIK_THETA["lang"]) == 18


def test_gewichte_bei_theta_null():
    assert np.allclose(TB.gewichte_je_tour(ZERO, "mittel")["gewichte"], [-2, 0, 2, 0, 2, -2])             # L - 16
    assert np.allclose(TB.gewichte_je_tour(ZERO, "rollout")["gewichte"], [0, 2, 4, 2, 4, 0])               # L - 14, alle nicht negativ
    assert np.allclose(TB.gewichte_je_tour(ZERO, "keiner")["gewichte"], [14, 16, 18, 16, 18, 14])
    with pytest.raises(ValueError):
        TB.gewichte_je_tour(ZERO, "pomo")


def test_ein_schritt_mit_zwei_touren_in_bruechen():
    erw = {f: TB.enumerate_step_exact(2, f) for f in TB.FORMEN_EXAKT}
    G = erw["keiner"]["wahrer_gradient"]
    assert G == [0, Fr(1, 6), Fr(-1, 6), 0]
    for f in ("keiner", "rollout", "loo"):
        assert erw[f]["erwartung"] == G
    assert erw["mittel"]["erwartung"] == [x / 2 for x in G]                                                    # (B-1)/B = 1/2
    assert erw["keiner"]["gesamtstreuung"] == Fr(5431, 36)
    assert erw["mittel"]["gesamtstreuung"] == Fr(7, 9)
    assert erw["rollout"]["gesamtstreuung"] == Fr(139, 36)
    assert erw["loo"]["gesamtstreuung"] == Fr(28, 9)
    assert erw["mittel"]["gesamtstreuung"] * 4 == erw["loo"]["gesamtstreuung"]                                # hochgerechnet 28/9
    with pytest.raises(ValueError):
        TB.enumerate_step_exact(2, "pomo")


@pytest.mark.parametrize("B", [2, 3, 4])
def test_mittel_ist_b_minus_1_durch_b_mal_leave_one_out_fuer_jede_auswahl(B):
    m, lo, k = TB.enumerate_step_exact(B, "mittel"), TB.enumerate_step_exact(B, "loo"), TB.enumerate_step_exact(B, "keiner")
    G = k["wahrer_gradient"]
    assert m["erwartung"] == [Fr(B - 1, B) * x for x in lo["erwartung"]] and lo["erwartung"] == G and k["erwartung"] == G
    assert m["gesamtstreuung"] * Fr(B, B - 1) ** 2 == lo["gesamtstreuung"]
    assert k["gesamtstreuung"] > 20 * lo["gesamtstreuung"]                                                    # ohne Referenzwert mehr als 20-mal so streuend


@pytest.mark.parametrize("B", [2, 3, 4])
def test_gleitkomma_aufzaehlung_gleicht_den_bruechen_bei_theta_null(B):
    for f in TB.FORMEN_EXAKT:
        exakt = TB.enumerate_step_exact(B, f)
        r = TB.enumerate_step(ZERO, B, f)
        assert np.allclose(r["erwartung"], [float(x) for x in exakt["erwartung"]], atol=1e-12)
        assert abs(r["gesamtstreuung"] - float(exakt["gesamtstreuung"])) < 1e-9


def test_faktor_und_richtung_der_zeilen():
    z = {r["form"]: r for r in TB.schritt_zeilen(ZERO, 2)}
    assert abs(z["keiner"]["faktor"] - 1) < 1e-12 and abs(z["mittel"]["faktor"] - 0.5) < 1e-12 and abs(z["rollout"]["faktor"] - 1) < 1e-12 and abs(z["loo"]["faktor"] - 1) < 1e-12
    assert math.isnan(z["pomo"]["faktor"]) and z["pomo"]["faktor_formel"] == 0.75                           # bei theta = 0 ist der Gradient von J_alle null: Faktor nicht messbar
    assert np.allclose(z["pomo"]["wahrer_gradient"], 0, atol=1e-14)
    assert all(abs(r["richtung"] - 1) < 1e-12 for r in z.values() if r["form"] != "pomo")                  # Erwartung zeigt in Richtung des Gradienten
    assert abs(TB.hochgerechnete_streuung(z["mittel"]) - 28 / 9) < 1e-12
    assert abs(z["keiner"]["gesamtstreuung"] / TB.hochgerechnete_streuung(z["mittel"]) - 5431 / 36 / (28 / 9)) < 1e-9            # 48,5 bei B = 2


@pytest.mark.parametrize("politik", C.POLITIKEN)
@pytest.mark.parametrize("B", [2, 3, 4])
def test_erwartungsfaktoren_gelten_fuer_jede_politik(politik, B):
    theta = C.POLITIK_THETA[politik]
    for r in TB.schritt_zeilen(theta, B):
        soll = {"keiner": 1, "rollout": 1, "loo": 1, "mittel": (B - 1) / B, "pomo": 0.75}[r["form"]]
        assert r["faktor_formel"] == soll
        if r["form"] == "pomo" and politik == "gleich":
            assert math.isnan(r["faktor"])                                                                 # Gradient 0, siehe oben
        else:
            assert abs(r["faktor"] - soll) < 1e-9, (politik, B, r["form"])


def test_pomo_mit_vier_startorten_alle_1296_kombinationen():
    p = TB.pomo_step(ZERO, True)
    n_ = TB.pomo_step(ZERO, False)
    assert math.isnan(p["faktor"]) and math.isnan(n_["faktor"])                                              # bei theta = 0 ist der Gradient von J_alle null
    for politik in ("kurz", "lang"):                                                                         # dort ist er nicht null: der Satz (N-1)/N ist nicht leer
        th = C.POLITIK_THETA[politik]
        assert np.linalg.norm(TB.wahrer_gradient_alle_starts(th)) > 0.05
        assert abs(TB.pomo_step(th, True)["faktor"] - 0.75) < 1e-9 and abs(TB.pomo_step(th, False)["faktor"] - 1.0) < 1e-9
    assert round(p["gesamtstreuung"], 3) == 0.575 and round(n_["gesamtstreuung"], 2) == 75.43
    assert n_["gesamtstreuung"] > 50 * p["gesamtstreuung"]


def test_pomo_streuung_exakt_in_bruechen_gleich_gleitkomma():
    """Gesamtstreuung 0,575 gegen 75,43 aus dem Handbeispiel, unabhängig in Brüchen nachgerechnet (alle 6^4 Kombinationen)."""
    Ls = [[Fr(x) for x in TB.laengen_ab(s)] for s in range(4)]
    Ss = [TB._scores_gleich(s) for s in range(4)]
    EP, EN, VP, VN = [Fr(0)] * 4, [Fr(0)] * 4, Fr(0), Fr(0)
    w = Fr(1, 6 ** 4)
    for idx in itertools.product(range(6), repeat=4):
        L = [Ls[s][idx[s]] for s in range(4)]
        mean = sum(L) / 4
        gp = [sum((L[s] - mean) * Ss[s][idx[s]][j] for s in range(4)) / 4 for j in range(4)]
        gn = [sum(L[s] * Ss[s][idx[s]][j] for s in range(4)) / 4 for j in range(4)]
        for j in range(4):
            EP[j] += w * gp[j]
            EN[j] += w * gn[j]
        VP += w * sum(x * x for x in gp)
        VN += w * sum(x * x for x in gn)
    VP -= sum(x * x for x in EP)
    VN -= sum(x * x for x in EN)
    assert abs(float(VP) - TB.pomo_step(ZERO, True)["gesamtstreuung"]) < 1e-10 and abs(float(VN) - TB.pomo_step(ZERO, False)["gesamtstreuung"]) < 1e-9
    assert round(float(VP), 3) == 0.575 and round(float(VN), 2) == 75.43


def test_randfaelle():
    with pytest.raises(ValueError):
        TB.enumerate_step(ZERO, 1, "mittel")
    with pytest.raises(ValueError):
        TB.enumerate_step(ZERO, 2, "pomo")
    r = TB.enumerate_step(ZERO, 1, "keiner")                                                               # eine Tour je Schritt: die Schätzung ist der Score mal L
    assert abs(r["faktor"] - 1) < 1e-12
    assert math.isnan(TB.hochgerechnete_streuung({"faktor_formel": 0.0, "gesamtstreuung": 1.0}))
