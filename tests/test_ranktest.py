"""Rangtests ohne scipy zur Laufzeit, mit scipy als Gegenprobe (Orakel O6, O7): exaktes Wilcoxon, Holm, Friedman (Schließformel), Quadratsummenzerlegung (auch per kleinste Quadrate)."""

import math

import numpy as np
import pytest

import bsl_ranktest as RT

stats = pytest.importorskip("scipy.stats")


def test_zehn_paare_gleiches_vorzeichen_geben_den_kleinsten_wert_2_durch_1024():
    w = RT.wilcoxon_exakt(np.arange(1, 11))
    assert abs(w["p"] - 2 / 1024) < 1e-15 and w["n"] == 10 and w["w_plus"] == 55 and w["w_minus"] == 0
    assert abs(RT.holm([w["p"]] + [1.0] * 5)[0] - 0.01171875) < 1e-15                    # Holm über 6 Vergleiche: 0,0117
    assert abs(RT.wilcoxon_exakt(np.arange(1, 9))["p"] - 2 / 256) < 1e-15                # acht Paare: 2/256, mal 6 = 0,047
    assert RT.holm([2 / 256] + [1.0] * 5)[0] < 0.05 < RT.holm([2 / 64] + [1.0] * 5)[0]


def test_wilcoxon_gleich_scipy_exakt():
    rng = np.random.default_rng(5)
    for _ in range(25):
        d = rng.normal(size=10)
        assert abs(RT.wilcoxon_exakt(d)["p"] - float(stats.wilcoxon(d).pvalue)) < 1e-12


def test_wilcoxon_nullen_werden_weggelassen_und_randfaelle():
    assert RT.wilcoxon_exakt([0.0, 0.0])["p"] == 1.0 and RT.wilcoxon_exakt([])["n"] == 0
    assert RT.wilcoxon_exakt([0.0, 1.0, 2.0, 3.0])["n"] == 3
    with pytest.raises(ValueError):
        RT.wilcoxon_exakt(np.arange(1, 16))
    # Gleichstände in den Beträgen: mittlere Ränge, p gültig
    p = RT.wilcoxon_exakt([1, -1, 2, 2, 3, -3, 4, 5])["p"]
    assert 0 < p <= 1


def test_holm_ist_monoton_und_gleich_der_handrechnung():
    assert np.allclose(RT.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert RT.holm([0.5]) == [0.5] and RT.holm([]) == []
    assert all(p <= 1 for p in RT.holm([0.9, 0.8, 0.7]))


@pytest.mark.parametrize("x,df", [(27.0, 3), (19.08, 3), (4.0, 2), (0.5, 1), (7.8, 3)])
def test_chi_quadrat_schliessformel_gleich_scipy(x, df):
    assert abs(RT.chi2_sf(x, df) - float(stats.chi2.sf(x, df))) < 1e-14
    assert RT.chi2_sf(0, 2) == 1.0
    with pytest.raises(ValueError):
        RT.chi2_sf(1.0, 5)


def test_friedman_gleich_scipy_mit_und_ohne_gleichstaende():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(10, 4)) + np.array([0, 0.3, 0.6, 0.9])
    r = RT.friedman(t)
    s = stats.friedmanchisquare(*t.T)
    assert abs(r["chi2"] - float(s.statistic)) < 1e-9 and abs(r["p"] - float(s.pvalue)) < 1e-9 and r["df"] == 3
    t2 = np.round(t, 0)                                            # viele Gleichstände
    r2, s2 = RT.friedman(t2), stats.friedmanchisquare(*t2.T)
    assert abs(r2["chi2"] - float(s2.statistic)) < 1e-9
    drei = RT.friedman(t[:, :3])
    assert drei["df"] == 2 and abs(drei["p"] - float(stats.friedmanchisquare(*t[:, :3].T).pvalue)) < 1e-9


def test_friedman_vollstaendige_ordnung_gibt_den_groessten_wert_und_p_wie_m1():
    t = np.tile([1.0, 2.0, 3.0, 4.0], (10, 1))
    r = RT.friedman(t)
    assert abs(r["chi2"] - 30.0) < 1e-12 and r["mittlere_raenge"] == [1.0, 2.0, 3.0, 4.0]
    assert abs(RT.chi2_sf(27.0, 3) / 5.89e-6 - 1) < 0.01 and abs(RT.chi2_sf(19.08, 3) / 2.63e-4 - 1) < 0.01          # M1: p = 5,9e-6 bzw. 2,6e-4


def test_paarweise_zaehlt_besser_und_holm_ueber_alle_paare():
    sp = {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "c": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}
    paare = RT.paarweise(sp)
    assert len(paare) == 3
    ab = next(p for p in paare if (p["a"], p["b"]) == ("a", "b"))
    assert ab["a_besser"] == 10 and ab["b_besser"] == 0 and abs(ab["diff_mittel"] + 1.0) < 1e-12 and abs(ab["p"] - 2 / 1024) < 1e-15
    assert abs(ab["p_holm"] - 6 / 1024) < 1e-15                                                         # drei Vergleiche: 3 * 2/1024


def test_quadratsummen_der_gekreuzten_tabellen_von_m1_und_kleinste_quadrate():
    c10 = np.array([[3.84, 5.38, 3.74, 4.06, 3.83], [4.38, 4.46, 3.36, 3.37, 3.59], [3.88, 3.63, 3.73, 3.69, 3.95], [4.29, 3.31, 3.70, 3.81, 3.65], [3.90, 4.26, 3.40, 3.70, 3.60]])
    c20 = np.array([[10.01, 10.36, 10.21, 9.30], [10.90, 9.11, 9.80, 9.88], [9.89, 10.10, 10.19, 9.73], [9.38, 9.13, 9.74, 10.96]])
    for tab, soll in ((c10, (13.1, 29.0, 57.9)), (c20, (1.8, 7.5, 90.7))):
        q = RT.quadratsummen(tab)
        assert abs(q["gewichte"] + q["daten"] + q["rest"] - 100) < 1e-9
        assert all(abs(a - b) < 0.3 for a, b in zip((q["gewichte"], q["daten"], q["rest"]), soll))
        z, s = tab.shape                                              # unabhängig: kleinste Quadrate mit Indikatorspalten
        X = np.zeros((z * s, 1 + (z - 1) + (s - 1)))
        X[:, 0] = 1
        for i in range(z):
            for j in range(s):
                if i > 0:
                    X[i * s + j, i] = 1
                if j > 0:
                    X[i * s + j, z - 1 + j] = 1
        y = tab.ravel()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rest = float(((y - X @ beta) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()) * 100
        assert abs(rest - q["rest"]) < 1e-9


def test_quadratsummen_randfall_alles_gleich():
    q = RT.quadratsummen(np.ones((3, 3)))
    assert q == {"gewichte": 0.0, "daten": 0.0, "rest": 0.0, "gesamt": 0.0} and not math.isnan(q["rest"])
