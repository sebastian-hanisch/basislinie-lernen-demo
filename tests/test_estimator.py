"""Experiment 1 (n = 6): Pool, Aufzählung, Streuung je Form (kleine Wiederholungszahl für die Testlaufzeit)."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_estimator as E1
import bsl_learn as LE


def test_pool_ist_deterministisch_und_gleichverteilt_im_einheitsquadrat():
    a, b, c = E1.pool_instanzen(0), E1.pool_instanzen(0), E1.pool_instanzen(1)
    assert a.shape == (16, 6, 2) and np.array_equal(a, b) and not np.array_equal(a, c) and a.min() >= 0 and a.max() < 1


def test_alle_touren_sind_permutationen_ab_dem_start():
    t = E1.alle_touren(6, 0)
    assert t.shape == (120, 6) and (t[:, 0] == 0).all() and all(sorted(r) == list(range(6)) for r in t) and len({tuple(r) for r in t}) == 120
    assert E1.alle_touren(6, 3).shape == (120, 6) and (E1.alle_touren(6, 3)[:, 0] == 3).all()


def test_exakter_gradient_fuer_pomo_ist_ein_anderes_ziel_und_summen_stimmen():
    P = LE.initial_params(8, 2)
    X = E1.pool_instanzen(0)[:3]
    g0, a0 = E1.exakter_gradient(P, X, alle_starts=False)
    g1, a1 = E1.exakter_gradient(P, X, alle_starts=True)
    assert a0 < 1e-12 and a1 < 1e-12 and g0.shape == g1.shape and not np.allclose(g0, g1)


def test_streuung_ohne_referenzwert_ist_groesser_als_mit_und_faktoren_aus_der_formel():
    P = LE.initial_params(8, 2)
    pool = E1.pool_instanzen(0)
    zeilen = E1.tabelle(P, 12, reps=40, seed=0, pool=pool)
    z = {r["form"]: r for r in zeilen}
    assert [r["form"] for r in zeilen] == list(C.FORMEN)
    for f in ("mittel", "rollout", "pomo"):
        assert z["keiner"]["rel_streuung"] > 2 * z[f]["rel_streuung"], f
    assert z["keiner"]["faktor"] == 1.0 and abs(z["mittel"]["faktor"] - 11 / 12) < 1e-15 and abs(z["pomo"]["faktor"] - 5 / 6) < 1e-15 and z["rollout"]["faktor"] == 1.0
    assert (z["pomo"]["instanzen_je_schritt"], z["mittel"]["instanzen_je_schritt"], z["rollout"]["baseline_touren_je_schritt"], z["keiner"]["baseline_touren_je_schritt"]) == (2, 12, 12, 0)
    assert all(0.0 <= r["anteil_negativ"] <= 1.0 for r in zeilen) and z["keiner"]["anteil_negativ"] == 0.0      # ohne Referenzwert ist jedes Gewicht eine positive Länge


def test_wiederholung_ist_deterministisch_und_pomo_braucht_vielfache_von_sechs():
    P = LE.initial_params(8, 2)
    pool = E1.pool_instanzen(0)
    a = E1.schaetzer_streuung(P, pool, "mittel", 6, reps=10, seed=1)
    b = E1.schaetzer_streuung(P, pool, "mittel", 6, reps=10, seed=1)
    assert a["rel_streuung"] == b["rel_streuung"]
    with pytest.raises(ValueError):
        E1.schaetzer_streuung(P, pool, "pomo", 7, reps=4)
