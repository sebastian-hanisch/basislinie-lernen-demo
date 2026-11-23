"""Referenzwerte als kleine Einheiten (DEMO-PLAYBOOK §1b), an der Handrechnung des Rechtecks 3 x 4 (Längen 14, 16, 18, 16, 18, 14, Mittel 16) und mit scipy als Gegenprobe der t-Regel."""

import itertools

import numpy as np
import pytest

import bsl_baseline as BL
import bsl_constants as C
import bsl_learn as LE
import bsl_net as N

L6 = np.array([[14.0, 16.0, 18.0, 16.0, 18.0, 14.0]])


def test_keiner_ist_null_und_gewichte_sind_die_laengen_durch_anzahl():
    assert BL.reference_value("keiner", L6) == 0.0
    assert np.allclose(BL.advantage_weights(L6, 0.0), L6 / 6)


def test_mittel_ist_das_mittel_aller_laengen_ein_skalar():
    b = BL.reference_value("mittel", L6)
    assert isinstance(b, float) and b == 16.0
    assert np.allclose(BL.advantage_weights(L6, b) * 6, [-2, 0, 2, 0, 2, -2])          # Gewichte (L - 16) des Handbeispiels


def test_mittel_ueber_den_batch_zaehlt_alle_instanzen_nicht_je_zeile():
    L = np.array([[10.0], [20.0], [30.0]])
    assert BL.reference_value("mittel", L) == 20.0


def test_pomo_ist_das_mittel_je_instanz_ueber_die_startorte():
    L = np.array([[1.0, 3.0, 5.0], [10.0, 10.0, 16.0]])
    b = BL.reference_value("pomo", L)
    assert b.shape == (2, 1) and np.allclose(b[:, 0], [3.0, 12.0])


def test_leave_one_out_ist_das_mittel_der_anderen():
    b = BL.reference_value("loo", np.array([[14.0, 18.0]]))
    assert np.allclose(b, [[18.0, 14.0]])
    L = np.array([[14.0, 16.0, 18.0, 16.0]])
    assert np.allclose(BL.advantage_weights(L, BL.reference_value("mittel", L)) * (4 / 3), BL.advantage_weights(L, BL.reference_value("loo", L)))   # Mittel = (B-1)/B mal Leave-one-out
    with pytest.raises(ValueError):
        BL.reference_value("loo", np.array([[1.0]]))


def test_rollout_braucht_die_laengen_des_referenznetzes_und_gibt_sie_zurueck():
    ref = np.array([[14.0], [14.0]])
    assert np.array_equal(BL.reference_value("rollout", np.array([[18.0], [16.0]]), ref), ref)
    with pytest.raises(ValueError):
        BL.reference_value("rollout", L6)
    with pytest.raises(ValueError):
        BL.reference_value("unbekannt", L6)


def test_rollout_gewichte_der_handrechnung_sind_nicht_negativ():
    w = BL.advantage_weights(L6, BL.reference_value("rollout", L6, np.array([[14.0]])))
    assert np.allclose(w * 6, [0, 2, 4, 2, 4, 0]) and (w >= 0).all()


def test_erwartungsfaktoren():
    assert BL.expected_gradient_scale("keiner") == BL.expected_gradient_scale("rollout") == BL.expected_gradient_scale("loo") == 1.0
    assert BL.expected_gradient_scale("mittel", 2) == 0.5 and abs(BL.expected_gradient_scale("mittel", 120) - 119 / 120) < 1e-15
    assert BL.expected_gradient_scale("pomo", 120, 20) == 0.95 and BL.expected_gradient_scale("pomo", 120, 10) == 0.9
    with pytest.raises(ValueError):
        BL.expected_gradient_scale("x")


def test_t_schranke_gleich_scipy():
    stats = pytest.importorskip("scipy.stats")
    assert abs(float(stats.t.ppf(0.95, 499)) - C.T_CRIT_499) < 1e-5


def test_austausch_regel_gleich_scipy_in_200_zufallsfaellen():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(11)
    gesehen = set()
    for _ in range(200):
        a = rng.normal(0.0, 1.0, 500)
        b = a + rng.normal(-0.06, 1.0, 500)               # b gegen a: aktuelles Netz b, Referenz a
        p = stats.ttest_rel(b, a, alternative="less").pvalue
        entscheidung = BL.rollout_should_replace(b, a)
        assert entscheidung == (p < 0.05)
        gesehen.add(entscheidung)
    assert gesehen == {True, False}                         # beide Zweige kommen vor


def test_austausch_randfaelle():
    gleich = np.ones(500)
    assert BL.paired_t(gleich, gleich) == 0.0 and BL.rollout_should_replace(gleich, gleich) is False
    assert BL.rollout_should_replace(gleich - 1.0, gleich) is True and BL.rollout_should_replace(gleich + 1.0, gleich) is False


def test_baseline_fn_je_form_am_netz():
    P = LE.initial_params(8, 3)
    X = np.random.default_rng(1).random((4, 5, 2))
    starts = np.zeros((4, 1), dtype=np.int64)
    L = N.tour_lengths(X, N.greedy_tours(P, X, starts))
    assert BL.make_baseline_fn("keiner")(X, starts, L) == 0.0
    assert BL.make_baseline_fn("mittel")(X, starts, L) == float(L.mean())
    fn = BL.make_baseline_fn("rollout", {"ref": P})
    assert np.allclose(fn(X, starts, L), L)                  # Referenznetz = dasselbe Netz: Greedy-Längen
    assert BL.make_baseline_fn("pomo")(X, np.tile(np.arange(5), (4, 1)), np.ones((4, 5))).shape == (4, 1)
    with pytest.raises(ValueError):
        BL.make_baseline_fn("unbekannt")


def test_konstanter_referenzwert_aendert_den_exakten_gradienten_nicht():
    """Gradient mit den Gewichten p L und p (L - b) über die vollständige Aufzählung aller 24 Touren: gleich (Summe der Scores ist 0)."""
    P = LE.initial_params(8, 5)
    X = np.random.default_rng(2).random((1, 5, 2))
    tours = np.array([(0,) + p for p in itertools.permutations(range(1, 5))])[None]            # (1, 24, 5)
    starts = np.zeros((1, 24), dtype=np.int64)
    H, _ = N.encode(P, X)
    out = N.decode(P, H, starts, mode="forced", forced=tours, keep=False)
    p = np.exp(out["logp"])
    assert abs(p.sum() - 1.0) < 1e-12
    L = N.tour_lengths(X, tours)
    _, G1 = N.loss_and_grad(P, X, starts, p * L, tours)
    _, G2 = N.loss_and_grad(P, X, starts, p * (L - 7.3), tours)
    for k in G1:
        assert np.allclose(G1[k], G2[k], atol=1e-10), k
