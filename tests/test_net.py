"""Das Zeigernetz in kleinen Einheiten (DEMO-PLAYBOOK §1b): Parameterzahlen, Kodierer und Dekodierer am Handbeispiel, Maske, Gradienten-Check gegen endliche Differenzen,
Eigenschaften (Umnummerieren, abgeschaltete Teile), Modelldatei."""

import itertools
import math

import numpy as np
import pytest
from conftest import zufallsinstanz

import bsl_constants as C
import bsl_net as N
from bsl_rngvec import UStream

VARIANTEN = {"voll": (1, True, True), "einbettung": (0, False, False), "attention": (1, True, False), "vorwaerts": (1, False, True)}


def _P(d=4, variante="voll", seed=5):
    layers, attn, ffn = VARIANTEN[variante]
    return N.init_params(d, UStream(seed), layers=layers, attn=attn, ffn=ffn)


# ------------------------------------------------------------------------------------------------------------- Parameter
@pytest.mark.parametrize("d,variante,erwartet", [(32, "voll", 12480), (16, "voll", 3168), (32, "einbettung", 4192), (32, "attention", 8288), (32, "vorwaerts", 8384), (2, "einbettung", 22)])
def test_parameterzahlen(d, variante, erwartet):
    P = _P(d, variante)
    assert N.n_params(P) == erwartet == N.param_count(d, *VARIANTEN[variante])
    info = N.describe(P)
    assert (info["d"], info["layers"], info["attn"], info["ffn"]) == (d, *VARIANTEN[variante])


def test_ohne_attention_und_vorwaertsnetz_gibt_es_keine_schicht():
    assert N.describe(N.init_params(8, UStream(1), layers=2, attn=False, ffn=False))["layers"] == 0
    assert N.param_count(8, 2, False, False) == N.param_count(8, 0, False, False)


def test_init_ist_deterministisch_und_haengt_vom_zufalls_seed_ab():
    a, b, c = _P(8, seed=3), _P(8, seed=3), _P(8, seed=4)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.allclose(a["W0"], c["W0"])
    assert np.all(a["b0"] == 0)


def test_init_streuung_ist_eins_durch_wurzel_eingang():
    P = N.init_params(32, UStream(9))
    for k, i in (("Wq0", 32), ("W10", 32), ("W20", 64), ("Wc", 96), ("Wkd", 32)):
        assert abs(P[k].std() - 1 / math.sqrt(i)) < 0.1 / math.sqrt(i)


def test_variantencode_trennt_die_varianten():
    assert [N.variant_code(*v) for v in VARIANTEN.values()] == [3, 0, 1, 2]
    assert N.variant_code(2, True, True) == 7


# ------------------------------------------------------------------------------------------------------------- Handbeispiel (Mini-Netz)
def test_einbettung_des_mini_netzes(hand_P, hand_x):
    H, _ = N.encode(hand_P, hand_x[None])
    soll = [(0.1, -0.1), (0.6, -0.6), (0.975, 0.15), (0.475, 0.65), (0.5375, 0.025)]
    assert np.allclose(H[0], soll, atol=1e-12)
    assert N.n_params(hand_P) == 22


def test_wahrscheinlichkeiten_des_ungelernten_mini_netzes_schritt_eins(hand_P, hand_x):
    ps, us = N.probabilities_along(hand_P, hand_x, [0, 4, 1, 2, 3], with_u=True)
    assert np.allclose(ps[0], [0, 0.5212, 0.2241, 0.0670, 0.1877], atol=1e-4)
    assert abs(us[0][1] - 1.44037) < 1e-4
    assert np.allclose(ps[1], [0, 0.7741, 0.2214, 0.0045, 0], atol=1e-4)           # Schritt 2 von Ort 4 aus
    assert np.allclose(ps[2], [0, 0, 0.9700, 0.0300, 0], atol=1e-4)
    assert np.allclose(ps[3], [0, 0, 0, 1, 0], atol=1e-12)                         # ein Ort frei: p = 1


def test_maske_besuchte_orte_haben_wahrscheinlichkeit_null_und_summe_eins(hand_P, hand_x):
    ps = N.probabilities_along(hand_P, hand_x, [0, 4, 1, 2, 3])
    besucht = np.zeros(5, dtype=bool)
    besucht[0] = True
    for t, ort in enumerate([4, 1, 2, 3]):
        assert np.all(ps[t][besucht] == 0.0) and abs(ps[t].sum() - 1.0) < 1e-12
        besucht[ort] = True


def test_greedy_tour_und_log_wahrscheinlichkeiten_des_mini_netzes(hand_P, hand_x):
    X = hand_x[None]
    starts = np.zeros((1, 1), dtype=np.int64)
    assert N.greedy_tours(hand_P, X, starts)[0, 0].tolist() == [0, 1, 4, 2, 3]
    H, _ = N.encode(hand_P, X, keep=False)
    for tour, soll in (([0, 1, 4, 2, 3], -0.99952), ([0, 4, 1, 2, 3], -1.95936)):
        out = N.decode(hand_P, H, starts, mode="forced", forced=np.array([[tour]]), keep=False)
        assert abs(out["logp"][0, 0] - soll) < 1e-4
    laenge = N.tour_lengths(np.array([C.HAND_PUNKTE]), np.array([[[0, 1, 4, 2, 3]]]))[0, 0]
    assert abs(laenge - 16.0) < 1e-12                                              # Optimum ist 15


def test_summe_der_wahrscheinlichkeiten_aller_touren_ist_eins(hand_P, hand_x):
    X = hand_x[None]
    touren = np.array([[(0,) + p for p in itertools.permutations(range(1, 5))]])
    assert touren.shape == (1, 24, 5)
    H, _ = N.encode(hand_P, X, keep=False)
    out = N.decode(hand_P, H, touren[:, :, 0], mode="forced", forced=touren, keep=False)
    assert abs(np.exp(out["logp"]).sum() - 1.0) < 1e-12
    # auch für ein größeres Netz mit Schicht: n = 6, alle 120 Touren ab Ort 0
    P = _P(6, "voll")
    x = zufallsinstanz(6, 1)[None]
    t6 = np.array([[(0,) + p for p in itertools.permutations(range(1, 6))]])
    H, _ = N.encode(P, x, keep=False)
    assert abs(np.exp(N.decode(P, H, t6[:, :, 0], mode="forced", forced=t6, keep=False)["logp"]).sum() - 1.0) < 1e-10


def test_step_logp_ist_die_einzelne_entscheidung_und_summiert_zur_tour(hand_P, hand_x):
    X = hand_x[None]
    H, _ = N.encode(hand_P, X, keep=False)
    starts = np.zeros((1, 1), dtype=np.int64)
    ctx = N.decoder_context(hand_P, H, starts)
    visited = np.zeros((1, 1, 5), dtype=bool)
    visited[0, 0, 0] = True
    lp = N.step_logp(hand_P, ctx, starts, visited)
    assert lp.shape == (1, 1, 5) and lp[0, 0, 0] == -np.inf
    assert np.allclose(np.exp(lp[0, 0, 1:]), [0.5212, 0.2241, 0.0670, 0.1877], atol=1e-4)
    tour = [0, 4, 1, 2, 3]
    visited[:] = False
    gesamt, cur = 0.0, np.array([[0]])
    visited[0, 0, 0] = True
    for ort in tour[1:]:
        gesamt += N.step_logp(hand_P, ctx, cur, visited)[0, 0, ort]
        visited[0, 0, ort] = True
        cur = np.array([[ort]])
    assert abs(gesamt - (-1.95936)) < 1e-4


# ------------------------------------------------------------------------------------------------------------- Einsatz
@pytest.mark.parametrize("variante", list(VARIANTEN))
def test_jede_tour_ist_eine_permutation_greedy_und_gezogen(variante):
    for n in (2, 3, 5, 8, 20):
        P = _P(8, variante, seed=n)
        X = np.stack([zufallsinstanz(n, 100 * n + i) for i in range(6)])
        starts = np.zeros((6, 1), dtype=np.int64)
        g = N.greedy_tours(P, X, starts)
        s = N.sample_tours(P, X, 3, UStream(n))
        assert N.is_permutation(g) and N.is_permutation(s)
        assert np.all(g[:, :, 0] == 0) and np.all(s[:, :, 0] == 0)


def test_tourlaenge_gleich_unabhaengiger_neuberechnung():
    P = _P(8)
    X = np.stack([zufallsinstanz(7, i) for i in range(5)])
    tours = N.sample_tours(P, X, 2, UStream(3), start=2)
    assert np.all(tours[:, :, 0] == 2)
    L = N.tour_lengths(X, tours)
    for m in range(5):
        for k in range(2):
            t = tours[m, k]
            soll = sum(math.hypot(*(X[m, t[i]] - X[m, t[(i + 1) % 7]])) for i in range(7))
            assert abs(L[m, k] - soll) < 1e-12


def test_ziehen_ist_deterministisch_und_folgt_den_wahrscheinlichkeiten(hand_P, hand_x):
    X = hand_x[None]
    a = N.sample_tours(hand_P, X, 5, UStream(8))
    assert np.array_equal(a, N.sample_tours(hand_P, X, 5, UStream(8)))
    assert not np.array_equal(a, N.sample_tours(hand_P, X, 5, UStream(9)))
    t = N.sample_tours(hand_P, X, 20000, UStream(1))[0]
    frei = np.bincount(t[:, 1], minlength=5) / 20000
    assert np.allclose(frei[1:], [0.5212, 0.2241, 0.0670, 0.1877], atol=0.015)    # erster Schritt wie berechnet


def test_greedy_gleichstand_nimmt_den_kleinsten_index():
    n = 4
    P = N.init_params(2, UStream(1), layers=0, attn=False, ffn=False)
    for k in P:
        P[k] = np.zeros_like(P[k])                                                 # alle Werte gleich: alle freien Orte gleich wahrscheinlich
    X = np.random.default_rng(0).random((1, n, 2))
    assert N.greedy_tours(P, X, np.zeros((1, 1), dtype=np.int64))[0, 0].tolist() == [0, 1, 2, 3]


def test_randfaelle_n2_und_n3():
    P = _P(4)
    X2 = zufallsinstanz(2, 1)[None]
    s = np.zeros((1, 1), dtype=np.int64)
    out = N.decode(P, N.encode(P, X2, keep=False)[0], s, mode="greedy", keep=False)
    assert out["tours"][0, 0].tolist() == [0, 1] and abs(out["logp"][0, 0]) < 1e-12        # eine Entscheidung mit p = 1
    F, G = N.loss_and_grad(P, X2, s, np.ones((1, 1)), np.array([[[0, 1]]]))
    assert abs(F) < 1e-12 and max(float(np.abs(g).max()) for g in G.values()) < 1e-12      # Gradient 0
    X3 = zufallsinstanz(3, 2)[None]
    assert sorted(N.greedy_tours(P, X3, s)[0, 0].tolist()) == [0, 1, 2]


# ------------------------------------------------------------------------------------------------------------- Gradienten-Check
def _gradient_abweichung(P, X, starts, w, tours, eps=1e-6):
    F0, G0 = N.loss_and_grad(P, X, starts, w, tours)
    schlimmste, zahl = 0.0, 0
    for k, a in P.items():
        flach = a.reshape(-1)
        for i in range(flach.size):
            alt = flach[i]
            flach[i] = alt + eps
            Fp, _ = N.loss_and_grad(P, X, starts, w, tours)
            flach[i] = alt - eps
            Fm, _ = N.loss_and_grad(P, X, starts, w, tours)
            flach[i] = alt
            num = (Fp - Fm) / (2 * eps)
            ana = G0[k].reshape(-1)[i]
            schlimmste = max(schlimmste, abs(ana - num) / max(1.0, abs(ana), abs(num)))
            zahl += 1
    return schlimmste, zahl


@pytest.mark.parametrize("variante", list(VARIANTEN))
def test_gradient_gegen_endliche_differenzen_alle_parameter(variante):
    P = _P(4, variante)
    r = np.random.default_rng(3)
    X = r.random((2, 5, 2))
    tours = np.array([[[0, 2, 4, 1, 3], [0, 3, 1, 4, 2]], [[1, 0, 2, 3, 4], [1, 4, 3, 2, 0]]])
    abw, zahl = _gradient_abweichung(P, X, tours[:, :, 0], r.normal(size=(2, 2)), tours)
    assert zahl == N.n_params(P) and abw < 1e-6


def test_gradient_des_mini_netzes_gegen_endliche_differenzen(hand_P, hand_x):
    X = np.stack([hand_x, hand_x])
    tours = np.array([[[0, 4, 1, 2, 3]], [[0, 1, 3, 2, 4]]])
    w = np.array([[-0.75], [0.75]])
    abw, zahl = _gradient_abweichung(hand_P, X, np.zeros((2, 1), dtype=np.int64), w, tours)
    assert zahl == 22 and abw < 1e-6


def test_gradient_mit_mehreren_starts_und_zwei_schichten():
    P = N.init_params(4, UStream(2), layers=2)
    r = np.random.default_rng(1)
    X = r.random((2, 5, 2))
    starts = np.tile(np.arange(5), (2, 1))
    tours = np.stack([np.stack([np.roll(np.arange(5), -s) for s in range(5)]) for _ in range(2)])
    abw, _ = _gradient_abweichung(P, X, starts, r.normal(size=(2, 5)), tours)
    assert abw < 1e-6


def test_gewicht_null_gibt_gradient_null():
    P = _P(4)
    X = zufallsinstanz(5, 2)[None]
    tours = np.array([[[0, 1, 2, 3, 4]]])
    F, G = N.loss_and_grad(P, X, tours[:, :, 0], np.zeros((1, 1)), tours)
    assert F == 0 and all(float(np.abs(g).max()) == 0.0 for g in G.values())


# ------------------------------------------------------------------------------------------------------------- Eigenschaften und Schalter
@pytest.mark.parametrize("variante", ["voll", "attention", "vorwaerts", "einbettung"])
def test_umnummerieren_vertauscht_nur_die_zeilen_der_einbettung(variante):
    P = _P(8, variante)
    X = np.stack([zufallsinstanz(9, i) for i in range(3)])
    perm = np.random.default_rng(4).permutation(9)
    H, _ = N.encode(P, X, keep=False)
    Hp, _ = N.encode(P, X[:, perm], keep=False)
    assert np.allclose(Hp, H[:, perm], atol=1e-12)


def test_greedy_folgt_der_umnummerierung():
    P = _P(8)
    X = zufallsinstanz(8, 3)[None]
    perm = np.random.default_rng(5).permutation(8)
    start_neu = int(np.where(perm == 0)[0][0])
    a = N.greedy_tours(P, X, np.zeros((1, 1), dtype=np.int64))[0, 0]
    b = N.greedy_tours(P, X[:, perm], np.full((1, 1), start_neu, dtype=np.int64))[0, 0]
    assert perm[b].tolist() == a.tolist()


def test_netz_mit_absoluten_koordinaten_ist_nicht_skaleninvariant():
    P = _P(8)
    X = np.stack([zufallsinstanz(12, i) for i in range(10)])
    starts = np.zeros((10, 1), dtype=np.int64)
    gleich = sum(np.array_equal(N.greedy_tours(P, X, starts)[i], N.greedy_tours(P, X * 10, starts)[i]) for i in range(10))
    assert gleich < 10                                                                # auf mindestens einer Instanz ändert sich die Tour


def test_abgeschaltete_teile_wirken_wie_fehlende_schluessel():
    X = zufallsinstanz(7, 8)[None]
    voll = _P(8, "voll")
    ohne_attn = {k: v for k, v in voll.items() if not k.startswith(("Wq", "Wk", "Wv", "Wo"))}
    attn_null = {**voll, "Wo0": np.zeros_like(voll["Wo0"])}
    assert N.describe(ohne_attn) == {"d": 8, "layers": 1, "attn": False, "ffn": True}
    assert np.allclose(N.encode(ohne_attn, X)[0], N.encode(attn_null, X)[0], atol=1e-12)       # Wo = 0 heißt: Attention ändert nichts
    ohne_ffn = {k: v for k, v in voll.items() if not k.startswith(("W1", "b1", "W2", "b2"))}
    ffn_null = {**voll, "W20": np.zeros_like(voll["W20"])}
    assert np.allclose(N.encode(ohne_ffn, X)[0], N.encode(ffn_null, X)[0], atol=1e-12)
    # jede Einheit mit Wirkung: Schlüssel entfernt ändert die Einbettung auf mindestens einer Instanz (Nullspalten-Regel)
    assert not np.allclose(N.encode(voll, X)[0], N.encode(ohne_attn, X)[0])
    assert not np.allclose(N.encode(voll, X)[0], N.encode(ohne_ffn, X)[0])
    assert not np.allclose(N.encode(ohne_attn, X)[0], N.encode(N.init_params(8, UStream(5), layers=0), X)[0])


def test_maske_wirkt_ohne_maske_waeren_besuchte_orte_wahrscheinlich(hand_P, hand_x):
    ps, us = N.probabilities_along(hand_P, hand_x, [0, 4, 1, 2, 3], with_u=True)
    unmaskiert = np.exp(us[1]) / np.exp(us[1]).sum()
    assert unmaskiert[0] > 0 and ps[1][0] == 0 and not np.allclose(unmaskiert, ps[1])


# ------------------------------------------------------------------------------------------------------------- Modelldatei
def test_modell_hin_und_zurueck(tmp_path):
    P = _P(8, "attention")
    pfad = tmp_path / "m.npz"
    N.save_model(pfad, P, {"steps": 12, "val_gap_greedy": 9.9})
    Q, meta = N.load_model(pfad)
    assert all(np.array_equal(P[k], Q[k]) for k in P) and set(P) == set(Q)
    assert meta["format"] == "znz-modell-1" and meta["steps"] == 12 and meta["n_params"] == N.n_params(P) and meta["attn"] is True and meta["ffn"] is False


def test_modell_laden_meldet_fehler(tmp_path):
    P = _P(8)
    pfad = tmp_path / "a.npz"
    N.save_model(pfad, P, {})
    with np.load(pfad, allow_pickle=False) as z:
        roh = {k: z[k] for k in z.files}
    for name, aend, text in (
        ("format", lambda r: r.update(meta=np.array('{"format": "x", "n_params": 1}')), "Format"),
        ("form", lambda r: r.update(Wc=np.zeros((5, 5))), "Form"),
        ("fehlt", lambda r: r.pop("Wkd"), "passen nicht"),
        ("nparams", lambda r: r.update(meta=np.array('{"format": "znz-modell-1", "n_params": 3}')), "n_params"),
        ("ohne_meta", lambda r: r.pop("meta"), "meta"),
    ):
        r = dict(roh)
        aend(r)
        np.savez(tmp_path / f"{name}.npz", **r)
        with pytest.raises(ValueError, match=text):
            N.load_model(tmp_path / f"{name}.npz")
