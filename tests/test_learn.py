"""Trainingsschleife mit vier Referenzwerten: Zähler gegen die Formeln (Plan O8), Zweig-Test (jede Form weicht ab, Nullspalten-Regel), gleiche Anfangsgewichte, Zeitwächter, Vektor-Rückfall."""

import numpy as np
import pytest

import bsl_baseline as BL
import bsl_constants as C
import bsl_learn as LE
import bsl_net as N
import bsl_posten as PO


def _lauf(form, steps=20, n=6, batch=12, d=8, **kw):
    return LE.train_form(n, form, steps, batch=batch, d=d, init_seed=kw.pop("init_seed", 4), data_seed=kw.pop("data_seed", 9), eval_every=0, **kw)


def test_alle_formen_starten_mit_denselben_gewichten():
    ps = [LE.initial_params(8, 4) for _ in C.FORMEN]
    assert all(all(np.array_equal(ps[0][k], p[k]) for k in p) for p in ps)
    P0, _, _ = LE.train_form(6, "mittel", 0, batch=12, d=8, init_seed=4, eval_every=0)
    assert all(np.array_equal(P0[k], ps[0][k]) for k in P0)                                   # null Schritte: nur die Initialisierung


def test_nach_20_schritten_weicht_jede_form_von_jeder_anderen_ab():
    erg = {f: _lauf(f)[0] for f in C.FORMEN}
    for i, a in enumerate(C.FORMEN):
        for b in C.FORMEN[i + 1:]:
            assert any(not np.allclose(erg[a][k], erg[b][k]) for k in erg[a]), (a, b)


def test_gleicher_zufalls_seed_gleiches_training_anderer_seed_andere_gewichte():
    a, b, c = _lauf("mittel")[0], _lauf("mittel")[0], _lauf("mittel", init_seed=5)[0]
    assert all(np.array_equal(a[k], b[k]) for k in a) and not np.allclose(a["W0"], c["W0"])


def test_zaehler_gegen_die_formeln():
    for form in C.FORMEN:
        _, _, c = _lauf(form, steps=200, n=6, batch=12)
        soll = PO.posten_je_lauf(form, 6, 200, 12)
        assert (c["instanzen"], c["touren"], c["baseline_touren"], c["kodierer"], c["pruefungen"]) == (soll["instanzen"], soll["touren"], soll["baseline_touren"], soll["kodierer"], soll["pruefungen"])
        assert c["schritte"] == 200 and c["abgebrochen"] is False
    assert PO.posten_je_lauf("pomo", 6, 200, 12)["instanzen"] == 2 * 200 and PO.posten_je_lauf("mittel", 6, 200, 12)["instanzen"] == 12 * 200
    assert _lauf("rollout", steps=200)[2]["pruefungen"] == 2 and _lauf("mittel", steps=200)[2]["ersetzungen"] == 0


def test_instanzen_je_schritt_der_formen_n10_und_n20():
    assert [PO.instanzen_je_schritt(f, 20) for f in C.FORMEN] == [120, 120, 120, 6]
    assert [PO.instanzen_je_schritt(f, 10) for f in C.FORMEN] == [120, 120, 120, 12]


def test_pomo_braucht_einen_batch_der_ein_vielfaches_von_n_ist():
    with pytest.raises(ValueError):
        LE.starts_fuer("pomo", 7, 12)
    m, st_ = LE.starts_fuer("pomo", 6, 12)
    assert m == 2 and st_.shape == (2, 6) and (st_[0] == np.arange(6)).all()
    m, st_ = LE.starts_fuer("mittel", 6, 12)
    assert m == 12 and st_.shape == (12, 1) and (st_ == 0).all()


def test_unbekannte_form_und_validierungskurve():
    with pytest.raises(ValueError):
        LE.train_form(6, "x", 1)
    xv = np.random.default_rng(1).random((5, 6, 2))
    _, hist, _ = LE.train_form(6, "mittel", 6, batch=12, d=8, val=(xv, np.full(5, 2.5)), eval_every=2, init_seed=1, data_seed=1)
    assert [h["step"] for h in hist] == [0, 2, 4, 6] and all(np.isfinite(h["val_gap"]) for h in hist)


def test_zeitwaechter_bricht_ab_und_das_zwischenmodell_bleibt_nutzbar():
    P, _, c = LE.train_form(6, "rollout", 50, batch=12, d=8, eval_every=0, max_seconds=0.0)
    assert c["abgebrochen"] is True and c["schritte"] == 0 and c["baseline_touren"] == 0
    X = np.random.default_rng(3).random((2, 6, 2))
    assert N.is_permutation(N.greedy_tours(P, X, np.zeros((2, 1), dtype=np.int64)))


def test_fortschrittsmeldung():
    meldungen = []
    LE.train_form(6, "mittel", 10, batch=12, d=8, eval_every=0, progress=lambda s, g: meldungen.append((s, g)), progress_every=4)
    assert meldungen == [(4, 10), (8, 10), (10, 10)]


def test_rollout_ersetzt_das_referenznetz_wenn_das_aktuelle_besser_ist(monkeypatch):
    monkeypatch.setattr(BL, "rollout_should_replace", lambda lc, lr, t_crit=C.T_CRIT_499: True)
    _, _, c = LE.train_form(6, "rollout", 200, batch=12, d=8, eval_every=0, init_seed=1, data_seed=1)
    assert c["ersetzungen"] == 2                                  # Prüfung bei Schritt 100 und 200


def test_pack_und_unpack_hin_und_zurueck():
    P = LE.initial_params(32, 7)
    v = LE.pack_params(P)
    assert v.dtype == np.float32 and v.size == 12480
    Q = LE.unpack_params(v)
    assert set(Q) == set(P) and all(np.allclose(P[k], Q[k], atol=1e-6) for k in P)
    with pytest.raises(ValueError):
        LE.unpack_params(v[:-1])


def test_training_mit_mittel_senkt_die_mittlere_laenge():
    P0 = LE.initial_params(16, 2)
    X = np.random.default_rng(5).random((200, 8, 2))
    s0 = np.zeros((200, 1), dtype=np.int64)
    vor = N.tour_lengths(X, N.greedy_tours(P0, X, s0)).mean()
    P, _, _ = LE.train_form(8, "mittel", 250, batch=60, d=16, lr=3e-3, init_seed=2, data_seed=2, eval_every=0)
    assert N.tour_lengths(X, N.greedy_tours(P, X, s0)).mean() < vor
