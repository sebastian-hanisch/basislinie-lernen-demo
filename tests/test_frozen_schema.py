"""Eingefrorene Daten (data/): Schema, Zählungen gegen die Formeln der Posten (Plan O8 an der Messung), Plausibilität der exakten Referenz, Vollständigkeit der Raster."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_frozen as FZ
import bsl_posten as PO

EXACT, V2, LR, CROSS, SHORT, COST = (FZ.lade(k) for k in ("exact", "v2", "lr", "cross", "short", "cost"))
pytestmark = pytest.mark.skipif(EXACT is None or V2 is None, reason="eingefrorene Messung fehlt")


def test_exakte_referenz_schema_und_plausibilitaet():
    assert FZ.pruefe_exact(EXACT) == [] and EXACT["format"] == C.FROZEN_FORMAT
    for n in C.N_MESS:
        e = EXACT[str(n)]
        for art in ("test", "val"):
            opt, nn, nn2 = (np.array(e[f"{art}_{k}"]) for k in ("opt", "nn", "nn2"))
            assert (opt > 0).all() and (opt <= nn + 1e-8).all() and (opt <= nn2 + 1e-8).all() and (nn2 <= nn + 1e-8).all()       # Optimum <= 2-opt <= Nächster Nachbar je Instanz
        assert min(e["test_k2"]) >= 1


def test_exakte_laengen_ein_ausschnitt_neu_gerechnet():
    import bsl_exact as EX
    import bsl_instances as I
    for n, idx in ((10, [0, 1, 500]), (20, [0, 7])):
        X = FZ.testsatz(n)
        for i in idx:
            assert abs(EX.held_karp(I.distance_matrix_np(X[i])) - EXACT[str(n)]["test_opt"][i]) < 1e-8, (n, i)
    assert len(FZ.valsatz(10)) == C.VAL_N[10] and len(FZ.testsatz(20)) == C.TEST_N[20]


def test_hauptreihe_raster_und_zaehler():
    for n in C.N_MESS:
        if str(n) not in V2:
            continue
        e = V2[str(n)]
        assert e["schritte"] == C.SCHRITTE_MESS[n]
        seeds = {f: [r["init_seed"] for r in e["formen"][f]["gemeinsam"]] for f in C.FORMEN}
        assert len({tuple(s) for s in seeds.values()}) == 1 and seeds["keiner"] == sorted(seeds["keiner"]) and seeds["keiner"][0] == C.GEWICHT_BASIS
        assert len(seeds["keiner"]) >= (10 if n == 10 else 6)
        for f in C.FORMEN:
            for r in e["formen"][f]["gemeinsam"]:
                assert FZ.pruefe_lauf({**r, "n": n, "form": f, "steps": e["schritte"]}) == [], (n, f)
                assert r["data_seed"] == r["init_seed"] + C.DATEN_BASIS - C.GEWICHT_BASIS and abs(r["lr"] - C.LERNRATE) < 1e-12
                soll = PO.posten_je_lauf(f, n, e["schritte"])
                assert (r["instanzen"], r["touren"], r["baseline_touren"], r["kodierer"], r["pruefungen"]) == (soll["instanzen"], soll["touren"], soll["baseline_touren"], soll["kodierer"], soll["pruefungen"])
                assert [s for s, _ in r["val_kurve"]] == list(range(0, e["schritte"] + 1, C.MESS_ALLE)) and r["schritte"] == e["schritte"]
                assert np.isfinite(r["test_gap"]) and np.isfinite(r["val_gap_ende"]) and r["test_gap"] > 0
        assert all(r["ersetzungen"] == 0 for f in ("keiner", "mittel", "pomo") for r in e["formen"][f]["gemeinsam"])
        assert all(0 <= r["ersetzungen"] <= e["schritte"] // 100 for r in e["formen"]["rollout"]["gemeinsam"])


def test_abgestimmte_laeufe_nur_wo_die_lernrate_abweicht():
    for n in C.N_MESS:
        if str(n) not in V2:
            continue
        for f in C.FORMEN:
            e = V2[str(n)]["formen"][f]
            if "abgestimmt" in e:
                assert abs(e["abgestimmt"]["lr"] - C.LERNRATE) > 1e-12 and e["abgestimmt"]["lr"] in C.LERNRATEN
                assert [r["init_seed"] for r in e["abgestimmt"]["laeufe"]] == [r["init_seed"] for r in e["gemeinsam"]]


@pytest.mark.skipif(LR is None, reason="Lernraten-Abstimmung fehlt")
def test_lernraten_raster_vollstaendig():
    for n in C.N_MESS:
        if str(n) not in LR:
            continue
        e = LR[str(n)]
        for f in C.LR_FORMEN[n]:
            for lr in C.LERNRATEN:
                zellen = [z for z in e["zellen"] if z["form"] == f and abs(z["lr"] - lr) < 1e-12]
                assert sorted(z["init_seed"] for z in zellen) == list(C.TUNE_GEWICHT), (n, f, lr)
        assert set(e["beste"]) == set(C.LR_FORMEN[n]) and all(v in C.LERNRATEN for v in e["beste"].values())
        assert e["schritte"] == C.LR_SCHRITTE[n]
    assert not set(C.TUNE_GEWICHT) & {C.GEWICHT_BASIS + s for s in range(C.SEEDS_MESS)} and not set(C.TUNE_DATEN) & {C.DATEN_BASIS + s for s in range(C.SEEDS_MESS)}


@pytest.mark.skipif(CROSS is None, reason="gekreuzte Messreihe fehlt")
def test_gekreuzte_tabellen_vollstaendig():
    for n in C.N_MESS:
        if str(n) not in CROSS:
            continue
        e = CROSS[str(n)]
        z, s = len(e["gewicht_seeds"]), len(e["daten_seeds"])
        assert z == s >= 3 and len(e["test_gap"]) == z and all(len(r) == s and all(v is not None for v in r) for r in e["test_gap"])
        assert len(e["kurven"]) == z * s and e["schritte"] == C.SCHRITTE_MESS[n]


@pytest.mark.skipif(SHORT is None, reason="Kurz-Läufe fehlen")
def test_kurz_laeufe_alle_zufalls_seeds_und_formen():
    for n in C.N_MESS:
        rs = SHORT[str(n)]
        assert len(rs) == C.SEEDS_MESS * len(C.FORMEN) and {r["form"] for r in rs} == set(C.FORMEN)
        assert all(r["schritte"] == C.KURZ_SCHRITTE[n] and [s for s, _ in r["val_kurve"]] == list(range(0, C.KURZ_SCHRITTE[n] + 1, C.KURZ_ALLE)) + ([C.KURZ_SCHRITTE[n]] if C.KURZ_SCHRITTE[n] % C.KURZ_ALLE else []) for r in rs)


@pytest.mark.skipif(COST is None, reason="Kostendatei fehlt")
def test_kostendatei_gleich_den_formeln():
    assert COST["annahmen"] == {"ops_je_bewertung": 8, "rueckwaerts_faktor": 2, "batch": 120, "d": 32}
    for n in C.N_MESS:
        if str(n) not in COST:
            continue
        for f, z in COST[str(n)]["formen"].items():
            assert z["ops"] == PO.ops_je_lauf(f, n, COST[str(n)]["steps"]) and z["touren"] == PO.posten_je_lauf(f, n, COST[str(n)]["steps"])["touren"]
