"""Rückfallmodelle (data/bsl_pretrained.npz): Größe unter 1,5 MB, Format, geladenes Modell gibt die gespeicherte Greedy-Lücke als Band (float32: ± 0,1 Prozentpunkte)."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_frozen as FZ
import bsl_net as N
import bsl_pretrained as PT
import bsl_train as T

pytestmark = pytest.mark.skipif(PT.meta() is None, reason="Rückfallmodelle fehlen")


def test_groesse_und_format():
    assert PT.NPZ.stat().st_size < 1.5e6
    m = PT.meta()
    assert m["format"] == C.PRETRAINED_FORMAT and m["n_params"] == 12480 and m["d"] == 32
    v = np.load(PT.NPZ)
    assert set(v.files) == set(m["modelle"]) and all(v[k].dtype == np.float32 and v[k].shape == (12480,) for k in v.files)
    kurz = [k for k in m["modelle"] if k.startswith("kurz_")]
    assert len(kurz) == 2 * 2 * len(C.FORMEN) and "lang_n10_mittel_s0" in m["modelle"]
    assert PT.seeds_kurz(10) == sorted(PT.seeds_kurz(10)) and len(PT.seeds_kurz(10)) == 2 and len(PT.seeds_kurz(20)) == 2


def test_jedes_modell_gibt_seine_gespeicherte_greedy_luecke_als_band():
    for key, info in PT.meta()["modelle"].items():
        art = key.split("_")[0]
        P = PT.lade(art, info["n"], info["form"], info["zufalls_seed"])
        assert P is not None and N.n_params(P) == 12480
        gap = T.gap_greedy(P, FZ.testsatz(info["n"]), FZ.exakte_laengen(info["n"], "test"))
        assert abs(gap - info["test_gap"]) <= 0.1, (key, gap, info["test_gap"])


def test_fehlendes_modell_gibt_none_und_falsche_laenge_valueerror():
    assert PT.lade("kurz", 10, "keiner", 99) is None and PT.verfuegbar("kurz", 10, "keiner", 99) is False
    import bsl_learn as LE
    with pytest.raises(ValueError):
        LE.unpack_params(np.zeros(5, dtype=np.float32))
