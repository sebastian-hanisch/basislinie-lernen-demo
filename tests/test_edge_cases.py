"""Randfälle (Plan Z5): Nenner 0, fehlende Messreihe, Läufe ohne Streuung, eine Form mit nur einem Lauf, verschieden lange Spalten, Zählung mit s = 0, Rückfall ohne Datei."""

import numpy as np
import pytest

import bsl_baseline as BL
import bsl_constants as C
import bsl_evaluation as V
import bsl_frozen as FZ
import bsl_pretrained as PT
import bsl_ranktest as RT
import bsl_tabular as TB


def test_messreihe_fehlt_oder_gibt_es_nicht_fuer_diese_groesse(tmp_path, monkeypatch):
    monkeypatch.setattr(FZ, "DATA", tmp_path)
    FZ.lade.cache_clear()
    try:
        assert FZ.lade("v2") is None and V.hat_messung(10) is False and V.lr_tabelle(10) is None and V.cross_daten(10) is None and V.kurz_tabelle(10, 0) is None
        with pytest.raises(FileNotFoundError):
            FZ.exakte_laengen(10, "test")
    finally:
        monkeypatch.undo()
        FZ.lade.cache_clear()
    assert V.hat_messung(5) is False and V.hat_messung(50) is False


def test_rangtests_mit_verschieden_langen_spalten_und_einem_lauf():
    tab = {"formen": {f: {"werte": [1.0] * (3 if f == "pomo" else 4)} for f in C.FORMEN}}
    with pytest.raises(ValueError):
        V.rang_tests(tab)
    assert RT.wilcoxon_exakt([1.0])["p"] == 1.0                      # ein Paar: kleinster möglicher p-Wert ist 1
    assert RT.quadratsummen(np.array([[1.0, 2.0], [3.0, 4.0]]))["rest"] == pytest.approx(0.0, abs=1e-12)     # additive Tabelle: kein Rest


def test_streuung_ohne_unterschied_in_den_ranglisten():
    t = np.tile([1.0, 1.0, 1.0, 1.0], (5, 1))
    r = RT.friedman(t)
    assert r["chi2"] == 0.0 and r["p"] == 1.0                        # alle Gleichstände: Korrektur nullt den Nenner, kein Absturz


def test_referenzwert_randfaelle():
    assert BL.paired_t(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert BL.paired_t(np.array([2.0, 3.0]), np.array([1.0, 2.0])) == float("inf")             # Differenz konstant 1, Streuung 0
    with pytest.raises(ValueError):
        TB.enumerate_step(np.zeros(4), 1, "loo")


def test_ohne_rueckfallmodelle_geben_die_funktionen_none(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "NPZ", tmp_path / "gibt_es_nicht.npz")
    monkeypatch.setattr(PT, "META", tmp_path / "gibt_es_nicht.json")
    PT._vektoren.cache_clear()
    PT.meta.cache_clear()
    try:
        assert PT.lade("lang", 10, "mittel", 0) is None and PT.meta_von("kurz", 10, "keiner", 1) is None and PT.seeds_kurz(10) == [] and PT.verfuegbar("kurz", 10, "keiner", 1) is False
    finally:
        monkeypatch.undo()
        PT._vektoren.cache_clear()
        PT.meta.cache_clear()


def test_einsatzkosten_und_kostenkurve_ohne_vergleichswert():
    kv = {"einheit": "schritte", "ziel": 5.0, "s": 3, "formen": {f: {"erreicht_x": None, "erreicht_schritt": None, "median": [9.0]} for f in C.FORMEN}}
    art, text = V.verdict_cost(kv)
    assert "nach keinem Messpunkt" in text and "erreicht das Ziel in keinem Messpunkt" in text
