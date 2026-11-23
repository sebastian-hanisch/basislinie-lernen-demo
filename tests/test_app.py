"""End-to-end-Rauchtest mit Streamlits AppTest: lädt app.py, klickt jedes Preset, fährt jeden Regler an seine Grenzen und klickt den Trainieren-Knopf (kurz). Geprüft wird, dass kein
Python-Fehler auftritt und die erwarteten Meldungen erscheinen (AppTest sieht keine Plotly-Fehler im Browser; die wurden im Browser-Pane angesehen)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import bsl_constants as C
import bsl_frozen as FZ
import bsl_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("select_slider", "number_input", "selectbox", "toggle", "multiselect")


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(elemente) -> str:
    return " ".join(e.value for e in elemente)


def _meldungen(at) -> str:
    return " ".join([_texte(at.success), _texte(at.info), _texte(at.warning), _texte(at.error)])


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_laedt_ohne_fehler_und_zeigt_den_exakten_schritt():
    at = _app()
    assert not at.error
    meldung = _meldungen(at)
    assert "Die Streuung ohne Referenzwert ist das 48,5-Fache der Streuung des Mittels" in meldung        # Handbeispiel B = 2, Politik gleich
    text = " ".join(m.value for m in at.markdown)
    assert "5431/36" in " ".join(c.value for c in at.caption) and "Hier steht kein Platzhalter" not in text and "TODO" not in text


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    assert not at.error


def test_jeder_regler_an_den_grenzen():
    at = _app()
    for key, spec in P.SETTING_SPECS.items():
        if spec.multi:
            werte = ([spec.options[0]], list(spec.options))
        elif spec.options is not None:
            werte = (spec.options[0], spec.options[-1])
        elif isinstance(spec.default, bool):
            werte = (True, False)
        else:
            werte = (spec.lo, spec.hi)
        for wert in werte:
            _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception, (key, wert, [str(e) for e in at.exception])
        _regler(at, key).set_value(list(spec.default) if spec.multi else spec.default).run(timeout=TIMEOUT)


def test_exakter_schritt_je_politik_und_b():
    at = _app()
    for politik in C.POLITIKEN:
        for b in C.TOUR_B:
            at.select_slider(key="tour_b_select").set_value(b).run(timeout=TIMEOUT)
            at.selectbox(key="politik_select").set_value(politik).run(timeout=TIMEOUT)
            assert not at.exception, (politik, b)
            assert f"{6 ** b} Kombinationen aufgezählt" in _meldungen(at)


def test_live_training_n10_zwei_formen_50_schritte():
    at = _app()
    at.sidebar.multiselect(key="formen_multi").set_value(["mittel", "pomo"]).run(timeout=TIMEOUT)
    at.select_slider(key="live_schritte_select").set_value(50).run(timeout=TIMEOUT)
    assert "Noch kein Lauf für diese Einstellung" in " ".join(c.value for c in at.caption)
    at.button(key="live_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    live = next(m.value for m in at.markdown if "Lücke des Endmodells auf dem Testsatz" in m.value)
    assert "| Mittel |" in live and "| POMO |" in live and "| Rollout |" not in live and "| keiner |" not in live and "Baseline-Touren" in live
    meldung = _meldungen(at)
    assert "Das ist ein einzelner Lauf (Zufalls-Seed 0, 50 Schritte, n = 10)" in meldung
    assert "| Mittel | " in live and "6 000, 6 000, 0 |" in live and "600, 6 000, 0 |" in live                 # Mittel: 6 000 Instanzen und Touren, POMO: 600 Instanzen und 6 000 Touren
    assert at.get("plotly_chart")


def test_live_ergebnis_gilt_nur_fuer_seine_einstellung():
    at = _app()
    at.sidebar.multiselect(key="formen_multi").set_value(["pomo"]).run(timeout=TIMEOUT)
    at.select_slider(key="live_schritte_select").set_value(50).run(timeout=TIMEOUT)
    at.button(key="live_knopf").click().run(timeout=TIMEOUT)
    assert "einzelner Lauf" in _meldungen(at)
    at.select_slider(key="live_schritte_select").set_value(100).run(timeout=TIMEOUT)
    assert "einzelner Lauf" not in _meldungen(at) and not at.exception


def test_zeitgrenze_meldet_nicht_gerechnete_formen(monkeypatch):
    monkeypatch.setattr(C, "LIVE_ZEITGRENZE", 0.0)
    at = _app()
    at.select_slider(key="live_schritte_select").set_value(50).run(timeout=TIMEOUT)
    at.button(key="live_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Nicht gerechnet (Zeitgrenze oder keine Daten): keiner, Mittel, Rollout, POMO" in _texte(at.warning)


def test_vorgerechneter_lauf_zeigt_kurven_und_karten_fuer_preset_seeds():
    if FZ.lade("short") is None:
        pytest.skip("Kurz-Läufe fehlen")
    at = _app()
    at.sidebar.button(key="preset_Kurzer Lauf n = 10").click().run(timeout=TIMEOUT)
    at.toggle(key="live_vorgerechnet_toggle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "einzelner Lauf (Zufalls-Seed" in _meldungen(at) and "300 Schritte, n = 10" in _meldungen(at)
    assert not [e for e in at.caption if "keine Modelle vor" in e.value]                           # Preset-Seed hat Rückfallmodelle
    at.sidebar.number_input(key="seed_input").set_value(9).run(timeout=TIMEOUT)
    assert not at.exception and any("keine Modelle vor" in c.value for c in at.caption)           # Seed 9 hat nur Kurven


def test_ohne_form_meldet_statt_zu_rechnen():
    at = _app()
    at.sidebar.multiselect(key="formen_multi").set_value([]).run(timeout=TIMEOUT)
    assert not at.exception and "mindestens eine Referenzwert-Form" in _texte(at.info)


def test_fehlende_messdaten_geben_hinweise_statt_fehlern(tmp_path, monkeypatch):
    monkeypatch.setattr(FZ, "DATA", tmp_path)
    FZ.lade.cache_clear()
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        at.run(timeout=TIMEOUT)
        assert not at.exception, [str(e) for e in at.exception]
        info = _texte(at.info)
        assert "Die eingefrorene Hauptreihe fehlt" in info and "Die eingefrorene Lernraten-Abstimmung fehlt" in info and "Die gekreuzte Messreihe fehlt" in info
    finally:
        monkeypatch.undo()
        FZ.lade.cache_clear()
