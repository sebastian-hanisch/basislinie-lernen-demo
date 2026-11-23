"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Die Tests unten stellen sicher, dass wirklich JEDER Regler der Seite
(Seitenleiste und Abschnitte) dort steht und seine Änderung die Hervorhebung aufhebt."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bsl_constants as C
import bsl_presets as P
import bsl_texts as TX

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input", "multiselect")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht auf der Seite")


def _anderer_wert(spec, wert):
    if spec.multi:
        return list(wert)[:-1] if len(wert) > 1 else list(spec.options)
    if spec.options is not None:
        return next(o for o in spec.options if o != wert)
    if isinstance(wert, bool):
        return not wert
    return spec.hi if wert != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value == TX.TITEL
    for wort in ("Konfiguration", "Parameterwahl", "Verfahrenswahl", "Wahlregeln"):
        assert wort not in at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [P.DEFAULT_PRESET] == ["Handbeispiel: vier Orte"]


def test_erste_expander_und_titel_der_abschnitte():
    at = _app()
    namen = [e.label for e in at.expander]
    assert namen[0] == "💬 In einfachen Worten" and namen[-1] == "📐 Mathematische Formulierung"
    assert "So funktioniert ein Referenzwert" in namen
    ueberschriften = [s.value for s in at.subheader]
    assert "🚧 Wo die Annahmen enden" in ueberschriften and sum(1 for u in ueberschriften if u.startswith("🔬")) == 6


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run(timeout=TIMEOUT)
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs_und_jeder_eintrag_hat_einen_regler():
    at = _app()
    seitenleiste = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    assert seitenleiste <= set(P.SETTING_SPECS), f"Regler {sorted(seitenleiste - set(P.SETTING_SPECS))} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    alle = {w.key for art in REGLER_ARTEN for w in getattr(at, art) if w.key and not w.key.startswith("preset_")}
    assert alle == set(P.SETTING_SPECS), (sorted(alle - set(P.SETTING_SPECS)), sorted(set(P.SETTING_SPECS) - alle))


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_des_vorher_aktiven_presets_auf(state_key):
    at = _app()
    start = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{start}").click().run(timeout=TIMEOUT)
    assert _typen(at)[start] == "primary"
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(spec, regler.value)).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    typen = _typen(at)
    assert typen[start] != "primary"
    # ist danach ein anderes Preset markiert (die Änderung führt genau auf dessen Werte), dann stimmt die Markierung mit den Reglern überein
    for name, typ in typen.items():
        if typ == "primary":
            for key, sk in P.PRESET_KEYS.items():
                assert P.normiere(P.SETTING_SPECS[sk], at.session_state[sk]) == P.normiere(P.SETTING_SPECS[sk], C.PRESETS[name][key]), (name, key)
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{start}").click().run(timeout=TIMEOUT)
    assert _typen(at)[start] == "primary"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_aktives_preset_zustandslos_jede_aenderung_hebt_auf(state_key, monkeypatch):
    """Ohne AppTest: eine Änderung jedes einzelnen Reglers hebt die Markierung jedes Presets auf, das gerade passt."""
    for name in C.PRESETS:
        zustand = {sk: (list(C.PRESETS[name][key]) if P.SETTING_SPECS[sk].multi else C.PRESETS[name][key]) for key, sk in P.PRESET_KEYS.items()}
        monkeypatch.setattr(P.st, "session_state", dict(zustand))
        assert P.aktives_preset() == name
        spec = P.SETTING_SPECS[state_key]
        P.st.session_state[state_key] = _anderer_wert(spec, zustand[state_key])
        assert P.aktives_preset() != name, (name, state_key)
