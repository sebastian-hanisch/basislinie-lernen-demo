"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink einrasten, Zufalls-Seeds der Kurz-Läufe."""

import pytest

import bsl_constants as C
import bsl_presets as P


def test_jedes_preset_hat_hilfetext_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.multi:
                assert set(preset[key]) <= set(spec.options) and preset[key], (name, key)
            elif spec.options is not None:
                assert preset[key] in spec.options, (name, key)
            elif isinstance(spec.default, bool):
                assert isinstance(preset[key], bool)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_standard_preset_ist_das_erste_und_gleich_den_standardeinstellungen():
    erstes = C.PRESETS[next(iter(C.PRESETS))]
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        assert P.normiere(spec, erstes[key]) == P.normiere(spec, spec.default), key
    assert erstes["n"] == 10 and erstes["seed"] == 0 and erstes["tour_b"] == 2 and erstes["politik"] == "gleich"


def test_setting_specs_decken_jeden_preset_schluessel_ab():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_sind_die_geplanten_sechs_mit_ihren_einstellungen():
    assert list(C.PRESETS) == ["Handbeispiel: vier Orte", "Kurzer Lauf n = 10", "Kurzer Lauf n = 20", "Sechs Zufalls-Seeds, n = 20", "Ist „keiner“ nur falsch eingestellt?", "Was kostet das Lernen?"]
    p = C.PRESETS
    assert (p["Handbeispiel: vier Orte"]["tour_b"], p["Handbeispiel: vier Orte"]["politik"]) == (2, "gleich")
    assert (p["Kurzer Lauf n = 10"]["n"], p["Kurzer Lauf n = 10"]["schritte"], p["Kurzer Lauf n = 10"]["lr"]) == (10, 300, 1e-3)
    assert (p["Kurzer Lauf n = 20"]["n"], p["Kurzer Lauf n = 20"]["schritte"]) == (20, 250)
    assert p["Kurzer Lauf n = 10"]["seed"] == C.PRESET_SEED_KURZ[10] and p["Kurzer Lauf n = 20"]["seed"] == C.PRESET_SEED_KURZ[20]
    assert p["Sechs Zufalls-Seeds, n = 20"]["n"] == 20 and p["Sechs Zufalls-Seeds, n = 20"]["verlaeufe"] is True
    assert p["Ist „keiner“ nur falsch eingestellt?"]["e3_lr"] == "abgestimmt" and p["Ist „keiner“ nur falsch eingestellt?"]["n"] == 10
    assert p["Was kostet das Lernen?"]["einheit"] == "ops" and p["Was kostet das Lernen?"]["n"] == 20


def test_einrasten_des_permalinks():
    spec = P.SETTING_SPECS["n_select"]
    assert P.clamp(spec, 12) == 10 and P.clamp(spec, 100) == 20 and P.clamp(spec, 1) == 10
    assert P.clamp(P.SETTING_SPECS["politik_select"], "unsinn") is None and P.clamp(P.SETTING_SPECS["politik_select"], "kurz") == "kurz"
    assert P.clamp(P.SETTING_SPECS["seed_input"], 50) == 9 and P.clamp(P.SETTING_SPECS["live_instanz_input"], -3) == 0
    assert P.clamp(P.SETTING_SPECS["live_lr_select"], 0.0005) == 0.0003
    assert P.clamp(P.SETTING_SPECS["formen_multi"], "keiner,pomo,quatsch") == ["keiner", "pomo"] and P.clamp(P.SETTING_SPECS["formen_multi"], "quatsch") is None
    assert P._bool("true") is True and P._bool("False") is False and P._float("0,003") == 0.003


def test_multi_vergleich_ist_reihenfolgeunabhaengig():
    spec = P.SETTING_SPECS["formen_multi"]
    assert P.normiere(spec, ["pomo", "keiner"]) == ("keiner", "pomo")


@pytest.mark.parametrize("n", C.N_MESS)
def test_zufalls_seed_der_kurzen_laeufe_liegt_im_bereich_und_ist_nicht_null_bei_n10(n):
    assert 0 <= C.PRESET_SEED_KURZ[n] <= C.SEED_MAX
    assert C.PRESETS["Kurzer Lauf n = 10"]["seed"] != C.PRESETS["Handbeispiel: vier Orte"]["seed"]          # sonst wären beide Presets nicht unterscheidbar
