"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Knopf (Standardmuster aus dem Demo-Portfolio).

SETTING_SPECS: alle Regler der Seite (Seitenleiste und Abschnitte); sie stehen im Permalink und entscheiden, welches Preset hervorgehoben ist. Regler, die nur bei einer Wahl wirken
(`live_schritte_select` und `live_lr_select` ohne „Vorgerechneten Lauf zeigen“), bleiben sichtbar und sind sonst abgeschaltet (disabled), sie verschwinden nicht.
Mehrfachauswahlen (Formen) stehen im Permalink als Kommaliste und werden beim Vergleich in der Reihenfolge von `FORMEN` normiert."""

import random
from dataclasses import dataclass

import streamlit as st

import bsl_constants as C


def _bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "wahr")


def _float(text) -> float:
    return float(str(text).replace(",", "."))


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None       # Auswahlregler: erlaubte Werte
    multi: bool = False                # Mehrfachauswahl aus `options`


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, options=C.N_MESS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "formen_multi": SettingSpec("formen", str, C.DEFAULT_FORMEN, options=C.FORMEN, multi=True),
    "tour_b_select": SettingSpec("b", int, C.DEFAULT_TOUR_B, options=C.TOUR_B),
    "politik_select": SettingSpec("politik", str, C.DEFAULT_POLITIK, options=C.POLITIKEN),
    "e1_b_select": SettingSpec("e1b", int, C.E1_B_STANDARD, options=C.E1_B),
    "e1_stand_select": SettingSpec("e1stand", str, C.DEFAULT_STAND, options=C.STAND_OPTIONEN),
    "live_schritte_select": SettingSpec("schritte", int, 300, options=C.SCHRITTE_LIVE),
    "live_lr_select": SettingSpec("lr", _float, C.LERNRATE, options=C.LERNRATEN),
    "live_vorgerechnet_toggle": SettingSpec("vorgerechnet", _bool, False),
    "live_instanz_input": SettingSpec("instanz", int, 0, 0, 29),
    "e3_metrik_select": SettingSpec("metrik", str, "endgap", options=C.METRIKEN),
    "e3_lr_select": SettingSpec("e3lr", str, "gemeinsam", options=C.LR_MODI),
    "e3_verlaeufe_toggle": SettingSpec("verlaeufe", _bool, False),
    "e4_formen_multi": SettingSpec("e4formen", str, C.FORMEN, options=C.FORMEN, multi=True),
    "e5_einheit_select": SettingSpec("einheit", str, C.DEFAULT_EINHEIT, options=C.EINHEITEN),
    "e6_k_select": SettingSpec("k", int, C.DEFAULT_LETZTE_K, options=C.LETZTE_K),
}
PRESET_KEYS = {
    "n": "n_select", "seed": "seed_input", "formen": "formen_multi", "tour_b": "tour_b_select", "politik": "politik_select", "e1_b": "e1_b_select", "e1_stand": "e1_stand_select",
    "schritte": "live_schritte_select", "lr": "live_lr_select", "vorgerechnet": "live_vorgerechnet_toggle", "instanz": "live_instanz_input", "metrik": "e3_metrik_select",
    "e3_lr": "e3_lr_select", "verlaeufe": "e3_verlaeufe_toggle", "e4_formen": "e4_formen_multi", "einheit": "e5_einheit_select", "k": "e6_k_select",
}
DEFAULT_PRESET = next(iter(C.PRESETS))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = list(spec.default) if spec.multi else spec.default


def bounds(state_key):
    """(Minimum, Maximum) eines Zahlenreglers."""
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key) -> tuple:
    """Erlaubte Werte eines Auswahlreglers."""
    return SETTING_SPECS[state_key].options


def normiere(spec: SettingSpec, wert):
    """Vergleichbare Form eines Wertes: Mehrfachauswahlen als Tupel in der Reihenfolge der Optionen."""
    if spec.multi:
        return tuple(o for o in spec.options if o in set(wert))
    return wert


def clamp(spec: SettingSpec, value):
    """Wert aus der Adresszeile auf den Regler abbilden: Zahlen begrenzen, Auswahlwerte einrasten (Zahl: nächster Wert; Text: nur gültige), Mehrfachauswahl: nur gültige Einträge."""
    if spec.multi:
        wanted = [v.strip() for v in str(value).split(",")]
        gueltig = [o for o in spec.options if o in wanted]
        return gueltig or None
    if spec.options is not None:
        if value in spec.options:
            return value
        if spec.caster in (int, _float):
            return min(spec.options, key=lambda o: (abs(o - value), o))
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = qp[spec.url_param] if spec.multi else spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = clamp(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            spec = SETTING_SPECS[state_key]
            st.query_params[spec.url_param] = ",".join(normiere(spec, value)) if spec.multi else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        spec = SETTING_SPECS[state_key]
        if key in C.PRESETS[name]:
            st.session_state[state_key] = list(C.PRESETS[name][key]) if spec.multi else C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            normiere(spec, st.session_state.get(state_key, spec.default)) == normiere(spec, preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
