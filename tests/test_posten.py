"""Trainingsposten und Einsatzkosten (Zählregel Q5): Formeln gegen die Handrechnung der Vormessung, Dekodierer in der O-Konvention von Stück 1 (35 184 statt 34 160)."""

import pytest

import bsl_constants as C
import bsl_cost as K
import bsl_posten as PO


def test_posten_je_lauf_gegen_die_vormessung():
    assert PO.posten_je_lauf("mittel", 10, 4000)["touren"] == 480_000 and PO.posten_je_lauf("mittel", 20, 8000)["touren"] == 960_000
    assert PO.posten_je_lauf("rollout", 10, 4000)["baseline_touren"] == 520_000                   # 120 je Schritt plus 40 Prüfungen zu 1 000
    assert PO.posten_je_lauf("rollout", 20, 8000)["baseline_touren"] == 1_040_000                 # 960 000 + 80 Prüfungen zu 1 000
    assert PO.posten_je_lauf("rollout", 20, 8000)["pruefungen"] == 80
    assert PO.posten_je_lauf("keiner", 20, 8000)["baseline_touren"] == 0 == PO.posten_je_lauf("pomo", 20, 8000)["baseline_touren"]
    assert PO.posten_je_lauf("pomo", 20, 8000)["instanzen"] == 48_000 and PO.posten_je_lauf("pomo", 10, 4000)["instanzen"] == 48_000
    assert PO.posten_je_lauf("mittel", 20, 8000)["instanzen"] == 960_000
    p = PO.posten_je_lauf("rollout", 20, 8000)
    assert p["kodierer"] == p["instanzen"] + p["baseline_touren"] == 960_000 + 1_040_000


def test_kodierer_und_dekodierer_der_o_konvention():
    assert K.ops_kodierer(20, 32) == 211_200 and K.ops_kodierer(10, 32) == 99_200
    assert K.ops_dekodierer_tour(20, 32) == 35_184                                                # nicht 34 160 (Plan nach M1)
    assert K.ops_netz_tour(20, 32) == 246_384 and K.ops_netz_tour(10, 32) == 113_704


def test_operationen_je_schritt_und_verhaeltnisse():
    e, t = 211_200, 35_184
    assert PO.ops_je_schritt("mittel", 20) == 3 * (120 * e + 120 * t) == PO.ops_je_schritt("keiner", 20)
    assert PO.ops_je_schritt("pomo", 20) == 3 * (6 * e + 120 * t)
    assert PO.ops_je_schritt("rollout", 20) == 3 * (120 * e + 120 * t) + 120 * (e + t) + 2 * 500 * (e + t) / 100
    assert round(PO.verhaeltnis_zum_mittel("pomo", 20), 2) == 0.19 and round(PO.verhaeltnis_zum_mittel("pomo", 10), 2) == 0.21
    assert round(PO.verhaeltnis_zum_mittel("rollout", 20), 2) == 1.36 and round(PO.verhaeltnis_zum_mittel("rollout", 10), 2) == 1.36
    assert PO.verhaeltnis_zum_mittel("mittel", 20) == 1.0


def test_operationen_je_lauf_gleich_schritte_mal_je_schritt_bei_vollen_pruefungen():
    for form in C.FORMEN:
        assert PO.ops_je_lauf(form, 20, 8000) == pytest.approx(8000 * PO.ops_je_schritt(form, 20), rel=1e-12)
    assert PO.ops_je_lauf("rollout", 20, 50) == 50 * 3 * (120 * 211_200 + 120 * 35_184) + 120 * 50 * 246_384        # keine Prüfung nach 50 Schritten


def test_einsatzkosten_k_und_o():
    e20 = PO.einsatz_kosten(20, 492.0)
    assert e20["netz"] == {"k": 190, "o": 246_384} and e20["nn"] == {"k": 190, "o": 1_520} and e20["nn_2opt"]["k"] == 190 + 492 and e20["nn_2opt"]["o"] == 8 * (190 + 492)
    e10 = PO.einsatz_kosten(10, 80.0)
    assert e10["netz"] == {"k": 45, "o": 113_704} and e10["nn"] == {"k": 45, "o": 360} and e10["nn_2opt"]["k"] == 125 and e10["nn_2opt"]["o"] == 1_000
    assert e20["netz"]["o"] / e20["nn"]["o"] > 160 and e20["netz"]["k"] == e20["nn"]["k"]                          # K gleich, O benachteiligt das Netz


def test_voller_pruefrunde_zugzahlen_q2():
    assert K.m_verschiedene_zuege(20) == 1850 and K.m_verschiedene_zuege(50) == 12875
