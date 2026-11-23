"""Verdikt-Funktionen als reine Funktionen: jede Zahl im Text, beide Zweige jeder Fassung, Ränder (Nenner 0, kein Unterschied, Ausreißer), Preset-Zeilen des exakten Schritts."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_evaluation as V
import bsl_ranktest as RT
import bsl_tabular as TB


def _tab(keiner, mittel, rollout, pomo, nn=10.0, nn2=1.0, n=20):
    formen = {}
    for f, w in zip(C.FORMEN, (keiner, mittel, rollout, pomo)):
        w = np.array(w, dtype=float)
        formen[f] = {"form": f, "werte": w.tolist(), "seeds": list(range(len(w))), "lr": 1e-3, "s": len(w), "mittel": float(w.mean()), "median": float(np.median(w)),
                     "sd": float(w.std(ddof=1)), "min": float(w.min()), "max": float(w.max()), "unter_nn": int((w < nn).sum()), "unter_2opt": int((w < nn2).sum()),
                     "posten": {}, "ops": 0, "ersetzungen": [0] * len(w)}
    return {"n": n, "metrik": "endgap", "modus": "gemeinsam", "nn": nn, "nn_2opt": nn2, "formen": formen, "steps": 8000}


def test_formate():
    assert V.de(3.14159) == "3,14" and V.de_vz(2.5, 1) == "+2,5" and V.de_vz(-2.5, 1) == "-2,5" and V.zahl(12480) == "12 480" and V.lr_text(0.001) == "0,001" and V.lr_text(3e-4) == "0,0003"


# ------------------------------------------------------------------------------------------------------------ exakter Schritt
@pytest.mark.parametrize("politik", C.POLITIKEN)
@pytest.mark.parametrize("B", C.TOUR_B)
def test_verdict_tabular_nennt_jede_zeile_und_den_faktor(politik, B):
    zeilen = TB.schritt_zeilen(TB.theta_von(politik), B)
    art, text = V.verdict_tabular(zeilen, B, politik)
    for name in ("keiner", "Mittel", "Rollout", "POMO", "Leave-one-out"):
        assert name in text
    assert f"{6 ** B} Kombinationen" in text and "-Fache der Streuung des Mittels" in text
    z = {r["form"]: r for r in zeilen}
    quot = z["keiner"]["gesamtstreuung"] / TB.hochgerechnete_streuung(z["mittel"])
    assert V.de(quot, 1) in text and (art == "success") == (quot >= C.STREUUNG_DEUTLICH)


def test_verdict_tabular_handbeispiel_zahlen_b2_und_b4():
    z2 = TB.schritt_zeilen(TB.theta_von("gleich"), 2)
    art, text = V.verdict_tabular(z2, 2, "gleich")
    assert art == "success" and "48,5-Fache" in text and "150,861" in text and "0,778" in text and "3,861" in text and "0,500-mal" in text
    z4 = TB.schritt_zeilen(TB.theta_von("gleich"), 4)
    assert "73,4-Fache" in V.verdict_tabular(z4, 4, "gleich")[1]


def test_verdict_tabular_ohne_deutlichen_unterschied_sagt_es():
    zeilen = [{"form": f, "faktor": 1.0 if f != "mittel" else 0.5, "faktor_formel": 1.0 if f != "mittel" else 0.5, "gesamtstreuung": s, "N": None} for f, s in (("keiner", 1.0), ("mittel", 0.25), ("rollout", 1.0), ("loo", 1.0), ("pomo", 1.0))]
    art, text = V.verdict_tabular(zeilen, 2, "gleich")
    assert art == "info" and "nicht deutlich" in text and "1,0-Fache" in text
    pomo = TB.schritt_zeilen(TB.theta_von("gleich"), 2)
    assert "Erwartungsfaktor 0,750 aus der Formel (der wahre Gradient ist hier null" in V.verdict_tabular(pomo, 2, "gleich")[1]


# ------------------------------------------------------------------------------------------------------------ Experiment 1
def _e1(werte):
    return [{"form": f, "rel_streuung": v} for f, v in zip(C.FORMEN, werte)]


def test_verdict_estimator_beide_zweige_und_der_rahmen():
    art, text = V.verdict_estimator(_e1([40.0, 4.0, 5.0, 2.0]), 200, 12, "ungelernt")
    assert art == "success" and "etwa 20-mal so stark" in text and "nicht auf die letzte Stelle belastbar" in text and "POMO" in text and "16 Instanzen mit 6 Orten" in text and "200 Wiederholungen" in text and "ungelernt" in text
    art, text = V.verdict_estimator(_e1([5.0, 4.0, 5.0, 4.5]), 200, 12, "vorgerechnet")
    assert art == "info" and "kein deutlicher Unterschied (Verhältnis 1,2" in text
    art, text = V.verdict_estimator(_e1([5.0, 0.0, 5.0, 4.5]), 200, 12, "ungelernt")                    # Nenner 0
    assert art == "info" and "nicht definiert" in text and "inf" not in text


# ------------------------------------------------------------------------------------------------------------ Experiment 2
def test_verdict_live_je_form_und_einzellauf_satz():
    art, text = V.verdict_live({"keiner": 25.0, "mittel": 8.0, "rollout": 7.0, "pomo": 6.5}, 10.0, 0.7, 10, 300, 3)
    assert art == "info" and "keiner endet über dem Nächsten Nachbarn (25,00 % gegen 10,00 %)" in text and "Mittel endet unter" in text
    assert "einzelner Lauf (Zufalls-Seed 3, 300 Schritte, n = 10)" in text and "Nächste Nachbar mit 2-opt liegt bei 0,70 %" in text
    assert "keiner lernt nicht" not in text
    art, text = V.verdict_live({"mittel": 8.0}, 10.0, 0.7, 20, 250, 0, {"mittel": 120})
    assert art == "success" and "Abgebrochen (Zeitgrenze): Mittel bei Schritt 120" in text


# ------------------------------------------------------------------------------------------------------------ Experiment 3
def _ranks(tab):
    cols = {f: tab["formen"][f]["werte"] for f in C.FORMEN}
    return {"friedman": RT.friedman(np.array([cols[f] for f in C.FORMEN]).T), "friedman_drei": RT.friedman(np.array([cols[f] for f in ("mittel", "rollout", "pomo")]).T),
            "paare": RT.paarweise(cols), "s": len(cols["keiner"])}


def test_verdict_seeds_ohne_nachweisbaren_unterschied():
    rng = np.random.default_rng(1)
    base = 8.0 + rng.normal(0, 0.5, 8)
    tab = _tab([17.0, 18.5, 19.0, 15.0, 18.0, 17.5, 19.5, 155.0], base, base + rng.normal(0, 0.3, 8), base + rng.normal(0, 0.3, 8))
    art, text = V.verdict_seeds(tab, _ranks(tab))
    assert "Ohne Referenzwert enden 1 von 8 Läufen unter dem Nächsten Nachbarn" not in text          # unter 10,0 liegt keiner
    assert "Ohne Referenzwert enden 0 von 8 Läufen unter dem Nächsten Nachbarn (Median 18,25 %, größter Wert 155,00 %" in text
    assert "Mittel, Rollout und POMO: 8, 8 und 8 von 8" in text and "kein nachweisbarer Unterschied" in text and "bei 8 Paaren und 6 Vergleichen ist 0,0469 der kleinste mögliche Wert" in text
    assert "Gilt für n = 20, 8 000 Schritte" in text and "8 Zufalls-Seeds" in text and "n = 50 anders abschneiden (nicht gemessen)" in text
    assert "Der Friedman-Test über alle vier Formen wird von „keiner“ bestimmt; nur über Mittel, Rollout und POMO: χ² = " in text         # vier Formen mit „keiner“, drei ohne


def test_verdict_seeds_mit_nachweisbarem_paar_nennt_richtung_und_differenz():
    mittel = np.linspace(5, 6, 10)
    tab = _tab(np.linspace(20, 30, 10), mittel + 0.6, mittel, mittel + 0.0001 * np.arange(10) * 0 + 0.3, nn=11.0, n=10)
    tab["steps"] = 4000
    art, text = V.verdict_seeds(tab, _ranks(tab))
    assert "Unter den drei Referenzwerten: " in text and "Rollout liegt 0,60 Prozentpunkte unter Mittel (Holm-p 0,0117, 10 Paare)" in text
    assert "Paarung gilt nur für die Anfangsgewichte" in text and "Gilt für n = 10, 4 000 Schritte" in text


# ------------------------------------------------------------------------------------------------------------ Experiment 4
def _lr(keiner_unter, delta):
    je = {lr: {"mittel": 30.0, "werte": [30.0, 30.0], "unter": 0, "von": 2} for lr in C.LERNRATEN}
    mi = {lr: {"mittel": 5.0 + (0.5 if lr != 1e-3 else 0.0), "werte": [5.0, 5.0], "unter": 2, "von": 2} for lr in C.LERNRATEN}
    if keiner_unter:
        je[3e-4] = {"mittel": 5.0 + delta, "werte": [5.0, 5.0 + 2 * delta], "unter": 2, "von": 2}
    k = {"lr": je, "beste": min(je, key=lambda lr: (je[lr]["mittel"], lr))}
    return {"n": 10, "schritte": 2000, "nn": 10.0, "formen": {"keiner": k, "mittel": {"lr": mi, "beste": 1e-3}}}


def test_verdict_lr_fassung_a_und_b():
    art, text = V.verdict_lr(_lr(False, 0.0))
    assert art == "info" and text.startswith("Keine der vier Lernraten bringt „keiner“ in allen Abstimmungs-Seeds unter den Nächsten Nachbarn") and "bei Lernrate 0,001" in text and "25,00 Prozentpunkte" in text
    assert "2 Zufalls-Seeds, 2 000 Schritte" in text
    art, text = V.verdict_lr(_lr(True, 1.5))
    assert art == "info" and "Mit Lernrate 0,0003 liegt „keiner“ in allen Abstimmungs-Seeds unter dem Nächsten Nachbarn und bleibt 1,50 Prozentpunkte hinter dem Mittel bei dessen bester Lernrate (0,001)" in text
    assert "mit der gemeinsamen Lernrate 0,001 lag „keiner“ in 0 von 2 Abstimmungs-Seeds darunter" in text and "ohne Referenzwert lernt nicht" not in text


# ------------------------------------------------------------------------------------------------------------ Experiment 5 und 6
def _kv(ziele):
    formen = {f: {"erreicht_x": x, "erreicht_schritt": None if x is None else 1000, "median": [10.0, 6.0]} for f, x in zip(C.FORMEN, ziele)}
    return {"einheit": "ops", "ziel": 7.5, "formen": formen, "s": 10, "n": 20}


def test_verdict_cost_faktoren_und_fehlender_messpunkt():
    art, text = V.verdict_cost(_kv([None, 8e11, 4.4e10, 1e11]))
    assert "Grobe Operationen (Annahme 3 mal Vorwärts)" in text and "Mittel das Ziel (7,50 % Median der Validierungslücke) nach 800 000 000 000" in text
    assert "POMO nach 100 000 000 000 (Faktor 8,00 gegenüber dem Mittel)" in text and "Rollout nach 44 000 000 000 (Faktor 18,18" in text and "keiner erreicht das Ziel in keinem Messpunkt" in text
    assert "belastbar ist nur die Größenordnung, nicht die zweite Stelle" in text and "hängen am ersten Messpunkt" in text and "Median über 10 Zufalls-Seeds" in text
    art, text = V.verdict_cost(_kv([1e9, None, 1e9, 1e9]))
    assert "nach keinem Messpunkt" in text and "Faktor" not in text.split("; ", 1)[1] and "(Faktor" not in text           # ohne Vergleichswert kein Faktor


def test_verdict_spread_mit_und_ohne_mittel_ueber_messpunkte():
    shares = {"gewichte": 1.8, "daten": 7.5, "rest": 90.7}
    art, text = V.verdict_spread(shares, 0.45, 0.40, 20, (4, 4), 5)
    assert "Der Gewichts-Seed erklärt 1,8 %, der Daten-Seed 7,5 %, der Rest 90,7 %" in text and "(Referenzwert Mittel, n = 20, 4 × 4 Läufe)" in text
    assert "ist so groß wie die Streuung zwischen Zufalls-Seeds (Standardabweichung 0,40)" in text and "Wechselwirkung und Rauschen, ohne Wiederholungszellen nicht trennbar" in text
    assert "Bei reinem Rauschen wären für den Gewichts-Seed schon 20,0 % der Quadratsumme zu erwarten (Freiheitsgrade 3 von 15)" in text and "nicht direkt vergleichbar" in text
    assert "25,0 % der Quadratsumme zu erwarten (Freiheitsgrade 2 von 8)" in V.verdict_spread(shares, 0.45, 0.40, 20, (3, 3), 5)[1]
    assert "16,7 % der Quadratsumme zu erwarten (Freiheitsgrade 4 von 24)" in V.verdict_spread(shares, 0.45, 0.40, 10, (5, 5), 5)[1]
    assert "kleiner als" in V.verdict_spread(shares, 0.1, 0.9, 20, (4, 4), 3)[1] and "größer als" in V.verdict_spread(shares, 0.9, 0.1, 20, (4, 4), 3)[1]
    art, text = V.verdict_spread(shares, float("nan"), 0.40, 20, (4, 4), 1)
    assert "Rauschen am Ende" not in text and "Mittel der letzten 1 Messpunkte" in text
