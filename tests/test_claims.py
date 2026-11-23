"""Jede Zahl der README gegen die eingefrorene Messung (data/) und die Module der Demo.

„Exakt aus der Datei“: die Auswertung der gespeicherten Läufe mit denselben Funktionen wie in der App ergibt genau die Zahl der README (auf die gedruckten Stellen). Zusätzlich Bänder
um die Zahlen der Vormessung (Erwartung aus `numpy.default_rng`, hier mit SplitMix64 neu gemessen). Die Läufe selbst rechnet tests/test_frozen_recompute.py an einem Ausschnitt als Band
neu; Zahlen des Handbeispiels stehen exakt in tests/test_tabular.py."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bsl_befunde as BF
import bsl_constants as C
import bsl_cost as K
import bsl_estimator as E1
import bsl_evaluation as V
import bsl_frozen as FZ
import bsl_posten as PO
import bsl_ranktest as RT
import bsl_tabular as TB
import bsl_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
de, zahl = V.de, V.zahl
MESSUNG = FZ.lade("v2") is not None and FZ.lade("exact") is not None
braucht_messung = pytest.mark.skipif(not MESSUNG, reason="eingefrorene Messung fehlt")


def steht(*teile):
    for teil in teile:
        assert teil in README, f"nicht in der README: {teil}"


# ------------------------------------------------------------------------------------------------------------- Handbeispiel und Zählung (exakt)
def test_handbeispiel_vier_orte_zahlen():
    steht("5431/36", "7/9", "139/36", "28/9", "150,86", "48,5")
    erw = {f: TB.enumerate_step_exact(2, f) for f in TB.FORMEN_EXAKT}
    assert str(erw["keiner"]["gesamtstreuung"]) == "5431/36" and str(erw["mittel"]["gesamtstreuung"]) == "7/9" and str(erw["rollout"]["gesamtstreuung"]) == "139/36" and str(erw["loo"]["gesamtstreuung"]) == "28/9"
    assert round(5431 / 36, 2) == 150.86 and round(5431 / 36 / (28 / 9), 1) == 48.5
    z4 = TB.schritt_zeilen(TB.theta_von("gleich"), 4)
    assert de(z4[0]["gesamtstreuung"] / TB.hochgerechnete_streuung(z4[1]), 1) == "73,4"
    steht("73,4", "0,575", "75,43")
    assert round(TB.pomo_step(TB.theta_von("gleich"), True)["gesamtstreuung"], 3) == 0.575 and round(TB.pomo_step(TB.theta_von("gleich"), False)["gesamtstreuung"], 2) == 75.43


def test_trainingsposten_und_operationen_der_readme():
    steht("480 000", "520 000", "960 000", "1 040 000", "211 200", "35 184", "246 384", "113 704")
    assert PO.posten_je_lauf("mittel", 10, 4000)["touren"] == 480_000 and PO.posten_je_lauf("rollout", 10, 4000)["baseline_touren"] == 520_000
    assert PO.posten_je_lauf("mittel", 20, 8000)["touren"] == 960_000 and PO.posten_je_lauf("rollout", 20, 8000)["baseline_touren"] == 1_040_000
    assert K.ops_kodierer(20, 32) == 211_200 and K.ops_dekodierer_tour(20, 32) == 35_184 and K.ops_netz_tour(20, 32) == 246_384 and K.ops_netz_tour(10, 32) == 113_704
    steht(f"{de(PO.verhaeltnis_zum_mittel('pomo', 20), 2)}-fach", f"{de(PO.verhaeltnis_zum_mittel('pomo', 10), 2)}-fach", f"{de(PO.verhaeltnis_zum_mittel('rollout', 20), 2)}-fach")
    assert (de(PO.verhaeltnis_zum_mittel("pomo", 20)), de(PO.verhaeltnis_zum_mittel("pomo", 10)), de(PO.verhaeltnis_zum_mittel("rollout", 20))) == ("0,19", "0,21", "1,36")


def test_statistik_der_readme():
    steht("2/1024", "0,0117", "2/64", "0,1875")
    assert abs(RT.wilcoxon_exakt(np.arange(1, 11))["p"] - 2 / 1024) < 1e-15 and abs(6 * 2 / 1024 - 0.01171875) < 1e-15
    assert abs(RT.wilcoxon_exakt(np.arange(1, 7))["p"] - 2 / 64) < 1e-15 and abs(6 * 2 / 64 - 0.1875) < 1e-15


def test_zahlen_der_parameter_in_der_readme():
    steht("12 480")
    assert C.BATCH == 120 and C.D_STANDARD == 32


# ------------------------------------------------------------------------------------------------------------- Texte
def test_annahmenblock_der_app_steht_wortgleich_in_der_readme():
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in README


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    m = re.search(r"(\d+) Tests in (\d+) Dateien", README)
    assert m, "Testzahl fehlt in der README"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=170).stdout
    zeilen = [z for z in out.splitlines() if "::" in z]
    assert int(m.group(1)) == len(zeilen)
    assert int(m.group(2)) == len(list((ROOT / "tests").glob("test_*.py")))


def test_verbotene_formulierungen_fehlen():
    dateien = ["app.py", "bsl_texts.py", "bsl_evaluation.py", "bsl_constants.py", "bsl_visualization.py", "bsl_presets.py"]
    texte = {"README": README, **{d: (ROOT / d).read_text(encoding="utf-8") for d in dateien}}
    for name, t in texte.items():
        for wort in (r"\bträgt\b", r"\btragen\b", "TLA", "Saatwert", "Schwanz", "läuft lokal", "Parameterwahl", "Verfahrenswahl", "Wahlregeln"):
            assert not re.findall(wort, t), (name, wort)
        assert "Merkmalsnetz mit 145 Parametern ordnet" not in t                                          # Q1: Netze K und Z
        assert "Netz braucht Attention" not in t and "Kodierer braucht Attention" not in t                  # Abnahme Stück 1: Ablation widerlegt die Zuschnitt-Aussage
    for name in ("INTRO", "SO_FUNKTIONIERT", "ZAEHLREGEL", "ERKLAERTABELLE", "ASSUMPTIONS_MARKDOWN"):
        assert " – " not in getattr(TX, name), name                                                        # kein Gedankenstrich im Fließtext der Anzeigetexte
    for name in ("README", "app.py", "bsl_texts.py", "bsl_evaluation.py"):
        assert not re.findall(r"(?<!Zufalls-)(?<!Gewichts-)(?<!Daten-)(?<!Abstimmungs-)\bSeeds?\b", texte[name]), name           # „Zufalls-Seed“, nie „Seed“ allein


def test_jeder_testverweis_der_readme_gibt_es():
    """Die Spalte „Test“ der Befundtabelle nennt Datei und Testfunktion: beide müssen existieren."""
    verweise = re.findall(r"`(test_[a-z_]+\.py)(?:::(test_[a-z0-9_]+))?`", README)
    assert len(verweise) >= 12
    for datei, funktion in verweise:
        pfad = ROOT / "tests" / datei
        assert pfad.exists(), datei
        if funktion:
            assert f"def {funktion}(" in pfad.read_text(encoding="utf-8"), (datei, funktion)


def test_titel_und_zeichenfolgen_der_namen():
    assert "Konfiguration" not in TX.TITEL and "Referenzwert" in TX.TITEL
    namen = [p.name for p in list(ROOT.glob("*.py")) + list(ROOT.glob("tests/*.py")) + list(ROOT.glob("tools/*.py")) + list(ROOT.glob("data/*"))] + [ROOT.name]
    for verboten in ("hub", "sso", "prod", "ass", "qa"):
        assert not [n for n in namen if verboten in n.lower()], verboten                                    # Streamlit lehnt solche Zeichenfolgen in Adressen ab


# ------------------------------------------------------------------------------------------------------------- Befunde (gemessen): Zeilen der README gegen die Daten

@braucht_messung
def test_befundtabelle_steht_wortgleich_in_der_readme():
    """Jede Zeile der Befundtabelle (aus den Daten mit denselben Funktionen wie die App gebildet) steht wörtlich in der README."""
    zeilen = BF.tabelle_zeilen()
    assert len(zeilen) >= 20
    for frage, ergebnis, test in zeilen:
        steht(f"| {frage} | {ergebnis} | {test} |")


@braucht_messung
def test_befund_gegner():
    """Bänder um die Vormessung: Nächster Nachbar 10,58 / 17,40 %, NN + 2-opt 0,68 / 2,78 %, Zugauswertungen 80 / 501 (Instanzmengen und Zufallsströme sind andere)."""
    bander = {10: dict(nn=(9.0, 12.0), zw=(0.3, 1.2), zug=(60, 110)), 20: dict(nn=(15.5, 19.5), zw=(2.0, 3.6), zug=(400, 600))}
    for n in C.N_MESS:
        g = FZ.nn_gaps(n, "test")
        assert bander[n]["nn"][0] <= g["nn"] <= bander[n]["nn"][1] and bander[n]["zw"][0] <= g["nn_2opt"] <= bander[n]["zw"][1]
        assert bander[n]["zug"][0] <= g["zugauswertungen_mittel"] <= bander[n]["zug"][1]
        assert g["zugauswertungen_min"] >= K.k_pruefrunde_2opt(n)                                        # mindestens eine volle Prüfrunde


@braucht_messung
def test_befund_keiner():
    """Zählung statt Urteilswort: wie viele Läufe ohne Referenzwert enden unter dem Nächsten Nachbarn, Mittel und Median getrennt; Band um die Vormessung (n = 20: Median 18,7 %, n = 10: 9,3 %)."""
    for n, (lo, hi) in ((10, (5.0, 14.0)), (20, (14.0, 24.0))):
        z = V.seed_tabelle(n, "endgap", "gemeinsam")["formen"]["keiner"]
        assert lo <= z["median"] <= hi
        assert z["unter_nn"] == sum(w < V.seed_tabelle(n, "endgap")["nn"] for w in z["werte"])


@braucht_messung
def test_befund_referenzwerte():
    """Mit Mittel, Rollout und POMO: Band um die Vormessung (n = 10: 3,84 / 3,27 / 3,25 %; n = 20: 9,86 / 9,62 / 9,52 %) plus 2 Standardabweichungen zwischen den Zufalls-Seeds."""
    plan = {10: {"mittel": 3.84, "rollout": 3.27, "pomo": 3.25}, 20: {"mittel": 9.86, "rollout": 9.62, "pomo": 9.52}}
    for n in ns_mit():
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        for f, soll in plan[n].items():
            z = t["formen"][f]
            assert abs(z["mittel"] - soll) <= max(1.5, 2 * z["sd"]), (n, f, z["mittel"], soll)
            assert z["unter_nn"] == z["s"] == len(z["werte"])                                          # in dieser Messung alle Läufe unter dem Nächsten Nachbarn


@braucht_messung
def test_befund_rangtests():
    for n in ns_mit():
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        r = V.rang_tests(t)
        assert r["friedman"]["df"] == 3 and 0 <= r["friedman"]["p"] <= 1 and len(r["paare"]) == 6
        kleinst = 2 / 2 ** r["s"] * 6
        assert all(p["p_holm"] >= kleinst - 1e-12 for p in r["paare"])                                  # kleinster möglicher Holm-p bei s Paaren und 6 Vergleichen
        assert all(p["a_besser"] + p["b_besser"] == p["n_paare"] for p in r["paare"] if p["n_paare"] == r["s"])


@braucht_messung
def test_friedman_nur_ueber_die_drei_referenzwerte_und_rangfolge_haengt_an_der_lernrate():
    """Der Friedman-Test der Tabelle läuft über vier Formen (mit „keiner“) und wird von „keiner“ bestimmt; nur über Mittel, Rollout und POMO (scipy als Gegenprobe) ist er bei n = 20 nicht
    nachweisbar. Die Rangfolge bei n = 10 hängt an der Lernrate: mit der je Form abgestimmten Lernrate liegt das Mittel nicht mehr hinter dem Rollout, und POMO liegt vor dem Rollout."""
    from scipy import stats
    for n, (chi2, p) in ((10, (10.40, 0.0055)), (20, (1.00, 0.6065))):
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        r = V.rang_tests(t)
        s = stats.friedmanchisquare(*[t["formen"][f]["werte"] for f in ("mittel", "rollout", "pomo")])
        assert abs(r["friedman_drei"]["chi2"] - s.statistic) < 1e-9 and abs(r["friedman_drei"]["p"] - s.pvalue) < 1e-9 and r["friedman_drei"]["df"] == 2
        assert (round(r["friedman_drei"]["chi2"], 2), round(r["friedman_drei"]["p"], 4)) == (chi2, p)
        assert r["friedman"]["p"] < 0.05 and r["friedman"]["df"] == 3                                    # über vier Formen: durch „keiner“ bestimmt
    steht("Friedman nur über Mittel, Rollout und POMO χ² = 10,40, p = 0,0055", "Friedman nur über Mittel, Rollout und POMO χ² = 1,00, p = 0,6065")
    rg, ra = V.rang_tests(V.seed_tabelle(10, "endgap", "gemeinsam")), V.rang_tests(V.seed_tabelle(10, "endgap", "abgestimmt"))
    assert BF._paar(rg, "mittel", "rollout")[1] < 0.05 and BF._paar(rg, "rollout", "pomo")[1] > 0.05          # gemeinsame Lernrate 0,001
    assert BF._paar(ra, "mittel", "rollout")[1] > 0.05 and BF._paar(ra, "mittel", "rollout")[0] < 0                # abgestimmt: Mittel nicht hinter dem Rollout
    assert BF._paar(ra, "mittel", "pomo")[1] < 0.05 and BF._paar(ra, "rollout", "pomo")[1] < 0.05 and BF._paar(ra, "rollout", "pomo")[0] > 0   # POMO besser als der Rollout
    steht("Bei n = 10 hängt die Rangfolge der drei Referenzwerte an der Lernrate")


@braucht_messung
def test_befund_lernrate():
    for n in C.N_MESS:
        lt = V.lr_tabelle(n)
        if lt is None:
            continue
        for f, z in lt["formen"].items():
            assert z["beste"] == min(z["lr"], key=lambda lr: (z["lr"][lr]["mittel"], lr))
            assert all(v["von"] == 2 for v in z["lr"].values())


@braucht_messung
def test_befund_abgestimmte_hauptreihe():
    for n in ns_mit():
        gem, abg = V.seed_tabelle(n, "endgap", "gemeinsam"), V.seed_tabelle(n, "endgap", "abgestimmt")
        for f in C.FORMEN:
            lr = V.lernrate_der_form(n, f, "abgestimmt")
            assert (abg["formen"][f]["lr"] == lr) and (lr == C.LERNRATE) == (abg["formen"][f]["werte"] == gem["formen"][f]["werte"])


@braucht_messung
def test_befund_kosten():
    for n in ns_mit():
        kv = V.kosten_kurven(n, "ops", "gemeinsam")
        assert kv["formen"]["mittel"]["erreicht_x"] is not None                                          # das Ziel liegt am Ende des Mittels: es erreicht es
        pomo = kv["formen"]["pomo"]["erreicht_x"]
        if pomo is not None:
            assert pomo < kv["formen"]["mittel"]["erreicht_x"] * 1.0001 or kv["formen"]["pomo"]["erreicht_schritt"] >= kv["formen"]["mittel"]["erreicht_schritt"]
        ziel = kv["formen"]["mittel"]["median"][-1] + C.ZIEL_AUFSCHLAG
        assert abs(kv["ziel"] - ziel) < 1e-12


@braucht_messung
def test_befund_streuung():
    for n in ns_mit():
        cd = V.cross_daten(n)
        if cd is None:
            continue
        sh = RT.quadratsummen(np.array(cd["test_gap"]))
        assert abs(sh["gewichte"] + sh["daten"] + sh["rest"] - 100) < 1e-9 and sh["rest"] >= 0


def test_befund_experiment1():
    """Ohne Referenzwert streut der Schritt am ungelernten Netz mindestens doppelt so stark wie mit dem besten der drei anderen (Band: Faktor 2, Festlegung)."""
    import bsl_net as N
    from bsl_rngvec import UStream
    from bsl_scenario import derive
    P = N.init_params(C.D_STANDARD, UStream(derive(C.ZWECK_E1_NETZ, 0)), layers=1, attn=True, ffn=True)
    z = {r["form"]: r["rel_streuung"] for r in E1.tabelle(P, C.E1_B_STANDARD, reps=C.E1_REPS, seed=0)}
    assert z["keiner"] > C.STREUUNG_DEUTLICH * min(z[f] for f in ("mittel", "rollout", "pomo"))


@braucht_messung
def test_befund_kurze_laeufe():
    d = FZ.lade("short")
    if d is None:
        pytest.skip("Kurz-Läufe fehlen")
    for n in C.N_MESS:
        rs = d[str(n)]
        assert len(rs) == 40 and all(r["schritte"] == C.KURZ_SCHRITTE[n] for r in rs)


@braucht_messung
def test_befund_austausche():
    for n in ns_mit():
        z = V.seed_tabelle(n, "endgap", "gemeinsam")["formen"]["rollout"]
        assert 0 <= min(z["ersetzungen"]) <= max(z["ersetzungen"]) <= z["posten"]["pruefungen"]


def ns_mit():
    return [n for n in C.N_MESS if V.hat_messung(n)]



@braucht_messung
def test_befund_der_kernfrage_steht_wortgleich_in_der_readme():
    steht(BF.hook())
    for n in ns_mit():
        lt = V.lr_tabelle(n)
        if lt:
            steht(V.verdict_lr(lt)[1])


@braucht_messung
def test_saetze_unter_der_befundtabelle_stehen_wortgleich_in_der_readme():
    saetze = BF.befund_saetze()
    assert len(saetze) >= 6
    for satz in saetze:
        steht(satz)


@braucht_messung
def test_korrekturen_gegenueber_der_vormessung_gelten_in_den_daten():
    """Jede Aussage des Abschnitts „Befunde und Korrekturen“, die eine Zählung oder einen Vergleich enthält, als Test (DEMO-PLAYBOOK §11a Punkt 19)."""
    t10 = V.seed_tabelle(10, "endgap", "gemeinsam")
    nn10 = t10["nn"]
    assert all(w < nn10 for w in t10["formen"]["keiner"]["werte"])                                          # 1: n = 10, alle Läufe ohne Referenzwert unter dem Nächsten Nachbarn
    assert max(t10["formen"]["keiner"]["werte"]) < 1.25 * nn10                                                # kein Ausreißer wie in der Vormessung (73 und 81 %)
    for n in ns_mit():
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        a = V.seed_tabelle(n, "endgap", "abgestimmt")
        assert max(t["formen"]["keiner"]["werte"]) < 1.25 * t["nn"]
        assert a["formen"]["keiner"]["mittel"] < t["formen"]["keiner"]["mittel"]                              # abgestimmte Lernrate verbessert „keiner“ ...
        assert a["formen"]["keiner"]["mittel"] - a["formen"]["mittel"]["mittel"] > 2.0                        # ... schließt den Abstand zum Mittel aber nicht (mehr als 2 Prozentpunkte)
        assert t["formen"]["keiner"]["mittel"] - t["formen"]["mittel"]["mittel"] > 2.0
        assert a["formen"]["keiner"]["mittel"] - a["formen"]["mittel"]["mittel"] > 3.0 and t["formen"]["keiner"]["mittel"] - t["formen"]["mittel"]["mittel"] > 3.0      # „um mehr als 3 Prozentpunkte“
    if V.hat_messung(20):
        t20, a20 = V.seed_tabelle(20, "endgap", "gemeinsam"), V.seed_tabelle(20, "endgap", "abgestimmt")
        assert t20["formen"]["keiner"]["unter_nn"] <= t20["formen"]["keiner"]["s"] // 2 + 1                  # n = 20, gemeinsame Lernrate: höchstens die Hälfte (plus einer) der Läufe darunter
        assert abs(t20["formen"]["keiner"]["median"] - t20["nn"]) < 1.5
        assert a20["formen"]["keiner"]["unter_nn"] > a20["formen"]["keiner"]["s"] // 2                       # abgestimmt: in den meisten Läufen darunter
    cross10 = V.cross_daten(10)
    if cross10:
        assert RT.quadratsummen(np.array(cross10["test_gap"]))["gewichte"] > 2 * 13.1                        # 2: mehr als das Doppelte des Anteils der Vormessung
        assert np.array(cross10["test_gap"]).shape == (5, 5)
    cross20 = V.cross_daten(20)
    if cross20:
        assert np.array(cross20["test_gap"]).shape == (3, 3)
        assert RT.quadratsummen(np.array(cross20["test_gap"]))["gewichte"] > 10 * 1.8                       # mehr als das Zehnfache des Anteils der Vormessung
    g10, ab10 = V.seed_tabelle(10, "endgap", "gemeinsam"), V.seed_tabelle(10, "endgap", "abgestimmt")
    assert ab10["formen"]["rollout"]["lr"] == 0.0003 and ab10["formen"]["rollout"]["mittel"] > g10["formen"]["rollout"]["mittel"]    # 3: abgestimmter Rollout schlechter als bei 0,001
    steht("35 184 statt 34 160", "0,19- statt 0,18-fach", "1,36-fach", "bleibt die Lücke um mehr als 3 Prozentpunkte über der mit Referenzwert")
    assert round((5431 / 36) / (7 / 9)) == 194 and "loo" not in C.FORMEN                                      # 48,5 gegen das hochgerechnete Mittel, 194 gegen das ungerechnete; „loo“ nur im exakten Schritt
    steht("ohne Hochrechnung, gegen die Streuung 7/9 des Mittels, wäre es das 194-Fache", "Leave-one-out nur im exakten Schritt, nicht in der Streuungsmessung am Netz",
          "Der Schätzer mit dem Mittel über den Batch ist (B − 1)/B mal der Schätzer mit Leave-one-out")
    assert "in Experiment 1" not in TX.ERKLAERTABELLE.split("| Leave-one-out |")[1].split(chr(10))[0] and "Leave-one-out" in TX.SO_FUNKTIONIERT and "Schätzer mit dem Leave-one-out-Referenzwert" in TX.SO_FUNKTIONIERT
    assert K.ops_dekodierer_tour(20, 32) == 35_184 and PO.verhaeltnis_zum_mittel("pomo", 20) < 0.195


@braucht_messung
def test_einschraenkungen_nach_der_aussagenpruefung_gelten_in_den_daten():
    """Die Sätze, die nach der unabhängigen Prüfung eingeschränkt wurden, als Zählung oder Rechnung (Zahlen der Prüfung unabhängig mit scipy nachgerechnet)."""
    from scipy import stats
    t20, a20 = V.seed_tabelle(20, "endgap", "gemeinsam"), V.seed_tabelle(20, "endgap", "abgestimmt")
    assert 0.5 < t20["formen"]["keiner"]["mittel"] - t20["nn"] < 1.0 and t20["formen"]["keiner"]["unter_nn"] == 2 and a20["formen"]["keiner"]["unter_nn"] == 5
    steht("im Mittel 0,7 Prozentpunkte über dem Nächsten Nachbarn (2 von 6 Läufen darunter)", "in 5 von 6 Läufen darunter")
    for n, soll in ((10, 83.29), (20, 170.45)):                                                          # Lernrate 0,003 ohne Referenzwert: das Netz lernt nicht
        lt = V.lr_tabelle(n)
        assert abs(lt["formen"]["keiner"]["lr"][0.003]["mittel"] - soll) < 0.005
        assert lt["formen"]["keiner"]["lr"][0.003]["mittel"] > 80
    steht("83,29 % bei n = 10 und 170,45 % bei n = 20")
    for n, (g0, g1), p_soll in ((10, (5, 5), 0.16), (20, (3, 3), 0.08)):                                 # F-Test auf den Gewichts-Seed aus den Anteilen der Quadratsumme
        sh = RT.quadratsummen(np.array(V.cross_daten(n)["test_gap"]))
        f = (sh["gewichte"] / (g0 - 1)) / (sh["rest"] / ((g0 - 1) * (g1 - 1)))
        assert abs(float(stats.f.sf(f, g0 - 1, (g0 - 1) * (g1 - 1))) - p_soll) < 0.005
        assert 100 * (g0 - 1) / (g0 * g1 - 1) < sh["gewichte"]
    steht("p = 0,16 (n = 10, 5 × 5 Läufe) und p = 0,08 (n = 20, 3 × 3 Läufe)", "16,7 % (5 × 5) bzw. 25,0 % (3 × 3)", "Freiheitsgrade 4 von 24 bzw. 2 von 8")
    assert "Zu n = 50 liegt in dieser Demo keine Messung vor." in TX.ASSUMPTIONS_MARKDOWN and "empfindlich" not in README
    steht("Critic-Netz, in der Active Search ein gleitendes Mittel")
    assert "Active Search" in TX.SO_FUNKTIONIERT
    assert "einzeln gezählt, und" not in TX.ZAEHLREGEL and "Multiplikations-Additions-Paare" in TX.ZAEHLREGEL
    z = BF.befund_saetze()
    satz = next(s for s in z if s.startswith("Kein Lauf der Hauptreihe"))
    assert "0 von 116 verschiedenen Läufen" in satz and "128 Zellen" in satz and "12 Läufe doppelt" in satz
    kosten = [s for s in z if "bis zum Ziel" in s]
    assert len(kosten) == 2 and all("belastbar ist nur die Größenordnung, nicht die zweite Stelle" in s for s in kosten)
    assert any("wieder über dem Ziel" in s for s in kosten)                                              # n = 10: die Median-Kurve des Mittels ist nicht monoton
