"""Konstanten: Netz und Training (Namen und Werte wie in der Quelle zeigernetz-demo), Referenzwert-Formen, Zufalls-Seeds je Zweck, Messgrößen, Verdikt-Schwellen, Regler, Farben, Presets.

Der erste Block (Netz, Instanzen, Zwecke 2 bis 6 und 8, 11, 24, 33) wird von den Kopien der Quelle gebraucht und hat dieselben Namen und Werte wie `znz_constants`.
Neue Zwecke dieser Demo liegen im reservierten Bereich 150 bis 199."""

# ---------------------------------------------------------------- Netz und Training (wie Quelle)
C_CLIP = 10.0
BATCH = 120                                # Touren je Trainingsschritt
LERNRATE = 1e-3
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
NORM_BEGRENZUNG = 1.0
ZEITWAECHTER_SEKUNDEN = 100.0
EVAL_ALLE = 100                            # Name der Quelle (wird von der kopierten Trainingsschleife als Vorgabe gelesen)
LETZTE_MESSPUNKTE = 3
VARIANTE_ATTN_FFN = {"voll": (True, True), "einbettung": (False, False), "attention": (True, False), "vorwaerts": (False, True)}
D_STANDARD = 32
N_HAND = 5
MODELL_FORMAT = "znz-modell-1"

# ---------------------------------------------------------------- Instanzen (wie Quelle)
FAMILIEN = ("gleich", "cluster", "gitter")
FAMILIE_HAND = "hand"
SEITE = 1000.0
CLUSTER_ZAHL, CLUSTER_SIGMA = 5, 40.0
GITTER_FAKTOR, GITTER_RAUSCHEN = 1.5, 0.1
VAL_OFFSET = 1000                           # Validierungsinstanzen haben Zufalls-Seeds ab 1000
HAND_PUNKTE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5))
HAND_SKALA = 4.0

# ---------------------------------------------------------------- Zwecke der Zufalls-Seeds (derive)
ZWECK_INIT, ZWECK_TRAIN_X, ZWECK_TRAIN_S, ZWECK_SAMPLE, ZWECK_BOOT, ZWECK_ZUFALL = 2, 3, 4, 5, 6, 8
ZWECK_INSTANZ, ZWECK_START, ZWECK_LAUF = 11, 33, 24
ZWECK_AUSTAUSCH = 150                       # 500 feste Instanzen der Austausch-Prüfung des Rollout-Referenznetzes (Schlüssel n, Daten-Seed)
ZWECK_POOL = 151                            # Instanzpool von Experiment 1 (n = 6)
ZWECK_WIEDERHOLUNG = 152                    # Wiederholungen von Experiment 1
ZWECK_E1_NETZ = 153                         # ungelerntes Netz von Experiment 1

# ---------------------------------------------------------------- Zählung (Q5)
OPS_JE_KANDIDAT = 8                         # Annahme: grobe Operationen je Kandidatenbewertung der klassischen Verfahren
RUECKWAERTS_FAKTOR = 2                      # Annahme: der Rückwärtspass kostet das Doppelte des Vorwärtspasses (Schritt = 3 mal Vorwärts)

# ---------------------------------------------------------------- Referenzwert-Formen
FORMEN = ("keiner", "mittel", "rollout", "pomo")
FORM_KURZ = {"keiner": "keiner", "mittel": "Mittel", "rollout": "Rollout", "pomo": "POMO", "loo": "Leave-one-out"}
FORM_ANZEIGE = {
    "keiner": "Kein Referenzwert",
    "mittel": "Mittel über den Batch (verschiedene Instanzen)",
    "rollout": "Greedy-Rollout (Referenznetz)",
    "pomo": "POMO-Mittel (alle Startorte)",
    "loo": "Leave-one-out (nur im exakten Schritt)",
}
ROLLOUT_ALLE = 100                          # Austausch-Prüfung des Referenznetzes alle 100 Schritte
ROLLOUT_INSTANZEN = 500                     # feste Instanzen der Prüfung
T_CRIT_499 = 1.647913                       # t_{0,95; 499}: Schwelle des gepaarten einseitigen t-Tests (p < 0,05) bei 500 Instanzen

# ---------------------------------------------------------------- Messung (eingefroren)
N_MESS = (10, 20)
SEEDS_MESS = 10                             # Zufalls-Seeds 0 bis 9 (Gewichte 1000 + s, Daten 2000 + s)
GEWICHT_BASIS, DATEN_BASIS = 1000, 2000
TUNE_GEWICHT, TUNE_DATEN = (9001, 9002), (9501, 9502)       # Abstimmungs-Seeds: nie dieselben wie in der Hauptreihe
CROSS_GEWICHT_BASIS, CROSS_DATEN_BASIS = 3000, 4000
SCHRITTE_MESS = {10: 4000, 20: 8000}
MESS_ALLE = 250                             # Messpunkt der eingefrorenen Verläufe
TEST_N = {10: 1000, 20: 300}                # Testinstanzen (Zufalls-Seed 0 bis Anzahl - 1)
VAL_N = {10: 200, 20: 200}                  # Validierungsinstanzen (Zufalls-Seed ab 1000)
LERNRATEN = (1e-4, 3e-4, 1e-3, 3e-3)
LR_SCHRITTE = {10: 2000, 20: 3000}
LR_FORMEN = {10: FORMEN, 20: ("keiner", "mittel")}
CROSS_GRID = {10: 5, 20: 4}                 # gekreuzt: Gewichts-Seeds mal Daten-Seeds (5 x 5 bzw. 4 x 4)
KURZ_SCHRITTE = {10: 300, 20: 250}          # Voreinstellung des Live-Trainings in Experiment 2
KURZ_ALLE = 50                              # Messpunkt des Live-Trainings
FROZEN_FORMAT = "bsl-frozen-1"
PRETRAINED_FORMAT = "bsl-pretrained-1"

# ---------------------------------------------------------------- Verdikte (gesetzt, nicht abgeleitet)
MARKE_KLAR, MARKE_GLEICH = 0.25, 0.10       # wie Quelle (znz_stats)
BOOTSTRAP_ZIEHUNGEN = 2000
HOLM_SCHWELLE = 0.05                        # Festlegung
RAUSCHGRENZE = 0.5                          # Prozentpunkte: Rauschen des Endpunkts eines Laufs (Vormessung)
ZIEL_AUFSCHLAG = 0.5                        # Ziel der Kostenkurve = Median des Mittels am Ende plus Rauschgrenze
STREUUNG_DEUTLICH = 2.0                     # Faktor für „deutlich kleiner“ im exakten Schritt und in Experiment 1
E1_N, E1_POOL, E1_REPS = 6, 16, 200         # Experiment 1: Orte, Instanzen im Pool, Wiederholungen
E1_B = (6, 12, 24, 48)
E1_B_STANDARD = 12

# ---------------------------------------------------------------- Regler
SEED_MAX = 9
DEFAULT_N, DEFAULT_SEED = 10, 0
DEFAULT_FORMEN = FORMEN
TOUR_B = (2, 3, 4)
DEFAULT_TOUR_B = 2
POLITIKEN = ("gleich", "kurz", "lang")
POLITIK_ANZEIGE = {"gleich": "Gleich (alle Orte gleich bewertet)", "kurz": "Kurze Touren bevorzugt", "lang": "Lange Touren bevorzugt"}
POLITIK_THETA = {"gleich": (0.0, 0.0, 0.0, 0.0), "kurz": (0.0, 1.0, 0.5, 0.0), "lang": (0.0, 0.5, 1.0, 0.0)}
DEFAULT_POLITIK = "gleich"
STAND_OPTIONEN = ("ungelernt", "vorgerechnet")
DEFAULT_STAND = "ungelernt"
SCHRITTE_LIVE = tuple(range(50, 1001, 50))
SCHRITTE_LIVE_MAX = 1000
LIVE_ZEITGRENZE = 60.0                      # Sekunden, Prüfung an jeder Messstelle
LERNRATE_STUFEN = LERNRATEN
METRIKEN = ("endgap", "letzte5")
METRIK_ANZEIGE = {"endgap": "Endlücke des Endmodells (Testsatz)", "letzte5": "Mittel der letzten 5 Messpunkte (Validierung)"}
LR_MODI = ("gemeinsam", "abgestimmt")
LR_MODUS_ANZEIGE = {"gemeinsam": "Gemeinsame Lernrate 0,001", "abgestimmt": "Je Form abgestimmt (Experiment 4)"}
EINHEITEN = ("schritte", "touren", "kodierer", "ops")
EINHEIT_ANZEIGE = {"schritte": "Schritte", "touren": "Gezogene Touren", "kodierer": "Kodierer-Durchläufe", "ops": "Grobe Operationen (Annahme 3 mal Vorwärts)"}
DEFAULT_EINHEIT = "ops"
LETZTE_K = (1, 3, 5)
DEFAULT_LETZTE_K = 1

# ---------------------------------------------------------------- Farben
FORM_FARBE = {"keiner": "#e45756", "mittel": "#4c78a8", "rollout": "#f58518", "pomo": "#2F6B65", "loo": "#7f7f7f"}
COLOR_NN = "#8c564b"
COLOR_2OPT = "#9467bd"
COLOR_OPT = "#4c78a8"
COLOR_GRAU = "rgba(150,150,150,0.75)"

# ---------------------------------------------------------------- Presets (das erste ist die Voreinstellung beim Öffnen; Zufalls-Seeds der Kurz-Läufe aus der Messung)
PRESET_SEED_KURZ = {10: 1, 20: 1}
_BASIS = {
    "n": DEFAULT_N, "seed": DEFAULT_SEED, "formen": DEFAULT_FORMEN, "tour_b": DEFAULT_TOUR_B, "politik": DEFAULT_POLITIK, "e1_b": E1_B_STANDARD, "e1_stand": DEFAULT_STAND,
    "schritte": 300, "lr": LERNRATE, "vorgerechnet": False, "instanz": 0, "metrik": "endgap", "e3_lr": "gemeinsam", "verlaeufe": False, "e4_formen": FORMEN,
    "einheit": DEFAULT_EINHEIT, "k": DEFAULT_LETZTE_K,
}
PRESETS = {
    "Handbeispiel: vier Orte": dict(_BASIS),
    "Kurzer Lauf n = 10": {**_BASIS, "seed": PRESET_SEED_KURZ[10]},
    "Kurzer Lauf n = 20": {**_BASIS, "n": 20, "seed": PRESET_SEED_KURZ[20], "schritte": 250},
    "Sechs Zufalls-Seeds, n = 20": {**_BASIS, "n": 20, "verlaeufe": True},
    "Ist „keiner“ nur falsch eingestellt?": {**_BASIS, "e3_lr": "abgestimmt"},
    "Was kostet das Lernen?": {**_BASIS, "n": 20, "einheit": "ops"},
}
PRESET_HELP = {
    "Handbeispiel: vier Orte": "Der exakte Lernschritt an vier Orten: B = 2 Touren, Politik „gleich“, alle 36 Paare aufgezählt. Die Zahlen sind Brüche und ändern sich nie.",
    "Kurzer Lauf n = 10": "Live-Training mit 10 Orten und 300 Schritten für alle vier Formen (Lernrate 0,001), mit einem Zufalls-Seed, bei dem „keiner“ nach diesem kurzen Lauf über dem Nächsten Nachbarn endet.",
    "Kurzer Lauf n = 20": "Live-Training mit 20 Orten und 250 Schritten für alle vier Formen (Lernrate 0,001), mit einem Zufalls-Seed, bei dem „keiner“ nach diesem kurzen Lauf über dem Nächsten Nachbarn endet.",
    "Sechs Zufalls-Seeds, n = 20": "Eingefrorene Hauptreihe bei n = 20 mit allen Verläufen je Zufalls-Seed.",
    "Ist „keiner“ nur falsch eingestellt?": "Die Hauptreihe bei n = 10 mit der je Form abgestimmten Lernrate; die Abstimmung steht bei Experiment 4.",
    "Was kostet das Lernen?": "Kosten bei n = 20 in groben Operationen (Annahme: Rückwärtspass doppelt so teuer wie Vorwärtspass).",
}
