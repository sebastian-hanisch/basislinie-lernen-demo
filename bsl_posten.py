"""Trainingsposten je Referenzwert (Zählregel Q5: Training ist der dritte Posten, nie in K oder O des Einsatzes eingerechnet).

Posten eines Laufs: Instanzen im Kodierer des lernenden Netzes, gezogene Touren, Baseline-Touren (Greedy-Touren des Referenznetzes beim Rollout: 120 je Schritt plus
2 mal 500 je Austausch-Prüfung alle 100 Schritte), Kodierer-Durchläufe (Instanzen plus Baseline-Touren: jede Baseline-Tour kodiert ihre Instanz beim Referenznetz einmal) und
grobe Operationen als Annahme (Rückwärtspass doppelt so teuer wie der Vorwärtspass, Schritt = 3 mal Vorwärts; die Baseline-Touren und die Prüfung brauchen nur Vorwärts).
Kodierer und Dekodierer in O nach `bsl_cost` (n = 20, d = 32: 211 200 und 35 184)."""

import bsl_constants as C
import bsl_cost as K


def instanzen_je_schritt(form: str, n: int, batch: int = C.BATCH) -> int:
    """Instanzen im Kodierer je Schritt: batch, bei POMO batch / n (eine Instanz hat n Startorte)."""
    return batch // n if form == "pomo" else batch


def posten_je_lauf(form: str, n: int, steps: int, batch: int = C.BATCH) -> dict:
    """Instanzen, gezogene Touren, Baseline-Touren und Kodierer-Durchläufe eines Laufs mit `steps` Schritten."""
    inst = instanzen_je_schritt(form, n, batch) * steps
    base = 0
    pruefungen = 0
    if form == "rollout":
        pruefungen = steps // C.ROLLOUT_ALLE
        base = batch * steps + 2 * C.ROLLOUT_INSTANZEN * pruefungen
    return {"instanzen": inst, "touren": batch * steps, "baseline_touren": base, "kodierer": inst + base, "pruefungen": pruefungen}


def ops_je_lauf(form: str, n: int, steps: int, d: int = C.D_STANDARD, batch: int = C.BATCH) -> int:
    """Grobe Operationen eines Laufs (Annahme: Schritt = (1 + RUECKWAERTS_FAKTOR) mal Vorwärts; Rollout-Baseline und Austausch-Prüfung nur Vorwärts)."""
    e, t = K.ops_kodierer(n, d), K.ops_dekodierer_tour(n, d)
    inst = instanzen_je_schritt(form, n, batch)
    gesamt = steps * (1 + C.RUECKWAERTS_FAKTOR) * (inst * e + batch * t)
    if form == "rollout":
        gesamt += batch * steps * (e + t) + 2 * C.ROLLOUT_INSTANZEN * (steps // C.ROLLOUT_ALLE) * (e + t)
    return int(gesamt)


def ops_je_schritt(form: str, n: int, d: int = C.D_STANDARD, batch: int = C.BATCH) -> float:
    """Grobe Operationen je Schritt (Rollout mit der Prüfung auf einen Schritt umgelegt)."""
    e, t = K.ops_kodierer(n, d), K.ops_dekodierer_tour(n, d)
    inst = instanzen_je_schritt(form, n, batch)
    gesamt = (1 + C.RUECKWAERTS_FAKTOR) * (inst * e + batch * t)
    if form == "rollout":
        gesamt += batch * (e + t) + 2 * C.ROLLOUT_INSTANZEN * (e + t) / C.ROLLOUT_ALLE
    return float(gesamt)


def verhaeltnis_zum_mittel(form: str, n: int, d: int = C.D_STANDARD) -> float:
    """Operationen je Schritt einer Form geteilt durch die des Mittels (POMO etwa 0,2-fach, Rollout etwa 1,36-fach)."""
    return ops_je_schritt(form, n, d) / ops_je_schritt("mittel", n, d)


def einsatz_kosten(n: int, zugauswertungen_nn_2opt: float, d: int = C.D_STANDARD) -> dict:
    """Einsatzkosten einer Tour in K und O (Leserichtung: K begünstigt das Netz, O benachteiligt es), Training steht daneben als dritter Posten.

    Netz: K = n(n-1)/2, O aus den Formen der Matrixprodukte; Nächster Nachbar: K gleich, O = 8 je Kandidatenbewertung (Annahme);
    Nächster Nachbar mit 2-opt: K = Konstruktion + mittlere Zahl der Zugauswertungen (aus der Messung), O = 8 je Kandidatenbewertung."""
    k_con = K.k_konstruktion(n)
    return {
        "netz": {"k": k_con, "o": K.ops_netz_tour(n, d)},
        "nn": {"k": k_con, "o": K.ops_kandidaten(k_con)},
        "nn_2opt": {"k": k_con + zugauswertungen_nn_2opt, "o": K.ops_kandidaten(k_con + zugauswertungen_nn_2opt)},
    }
