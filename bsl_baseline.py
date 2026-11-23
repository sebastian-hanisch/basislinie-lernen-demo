"""Referenzwerte (Baselines) des REINFORCE-Schritts, je Form eine kleine Einheit.

Ein Schritt mit Längen L (M Instanzen, K Touren je Instanz) und Referenzwert b hat die Gewichte w = (L - b) / L.size. Der Referenzwert darf ein Skalar oder ein Feld
der Form (M, K) sein (Schnittstelle von `bsl_train.reinforce_step`, `baseline_fn(X, starts, L)`).

* keiner: b = 0.
* mittel: b = Mittel aller Längen des Schritts (verschiedene Instanzen). Es enthält die eigene Tour: der Erwartungswert des Schritts wird mit (B - 1) / B multipliziert.
* rollout: b = Länge der Greedy-Tour eines Referenznetzes auf derselben Instanz (hängt nicht von der gezogenen Tour ab).
* pomo: b = Mittel der Längen aller Touren derselben Instanz (eine Tour je Startort); Faktor (N - 1) / N.
* loo (Leave-one-out): b_i = Mittel der anderen B - 1 Längen; nur im exakten Schritt, nicht in der Streuungsmessung am Netz (Experiment 1) und nicht im Training.

Das Referenznetz des Rollouts wird alle 100 Schritte durch das aktuelle Netz ersetzt, wenn dieses auf 500 festen Instanzen im gepaarten einseitigen t-Test (p < 0,05)
kürzere Greedy-Touren baut; die Entscheidung ist t < -t_crit (kein scipy zur Laufzeit; Konstante mit scipy geprüft)."""

import math

import numpy as np

import bsl_constants as C
import bsl_net as N


def reference_value(form: str, L: np.ndarray, ref_L: np.ndarray | None = None):
    """Referenzwert einer Form für die Längen L (M, K): Skalar oder Feld (M, 1) / (M, K). `ref_L` (M, K) sind die Greedy-Längen des Referenznetzes (nur Rollout)."""
    L = np.asarray(L, dtype=np.float64)
    if form == "keiner":
        return 0.0
    if form == "mittel":
        return float(L.mean())
    if form == "pomo":
        return L.mean(axis=1, keepdims=True)
    if form == "rollout":
        if ref_L is None:
            raise ValueError("Der Rollout braucht die Greedy-Längen des Referenznetzes")
        return np.asarray(ref_L, dtype=np.float64)
    if form == "loo":
        s = L.sum()
        if L.size < 2:
            raise ValueError("Leave-one-out braucht mindestens zwei Touren")
        return (s - L) / (L.size - 1)
    raise ValueError(f"Unbekannte Form {form!r}")


def advantage_weights(L: np.ndarray, b) -> np.ndarray:
    """Gewichte (L - b) / (Zahl der Touren) des Schritts; positiv heißt: die Tour wird unwahrscheinlicher gemacht (Länge ist die Strafe)."""
    L = np.asarray(L, dtype=np.float64)
    return (L - b) / L.size


def expected_gradient_scale(form: str, B: int = C.BATCH, N_starts: int = 1) -> float:
    """Faktor, mit dem der Erwartungswert des Schritts den wahren Gradienten von J skaliert: 1 (keiner, Rollout, Leave-one-out), (B - 1) / B (Mittel), (N - 1) / N (POMO)."""
    if form in ("keiner", "rollout", "loo"):
        return 1.0
    if form == "mittel":
        return (B - 1) / B
    if form == "pomo":
        return (N_starts - 1) / N_starts
    raise ValueError(f"Unbekannte Form {form!r}")


def paired_t(laenge_aktuell: np.ndarray, laenge_referenz: np.ndarray) -> float:
    """t-Statistik des gepaarten Tests auf die Differenz (aktuell minus Referenz); negativ heißt: das aktuelle Netz ist kürzer."""
    dif = np.asarray(laenge_aktuell, dtype=np.float64) - np.asarray(laenge_referenz, dtype=np.float64)
    m = dif.size
    sd = float(dif.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(dif.mean()) == 0.0 else math.copysign(math.inf, float(dif.mean()))
    return float(dif.mean()) / (sd / math.sqrt(m))


def rollout_should_replace(Lc: np.ndarray, Lb: np.ndarray, t_crit: float = C.T_CRIT_499) -> bool:
    """Ersetzt das aktuelle Netz das Referenznetz? Ja, wenn t < -t_crit (einseitig, p < 0,05; bei 500 Instanzen t_crit = 1,6479)."""
    return paired_t(Lc, Lb) < -t_crit


def greedy_reference_lengths(P_ref: dict, X: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Längen der Greedy-Touren des Referenznetzes auf X (M, n, 2) mit den Startorten `starts` (M, K): Form (M, K)."""
    return N.tour_lengths(X, N.greedy_tours(P_ref, X, starts))


def make_baseline_fn(form: str, state: dict | None = None):
    """`baseline_fn(X, starts, L)` für `bsl_train.reinforce_step`. Beim Rollout liest sie das Referenznetz aus `state["ref"]` (der Aufrufer ersetzt es alle 100 Schritte)."""
    if form == "rollout":
        def _rollout(X, starts, L):
            return reference_value("rollout", L, greedy_reference_lengths(state["ref"], X, starts))
        return _rollout
    if form in ("keiner", "mittel", "pomo", "loo"):
        return lambda X, starts, L: reference_value(form, L)
    raise ValueError(f"Unbekannte Form {form!r}")
