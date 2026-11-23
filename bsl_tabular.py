"""Vier Orte, ein Lernschritt exakt: die Mini-Politik (eine Bewertung theta_j je Ort, Softmax über die noch offenen Orte, ein Zeigernetz ohne Kodierer).

Ort 0 bis 3 im Rechteck 3 x 4 ((0,0), (3,0), (3,4), (0,4)); Wege 3, 4, 5. Von Ort 0 aus gibt es 3! = 6 Touren mit den Längen 14, 16, 18, 16, 18, 14 (Reihenfolge der Aufzählung:
(0,1,2,3), (0,1,3,2), (0,2,1,3), (0,2,3,1), (0,3,1,2), (0,3,2,1)). Alle Erwartungswerte und Streuungen entstehen durch Aufzählen aller 6^B (bzw. 6^4 bei POMO) Kombinationen
mit ihrer Wahrscheinlichkeit, kein Zufall. Bei theta = 0 gibt es eine zweite Rechnung mit Brüchen (`enumerate_step_exact`), die Zahlen 5431/36, 7/9, 139/36, 28/9 liefert.

Streuung = Gesamtvarianz = Spur der Kovarianzmatrix der vier Komponenten des Schritts (Gradient nach theta_0 bis theta_3)."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np

import bsl_constants as C

PUNKTE = ((0, 0), (3, 0), (3, 4), (0, 4))
ABSTAND = tuple(tuple(round(math.hypot(a[0] - b[0], a[1] - b[1])) for b in PUNKTE) for a in PUNKTE)   # 3, 4, 5 sind ganzzahlig
ORTE = 4
FORMEN_EXAKT = ("keiner", "mittel", "rollout", "loo")


def touren_ab(start: int) -> list[tuple]:
    """Alle 3! = 6 Touren ab `start` (Reihenfolge: lexikographisch über die übrigen Orte)."""
    rest = [c for c in range(ORTE) if c != start]
    return [(start,) + p for p in itertools.permutations(rest)]


def laenge(tour) -> int:
    """Länge der geschlossenen Tour."""
    return sum(ABSTAND[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def laengen_ab(start: int = 0) -> list[int]:
    return [laenge(t) for t in touren_ab(start)]


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def log_wahrscheinlichkeit(theta, tour) -> float:
    """log p der Tour: je Schritt Softmax über die noch offenen Orte, zustandsunabhängige Bewertungen theta_j."""
    theta = np.asarray(theta, dtype=np.float64)
    besucht = {tour[0]}
    lp = 0.0
    for c in tour[1:]:
        offen = [j for j in range(ORTE) if j not in besucht]
        lp += math.log(float(_softmax(theta[offen])[offen.index(c)]))
        besucht.add(c)
    return lp


def wahrscheinlichkeiten(theta, start: int = 0) -> np.ndarray:
    """Wahrscheinlichkeit jeder der sechs Touren ab `start`."""
    return np.array([math.exp(log_wahrscheinlichkeit(theta, t)) for t in touren_ab(start)])


def score(theta, tour) -> np.ndarray:
    """Gradient von log p nach theta (vier Komponenten): je Schritt Anzeige(gewählt) minus Softmax-Wahrscheinlichkeit der offenen Orte; die Summe der Komponenten ist 0."""
    theta = np.asarray(theta, dtype=np.float64)
    g = np.zeros(ORTE)
    besucht = {tour[0]}
    for c in tour[1:]:
        offen = [j for j in range(ORTE) if j not in besucht]
        p = _softmax(theta[offen])
        for k, j in enumerate(offen):
            g[j] -= p[k]
        g[c] += 1.0
        besucht.add(c)
    return g


def scores_ab(theta, start: int = 0) -> np.ndarray:
    return np.array([score(theta, t) for t in touren_ab(start)])


def erwartete_laenge(theta, start: int = 0) -> float:
    """J(theta) = Erwartungswert der Länge der gezogenen Tour ab `start`."""
    return float(wahrscheinlichkeiten(theta, start) @ np.array(laengen_ab(start), dtype=np.float64))


def wahrer_gradient(theta, start: int = 0) -> np.ndarray:
    """Gradient von J nach theta: Summe p L Score (Aufzählung aller Touren)."""
    p = wahrscheinlichkeiten(theta, start)
    L = np.array(laengen_ab(start), dtype=np.float64)
    return (p * L) @ scores_ab(theta, start)


def wahrer_gradient_alle_starts(theta) -> np.ndarray:
    """Gradient von J_alle = Mittel über die vier Startorte (Ziel des POMO-Schritts)."""
    return sum(wahrer_gradient(theta, s) for s in range(ORTE)) / ORTE


def greedy_laenge(theta, start: int = 0) -> int:
    """Länge der Greedy-Tour (je Schritt der offene Ort mit der größten Bewertung, Gleichstand: kleinster Ort) ab `start`."""
    theta = np.asarray(theta, dtype=np.float64)
    tour, besucht = [start], {start}
    while len(tour) < ORTE:
        offen = [j for j in range(ORTE) if j not in besucht]
        best = max(offen, key=lambda j: (theta[j], -j))
        tour.append(best)
        besucht.add(best)
    return laenge(tour)


def gewichte_je_tour(theta, form: str, start: int = 0) -> dict:
    """Gewichte (L - b) der sechs Touren bei dem Referenzwert, der im Mittel herauskommt (Anzeige der Balken): keiner b = 0; Mittel und Leave-one-out b = Erwartungswert der Länge;
    Rollout b = Länge der Greedy-Tour. Rückgabe {"laengen", "p", "b", "gewichte"}."""
    L = np.array(laengen_ab(start), dtype=np.float64)
    p = wahrscheinlichkeiten(theta, start)
    if form == "keiner":
        b = 0.0
    elif form in ("mittel", "loo"):
        b = float(p @ L)
    elif form == "rollout":
        b = float(greedy_laenge(theta, start))
    else:
        raise ValueError(f"Unbekannte Form {form!r}")
    return {"laengen": L, "p": p, "b": b, "gewichte": L - b}


def _gegen_gradient(E, G) -> dict:
    """Anteil des Erwartungswerts am wahren Gradienten (<E, G> / <G, G>) und Kosinus der Richtung; NaN, wenn der wahre Gradient (fast) null ist
    (POMO bei theta = 0: die Mittel über die Starts heben sich auf, der Satz „Erwartung gleich (N-1)/N mal Gradient“ sagt dort nichts)."""
    gg = float(G @ G)
    if gg < 1e-18:
        return {"faktor": float("nan"), "richtung": float("nan")}
    ee = float(E @ E)
    return {"faktor": float(E @ G) / gg, "richtung": float(E @ G) / math.sqrt(ee * gg) if ee > 0 else float("nan")}


def enumerate_step(theta, B: int, form: str, start: int = 0) -> dict:
    """Ein Schritt mit B Touren (gezogen, mit Zurücklegen, ab `start`), alle 6^B Kombinationen aufgezählt.

    Der Schritt ist g = (1/B) Summe_i (L_i - b_i) Score_i. Rückgabe: erwartung (4,), gesamtstreuung (Spur der Kovarianz), wahrer_gradient (4,), faktor
    (Anteil des Erwartungswerts am wahren Gradienten, <E, G> / <G, G>), richtung (Kosinus zwischen E und G)."""
    if form not in FORMEN_EXAKT:
        raise ValueError(f"Unbekannte Form {form!r}")
    if B < 2 and form in ("mittel", "loo"):
        raise ValueError("Mittel und Leave-one-out brauchen mindestens zwei Touren")
    L = np.array(laengen_ab(start), dtype=np.float64)
    p = wahrscheinlichkeiten(theta, start)
    S = scores_ab(theta, start)
    idx = np.array(list(itertools.product(range(6), repeat=B)))                     # (6^B, B)
    Li, Si = L[idx], S[idx]                                                          # (6^B, B), (6^B, B, 4)
    w = p[idx].prod(axis=1)
    if form == "keiner":
        b = np.zeros_like(Li)
    elif form == "mittel":
        b = np.repeat(Li.mean(axis=1, keepdims=True), B, axis=1)
    elif form == "rollout":
        b = np.full_like(Li, float(greedy_laenge(theta, start)))
    else:
        b = (Li.sum(axis=1, keepdims=True) - Li) / (B - 1)
    g = ((Li - b)[:, :, None] * Si).sum(axis=1) / B                                  # (6^B, 4)
    E = (w[:, None] * g).sum(axis=0)
    var = float((w[:, None] * g * g).sum() - (E * E).sum())
    G = wahrer_gradient(theta, start)
    return {"erwartung": E, "gesamtstreuung": var, "wahrer_gradient": G, **_gegen_gradient(E, G), "faktor_formel": 1.0 if form != "mittel" else (B - 1) / B}


def pomo_step(theta, mit_referenzwert: bool = True) -> dict:
    """POMO mit N = 4 Startorten: je Start eine Tour, Referenzwert = Mittel der vier Längen (oder keiner); alle 6^4 = 1 296 Kombinationen aufgezählt.

    Ziel ist der Gradient von J_alle (Mittel über die Starts). Rückgabe wie `enumerate_step`."""
    Ls = [np.array(laengen_ab(s), dtype=np.float64) for s in range(ORTE)]
    ps = [wahrscheinlichkeiten(theta, s) for s in range(ORTE)]
    Ss = [scores_ab(theta, s) for s in range(ORTE)]
    idx = np.array(list(itertools.product(range(6), repeat=ORTE)))                  # (1296, 4)
    Li = np.stack([Ls[s][idx[:, s]] for s in range(ORTE)], axis=1)                  # (1296, 4)
    Si = np.stack([Ss[s][idx[:, s]] for s in range(ORTE)], axis=1)                  # (1296, 4, 4)
    w = np.prod(np.stack([ps[s][idx[:, s]] for s in range(ORTE)], axis=1), axis=1)
    b = Li.mean(axis=1, keepdims=True) if mit_referenzwert else np.zeros_like(Li)
    g = ((Li - b)[:, :, None] * Si).sum(axis=1) / ORTE
    E = (w[:, None] * g).sum(axis=0)
    var = float((w[:, None] * g * g).sum() - (E * E).sum())
    G = wahrer_gradient_alle_starts(theta)
    return {"erwartung": E, "gesamtstreuung": var, "wahrer_gradient": G, **_gegen_gradient(E, G), "faktor_formel": (ORTE - 1) / ORTE if mit_referenzwert else 1.0}


def schritt_zeilen(theta, B: int) -> list[dict]:
    """Alle Zeilen der Primäransicht für Politik theta und B Touren: keiner, Mittel, Rollout, Leave-one-out (Zusatzzeile) und POMO mit N = 4."""
    zeilen = []
    for form in FORMEN_EXAKT:
        r = enumerate_step(theta, B, form)
        zeilen.append({"form": form, "B": B, "N": None, **r})
    pk = pomo_step(theta, True)
    pn = pomo_step(theta, False)
    zeilen.append({"form": "pomo", "B": None, "N": ORTE, **pk, "streuung_ohne": pn["gesamtstreuung"]})
    return zeilen


def hochgerechnete_streuung(zeile: dict) -> float:
    """Streuung auf gleiche Erwartung hochgerechnet: Gesamtstreuung geteilt durch das Quadrat des Faktors aus der Formel (1, (B-1)/B, (N-1)/N)."""
    f = zeile["faktor_formel"]
    return zeile["gesamtstreuung"] / (f * f) if f else float("nan")


# ------------------------------------------------------------------------------------------------------------------ dieselbe Aufzählung in Brüchen (theta = 0)
def _scores_gleich(start: int = 0) -> list[list[Fr]]:
    """Score bei theta = 0 in Brüchen: je Schritt Anzeige(gewählt) minus 1 / (Zahl offener Orte)."""
    out = []
    for tour in touren_ab(start):
        g = [Fr(0)] * ORTE
        besucht = {tour[0]}
        for c in tour[1:]:
            offen = [j for j in range(ORTE) if j not in besucht]
            for j in offen:
                g[j] -= Fr(1, len(offen))
            g[c] += 1
            besucht.add(c)
        out.append(g)
    return out


def enumerate_step_exact(B: int, form: str) -> dict:
    """Wie `enumerate_step` bei theta = 0, aber in Brüchen: {"erwartung": [Fraction]*4, "gesamtstreuung": Fraction, "wahrer_gradient": [Fraction]*4}."""
    L = [Fr(x) for x in laengen_ab(0)]
    S = _scores_gleich(0)
    greedy = Fr(laenge((0, 1, 2, 3)))          # bei theta = 0 baut Greedy die Tour mit dem kleinsten Index
    G = [sum(L[i] * S[i][j] for i in range(6)) / 6 for j in range(ORTE)]
    E, E2 = [Fr(0)] * ORTE, Fr(0)
    wgt = Fr(1, 6 ** B)
    for idx in itertools.product(range(6), repeat=B):
        Ls = [L[i] for i in idx]
        g = [Fr(0)] * ORTE
        for k, i in enumerate(idx):
            if form == "keiner":
                b = Fr(0)
            elif form == "mittel":
                b = sum(Ls) / B
            elif form == "rollout":
                b = greedy
            elif form == "loo":
                b = (sum(Ls) - Ls[k]) / (B - 1)
            else:
                raise ValueError(form)
            for j in range(ORTE):
                g[j] += (Ls[k] - b) * S[i][j] / B
        for j in range(ORTE):
            E[j] += wgt * g[j]
        E2 += wgt * sum(x * x for x in g)
    return {"erwartung": E, "gesamtstreuung": E2 - sum(x * x for x in E), "wahrer_gradient": G}


def theta_von(politik: str) -> tuple:
    return C.POLITIK_THETA[politik]
