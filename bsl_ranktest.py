"""Rangtests für den Vergleich der Referenzwert-Formen über Zufalls-Seeds, ohne scipy: exaktes Wilcoxon-Vorzeichen-Rang-Test (Aufzählung aller Vorzeichen, n <= 14), Holm, Friedman mit
Schließformel für 2 und 3 Freiheitsgrade, Zerlegung der Quadratsumme einer gekreuzten Tabelle (Gewichts-Seed mal Daten-Seed).

Gegenprobe der Tests: scipy.stats (nur in tests/, requirements-dev.txt)."""

import math

import numpy as np


def _raenge(x: np.ndarray) -> np.ndarray:
    """Mittlere Ränge (Gleichstände bekommen den Mittelwert ihrer Plätze), Rang 1 = kleinster Wert."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon_exakt(d) -> dict:
    """Exakter zweiseitiger Vorzeichen-Rang-Test auf Differenzen d (Nullen werden weggelassen). Aufzählung aller 2^n Vorzeichenverteilungen der Ränge.

    Rückgabe {"p", "n", "w_plus", "w_minus"}; p = Anteil der Vorzeichenverteilungen, deren Rangsumme mindestens so weit vom Mittel n(n+1)/4 liegt wie die beobachtete.
    Zehn Paare mit gleichem Vorzeichen geben p = 2/1024."""
    d = np.asarray(d, dtype=np.float64)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"p": 1.0, "n": 0, "w_plus": 0.0, "w_minus": 0.0}
    if n > 14:
        raise ValueError("Exakter Test nur bis 14 Paare")
    r = _raenge(np.abs(d))
    w_plus = float(r[d > 0].sum())
    masks = np.arange(1 << n)
    summen = np.zeros(1 << n)
    for i in range(n):
        summen += ((masks >> i) & 1) * r[i]
    mitte = r.sum() / 2
    p = float(np.mean(np.abs(summen - mitte) >= abs(w_plus - mitte) - 1e-9))
    return {"p": p, "n": n, "w_plus": w_plus, "w_minus": float(r.sum() - w_plus)}


def holm(ps) -> list[float]:
    """Holm-Anpassung der p-Werte (in der Reihenfolge der Eingabe)."""
    ps = list(ps)
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [0.0] * m
    lauf = 0.0
    for rang, i in enumerate(order):
        lauf = max(lauf, min(1.0, (m - rang) * ps[i]))
        out[i] = lauf
    return out


def chi2_sf(x: float, df: int) -> float:
    """Obere Wahrscheinlichkeit der Chi-Quadrat-Verteilung (Schließformeln für df = 1, 2, 3)."""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    if df == 3:
        return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    raise ValueError("Nur df = 1, 2, 3")


def friedman(tabelle) -> dict:
    """Friedman-Test: Zeilen = Blöcke (Zufalls-Seeds), Spalten = Formen. Mit Korrektur für Gleichstände. Rückgabe {"chi2", "df", "p", "mittlere_raenge"}."""
    t = np.asarray(tabelle, dtype=np.float64)
    nb, k = t.shape
    r = np.vstack([_raenge(row) for row in t])
    rs = r.sum(axis=0)
    stat = 12.0 * float((rs ** 2).sum()) / (nb * k * (k + 1)) - 3.0 * nb * (k + 1)
    ties = 0.0
    for row in t:
        _, c = np.unique(row, return_counts=True)
        ties += float((c ** 3 - c).sum())
    korr = 1.0 - ties / (nb * k * (k * k - 1))
    chi2 = stat / korr if korr > 0 else 0.0
    return {"chi2": chi2, "df": k - 1, "p": chi2_sf(chi2, k - 1), "mittlere_raenge": (rs / nb).tolist()}


def paarweise(spalten: dict) -> list[dict]:
    """Alle Paare gepaarter Spalten (Name -> Werte je Zufalls-Seed, gleiche Reihenfolge): Wilcoxon exakt, Differenz der Mittel (a minus b) und Holm über alle Paare."""
    namen = list(spalten)
    paare = [(a, b) for i, a in enumerate(namen) for b in namen[i + 1:]]
    roh = []
    for a, b in paare:
        d = np.asarray(spalten[a], dtype=np.float64) - np.asarray(spalten[b], dtype=np.float64)
        w = wilcoxon_exakt(d)
        roh.append({"a": a, "b": b, "p": w["p"], "diff_mittel": float(d.mean()), "n_paare": w["n"], "a_besser": int((d < 0).sum()), "b_besser": int((d > 0).sum())})
    for r, ph in zip(roh, holm([r["p"] for r in roh])):
        r["p_holm"] = ph
    return roh


def quadratsummen(tabelle) -> dict:
    """Zerlegung der Gesamtquadratsumme einer Tabelle (Zeilen = Gewichts-Seeds, Spalten = Daten-Seeds) in Zeilen, Spalten und Rest, in Prozent.

    Der Rest enthält die Wechselwirkung und das Rauschen (ohne Wiederholungszellen nicht trennbar)."""
    t = np.asarray(tabelle, dtype=np.float64)
    z, s = t.shape
    g = t.mean()
    ss_z = s * float(((t.mean(axis=1) - g) ** 2).sum())
    ss_s = z * float(((t.mean(axis=0) - g) ** 2).sum())
    ss_t = float(((t - g) ** 2).sum())
    if ss_t == 0:
        return {"gewichte": 0.0, "daten": 0.0, "rest": 0.0, "gesamt": 0.0}
    return {"gewichte": 100 * ss_z / ss_t, "daten": 100 * ss_s / ss_t, "rest": 100 * (ss_t - ss_z - ss_s) / ss_t, "gesamt": ss_t}
