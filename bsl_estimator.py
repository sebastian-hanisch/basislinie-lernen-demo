"""Experiment 1: Streut der Schätzer mit Referenzwert weniger? Am echten Zeigernetz, n = 6 Orte, exakter Gradient durch Aufzählung aller Touren.

Pool: 16 feste Instanzen (n = 6, gleichverteilt im Einheitsquadrat, Blockstrom). Je Instanz gibt es 5! = 120 Touren ab Ort 0. Der exakte Gradient von J = Mittel über den Pool des
Erwartungswerts der Tourlänge ist Summe p L Score, eine Vorwärts- und Rückwärtsrechnung mit den Gewichten w = p L / 16 (der Gradient ist linear in w). Für POMO ist das Ziel der
Gradient von J_alle (Mittel über alle sechs Startorte), mit 720 Touren je Instanz.

Der Schätzer eines Schritts zieht B (Instanz, Tour)-Paare (bei POMO B / 6 Instanzen mit je sechs Startorten), bildet den Referenzwert und rechnet den Gradienten wie das Training.
Die Streuung ist die Gesamtvarianz der Schätzer über R Wiederholungen geteilt durch das Quadrat der Norm des exakten Gradienten des jeweiligen Ziels. Der Erwartungsfaktor
((B-1)/B, (N-1)/N) kommt aus der Formel, nicht aus den Wiederholungen: mit R = 200 ist ein Faktor 0,92 bei dieser Streuung nicht von 1 zu trennen."""

import itertools
import math

import numpy as np

import bsl_baseline as BL
import bsl_constants as C
import bsl_instances as I
import bsl_net as N
from bsl_rngvec import UStream
from bsl_scenario import derive


def pool_instanzen(seed: int = 0, count: int = C.E1_POOL, n: int = C.E1_N) -> np.ndarray:
    """Pool aus `count` Instanzen mit n Orten (Form (count, n, 2))."""
    return I.train_batch(UStream(derive(C.ZWECK_POOL, seed, n)), count, n)


def alle_touren(n: int, start: int) -> np.ndarray:
    """Alle (n-1)! Touren ab `start` (Form ((n-1)!, n))."""
    rest = [c for c in range(n) if c != start]
    return np.array([(start,) + p for p in itertools.permutations(rest)], dtype=np.int64)


def _flach(G: dict, schluessel: list) -> np.ndarray:
    return np.concatenate([G[k].ravel() for k in schluessel])


def exakter_gradient(P: dict, X: np.ndarray, alle_starts: bool = False) -> tuple[np.ndarray, float]:
    """Exakter Gradient von J durch Aufzählung (Vektor aller Parameter in der Reihenfolge von `P`) und die größte Abweichung der Wahrscheinlichkeitssumme je Instanz und Start von 1."""
    M, n = X.shape[0], X.shape[1]
    starts_liste = list(range(n)) if alle_starts else [0]
    tours = np.concatenate([alle_touren(n, s) for s in starts_liste])                  # (K, n)
    K = tours.shape[0]
    forced = np.broadcast_to(tours, (M, K, n)).copy()
    starts = np.broadcast_to(tours[:, 0], (M, K)).copy()
    H, _ = N.encode(P, X)
    out = N.decode(P, H, starts, mode="forced", forced=forced, keep=False)
    p = np.exp(out["logp"])
    L = N.tour_lengths(X, forced)
    je_start = K // len(starts_liste)
    summen = p.reshape(M, len(starts_liste), je_start).sum(axis=2)
    w = p * L / (M * len(starts_liste))
    _, G = N.loss_and_grad(P, X, starts, w, forced)
    return _flach(G, list(P)), float(np.abs(summen - 1.0).max())


def schaetzer_gradient(P: dict, X: np.ndarray, starts: np.ndarray, us: UStream, form: str) -> tuple[np.ndarray, float]:
    """Gradient eines Schritts (wie das Training, ohne Adam) und der Anteil der Touren mit negativem Gewicht (kürzer als der Referenzwert, werden wahrscheinlicher)."""
    H, ecache = N.encode(P, X)
    out = N.decode(P, H, starts, mode="sample", u_stream=us)
    L = N.tour_lengths(X, out["tours"])
    ref = BL.greedy_reference_lengths(P, X, starts) if form == "rollout" else None
    b = BL.reference_value(form, L, ref)
    w = BL.advantage_weights(L, b)
    Gd, dH = N.decode_backward(P, out["cache"], w)
    G = {**N.encode_backward(P, ecache, dH), **Gd}
    return _flach(G, list(P)), float(np.mean((L - b) < 0))


def schaetzer_streuung(P: dict, pool: np.ndarray, form: str, B: int, reps: int = C.E1_REPS, seed: int = 0, ziel_norm2: float | None = None, progress=None) -> dict:
    """Gesamtvarianz des Schätzers über `reps` Wiederholungen (ddof = 1), relativ zur quadrierten Norm des exakten Gradienten (`ziel_norm2`, sonst aus `exakter_gradient`)."""
    n = pool.shape[1]
    pomo = form == "pomo"
    if pomo and B % n:
        raise ValueError("Bei POMO muss B ein Vielfaches von n sein")
    if ziel_norm2 is None:
        g, _ = exakter_gradient(P, pool, alle_starts=pomo)
        ziel_norm2 = float(g @ g)
    m = B // n if pomo else B
    starts = np.tile(np.arange(n, dtype=np.int64), (m, 1)) if pomo else np.zeros((m, 1), dtype=np.int64)
    us = UStream(derive(C.ZWECK_WIEDERHOLUNG, C.FORMEN.index(form), B, seed))
    sel = UStream(derive(C.ZWECK_WIEDERHOLUNG, 100 + C.FORMEN.index(form), B, seed))
    gs, neg = [], []
    for r in range(reps):
        idx = np.minimum((sel.uniform((m,)) * pool.shape[0]).astype(np.int64), pool.shape[0] - 1)
        g, anteil = schaetzer_gradient(P, pool[idx], starts, us, form)
        gs.append(g)
        neg.append(anteil)
        if progress is not None and (r + 1) % 25 == 0:
            progress(r + 1, reps)
    G = np.array(gs)
    var = float(((G - G.mean(axis=0)) ** 2).sum() / (reps - 1))
    return {"form": form, "B": B, "rel_streuung": var / ziel_norm2, "anteil_negativ": float(np.mean(neg)), "faktor": BL.expected_gradient_scale(form, B, n),
            "reps": reps, "instanzen_je_schritt": m, "touren_je_schritt": B, "baseline_touren_je_schritt": B if form == "rollout" else 0}


def tabelle(P: dict, B: int, reps: int = C.E1_REPS, seed: int = 0, pool: np.ndarray | None = None, progress=None) -> list[dict]:
    """Eine Zeile je Form (keiner, Mittel, Rollout, POMO) für `B` Touren je Schritt."""
    pool = pool_instanzen() if pool is None else pool
    g0, _ = exakter_gradient(P, pool, alle_starts=False)
    g1, _ = exakter_gradient(P, pool, alle_starts=True)
    zeilen = []
    for form in C.FORMEN:
        z = g1 @ g1 if form == "pomo" else g0 @ g0
        zeilen.append(schaetzer_streuung(P, pool, form, B, reps, seed, float(z), progress))
        zeilen[-1]["gradient_norm"] = math.sqrt(float(z))
    return zeilen
