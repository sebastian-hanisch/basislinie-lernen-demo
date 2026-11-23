"""Orakel O3/O4: Erwartungswert des Lernschritts am echten Netz (n = 5, d = 8, Pool aus zwei Instanzen, B = 2, alle 48^2 = 2 304 Kombinationen aus Instanz und Tour aufgezählt).

Prüfling: Rückwärtspass mit den Gewichten (L - b)/B. Orakel (anderer Rechenweg): Gradient von J durch zentrale endliche Differenzen der reinen Vorwärtsrechnung über 40 zufällige Koordinaten.
Erwartung: keiner und Rollout unverzerrt, Mittel über den Batch mit dem Faktor (B-1)/B = 1/2."""

import itertools

import numpy as np

import bsl_baseline as BL
import bsl_estimator as E1
import bsl_learn as LE
import bsl_net as N


def _setup(n=5, d=8, m=2, seed=3):
    P = LE.initial_params(d, seed)
    X = np.random.default_rng(seed).random((m, n, 2))
    tours = E1.alle_touren(n, 0)
    return P, X, tours


def _J(P, X, tours):
    M, K = X.shape[0], tours.shape[0]
    forced = np.broadcast_to(tours, (M, K, X.shape[1])).copy()
    starts = np.zeros((M, K), dtype=np.int64)
    H, _ = N.encode(P, X)
    out = N.decode(P, H, starts, mode="forced", forced=forced, keep=False)
    p = np.exp(out["logp"])
    return float((p * N.tour_lengths(X, forced)).sum() / M), p


def _flach(P, G):
    return np.concatenate([G[k].ravel() for k in P])


def test_exakter_gradient_gleich_endlicher_differenz_und_wahrscheinlichkeiten_summieren_zu_eins():
    P, X, tours = _setup()
    g, abw = E1.exakter_gradient(P, X)
    assert abw < 1e-12
    keys = list(P)
    sizes = [P[k].size for k in keys]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(1)
    for idx in rng.choice(int(offs[-1]), 40, replace=False):
        k = next(i for i in range(len(keys)) if offs[i] <= idx < offs[i + 1])
        pos = np.unravel_index(int(idx - offs[k]), P[keys[k]].shape)
        h = 1e-6
        Pp = {a: v.copy() for a, v in P.items()}
        Pm = {a: v.copy() for a, v in P.items()}
        Pp[keys[k]][pos] += h
        Pm[keys[k]][pos] -= h
        fd = (_J(Pp, X, tours)[0] - _J(Pm, X, tours)[0]) / (2 * h)
        assert abs(g[idx] - fd) <= 1e-6 * max(1.0, abs(fd)), (keys[k], pos)


def test_erwartung_der_schaetzer_aus_aufzaehlung_aller_kombinationen():
    P, X, tours = _setup()
    M, K, n = X.shape[0], tours.shape[0], X.shape[1]
    G, _ = E1.exakter_gradient(P, X)
    _, p = _J(P, X, tours)                                                # (M, K)
    L = N.tour_lengths(X, np.broadcast_to(tours, (M, K, n)))
    H, _ = N.encode(P, X)
    greedy = BL.greedy_reference_lengths(P, X, np.zeros((M, 1), dtype=np.int64))[:, 0]
    # Score je (Instanz, Tour): Rückwärtspass mit Gewicht eins auf genau dieser Tour
    S = np.zeros((M, K, G.size))
    forced = np.broadcast_to(tours, (M, K, n)).copy()
    starts = np.zeros((M, K), dtype=np.int64)
    for m in range(M):
        for k in range(K):
            w = np.zeros((M, K))
            w[m, k] = 1.0
            _, Gk = N.loss_and_grad(P, X, starts, w, forced)
            S[m, k] = _flach(P, Gk)
    samples = [(m, k) for m in range(M) for k in range(K)]
    prob = np.array([p[m, k] / M for m, k in samples])
    assert abs(prob.sum() - 1) < 1e-12 and len(samples) ** 2 == 2304
    Ls = np.array([L[m, k] for m, k in samples])
    Sm = np.array([S[m, k] for m, k in samples])
    Gr = np.array([greedy[m] for m, _ in samples])
    erw = {f: np.zeros(G.size) for f in ("keiner", "mittel", "rollout")}
    for a, b in itertools.product(range(len(samples)), repeat=2):
        w = prob[a] * prob[b]
        mean = (Ls[a] + Ls[b]) / 2
        erw["keiner"] += w * (Ls[a] * Sm[a] + Ls[b] * Sm[b]) / 2
        erw["mittel"] += w * ((Ls[a] - mean) * Sm[a] + (Ls[b] - mean) * Sm[b]) / 2
        erw["rollout"] += w * ((Ls[a] - Gr[a]) * Sm[a] + (Ls[b] - Gr[b]) * Sm[b]) / 2
    nrm = np.linalg.norm(G)
    assert np.linalg.norm(erw["keiner"] - G) < 1e-9 * nrm
    assert np.linalg.norm(erw["rollout"] - G) < 1e-9 * nrm
    assert np.linalg.norm(erw["mittel"] - 0.5 * G) < 1e-9 * nrm                                       # (B-1)/B mit B = 2
    assert BL.expected_gradient_scale("mittel", 2) == 0.5


def test_schaetzer_gradient_ist_der_rueckwaertspass_des_trainings_ohne_adam():
    """`schaetzer_gradient` und `reinforce_step` sehen dasselbe Signal: bei Lernrate 0 bleiben die Gewichte, die Gradientennorm stimmt überein."""
    import bsl_train as T
    from bsl_rngvec import UStream
    P, X, _ = _setup(n=5, d=8, m=4)
    starts = np.zeros((4, 1), dtype=np.int64)
    g, anteil = E1.schaetzer_gradient(P, X, starts, UStream(7), "mittel")
    opt = T.Adam({k: v for k, v in P.items()}, lr=0.0, clip=None)
    r = T.reinforce_step({k: v.copy() for k, v in P.items()}, opt, X, starts, UStream(7), BL.make_baseline_fn("mittel"))
    assert abs(np.linalg.norm(g) - r["grad_norm"]) < 1e-9 * max(1.0, r["grad_norm"]) and 0.0 <= anteil <= 1.0
