"""Orakel-Tests (ORACLE-PLAYBOOK §2, Plan Z5): unabhängige Rechenwege für die Kernroutinen, die diese Demo selbst verwendet.

* O2: Held-Karp (`bsl_exact.held_karp`, kopiert) gegen Aufzählung aller Rundtouren bei n <= 8 und gegen eine eigene Teilmengen-Rekursion in reinem Python.
* O9: jede Tour des Netzes ist eine Permutation, die Länge stimmt mit der Neuberechnung mit `math.hypot` überein; Nächster Nachbar gegen Aufzählung der Wahl je Schritt.
* O3/O4 (Erwartung der Schätzer gegen endliche Differenz) stehen in tests/test_oracle_expectation.py und tests/test_tabular.py, O6/O7 in tests/test_ranktest.py (scipy),
  O8 (Zähler gegen Formeln) in tests/test_learn.py und tests/test_posten.py, O1 (Gradienten-Check gegen endliche Differenzen) in tests/test_net.py (Kopie aus der Quelle)."""

import itertools
import math

import numpy as np
import pytest

import bsl_exact as EX
import bsl_gegner as G
import bsl_instances as I
import bsl_learn as LE
import bsl_net as N


def _alle_touren(D):
    n = len(D)
    return min(D[0][p[0]] + D[p[-1]][0] + sum(D[a][b] for a, b in zip(p, p[1:])) for p in itertools.permutations(range(1, n)))


@pytest.mark.parametrize("n,seed", [(3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6)])
def test_o2_held_karp_gleich_aufzaehlung(n, seed):
    x = np.random.default_rng(seed).random((n, 2))
    D = I.distance_matrix_np(x)
    assert abs(EX.held_karp(D) - _alle_touren(D.tolist())) < 1e-12


def test_o2_held_karp_gleich_eigener_rekursion_in_reinem_python():
    x = np.random.default_rng(9).random((9, 2))
    d = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in x] for a in x]
    n = len(d)
    best = {}
    for j in range(1, n):
        best[(1 << j, j)] = d[0][j]
    for size in range(2, n):
        for sub in itertools.combinations(range(1, n), size):
            mask = sum(1 << j for j in sub)
            for j in sub:
                best[(mask, j)] = min(best[(mask ^ (1 << j), i)] + d[i][j] for i in sub if i != j)
    ziel = min(best[(sum(1 << j for j in range(1, n)), j)] + d[j][0] for j in range(1, n))
    assert abs(EX.held_karp(I.distance_matrix_np(x)) - ziel) < 1e-12


def test_o2_held_karp_tour_hat_die_optimale_laenge():
    x = np.random.default_rng(4).random((7, 2))
    D = I.distance_matrix_np(x)
    laenge, tour = EX.held_karp_tour(D)
    assert sorted(tour) == list(range(7)) and tour[0] == 0
    assert abs(sum(D[tour[i], tour[(i + 1) % 7]] for i in range(7)) - laenge) < 1e-12 and abs(laenge - EX.held_karp(D)) < 1e-12


@pytest.mark.parametrize("form", ["keiner", "mittel", "rollout", "pomo"])
def test_o9_touren_nach_kurzem_training_sind_permutationen_mit_stimmender_laenge(form):
    n = 8
    P, _, _ = LE.train_form(n, form, 8, batch=16, d=8, init_seed=3, data_seed=3, eval_every=0)
    X = np.random.default_rng(7).random((5, n, 2))
    starts = np.zeros((5, 1), dtype=np.int64)
    tours = N.greedy_tours(P, X, starts)
    assert N.is_permutation(tours)
    L = N.tour_lengths(X, tours)
    for m in range(5):
        t = tours[m, 0]
        assert abs(sum(math.hypot(*(X[m, t[i]] - X[m, t[(i + 1) % n]])) for i in range(n)) - L[m, 0]) < 1e-12


def test_o9_naechster_nachbar_gegen_aufzaehlung_der_wahl_je_schritt():
    x = np.random.default_rng(2).random((9, 2))
    D = I.distance_lists(x)
    tour, k = G.nn_tour_k(D, 0)
    besucht = [0]
    for _ in range(8):
        offen = [j for j in range(9) if j not in besucht]
        besucht.append(min(offen, key=lambda j: (D[besucht[-1]][j], j)))
    assert tour == besucht and k == 9 * 8 // 2


def test_o9_zwei_opt_endet_in_einem_optimum_und_senkt_die_laenge():
    x = np.random.default_rng(5).random((12, 2))
    D = I.distance_lists(x)
    tour, _ = G.nn_tour_k(D, 0)
    neu, laenge, _ = G.two_opt_k(D, tour)
    n = len(neu)
    assert sorted(neu) == list(range(n)) and laenge <= sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n)) + 1e-12
    for i in range(n - 1):                                                                       # kein verbessernder Zug übrig (Gegenprobe zum Abbruch)
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a, b, c, d_ = neu[i], neu[i + 1], neu[j], neu[(j + 1) % n]
            assert D[a][c] + D[b][d_] - D[a][b] - D[c][d_] >= -1e-9
