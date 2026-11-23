"""Ein Lauf der Hauptreihe wird neu gerechnet und als Band mit der eingefrorenen Messung verglichen (nie bytegleich: Plattformen, Linux rechnet im letzten Bit anders).

n = 10, Form Mittel, Zufalls-Seed 0, die ersten 250 Schritte (ein Messpunkt): erlaubt ist ein Abstand von höchstens dem Dreifachen der Standardabweichung zwischen den Zufalls-Seeds
der Hauptreihe an diesem Messpunkt, mindestens 0,5 Prozentpunkte."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_frozen as FZ
import bsl_learn as LE
import bsl_train as T

V2 = FZ.lade("v2")
pytestmark = pytest.mark.skipif(V2 is None or "10" not in V2, reason="eingefrorene Hauptreihe fehlt")


def test_lauf_n10_mittel_250_schritte_im_band():
    runs = sorted(V2["10"]["formen"]["mittel"]["gemeinsam"], key=lambda r: r["init_seed"])
    gaps250 = [dict((s, g) for s, g in r["val_kurve"])[250] for r in runs]
    soll = gaps250[0]
    band = max(3.0 * float(np.std(gaps250, ddof=1)), 0.5)
    xv, ov = FZ.valsatz(10), FZ.exakte_laengen(10, "val")
    P, hist, c = LE.train_form(10, "mittel", 250, init_seed=C.GEWICHT_BASIS, data_seed=C.DATEN_BASIS, val=(xv, ov), eval_every=250)
    assert [h["step"] for h in hist] == [0, 250] and c["schritte"] == 250
    assert abs(hist[1]["val_gap"] - soll) <= band, (hist[1]["val_gap"], soll, band)
    assert abs(hist[0]["val_gap"] - runs[0]["val_kurve"][0][1]) <= band                                  # Anfangswerte vor dem Training (Gewichte aus der Normalverteilung)
    assert abs(T.gap_greedy(P, xv, ov) - hist[1]["val_gap"]) < 1e-9
