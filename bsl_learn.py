"""Trainingsschleife mit wählbarem Referenzwert (keiner, Mittel, Rollout, POMO), Zähler, Zeitwächter und Validierungskurve.

Baut auf der Kopie `bsl_train` auf (Adam, `reinforce_step` mit `baseline_fn`, `gap_greedy`). Je Schritt `batch` Touren:
keiner, Mittel, Rollout: `batch` frische Instanzen mit je einer gezogenen Tour ab Ort 0; POMO: `batch / n` Instanzen mit je n Touren (ein Startort je Ort).
Zufall wie in der Quelle: Gewichte `UStream(derive(ZWECK_INIT, d, kodierercode, init_seed))` (bei gleichem Zufalls-Seed starten alle Formen mit denselben Gewichten),
Instanzen `UStream(derive(ZWECK_TRAIN_X, n, data_seed))`, Ziehungen `UStream(derive(ZWECK_TRAIN_S, n, data_seed))`, Instanzen der Austausch-Prüfung des Rollouts
`UStream(derive(ZWECK_AUSTAUSCH, n, data_seed))`. Bei POMO sind der Instanz- und der Ziehungsstrom andere (andere Zahl je Schritt); die Paarung gilt nur für die Anfangsgewichte."""

import time

import numpy as np

import bsl_baseline as BL
import bsl_constants as C
import bsl_net as N
import bsl_posten as PO
import bsl_train as T
from bsl_instances import train_batch
from bsl_rngvec import UStream
from bsl_scenario import derive


def initial_params(d: int, init_seed: int) -> dict:
    """Anfangsgewichte (eine Schicht mit Attention und Vorwärtsnetz), gleich für jede Form bei gleichem Zufalls-Seed."""
    return N.init_params(d, UStream(derive(C.ZWECK_INIT, d, N.variant_code(1, True, True), init_seed)), layers=1, attn=True, ffn=True)


def starts_fuer(form: str, n: int, batch: int) -> tuple[int, np.ndarray]:
    """(Zahl der Instanzen je Schritt, Startorte (M, K)): POMO startet in jedem Ort."""
    if form == "pomo":
        if batch % n:
            raise ValueError("Bei POMO muss der Batch ein Vielfaches von n sein")
        m = batch // n
        return m, np.tile(np.arange(n, dtype=np.int64), (m, 1))
    return batch, np.zeros((batch, 1), dtype=np.int64)


def train_form(n: int, form: str, steps: int, batch: int = C.BATCH, d: int = C.D_STANDARD, lr: float = C.LERNRATE, init_seed: int = 0, data_seed: int = 0,
               val=None, eval_every: int = C.MESS_ALLE, progress=None, max_seconds: float | None = None, progress_every: int = 25, P0: dict | None = None):
    """Trainiert ein Zeigernetz mit REINFORCE und der Referenzwert-Form `form`. Rückgabe (P, history, counts).

    val: None oder (X_val, opt_val); history = [{"step", "val_gap", "touren"}] bei Schritt 0, alle `eval_every` Schritte und am Ende.
    progress(step, steps) wird alle `progress_every` Schritte und am Ende aufgerufen; `max_seconds`: Zeitwächter, danach endet die Schleife (counts["abgebrochen"]).
    counts: instanzen, touren (gezogen), baseline_touren, kodierer, pruefungen, ersetzungen (Austausche des Referenznetzes), schritte, abgebrochen, sekunden."""
    if form not in C.FORMEN:
        raise ValueError(f"Unbekannte Form {form!r}")
    P = N.copy_params(P0) if P0 is not None else initial_params(d, init_seed)
    opt = T.Adam(P, lr)
    xs = UStream(derive(C.ZWECK_TRAIN_X, n, data_seed))
    ss = UStream(derive(C.ZWECK_TRAIN_S, n, data_seed))
    m, starts = starts_fuer(form, n, batch)
    state = {"ref": N.copy_params(P) if form == "rollout" else None}
    fn = BL.make_baseline_fn(form, state)
    if form == "rollout":
        ex = train_batch(UStream(derive(C.ZWECK_AUSTAUSCH, n, data_seed)), C.ROLLOUT_INSTANZEN, n)
        ex_starts = np.zeros((C.ROLLOUT_INSTANZEN, 1), dtype=np.int64)
    ersetzungen = 0
    history = []
    t0 = time.perf_counter()

    def messen(step):
        if val is not None and len(val[0]) > 0:
            history.append({"step": step, "val_gap": T.gap_greedy(P, val[0], val[1]), "touren": step * batch})

    messen(0)
    erledigt = 0
    abgebrochen = False
    for step in range(1, steps + 1):
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            abgebrochen = True
            break
        T.reinforce_step(P, opt, train_batch(xs, m, n), starts, ss, fn)
        erledigt = step
        if form == "rollout" and step % C.ROLLOUT_ALLE == 0:
            lc = N.tour_lengths(ex, N.greedy_tours(P, ex, ex_starts))
            lr_ = N.tour_lengths(ex, N.greedy_tours(state["ref"], ex, ex_starts))
            if BL.rollout_should_replace(lc, lr_):
                state["ref"] = N.copy_params(P)
                ersetzungen += 1
        if eval_every and step % eval_every == 0:
            messen(step)
        if progress is not None and (step % progress_every == 0 or step == steps):
            progress(step, steps)
    if eval_every and erledigt % eval_every != 0:
        messen(erledigt)
    posten = PO.posten_je_lauf(form, n, erledigt, batch)
    counts = {**posten, "ersetzungen": ersetzungen, "schritte": erledigt, "abgebrochen": abgebrochen, "sekunden": time.perf_counter() - t0}
    return P, history, counts


# ------------------------------------------------------------------------------------------------------------------ Parametervektor (Rückfallmodelle)
def param_keys(d: int = C.D_STANDARD) -> list[str]:
    return list(N.expected_shapes(d, 1, True, True))


def pack_params(P: dict, d: int = C.D_STANDARD) -> np.ndarray:
    """Alle Parameter als ein Vektor float32 in der festen Reihenfolge von `bsl_net.expected_shapes` (12 480 Werte bei d = 32)."""
    return np.concatenate([np.asarray(P[k], dtype=np.float64).ravel() for k in param_keys(d)]).astype(np.float32)


def unpack_params(vec: np.ndarray, d: int = C.D_STANDARD) -> dict:
    """Umkehrung von `pack_params` (float64)."""
    shapes = N.expected_shapes(d, 1, True, True)
    if vec.size != sum(int(np.prod(s)) for s in shapes.values()):
        raise ValueError("Der Parametervektor passt nicht zu d")
    out, pos = {}, 0
    for k, s in shapes.items():
        cnt = int(np.prod(s))
        out[k] = np.asarray(vec[pos:pos + cnt], dtype=np.float64).reshape(s)
        pos += cnt
    return out
