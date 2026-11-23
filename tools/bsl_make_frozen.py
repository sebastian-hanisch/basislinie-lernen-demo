"""Erzeugt die eingefrorenen Messdaten der Demo in Stufen (je Lauf eine Zwischendatei in tools/_zwischen/, wieder aufnehmbar).

Aufruf (Test-venv, ein Prozess, BLAS-Threads 1):
  tools/bsl_make_frozen.py --stage exact            exakte Längen, Nächster Nachbar, 2-opt je Test- und Validierungsmenge (data/bsl_exact_ref.json)
  tools/bsl_make_frozen.py --stage kurz             kurze Läufe wie das Live-Training (Experiment 2), 10 Zufalls-Seeds x 4 Formen x n in {10, 20}
  tools/bsl_make_frozen.py --stage lr [--n 10]      Abstimmung der Lernrate auf eigenen Seeds (Experiment 4)
  tools/bsl_make_frozen.py --stage main --n 10      Hauptreihe, Lernrate 0,001 (`--seeds` Zahl der Zufalls-Seeds, seed-weise über alle Formen)
  tools/bsl_make_frozen.py --stage tuned --n 10     Hauptreihe mit der je Form abgestimmten Lernrate, nur für Formen, bei denen sie von 0,001 abweicht
  tools/bsl_make_frozen.py --stage cross --n 10     gekreuzt: Gewichts-Seed mal Daten-Seed, Form Mittel (`--grid` Seitenlänge)
  tools/bsl_make_frozen.py --stage assemble         die JSON-Dateien in data/ aus den Zwischendateien
  tools/bsl_make_frozen.py --stage kette            alle Läufe der Reihe nach (Größen siehe Kommandozeile), danach assemble
Gewichts- und Daten-Seeds: Hauptreihe 1000 + s und 2000 + s, Abstimmung 9001, 9002 und 9501, 9502, gekreuzt 3000 + i und 4000 + j."""
# ruff: noqa: E402
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import bsl_constants as C
import bsl_exact as EX
import bsl_frozen as FZ
import bsl_gegner as G
import bsl_instances as I
import bsl_learn as LE
import bsl_posten as PO
import bsl_scenario as S

ZW = ROOT / "tools" / "_zwischen"
DATA = ROOT / "data"


def _log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ exakte Wahrheit
def stage_exact() -> None:
    ziel = DATA / "bsl_exact_ref.json"
    if ziel.exists() and not FZ.pruefe_exact(json.loads(ziel.read_text(encoding="utf-8"))):
        _log("exact: vorhanden")
        return
    out = {"format": C.FROZEN_FORMAT, "quelle": "Held-Karp (bsl_exact), Nächster Nachbar und 2-opt (bsl_gegner), Einheitsquadrat"}
    for n in C.N_MESS:
        e = {}
        for art, X in (("test", FZ.testsatz(n)), ("val", FZ.valsatz(n))):
            opt, nn, nn2, k2 = [], [], [], []
            t0 = time.perf_counter()
            for x in X:
                D = I.distance_matrix_np(x)
                opt.append(round(EX.held_karp(D), 9))
                Dl = I.distance_lists(x)
                tour, _ = G.nn_tour_k(Dl, 0)
                nn.append(round(S.tour_cost(Dl, tour), 9))
                _, ln, k = G.two_opt_k(Dl, tour)
                nn2.append(round(ln, 9))
                k2.append(int(k))
            e[f"{art}_opt"], e[f"{art}_nn"], e[f"{art}_nn2"] = opt, nn, nn2
            if art == "test":
                e["test_k2"] = k2
            _log(f"exact n = {n} {art}: {len(X)} Instanzen in {time.perf_counter() - t0:.0f} s")
        out[str(n)] = e
    _json(ziel, out)


# ------------------------------------------------------------------------------------------------------------------ Läufe
def lauf_id(stage, n, form, lr, w, d, steps) -> str:
    return f"{stage}_n{n}_{form}_lr{lr:g}_w{w}_d{d}_s{steps}"


def fuehre_aus(stage, n, form, lr, w, d, steps, eval_every, test=True, params=True) -> dict:
    """Ein Lauf, falls die Zwischendatei fehlt; gibt den Datensatz zurück."""
    rid = lauf_id(stage, n, form, lr, w, d, steps)
    pf = ZW / f"{rid}.json"
    if pf.exists():
        return json.loads(pf.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    rec, P = FZ.messlauf(n, form, steps, lr, w, d, eval_every=eval_every, test=test)
    rec["stage"] = stage
    _json(pf, rec)
    if params:
        np.save(ZW / f"{rid}.npy", LE.pack_params(P))
    _log(f"{rid}: val {rec['val_gap_ende']:.2f} %" + (f", test {rec['test_gap']:.2f} %" if test else "") + f", {time.perf_counter() - t0:.0f} s")
    return rec


def stage_kurz(seeds=C.SEEDS_MESS) -> None:
    for n in C.N_MESS:
        for s in range(seeds):
            for form in C.FORMEN:
                fuehre_aus("kurz", n, form, C.LERNRATE, C.GEWICHT_BASIS + s, C.DATEN_BASIS + s, C.KURZ_SCHRITTE[n], C.KURZ_ALLE)


def stage_lr(ns) -> None:
    for n in ns:
        for form in C.LR_FORMEN[n]:
            for lr in C.LERNRATEN:
                for w, d in zip(C.TUNE_GEWICHT, C.TUNE_DATEN):
                    fuehre_aus("lr", n, form, lr, w, d, C.LR_SCHRITTE[n], 500, test=False, params=False)


def beste_lernraten(n: int) -> dict:
    """Je Form die Lernrate mit der kleinsten mittleren Validierungslücke des Endmodells über die zwei Abstimmungs-Seeds (Gleichstand: die kleinere Lernrate)."""
    out = {}
    for form in C.LR_FORMEN[n]:
        mittel = {}
        for lr in C.LERNRATEN:
            werte = []
            for w, d in zip(C.TUNE_GEWICHT, C.TUNE_DATEN):
                pf = ZW / f"{lauf_id('lr', n, form, lr, w, d, C.LR_SCHRITTE[n])}.json"
                if pf.exists():
                    werte.append(json.loads(pf.read_text(encoding="utf-8"))["val_gap_ende"])
            if len(werte) == len(C.TUNE_GEWICHT):
                mittel[lr] = float(np.mean(werte))
        if mittel:
            out[form] = min(mittel, key=lambda lr: (mittel[lr], lr))
    return out


def stage_main(n: int, seeds: int) -> None:
    for s in range(seeds):
        for form in C.FORMEN:
            fuehre_aus("main", n, form, C.LERNRATE, C.GEWICHT_BASIS + s, C.DATEN_BASIS + s, C.SCHRITTE_MESS[n], C.MESS_ALLE)


def stage_tuned(n: int, seeds: int) -> None:
    beste = beste_lernraten(n)
    _log(f"tuned n = {n}: beste Lernraten {beste}")
    for form, lr in beste.items():
        if abs(lr - C.LERNRATE) < 1e-12:
            continue
        for s in range(seeds):
            fuehre_aus("main", n, form, lr, C.GEWICHT_BASIS + s, C.DATEN_BASIS + s, C.SCHRITTE_MESS[n], C.MESS_ALLE)


def stage_cross(n: int, grid: int) -> None:
    for i in range(grid):
        for j in range(grid):
            fuehre_aus("cross", n, "mittel", C.LERNRATE, C.CROSS_GEWICHT_BASIS + i, C.CROSS_DATEN_BASIS + j, C.SCHRITTE_MESS[n], C.MESS_ALLE, params=False)


# ------------------------------------------------------------------------------------------------------------------ Zusammenstellen
def _lade(stage: str, **filt) -> list[dict]:
    out = []
    for pf in sorted(ZW.glob(f"{stage}_*.json")):
        r = json.loads(pf.read_text(encoding="utf-8"))
        if all(r.get(k) == v for k, v in filt.items()):
            out.append(r)
    return out


def kompakt(r: dict) -> dict:
    keys = ("init_seed", "data_seed", "lr", "val_kurve", "val_gap_ende", "test_gap", "instanzen", "touren", "baseline_touren", "kodierer", "pruefungen", "ersetzungen", "schritte", "sekunden")
    return {k: r[k] for k in keys if k in r}


def stage_assemble() -> None:
    DATA.mkdir(exist_ok=True)
    ref = json.loads((DATA / "bsl_exact_ref.json").read_text(encoding="utf-8"))
    # Hauptreihe
    v2 = {"format": C.FROZEN_FORMAT, "lernrate_gemeinsam": C.LERNRATE, "batch": C.BATCH, "messpunkt_alle": C.MESS_ALLE}
    for n in C.N_MESS:
        je_form = {}
        for form in C.FORMEN:
            lrs = sorted({r["lr"] for r in _lade("main", n=n, form=form)})
            gem = sorted((r for r in _lade("main", n=n, form=form) if abs(r["lr"] - C.LERNRATE) < 1e-12), key=lambda r: r["init_seed"])
            eintrag = {"gemeinsam": [kompakt(r) for r in gem]}
            for lr in lrs:
                if abs(lr - C.LERNRATE) >= 1e-12:
                    ab = sorted((r for r in _lade("main", n=n, form=form) if r["lr"] == lr), key=lambda r: r["init_seed"])
                    eintrag["abgestimmt"] = {"lr": lr, "laeufe": [kompakt(r) for r in ab]}
            je_form[form] = eintrag
        if any(je_form[f]["gemeinsam"] for f in C.FORMEN):
            v2[str(n)] = {"schritte": C.SCHRITTE_MESS[n], "formen": je_form, "test_n": C.TEST_N[n], "val_n": C.VAL_N[n]}
    _json(DATA / "bsl_frozen_v2.json", v2)
    # Lernraten
    lr_d = {"format": C.FROZEN_FORMAT, "gewicht_seeds": list(C.TUNE_GEWICHT), "daten_seeds": list(C.TUNE_DATEN), "lernraten": list(C.LERNRATEN)}
    for n in C.N_MESS:
        zellen = []
        for r in _lade("lr", n=n):
            zellen.append({"form": r["form"], "lr": r["lr"], "init_seed": r["init_seed"], "val_gap_ende": r["val_gap_ende"], "val_kurve": r["val_kurve"], "schritte": r["steps"]})
        if zellen:
            lr_d[str(n)] = {"schritte": C.LR_SCHRITTE[n], "zellen": zellen, "beste": {f: lr for f, lr in beste_lernraten(n).items()}}
    _json(DATA / "bsl_frozen_lr.json", lr_d)
    # gekreuzt
    cr = {"format": C.FROZEN_FORMAT, "form": "mittel"}
    for n in C.N_MESS:
        rs = _lade("cross", n=n)
        if rs:
            gw = sorted({r["init_seed"] for r in rs})
            gd = sorted({r["data_seed"] for r in rs})
            tab = [[next((r["test_gap"] for r in rs if r["init_seed"] == w and r["data_seed"] == d), None) for d in gd] for w in gw]
            kurven = [{"init_seed": r["init_seed"], "data_seed": r["data_seed"], "val_kurve": r["val_kurve"]} for r in rs]
            cr[str(n)] = {"schritte": C.SCHRITTE_MESS[n], "gewicht_seeds": gw, "daten_seeds": gd, "test_gap": tab, "kurven": kurven}
    _json(DATA / "bsl_frozen_cross.json", cr)
    # kurze Läufe
    ku = {"format": C.FROZEN_FORMAT, "schritte": {str(n): C.KURZ_SCHRITTE[n] for n in C.N_MESS}, "messpunkt_alle": C.KURZ_ALLE}
    for n in C.N_MESS:
        rs = _lade("kurz", n=n)
        if rs:
            ku[str(n)] = [{"form": r["form"], **kompakt(r)} for r in sorted(rs, key=lambda r: (r["init_seed"], C.FORMEN.index(r["form"])))]
    _json(DATA / "bsl_frozen_short.json", ku)
    # Posten und Annahmen (Übergabe an Stück 6)
    cost = {"format": C.FROZEN_FORMAT, "annahmen": {"ops_je_bewertung": C.OPS_JE_KANDIDAT, "rueckwaerts_faktor": C.RUECKWAERTS_FAKTOR, "batch": C.BATCH, "d": C.D_STANDARD}}
    for n in C.N_MESS:
        if str(n) not in v2:
            continue
        steps = C.SCHRITTE_MESS[n]
        formen = {}
        for form in C.FORMEN:
            gem = v2[str(n)]["formen"][form]["gemeinsam"]
            if not gem:
                continue
            gaps = [r["test_gap"] for r in gem]
            formen[form] = {**PO.posten_je_lauf(form, n, steps), "ops": PO.ops_je_lauf(form, n, steps), "endluecke_mittel_prozent": round(float(np.mean(gaps)), 4),
                            "endluecke_median_prozent": round(float(np.median(gaps)), 4), "seeds": len(gaps)}
        cost[str(n)] = {"steps": steps, "batch": C.BATCH, "formen": formen}
    _json(DATA / "bsl_frozen_cost.json", cost)
    _log(f"assemble: fertig (exakte Referenz n = 10: {len(ref['10']['test_opt'])} Testinstanzen)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["exact", "kurz", "lr", "main", "tuned", "cross", "assemble", "kette"])
    ap.add_argument("--n", type=int, choices=[10, 20])
    ap.add_argument("--seeds", type=int, default=C.SEEDS_MESS)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--seeds20", type=int, default=8)
    ap.add_argument("--grid20", type=int, default=3)
    a = ap.parse_args()
    ns = [a.n] if a.n else list(C.N_MESS)
    if a.stage == "exact":
        stage_exact()
    elif a.stage == "kurz":
        stage_kurz()
    elif a.stage == "lr":
        stage_lr(ns)
    elif a.stage == "main":
        stage_main(a.n, a.seeds)
    elif a.stage == "tuned":
        stage_tuned(a.n, a.seeds)
    elif a.stage == "cross":
        stage_cross(a.n, a.grid or C.CROSS_GRID[a.n])
    elif a.stage == "assemble":
        stage_assemble()
    else:
        stage_exact()
        stage_kurz()
        stage_lr([10])
        stage_main(10, C.SEEDS_MESS)
        stage_tuned(10, C.SEEDS_MESS)
        stage_cross(10, C.CROSS_GRID[10])
        stage_lr([20])
        stage_main(20, a.seeds20)
        stage_tuned(20, a.seeds20)
        stage_cross(20, a.grid20)
        stage_assemble()


if __name__ == "__main__":
    main()
