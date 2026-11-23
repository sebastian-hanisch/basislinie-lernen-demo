"""Plotly-Figuren: alle Achsen fest (fixedrange), kein Achsentitel an der x-Achse (er löste im Browser beim ersten Zeichnen Konsolenmeldungen „<text> attribute y: Expected length,
-Infinity“ aus, im Browser-Pane an `streifen`, `lr_heatmap` und `kosten_kurven` gefunden und behoben; der Titel steht als Annotation), Figuren ohne Ausnahme für die Randfälle."""

import numpy as np
import pytest

import bsl_constants as C
import bsl_evaluation as V
import bsl_frozen as FZ
import bsl_tabular as TB
import bsl_visualization as VIS

MESSUNG = FZ.lade("v2") is not None and "10" in FZ.lade("v2")


def _achsen(fig):
    return [fig.layout[k] for k in fig.layout if k.startswith(("xaxis", "yaxis"))]


def _alle_figuren():
    figs = {"gewichte": VIS.gewichte_balken(C.POLITIK_THETA["lang"]), "tabular": VIS.streuung_balken_tabular(TB.schritt_zeilen(C.POLITIK_THETA["gleich"], 2)),
            "estimator": VIS.estimator_balken([{"form": f, "rel_streuung": v} for f, v in zip(C.FORMEN, (30.0, 3.0, 4.0, 2.0))]),
            "anteile": VIS.anteile_balken({"gewichte": 10.0, "daten": 20.0, "rest": 70.0}),
            "live": VIS.lernkurven_live({"mittel": [(0, 50.0), (50, 9.0)], "pomo": [(0, 50.0), (50, 8.0)]}, 10.0, 1.0),
            "karten": VIS.karten_formen(np.random.default_rng(1).random((8, 2)), list(range(8)), {"mittel": list(range(8)), "pomo": list(range(7, -1, -1))})}
    if MESSUNG:
        tab = V.seed_tabelle(10, "endgap", "gemeinsam")
        figs["streifen"] = VIS.streifen(tab)
        figs["verlaeufe"] = VIS.verlaeufe_je_seed(10, {f: V.laeufe(10, f) for f in C.FORMEN}, tab["nn"])
        figs["kosten"] = VIS.kosten_kurven(V.kosten_kurven(10, "ops"))
        lt = V.lr_tabelle(10)
        if lt:
            figs["heatmap"] = VIS.lr_heatmap(lt)
    return figs


FIGUREN = _alle_figuren()


@pytest.mark.parametrize("name", list(FIGUREN))
def test_achsen_fest_und_kein_x_achsentitel(name):
    fig = FIGUREN[name]
    for ax in _achsen(fig):
        assert ax.fixedrange is True, (name, ax)
    for k in fig.layout:
        if k.startswith("xaxis"):
            assert fig.layout[k].title.text is None, (name, k)


def test_streifen_markiert_ausreisser_am_rand_mit_wert():
    if not MESSUNG:
        pytest.skip("Messung fehlt")
    tab = V.seed_tabelle(10, "endgap", "gemeinsam")
    tab["formen"]["keiner"]["werte"][0] = 155.0
    fig = VIS.streifen(tab, ymax=20.0)
    texte = [t for tr in fig.data if tr.text is not None and tr.mode == "markers+text" for t in tr.text]
    assert "155 %" in texte


def test_horizontale_marken_sind_formen_mit_papierkoordinaten_und_ohne_leere_annotation():
    fig = VIS.lernkurven_live({"mittel": [(0, 5.0), (50, 4.0)]}, 10.0, 1.0)
    assert len(fig.layout.shapes) == 2 and all(s.xref == "x domain" for s in fig.layout.shapes)
    fig2 = VIS.verlaeufe_je_seed(10, {"mittel": [{"val_kurve": [[0, 9.0], [250, 5.0]]}]}, 10.0)
    assert len(fig2.layout.shapes) == 1 and not [a for a in fig2.layout.annotations if a.text == ""]
