"""Plotly-Figuren der Demo. Alle Achsen fest (fixedrange). Horizontale Marken als Formen plus Annotation mit festen Koordinaten (nicht `add_hline` mit Annotation)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bsl_constants as C
import bsl_tabular as TB


def _fest(fig, hoehe: int, x_titel: str | None = None, y_titel: str | None = None):
    """Feste Achsen, Legende über dem Bild. Der Titel der x-Achse steht als Annotation unter dem Bild: ein Achsentitel an der x-Achse erzeugte beim ersten Zeichnen im Browser Konsolenmeldungen
    („<text> attribute y: Expected length, -Infinity“, Plotly, Titel der x-Achse bei fester Höhe); die Annotation mit Papierkoordinaten tut das nicht."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, title=y_titel)
    fig.update_layout(height=hoehe, margin=dict(l=10, r=10, t=40, b=48 if x_titel else 10), legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom"), dragmode=False)
    if x_titel:
        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.0, yshift=-30, text=x_titel, showarrow=False, xanchor="center", yanchor="top", font=dict(size=12))
    return fig


def _hlinie(fig, y: float, text: str, farbe: str, links: bool = False, row=None, col=None):
    """Gestrichelte waagerechte Marke über die ganze Breite mit Beschriftung in Papierkoordinaten."""
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_shape(type="line", xref="x domain", x0=0, x1=1, yref="y", y0=y, y1=y, line=dict(color=farbe, dash="dash", width=1.5), **kw)
    if text:
        fig.add_annotation(xref="x domain", x=0.0 if links else 1.0, yref="y", y=y, text=text, showarrow=False, xanchor="left" if links else "right", yanchor="bottom",
                           font=dict(size=11, color=farbe), **kw)


def _tour_linien(x, tour, farbe, breite=3, dash=None, name=None, gruppe=True):
    xs = [x[i][0] for i in list(tour) + [tour[0]]]
    ys = [x[i][1] for i in list(tour) + [tour[0]]]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, showlegend=name is not None, line=dict(color=farbe, width=breite, dash=dash), hoverinfo="skip")


# ---------------------------------------------------------------------------------------------------------------- Primäransicht: vier Orte
def gewichte_balken(theta, B: int | None = None) -> go.Figure:
    """Die sechs Touren (Länge, Wahrscheinlichkeit in der Beschriftung) und je Referenzwert das Gewicht L - b (bei dem Referenzwert, der im Mittel herauskommt)."""
    p = TB.wahrscheinlichkeiten(theta, 0)
    L = TB.laengen_ab(0)
    touren = TB.touren_ab(0)
    namen = [f"{''.join(str(c) for c in t)}<br>L = {ll}, p = {100 * q:.1f} %" for t, ll, q in zip(touren, L, p)]
    fig = go.Figure()
    for form in ("keiner", "mittel", "rollout"):
        w = TB.gewichte_je_tour(theta, form)
        fig.add_trace(go.Bar(x=namen, y=w["gewichte"].tolist(), name=f"{C.FORM_KURZ[form]} (b = {w['b']:.2f})".replace(".", ","), marker_color=C.FORM_FARBE[form]))
    fig.update_layout(barmode="group")
    return _fest(fig, 340, "Tour (Ort 0 zuerst)", "Gewicht L − b")


def streuung_balken_tabular(zeilen: list[dict]) -> go.Figure:
    """Gesamtstreuung je Form (logarithmische Achse), auf gleiche Erwartung hochgerechnet für das Mittel und POMO."""
    namen, werte, farben = [], [], []
    for r in zeilen:
        namen.append(C.FORM_KURZ[r["form"]] + (" (hochgerechnet)" if r["form"] in ("mittel", "pomo") else ""))
        werte.append(TB.hochgerechnete_streuung(r))
        farben.append(C.FORM_FARBE[r["form"]])
    fig = go.Figure(go.Bar(x=namen, y=werte, marker_color=farben, text=[f"{v:.2f}".replace(".", ",") for v in werte], textposition="outside"))
    fig.update_yaxes(type="log")
    return _fest(fig, 300, None, "Gesamtstreuung (log)")


# ---------------------------------------------------------------------------------------------------------------- Experiment 1
def estimator_balken(zeilen: list[dict]) -> go.Figure:
    namen = [C.FORM_KURZ[r["form"]] for r in zeilen]
    werte = [r["rel_streuung"] for r in zeilen]
    fig = go.Figure(go.Bar(x=namen, y=werte, marker_color=[C.FORM_FARBE[r["form"]] for r in zeilen], text=[f"{v:.1f}".replace(".", ",") for v in werte], textposition="outside"))
    fig.update_yaxes(type="log")
    return _fest(fig, 300, None, "relative Streuung (log)")


# ---------------------------------------------------------------------------------------------------------------- Experiment 2
def lernkurven_live(verlaeufe: dict, nn: float, nn2: float, ymax: float | None = None) -> go.Figure:
    """Validierungslücke gegen Schritt je Form; Linien für den Nächsten Nachbarn und den Nächsten Nachbarn mit 2-opt (Schritt 0 liegt oben außerhalb)."""
    fig = go.Figure()
    for form, h in verlaeufe.items():
        fig.add_trace(go.Scatter(x=[s for s, _ in h], y=[g for _, g in h], mode="lines+markers", name=C.FORM_KURZ[form], line=dict(color=C.FORM_FARBE[form], width=2.5)))
    _hlinie(fig, nn, "Nächster Nachbar", C.COLOR_NN)
    _hlinie(fig, nn2, "Nächster Nachbar + 2-opt", C.COLOR_2OPT)
    fig.update_yaxes(range=[0, ymax if ymax else 3.0 * nn])
    return _fest(fig, 360, "Trainingsschritte", "Validierungslücke in %")


def karten_formen(x: np.ndarray, opt_tour, touren: dict) -> go.Figure:
    """Je Form eine Karte mit der Greedy-Tour des Netzes (Farbe der Form) und der optimalen Tour (blau gestrichelt) auf derselben Instanz."""
    formen = list(touren)
    fig = make_subplots(rows=1, cols=len(formen), subplot_titles=[C.FORM_KURZ[f] for f in formen], horizontal_spacing=0.03)
    for i, f in enumerate(formen, start=1):
        fig.add_trace(_tour_linien(x, opt_tour, C.COLOR_OPT, 2, "dash"), row=1, col=i)
        fig.add_trace(_tour_linien(x, touren[f], C.FORM_FARBE[f], 3), row=1, col=i)
        fig.add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", marker=dict(size=7, color="#ffffff", line=dict(width=1.5, color="#444444")), showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(range=[-0.05, 1.05], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, row=1, col=i)
        fig.update_yaxes(range=[-0.05, 1.05], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, row=1, col=i)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=False, dragmode=False)
    return fig


# ---------------------------------------------------------------------------------------------------------------- Experiment 3
def streifen(tab: dict, ymax: float | None = None) -> go.Figure:
    """Endwerte der Läufe je Form (ein Punkt je Zufalls-Seed) mit Mittel (Raute) und Median (Strich); Werte über `ymax` stehen als Dreieck am oberen Rand mit ihrem Wert."""
    fig = go.Figure()
    alle = [v for f in C.FORMEN for v in tab["formen"][f]["werte"]]
    obere = ymax if ymax else min(max(alle) * 1.05, 2.5 * max(tab["nn"], 1e-9))
    for i, f in enumerate(C.FORMEN):
        z = tab["formen"][f]
        w = np.array(z["werte"])
        jitter = (np.arange(len(w)) - (len(w) - 1) / 2) * 0.05
        innen = w <= obere
        fig.add_trace(go.Scatter(x=i + jitter[innen], y=w[innen], mode="markers", name=C.FORM_KURZ[f], marker=dict(size=9, color=C.FORM_FARBE[f], opacity=0.8),
                                 text=[f"Zufalls-Seed {s}" for s, k in zip(z["seeds"], innen) if k], hoverinfo="text+y"))
        if (~innen).any():
            fig.add_trace(go.Scatter(x=i + jitter[~innen], y=[obere] * int((~innen).sum()), mode="markers+text", marker=dict(size=10, symbol="triangle-up", color=C.FORM_FARBE[f]),
                                     text=[f"{v:.0f} %".replace(".", ",") for v in w[~innen]], textposition="bottom center", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[i], y=[z["mittel"] if z["mittel"] <= obere else obere], mode="markers", showlegend=False,
                                 marker=dict(size=14, symbol="diamond-open", color="#222222", line=dict(width=2)), hovertext=f"Mittel {z['mittel']:.2f}", hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[i], y=[z["median"]], mode="markers", showlegend=False, marker=dict(size=16, symbol="line-ew", color="#222222", line=dict(width=2, color="#222222")),
                                 hovertext=f"Median {z['median']:.2f}", hoverinfo="text"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(C.FORMEN))), ticktext=[C.FORM_KURZ[f] for f in C.FORMEN], range=[-0.6, len(C.FORMEN) - 0.4])
    fig.update_yaxes(range=[0, obere * 1.08])
    _hlinie(fig, tab["nn"], "Nächster Nachbar", C.COLOR_NN)
    _hlinie(fig, tab["nn_2opt"], "Nächster Nachbar + 2-opt", C.COLOR_2OPT)
    return _fest(fig, 400, "Raute: Mittel, Strich: Median", "Lücke zur exakten Länge in %")


def verlaeufe_je_seed(n: int, laeufe_je_form: dict, nn: float, ymax: float | None = None) -> go.Figure:
    """Validierungskurven aller Zufalls-Seeds, eine Teilfigur je Form."""
    formen = list(laeufe_je_form)
    fig = make_subplots(rows=1, cols=len(formen), subplot_titles=[C.FORM_KURZ[f] for f in formen], shared_yaxes=True, horizontal_spacing=0.02)
    for i, f in enumerate(formen, start=1):
        for r in laeufe_je_form[f]:
            fig.add_trace(go.Scatter(x=[s for s, _ in r["val_kurve"]], y=[g for _, g in r["val_kurve"]], mode="lines", line=dict(color=C.FORM_FARBE[f], width=1.2), opacity=0.6,
                                     showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(fixedrange=True, row=1, col=i)
        fig.update_yaxes(fixedrange=True, range=[0, ymax if ymax else 3 * nn], row=1, col=i)
        _hlinie(fig, nn, "", C.COLOR_NN, row=1, col=i)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), dragmode=False, showlegend=False)
    return fig


# ---------------------------------------------------------------------------------------------------------------- Experiment 4
def lr_heatmap(tab: dict) -> go.Figure:
    """Mittlere Validierungslücke des Endmodells je Form (Zeile) und Lernrate (Spalte); die beste Lernrate je Form ist mit einem Stern markiert."""
    formen = list(tab["formen"])
    z, text = [], []
    for f in formen:
        je = tab["formen"][f]["lr"]
        z.append([je[lr]["mittel"] if lr in je else None for lr in C.LERNRATEN])
        text.append([(f"{je[lr]['mittel']:.1f}".replace(".", ",") + (" ★" if lr == tab["formen"][f]["beste"] else "") + f" ({je[lr]['unter']}/{je[lr]['von']})") if lr in je else "" for lr in C.LERNRATEN])
    zmax = min(60.0, max(v for r in z for v in r if v is not None))
    fig = go.Figure(go.Heatmap(z=z, x=[str(lr).replace(".", ",") for lr in C.LERNRATEN], y=[C.FORM_KURZ[f] for f in formen], text=text, texttemplate="%{text}", colorscale="RdYlGn_r",
                               zmin=0, zmax=zmax, showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return _fest(fig, 80 + 55 * len(formen), "Lernrate", None)


# ---------------------------------------------------------------------------------------------------------------- Experiment 5
def kosten_kurven(kv: dict) -> go.Figure:
    """Median der Validierungslücke gegen die gewählte Einheit (logarithmische x-Achse) mit Band min bis max; waagerechte Marke beim Ziel."""
    fig = go.Figure()
    for form in C.FORMEN:
        k = kv["formen"][form]
        x = [max(v, 1.0) for v in k["x"]]
        fig.add_trace(go.Scatter(x=x + x[::-1], y=k["max"] + k["min"][::-1], fill="toself", line=dict(width=0), fillcolor=C.FORM_FARBE[form], opacity=0.12, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=k["median"], mode="lines+markers", name=C.FORM_KURZ[form], line=dict(color=C.FORM_FARBE[form], width=2.5)))
    _hlinie(fig, kv["ziel"], "Ziel", "#444444", links=True)
    ymax = 2.5 * kv["ziel"]
    fig.update_yaxes(range=[0, max(ymax, 12)])
    fig.update_xaxes(type="log")
    return _fest(fig, 380, C.EINHEIT_ANZEIGE[kv["einheit"]] + " (logarithmisch)", "Median der Validierungslücke in %")


# ---------------------------------------------------------------------------------------------------------------- Experiment 6
def anteile_balken(shares: dict) -> go.Figure:
    fig = go.Figure(go.Bar(x=["Gewichts-Seed", "Daten-Seed", "Rest"], y=[shares["gewichte"], shares["daten"], shares["rest"]],
                           marker_color=["#4c78a8", "#f58518", "#999999"], text=[f"{v:.1f} %".replace(".", ",") for v in (shares["gewichte"], shares["daten"], shares["rest"])], textposition="outside"))
    fig.update_yaxes(range=[0, 110])
    return _fest(fig, 280, None, "Anteil der Quadratsumme in %")
