"""Welcher Referenzwert lässt ein Netz Touren lernen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein von Hand in numpy gebautes Zeigernetz (kopiert aus zeigernetz-demo) lernt mit REINFORCE Rundtouren; verglichen werden vier Referenzwerte (kein, Mittel über den Batch, Greedy-Rollout, POMO-Mittel):
ein Lernschritt exakt an vier Orten, Streuung des Schätzers am Netz, Live-Training, mehrere Zufalls-Seeds, Lernrate, Kosten und Streuungsquellen. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""
# ruff: noqa: E402  (die BLAS-Umgebungsvariablen müssen vor dem ersten Import von numpy stehen)

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # vor dem ersten Import von numpy: ein BLAS-Thread (mehr Threads kamen bei den kleinen Matrizen kaum voran)

import math
import time

import numpy as np
import streamlit as st

import bsl_constants as C
import bsl_estimator as E1
import bsl_evaluation as V
import bsl_exact as EX
import bsl_frozen as FZ
import bsl_instances as I
import bsl_layout as L
import bsl_learn as LE
import bsl_net as N
import bsl_posten as PO
import bsl_presets as PR
import bsl_pretrained as PT
import bsl_ranktest as RT
import bsl_tabular as TB
import bsl_texts as TX
import bsl_train as T
import bsl_visualization as VIS
import modul_frisch
from bsl_rngvec import UStream
from bsl_scenario import derive

modul_frisch.aktualisieren(__file__)

st.set_page_config(page_title="Welcher Referenzwert lässt ein Netz Touren lernen? – Sebastian Hanisch", layout="wide")


# ---------------------------------------------------------------------------------------------------------------- Zwischenspeicher
@st.cache_data(show_spinner=False)
def _zeilen_tabular(politik, B):
    return TB.schritt_zeilen(TB.theta_von(politik), B)


@st.cache_data(show_spinner=False)
def _bruch_tabular(B):
    out = {}
    for form in TB.FORMEN_EXAKT:
        r = TB.enumerate_step_exact(B, form)
        out[form] = {"streuung": str(r["gesamtstreuung"]), "erwartung": [str(x) for x in r["erwartung"]]}
    return out


@st.cache_data(show_spinner=False)
def _e1_netz(stand):
    if stand == "vorgerechnet":
        return PT.lade("lang", 10, "mittel", 0)
    return N.init_params(C.D_STANDARD, UStream(derive(C.ZWECK_E1_NETZ, 0)), layers=1, attn=True, ffn=True)


@st.cache_data(show_spinner=False)
def _e1_tabelle(B, stand, seed):
    P = _e1_netz(stand)
    return E1.tabelle(P, B, reps=C.E1_REPS, seed=seed)


@st.cache_data(show_spinner=False)
def _exakte_tour(n, idx):
    """Optimale Tour (Held-Karp, ab Ort 0) der Testinstanz idx."""
    x = FZ.testsatz(n)[idx]
    return EX.held_karp_tour(I.distance_matrix_np(x))


# ---------------------------------------------------------------------------------------------------------------- Kopf, Einleitung, Erklärung
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title(TX.TITEL)
st.markdown(TX.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/referenzwert-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Das Netz würfelt eine Tour und erfährt danach nur ihre Länge. Ob das viel oder wenig war, zeigt erst ein Vergleich: mit dem Durchschnitt der Touren desselben Schritts, mit der Tour, die eine ältere Fassung des Netzes ohne Würfeln fährt, oder mit dem Durchschnitt aller Touren derselben Orte von jedem Startort aus. Dieser Vergleichswert heißt Referenzwert (englisch Baseline).

Ohne ihn bekommt jede Tour nur ihre Länge als Rückmeldung, und das Lernen schwankt stark von Schritt zu Schritt. Der Referenzwert ändert im Durchschnitt nicht, in welche Richtung das Netz verschoben wird (beim Mittel und bei POMO macht er den Schritt nur kürzer), er verkleinert das Schwanken, und er kostet je nach Bauart unterschiedlich viel Rechenarbeit. Auch ohne Referenzwert lernt das Netz, in dieser Messung nur schlechter und unsicherer.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/referenzwert-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert ein Referenzwert", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

with st.expander("Wie hier gezählt wird (K, O und Training)"):
    st.markdown(TX.ZAEHLREGEL)

with st.expander("Benannte Verfahren kurz erklärt"):
    st.markdown(TX.ERKLAERTABELLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider, einfach kopieren, um ein Szenario zu teilen.")

PR.load_permalink_settings()
PR.init_session_state_defaults()

# ---------------------------------------------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    PR.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Szenario")
    n = st.select_slider("Orte n", options=PR.options("n_select"), key="n_select",
                         help="Größe von Training, Test und Hauptreihe: 10 oder 20 Orte. Der exakte Schritt (vier Orte) und Experiment 1 (sechs Orte) haben eigene Größen.")
    seed = st.number_input("Zufalls-Seed (Gewichte und Daten)", *PR.bounds("seed_input"), key="seed_input", step=1,
                           help="0 bis 9. Gewichte: Zufalls-Seed 1000 + s, Daten: 2000 + s. Der Zufalls-Seed der Gewichte bestimmt die Anfangsgewichte, der der Daten die Trainingsinstanzen und Ziehungen; alle Formen starten mit denselben Gewichten.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=PR.randomize_seed)
    formen = st.multiselect("Referenzwert-Formen", options=PR.options("formen_multi"), format_func=C.FORM_ANZEIGE.get, key="formen_multi",
                            help="Welche Formen das Live-Training (Experiment 2) und die Karten vergleicht. Die eingefrorenen Messreihen zeigen immer alle vier.")

n, seed = int(n), int(seed)
formen = [f for f in C.FORMEN if f in formen]

# ---------------------------------------------------------------------------------------------------------------- Primäransicht: ein Schritt exakt
st.markdown(f"## 🧭 {TX.KERNFRAGEN[0]}")
st.markdown("**Ein Trainingsschritt exakt an vier Orten.** Rechteck 3 × 4, Start in Ort 0; die sechs möglichen Touren haben die Längen 14, 16, 18, 16, 18, 14, das Optimum ist 14. "
            "Die Mini-Politik bewertet jeden Ort mit einer Zahl θ und wählt unter den noch offenen Orten per Softmax (ein Zeigernetz ohne Kodierer). Sie kann die beste Tour nicht darstellen, es geht nur um den Schätzer. "
            "Alle Zahlen entstehen durch Aufzählen aller möglichen Ziehungen, nicht durch Würfeln. Regler dieses Abschnitts: Touren je Schritt B und Politik.")
c1, c2 = st.columns(2)
with c1:
    B = st.select_slider("Touren je Schritt B", options=PR.options("tour_b_select"), key="tour_b_select",
                         help="Wie viele Touren ein Schritt zieht (mit Zurücklegen). Aufgezählt werden alle 6^B Kombinationen.")
with c2:
    politik = st.selectbox("Politik", options=PR.options("politik_select"), format_func=C.POLITIK_ANZEIGE.get, key="politik_select",
                           help="Feste Bewertungen θ der vier Orte: alle gleich, kurze Touren bevorzugt oder lange Touren bevorzugt.")
B = int(B)
theta = TB.theta_von(politik)
zeilen = _zeilen_tabular(politik, B)
p6 = TB.wahrscheinlichkeiten(theta, 0)
st.markdown("| Tour (ab Ort 0) | Länge | Wahrscheinlichkeit |\n|---|---|---|\n" + "\n".join(
    f"| {''.join(str(c) for c in t)} | {ll} | {V.de(100 * q, 1)} % |" for t, ll, q in zip(TB.touren_ab(0), TB.laengen_ab(0), p6)))
st.plotly_chart(VIS.gewichte_balken(theta), width="stretch", key="tab_gewichte")
st.caption("Gewichte L − b der sechs Touren bei dem Referenzwert, der im Mittel herauskommt (keiner: b = 0; Mittel: b = Erwartungswert der Länge; Rollout: b = Länge der Greedy-Tour der Politik). "
           "Positive Gewichte machen eine Tour unwahrscheinlicher, negative wahrscheinlicher. Beim Rollout ist kein Gewicht negativ, wenn die Greedy-Tour die kürzeste ist.")
tabelle = ["| Form | Erwartung des Schritts (Anteil am wahren Gradienten) | Gesamtstreuung | auf gleiche Erwartung hochgerechnet |", "|---|---|---|---|"]
for r in zeilen:
    name = C.FORM_KURZ[r["form"]] + (f" (N = {r['N']} Startorte)" if r["form"] == "pomo" else "")
    anteil = V.de(r["faktor"], 3) if not math.isnan(r["faktor"]) else f"nicht messbar (wahrer Gradient 0; Formel {V.de(r['faktor_formel'], 3)})"
    tabelle.append(f"| {name} | {anteil} | {V.de(r['gesamtstreuung'], 3)} | {V.de(TB.hochgerechnete_streuung(r), 3)} |")
st.markdown("\n".join(tabelle))
art, text = V.verdict_tabular(zeilen, B, politik)
getattr(st, art)(text)
if politik == "gleich":
    br = _bruch_tabular(B)
    st.caption("Bei „gleich“ sind die Gesamtstreuungen exakte Brüche: keiner " + br["keiner"]["streuung"] + ", Mittel " + br["mittel"]["streuung"] + ", Rollout " + br["rollout"]["streuung"]
               + ", Leave-one-out " + br["loo"]["streuung"] + ". Die Zeile POMO nimmt vier Startorte mit je einer Tour, ihr Ziel ist der Gradient des Mittels über alle Startorte.")
st.caption(f"Leave-one-out (b für jede Tour das Mittel der anderen) steht nur hier als Zusatzzeile; der Schätzer mit dem Mittel über den Batch ist (B − 1)/B = {V.de((B - 1) / B, 3)} mal der Schätzer mit Leave-one-out (für jede Auswahl von Touren aufgezählt; der Referenzwert selbst ist kein Vielfaches).")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E1: Streuung am Netz
st.subheader(f"🔬 {TX.KERNFRAGEN[0]} Am Zeigernetz")
st.markdown(f"Dasselbe am echten Netz: {C.E1_N} Orte, ein Pool aus {C.E1_POOL} festen Instanzen, der exakte Gradient durch Aufzählung aller 120 Touren je Instanz (bei POMO aller 720 Touren), "
            f"{C.E1_REPS} Wiederholungen je Form. Regler dieses Abschnitts: Touren je Schritt B und Stand des Netzes.")
c1, c2 = st.columns(2)
with c1:
    e1_b = st.select_slider("Touren je Schritt B (Vielfache von 6)", options=PR.options("e1_b_select"), key="e1_b_select",
                            help="Bei POMO sind es B / 6 Instanzen mit je sechs Startorten.")
with c2:
    e1_stand = st.selectbox("Stand des Netzes", options=PR.options("e1_stand_select"), key="e1_stand_select",
                            help="Ungelernt: zufällige Gewichte. Vorgerechnet: das nach 4 000 Schritten mit dem Mittel bei n = 10 trainierte Netz (Zufalls-Seed 0), hier auf n = 6 eingesetzt.")
e1_b = int(e1_b)
if e1_stand == "vorgerechnet" and PT.lade("lang", 10, "mittel", 0) is None:
    st.info("Das vorgerechnete Netz fehlt (data/bsl_pretrained.npz).")
else:
    with st.spinner(f"Experiment 1: {C.E1_REPS} Wiederholungen × 4 Formen × {e1_b} Touren = {V.zahl(C.E1_REPS * 4 * e1_b)} gezogene Touren"):
        e1 = _e1_tabelle(e1_b, e1_stand, 0)
    st.markdown("| Form | relative Streuung (Gesamtvarianz geteilt durch das Quadrat der Norm des exakten Gradienten) | Anteil der Touren mit negativem Gewicht | Erwartungsfaktor (aus der Formel) | Instanzen im Kodierer je Schritt | gezogene Touren | Baseline-Touren |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(f"| {C.FORM_KURZ[r['form']]} | {V.de(r['rel_streuung'], 2)} | {V.de(100 * r['anteil_negativ'], 0)} % | {V.de(r['faktor'], 3)} | {r['instanzen_je_schritt']} | {r['touren_je_schritt']} | {r['baseline_touren_je_schritt']} |" for r in e1))
    st.plotly_chart(VIS.estimator_balken(e1), width="stretch", key="e1_balken")
    art, text = V.verdict_estimator(e1, C.E1_REPS, e1_b, e1_stand)
    getattr(st, art)(text)
    st.caption("Der Erwartungsfaktor kommt aus der Formel und nicht aus den Wiederholungen: mit 200 Wiederholungen ist ein Faktor 0,92 bei dieser Streuung nicht von 1 zu trennen. "
               "Beim Rollout ist das Referenznetz hier das aktuelle Netz (frisch ersetzt). Die Rangfolge der drei Referenzwerte untereinander gilt nur für diesen Pool und diesen Netzzustand.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E2: Live-Training
st.subheader(f"🔬 {TX.KERNFRAGEN[1]}")
st.markdown("Vier Netze mit denselben Anfangsgewichten lernen mit den vier Referenzwerten, live. Regler dieses Abschnitts: Schritte, Lernrate, vorgerechneter Lauf und die Instanz der Karte; n, Zufalls-Seed und Formen stehen in der Seitenleiste.")
c1, c2, c3 = st.columns(3)
with c3:
    vorgerechnet = st.toggle("Vorgerechneten Lauf zeigen statt live rechnen", key="live_vorgerechnet_toggle",
                             help="Zeigt den eingefrorenen kurzen Lauf mit den Voreinstellungen (n = 10: 300 Schritte, n = 20: 250 Schritte, Lernrate 0,001) für den gewählten Zufalls-Seed.")
with c1:
    schritte = st.select_slider("Schritte", options=PR.options("live_schritte_select"), key="live_schritte_select", disabled=vorgerechnet,
                                help="Trainingsschritte zu je 120 Touren (bei n = 10 Standard 300, bei n = 20 Standard 250); Zeitgrenze 60 Sekunden für alle Formen zusammen.")
with c2:
    lr = st.select_slider("Lernrate", options=PR.options("live_lr_select"), key="live_lr_select", disabled=vorgerechnet, format_func=V.lr_text,
                          help="Für alle Formen gleich; die Abstimmung je Form steht bei Experiment 4.")
schritte, lr = int(schritte), float(lr)
instanz = int(st.number_input("Instanz der Karte (Testinstanz, Zufalls-Seed 0 bis 29)", *PR.bounds("live_instanz_input"), key="live_instanz_input", step=1))
nn_t, nn_v = None, None
if FZ.lade("exact") is not None:
    nn_t, nn_v = FZ.nn_gaps(n, "test"), FZ.nn_gaps(n, "val")
live_key = (n, seed, tuple(formen), schritte, lr)
if not formen:
    st.info("Wähle in der Seitenleiste mindestens eine Referenzwert-Form.")
elif nn_t is None:
    st.info("Die exakten Referenzlängen fehlen (data/bsl_exact_ref.json).")
else:
    ergebnis = None
    if vorgerechnet:
        kz = V.kurz_tabelle(n, seed)
        if kz is None:
            st.info("Für diesen Zufalls-Seed gibt es keinen vorgerechneten Lauf (Zufalls-Seed 0 bis 9 sind eingefroren).")
        else:
            schritte_v = C.KURZ_SCHRITTE[n]
            ergebnis = {}
            for f in formen:
                if f in kz:
                    ergebnis[f] = {"hist": [tuple(p) for p in kz[f]["val_kurve"]], "gap": kz[f]["test_gap"], "counts": kz[f],
                                   "P": PT.lade("kurz", n, f, seed)}
            schritte = schritte_v
            lr = C.LERNRATE
    else:
        posten = [PO.posten_je_lauf(f, n, schritte) for f in formen]
        touren = sum(p["touren"] for p in posten)
        base = sum(p["baseline_touren"] for p in posten)
        knopf = f"🏋️ Trainieren (n = {n} · {len(formen)} Formen · {V.zahl(schritte)} Schritte · {C.BATCH} Touren = {V.zahl(touren)} gezogene Touren" + (f", dazu {V.zahl(base)} Baseline-Touren des Rollouts)" if base else ")")
        if st.button(knopf, key="live_knopf"):
            xv, ov = FZ.valsatz(n), FZ.exakte_laengen(n, "val")
            xt, ot = FZ.testsatz(n), FZ.exakte_laengen(n, "test")
            balken = st.progress(0.0, text="Anfangsgewichte")
            res, t0 = {}, time.perf_counter()
            with st.spinner(f"Training: n = {n}, {len(formen)} Formen, {V.zahl(schritte)} Schritte, {V.zahl(touren)} gezogene Touren"):
                for i, f in enumerate(formen):
                    rest = C.LIVE_ZEITGRENZE - (time.perf_counter() - t0)
                    if rest <= 0:
                        res[f] = None
                        continue

                    def _fortschritt(s, gesamt, i=i, f=f):
                        balken.progress((i + s / gesamt) / len(formen), text=f"{C.FORM_KURZ[f]}: Schritt {V.zahl(s)} von {V.zahl(gesamt)}")

                    P_f, hist, counts = LE.train_form(n, f, schritte, lr=lr, init_seed=C.GEWICHT_BASIS + seed, data_seed=C.DATEN_BASIS + seed, val=(xv, ov), eval_every=C.KURZ_ALLE,
                                                      progress=_fortschritt, max_seconds=rest)
                    res[f] = {"P": P_f, "hist": [(h["step"], h["val_gap"]) for h in hist], "gap": T.gap_greedy(P_f, xt, ot), "counts": counts}
            st.session_state["live"] = {"key": live_key, "res": res}
            st.rerun()
        gespeichert = st.session_state.get("live")
        if gespeichert and gespeichert["key"] == live_key:
            ergebnis = gespeichert["res"]
        else:
            st.caption("Noch kein Lauf für diese Einstellung. Der Knopf rechnet live; ein vorgerechneter Lauf (Schalter oben) zeigt dasselbe ohne Wartezeit.")
    if ergebnis:
        fertig = {f: r for f, r in ergebnis.items() if r is not None}
        fehlt = [C.FORM_KURZ[f] for f in formen if ergebnis.get(f) is None]
        if fehlt:
            st.warning("Nicht gerechnet (Zeitgrenze oder keine Daten): " + ", ".join(fehlt) + ".")
        abg = {f: r["counts"]["schritte"] for f, r in fertig.items() if r["counts"].get("abgebrochen")}
        if fertig:
            st.plotly_chart(VIS.lernkurven_live({f: r["hist"] for f, r in fertig.items()}, nn_v["nn"], nn_v["nn_2opt"], 3.0 * nn_v["nn"]), width="stretch", key=f"live_kurven_{n}")
            st.caption(f"Validierungslücke auf {C.VAL_N[n]} festen Instanzen, Messpunkt alle {C.KURZ_ALLE} Schritte; vor dem Training (Schritt 0) liegt sie weit über dem Bild.")
            st.markdown("| Form | Lücke des Endmodells auf dem Testsatz | Trainingsposten (Instanzen im Kodierer, gezogene Touren, Baseline-Touren) |\n|---|---|---|\n" + "\n".join(
                f"| {C.FORM_KURZ[f]} | {V.de(r['gap'])} % | {V.zahl(r['counts']['instanzen'])}, {V.zahl(r['counts']['touren'])}, {V.zahl(r['counts']['baseline_touren'])} |" for f, r in fertig.items()))
            art, text = V.verdict_live({f: r["gap"] for f, r in fertig.items()}, nn_t["nn"], nn_t["nn_2opt"], n, schritte, seed, abg or None)
            getattr(st, art)(text)
            st.caption(f"Testsatz: {C.TEST_N[n]} Instanzen mit {n} Orten (Mittel der Lücken je Instanz). Zeit ist hier keine Aussage über das Verfahren"
                       + (f"; dieser Lauf brauchte {V.de(sum(r['counts']['sekunden'] for r in fertig.values()), 0)} Sekunden." if not vorgerechnet else "."))
            if all(r.get("P") is not None for r in fertig.values()) and len(fertig) > 0:
                xk = FZ.testsatz(n)[instanz]
                opt_tour = _exakte_tour(n, instanz)[1]
                tours = {f: [int(c) for c in N.greedy_tours(r["P"], xk[None], np.zeros((1, 1), dtype=np.int64))[0, 0]] for f, r in fertig.items()}
                st.plotly_chart(VIS.karten_formen(xk, opt_tour, tours), width="stretch", key=f"live_karten_{n}_{instanz}")
                st.caption("Greedy-Tour jedes Netzes auf der gewählten Testinstanz (Farbe der Form) und die optimale Tour (blau gestrichelt).")
            elif vorgerechnet:
                st.caption("Zu diesem Zufalls-Seed liegen keine Modelle vor (Karten nur für die Zufalls-Seeds " + ", ".join(map(str, PT.seeds_kurz(n))) + "); die Kurven und Lücken stehen trotzdem.")
            ek = PO.einsatz_kosten(n, nn_t["zugauswertungen_mittel"])
            st.markdown("**Einsatzkosten einer Tour** (K und O nebeneinander; das Training steht als dritter Posten oben):\n\n| Verfahren | K (Kandidatenbewertungen) | O (grobe Operationen) |\n|---|---|---|\n"
                        f"| Netz, greedy | {V.zahl(ek['netz']['k'])} | {V.zahl(ek['netz']['o'])} |\n| Nächster Nachbar | {V.zahl(ek['nn']['k'])} | {V.zahl(ek['nn']['o'])} |\n"
                        f"| Nächster Nachbar mit 2-opt | {V.zahl(ek['nn_2opt']['k'])} (Mittel der Messung) | {V.zahl(ek['nn_2opt']['o'])} |")
            st.caption("K begünstigt das Netz (ein Zeigerwert ist teurer als ein Vergleich), O benachteiligt es; 8 Operationen je Bewertung der klassischen Verfahren sind eine Annahme (Zehnerpotenzen).")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E3: zehn Zufalls-Seeds
st.subheader(f"🔬 {TX.KERNFRAGEN[2]}")
if not V.hat_messung(n):
    st.info("Die eingefrorene Hauptreihe fehlt (data/bsl_frozen_v2.json).")
else:
    st.markdown("Eingefroren: die Hauptreihe mit allen vier Formen und mehreren Zufalls-Seeds (Gewichte 1000 + s, Daten 2000 + s), Batch 120, getestet auf "
                f"{C.TEST_N[n]} Instanzen mit exakter Wahrheit. Regler dieses Abschnitts: Metrik, Lernrate und Verläufe; n kommt aus der Seitenleiste.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metrik = st.selectbox("Metrik", options=PR.options("e3_metrik_select"), format_func=C.METRIK_ANZEIGE.get, key="e3_metrik_select")
    with c2:
        lr_modus = st.selectbox("Lernrate", options=PR.options("e3_lr_select"), format_func=C.LR_MODUS_ANZEIGE.get, key="e3_lr_select",
                                help="Gemeinsam: 0,001 für alle Formen. Abgestimmt: für Formen, bei denen Experiment 4 eine andere Lernrate fand, die Läufe mit dieser; sonst die gemeinsamen.")
    with c3:
        verlaeufe = st.toggle("Verläufe je Zufalls-Seed zeigen", key="e3_verlaeufe_toggle")
    tab = V.seed_tabelle(n, metrik, lr_modus)
    ranks = V.rang_tests(tab)
    st.plotly_chart(VIS.streifen(tab), width="stretch", key=f"e3_streifen_{n}_{metrik}_{lr_modus}")
    if verlaeufe:
        st.plotly_chart(VIS.verlaeufe_je_seed(n, {f: V.laeufe(n, f, lr_modus) for f in C.FORMEN}, FZ.nn_gaps(n, "val")["nn"]), width="stretch", key=f"e3_verlaeufe_{n}_{lr_modus}")
        st.caption("Validierungslücke je Lauf (ein Strich je Zufalls-Seed), rote Linie: Nächster Nachbar auf den Validierungsinstanzen.")
    kopf = "| Form | Lernrate | Mittel | Median | Standardabweichung | kleinster | größter | unter dem Nächsten Nachbarn | unter NN + 2-opt | Instanzen im Kodierer, gezogene Touren, Baseline-Touren je Lauf |\n|---|---|---|---|---|---|---|---|---|---|\n"
    st.markdown(kopf + "\n".join(
        f"| {C.FORM_KURZ[f]} | {V.lr_text(z['lr'])} | {V.de(z['mittel'])} % | {V.de(z['median'])} % | {V.de(z['sd'])} | {V.de(z['min'])} % | {V.de(z['max'])} % | {z['unter_nn']} von {z['s']} | {z['unter_2opt']} von {z['s']} | "
        f"{V.zahl(z['posten']['instanzen'])}, {V.zahl(z['posten']['touren'])}, {V.zahl(z['posten']['baseline_touren'])} |" for f, z in tab["formen"].items()))
    st.markdown("**Rangtests über die Zufalls-Seeds** (Friedman über alle vier Formen: χ² = " + V.de(ranks["friedman"]["chi2"], 2) + f", {ranks['friedman']['df']} Freiheitsgrade, p = " + V.de(ranks["friedman"]["p"], 4)
                + f", bestimmt vom Abstand der Form „keiner“; nur über Mittel, Rollout und POMO: χ² = " + V.de(ranks["friedman_drei"]["chi2"], 2) + f", {ranks['friedman_drei']['df']} Freiheitsgrade, p = " + V.de(ranks["friedman_drei"]["p"], 4)
                + "; paarweise exaktes Wilcoxon mit Holm über 6 Vergleiche, drei davon gegen „keiner“):")
    st.markdown("| Paar | Differenz der Mittel (erste minus zweite Form, Prozentpunkte) | Läufe, in denen die erste Form besser ist | exaktes p | Holm-p |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {C.FORM_KURZ[p['a']]} gegen {C.FORM_KURZ[p['b']]} | {V.de_vz(p['diff_mittel'])} | {p['a_besser']} von {p['n_paare']} | {V.de(p['p'], 4)} | {V.de(p['p_holm'], 4)} |" for p in ranks["paare"]))
    art, text = V.verdict_seeds(tab, ranks)
    getattr(st, art)(text)
    st.caption(f"Mittel und Median stehen beide da: ein einzelner Ausreißer bestimmt das Mittel. Der Rollout ersetzte sein Referenznetz in den Läufen {min(tab['formen']['rollout']['ersetzungen'])} bis {max(tab['formen']['rollout']['ersetzungen'])} Mal (Prüfung alle 100 Schritte). "
               "Die Paarung gilt nur für die Anfangsgewichte (die Instanzströme von POMO sind andere).")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E4: Lernrate
st.subheader(f"🔬 {TX.KERNFRAGEN[3]}")
lt = V.lr_tabelle(n)
if lt is None:
    st.info("Die eingefrorene Lernraten-Abstimmung fehlt (data/bsl_frozen_lr.json).")
else:
    st.markdown("Jede Form bekommt ihre eigene Lernrate, gewählt auf getrennten Abstimmungs-Seeds (Gewichte 9001 und 9002, Daten 9501 und 9502), nie auf denen der Hauptreihe. Eingefroren. "
                "Regler dieses Abschnitts: die angezeigten Formen; n kommt aus der Seitenleiste (bei n = 20 nur „keiner“ und Mittel).")
    gezeigt = st.multiselect("Anzeige je Form", options=PR.options("e4_formen_multi"), format_func=C.FORM_KURZ.get, key="e4_formen_multi")
    sicht = {**lt, "formen": {f: z for f, z in lt["formen"].items() if f in gezeigt}}
    if sicht["formen"]:
        st.plotly_chart(VIS.lr_heatmap(sicht), width="stretch", key=f"e4_heat_{n}")
        st.caption(f"Mittlere Validierungslücke des Endmodells in Prozent nach {V.zahl(lt['schritte'])} Schritten (Mittel über die Abstimmungs-Seeds); in Klammern die Zahl der Abstimmungs-Seeds unter dem Nächsten Nachbarn ({V.de(lt['nn'])} % auf den Validierungsinstanzen); ★ beste Lernrate je Form.")
    else:
        st.info("Wähle mindestens eine Form.")
    art, text = V.verdict_lr(lt)
    getattr(st, art)(text)
    if V.hat_messung(n):
        abw = [f for f in lt["formen"] if abs(lt["formen"][f]["beste"] - C.LERNRATE) > 1e-12]
        st.caption("Formen, bei denen die abgestimmte Lernrate von 0,001 abweicht: " + (", ".join(C.FORM_KURZ[f] for f in abw) if abw else "keine") + ". Die Abstimmung läuft kürzer als die Hauptreihe (n = 10: 2 000 gegen 4 000 Schritte, n = 20: 3 000 gegen 8 000) und auf nur zwei Zufalls-Seeds; die Hauptreihe mit der abgestimmten Lernrate steht bei Experiment 3 (Regler „Lernrate“), und ihr Ergebnis kann vom Ergebnis der Abstimmung abweichen.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E5: Kosten
st.subheader(f"🔬 {TX.KERNFRAGEN[4]}")
if not V.hat_messung(n):
    st.info("Die eingefrorene Hauptreihe fehlt (data/bsl_frozen_v2.json).")
else:
    st.markdown("Aus den Verläufen von Experiment 3 (kein eigener Lauf): der Median der Validierungslücke gegen die gewählte Zähleinheit. Regler dieses Abschnitts: die Einheit; n kommt aus der Seitenleiste.")
    einheit = st.selectbox("Einheit der x-Achse", options=PR.options("e5_einheit_select"), format_func=C.EINHEIT_ANZEIGE.get, key="e5_einheit_select")
    kv = V.kosten_kurven(n, einheit, "gemeinsam")
    st.plotly_chart(VIS.kosten_kurven(kv), width="stretch", key=f"e5_kurven_{n}_{einheit}")
    tab5 = V.seed_tabelle(n, "endgap", "gemeinsam")
    st.markdown("| Form | Instanzen im Kodierer | gezogene Touren | Baseline-Touren | grobe Operationen (Annahme) | erster Messpunkt, an dem der Median das Ziel erreicht |\n|---|---|---|---|---|---|\n" + "\n".join(
        f"| {C.FORM_KURZ[f]} | {V.zahl(z['posten']['instanzen'])} | {V.zahl(z['posten']['touren'])} | {V.zahl(z['posten']['baseline_touren'])} | {V.zahl(z['ops'])} | "
        + (f"Schritt {V.zahl(kv['formen'][f]['erreicht_schritt'])}" if kv["formen"][f]["erreicht_schritt"] is not None else "keiner") + " |" for f, z in tab5["formen"].items()))
    st.caption(f"Posten je Lauf mit {V.zahl(tab5['steps'])} Schritten bei n = {n}. Ziel = Median der Validierungslücke des Mittels am Ende ({V.de(kv['formen']['mittel']['median'][-1])} %) plus {V.de(C.ZIEL_AUFSCHLAG, 1)} Prozentpunkte Rauschgrenze. "
               "Annahmen: Rückwärtspass doppelt so teuer wie Vorwärtspass, Rollout-Baseline und Austausch-Prüfung nur vorwärts. Zeit steht nicht als Befund da.")
    art, text = V.verdict_cost(kv)
    getattr(st, art)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E6: Streuung
st.subheader(f"🔬 {TX.KERNFRAGEN[5]}")
cd = V.cross_daten(n)
if cd is None or not V.hat_messung(n):
    st.info("Die gekreuzte Messreihe fehlt (data/bsl_frozen_cross.json).")
else:
    st.markdown("Referenzwert Mittel, gekreuzt: Gewichts-Seed i mal Daten-Seed j (alle Gewichte gegen alle Daten). Das Endpunkt-Rauschen kommt aus den Verläufen von Experiment 3. "
                "Regler dieses Abschnitts: Mittel der letzten k Messpunkte; n kommt aus der Seitenleiste.")
    k = st.select_slider("Mittel der letzten k Messpunkte", options=PR.options("e6_k_select"), key="e6_k_select", help="Wirkt auf die Streuung zwischen Zufalls-Seeds und das Rauschen innerhalb eines Laufs.")
    k = int(k)
    tabc = np.array(cd["test_gap"], dtype=float)
    st.markdown("| Gewichts-Seed \\ Daten-Seed | " + " | ".join(str(d) for d in cd["daten_seeds"]) + " |\n|---|" + "---|" * len(cd["daten_seeds"]) + "\n" + "\n".join(
        f"| {w} | " + " | ".join(V.de(v) + " %" for v in row) + " |" for w, row in zip(cd["gewicht_seeds"], tabc)))
    shares = RT.quadratsummen(tabc)
    st.plotly_chart(VIS.anteile_balken(shares), width="stretch", key=f"e6_anteile_{n}")
    rs = V.rauschen(n, k)
    art, text = V.verdict_spread(shares, rs["innen_sd"], rs["zwischen_sd"], n, tabc.shape, k)
    getattr(st, art)(text)
    st.caption("Lücke des Endmodells auf dem Testsatz in Prozent. Mit dem Mittel über k Messpunkten sinkt die Streuung nur, wenn die Zahlen oben es zeigen.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Grenzen und Formeln
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/referenzwert-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Instanz $x$ mit $n$ Orten, ein Netz mit Parametern $\theta$, das eine Tour $\pi$ als Permutation baut, und deren Länge $L(\pi,x)$. Gesucht sind Parameter, bei denen die mittlere Länge klein ist. Die Gleichungen beschreiben den Lernschritt vollständig bis auf Adam, die Begrenzung der Gradientennorm und den Aufbau des Netzes; diese stehen in `zeigernetz-demo` und sind hier nur genannt.

| Symbol | Bedeutung |
|---|---|
| $x$, $n$ | Instanz (Koordinaten im Einheitsquadrat), Zahl der Orte |
| $\theta$ | Parameter des Netzes (12 480 bei $d = 32$) |
| $\pi$, $L$ | Tour, Länge der geschlossenen Tour |
| $p_\theta(\pi \mid x)$ | Wahrscheinlichkeit, mit der das Netz die Tour $\pi$ baut |
| $J$ | mittlere Tourlänge, die das Netz senken soll |
| $b$ | Referenzwert (Baseline); hängt nicht von $\pi$ ab, außer beim Mittel und bei POMO über die eigene Tour (siehe Verzerrung) |
| $B$, $N$ | Touren je Schritt (120), Startorte je Instanz bei POMO ($N = n$) |
| $\hat g$ | Schätzer des Gradienten aus den $B$ gezogenen Touren |
| $\bar L$ | Mittel der $B$ Tourlängen eines Schritts |
| $\bar\theta$ | Parameter des Referenznetzes beim Rollout |
| $u_j$ | Zeigerwert des Netzes für Ort $j$ im aktuellen Schritt (nach der Beschneidung mit $10 \tanh$) |
| $s$ | Score $\nabla_\theta \log p_\theta(\pi \mid x)$ |
| $I$, $E$, $T$ | Instanzen je Schritt, Operationen des Kodierers, des Dekodierers je Tour |
| $L^\star$ | exakte kürzeste Tour (Held-Karp) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/referenzwert-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(TX.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html)."
)

PR.sync_query_params({"n_select": n, "seed_input": seed, "formen_multi": formen, "tour_b_select": B, "politik_select": politik, "e1_b_select": e1_b, "e1_stand_select": e1_stand,
                      "live_schritte_select": schritte if not vorgerechnet else st.session_state.get("live_schritte_select", 300), "live_lr_select": st.session_state.get("live_lr_select", C.LERNRATE),
                      "live_vorgerechnet_toggle": vorgerechnet, "live_instanz_input": instanz, "e3_metrik_select": st.session_state.get("e3_metrik_select", "endgap"),
                      "e3_lr_select": st.session_state.get("e3_lr_select", "gemeinsam"), "e3_verlaeufe_toggle": st.session_state.get("e3_verlaeufe_toggle", False),
                      "e4_formen_multi": st.session_state.get("e4_formen_multi", list(C.FORMEN)), "e5_einheit_select": st.session_state.get("e5_einheit_select", C.DEFAULT_EINHEIT),
                      "e6_k_select": st.session_state.get("e6_k_select", C.DEFAULT_LETZTE_K)})
