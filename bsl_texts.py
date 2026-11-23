"""Anzeigetexte der App: Titel, Einleitung, Erklärungen, Kernfragen, Grenzen, Formeln. Texte mit Zahlen stehen in `bsl_evaluation` (Verdikte) oder werden in der App aus den Daten gebildet."""

TITEL = "📏 Welcher Referenzwert lässt ein Netz Touren lernen?"

INTRO = """
Ein **Zeigernetz (Pointer Network)** lernt Rundtouren durch **REINFORCE**: es zieht Touren, die Länge ist die Strafe, und der Schritt verstärkt Touren, die kürzer sind als ein **Referenzwert (Baseline)**.
Wie dieser Referenzwert gebildet wird, entscheidet, ob das Lernen verlässlich gelingt und was es kostet: nichts, das Mittel über den Batch, die Greedy-Tour eines Referenznetzes (**Rollout**) oder das Mittel über alle Startorte
derselben Instanz (**POMO**). Unten stehen ein exakt aufgezählter Schritt, ein Live-Training mit vier Netzen, die eingefrorene Messreihe über mehrere Zufalls-Seeds je Form, danach Lernrate, Kosten und Streuung.
Wer zuerst ohne Fachbegriffe lesen will, öffnet **💬 In einfachen Worten**. Unter den Experimenten stehen **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**. Das Zeigernetz selbst baut die Demo `zeigernetz-demo` (die erste Demo dieser Reihe, noch ohne Live-Link); hier wird es kopiert und nur das Lernsignal getauscht.
"""

SO_FUNKTIONIERT = """
Ein Trainingsschritt zieht Touren, misst ihre Längen L und verschiebt die Gewichte des Netzes so, dass Touren mit **L − b** unter 0 wahrscheinlicher werden und Touren mit L − b über 0 unwahrscheinlicher. Der **Referenzwert b** ist das, was die vier Formen unterscheidet:

| Form | b (in Worten) | hängt b von der gezogenen Tour selbst ab? | Faktor auf den Erwartungswert des Schritts | Kosten je Schritt |
|---|---|---|---|---|
| Kein Referenzwert | 0 | nein | 1 | nur Ziehen und Rückwärtsrechnung |
| Mittel über den Batch (verschiedene Instanzen) | Mittel der Längen aller Touren des Schritts | ja (die eigene Tour steckt im Mittel) | (B − 1) / B | wie keiner |
| Greedy-Rollout (Referenznetz) | Länge der Greedy-Tour eines Referenznetzes auf derselben Instanz | nein | 1 | je Schritt B zusätzliche Greedy-Touren, dazu eine Prüfung alle 100 Schritte |
| POMO-Mittel (alle Startorte) | Mittel der Längen aller Touren derselben Instanz, eine Tour je Startort | ja | (N − 1) / N | nur B / n Instanzen im Kodierer statt B |

B ist die Zahl der Touren je Schritt (hier 120), N die der Startorte je Instanz (bei POMO gleich n). Das Mittel über den Batch ist in der Literatur keine benannte Variante (dort stehen ein gleitendes Mittel über die Zeit in Bellos Active Search, ein gelernter Critic, der Rollout und das Mittel über die Startorte);
mathematisch ist der Schätzer mit diesem Mittel (B − 1) / B mal der Schätzer mit dem Leave-one-out-Referenzwert (der Referenzwert selbst ist kein Vielfaches), siehe die Formeln unten.
"""

ZAEHLREGEL = """
**K** zählt Kandidatenbewertungen: eine Konstruktion kostet n(n − 1)/2 (Nächster Nachbar und Netz gleich, ein Zeigerwert ist eine Bewertung), jedes in 2-opt bewertete Kantenpaar eine weitere. **O** zählt grobe Operationen:
Multiplikations-Additions-Paare der Matrixprodukte des Netzes (a·b·c je Produkt, aus den Formen gezählt, nicht Multiplikationen und Additionen einzeln) und 8 je Kandidatenbewertung der klassischen Verfahren (Annahme, nur auf Zehnerpotenzen belastbar).
K begünstigt das Netz (ein Zeigerwert ist teurer als ein Vergleich), O benachteiligt es. Das **Training** ist ein dritter Posten, der neben K und O steht und nie in sie eingerechnet wird:
hier in Instanzen im Kodierer, gezogenen Touren, Baseline-Touren und groben Operationen (Annahme: der Rückwärtspass kostet das Doppelte des Vorwärtspasses, ein Schritt also das Dreifache).
"""

ERKLAERTABELLE = """
| Name | Was es hier ist |
|---|---|
| Zeigernetz (Pointer Network) | Netz, das je Schritt auf einen noch freien Ort zeigt; hier handgeschrieben in numpy, d = 32, eine Schicht, 12 480 Parameter (aus `zeigernetz-demo` kopiert) |
| REINFORCE | Lernverfahren: Touren ziehen, Länge als Strafe, Gewichte gegen den Gradienten von (L − b) mal log p verschieben |
| Referenzwert (Baseline) | Vergleichswert b, der von der Länge abgezogen wird; ändert nicht den Erwartungswert des Schritts (solange er nicht von der gezogenen Tour abhängt), nur die Streuung |
| Greedy-Rollout | b ist die Länge der Greedy-Tour eines Referenznetzes; es wird alle 100 Schritte ersetzt, wenn das aktuelle Netz auf 500 festen Instanzen im gepaarten einseitigen t-Test (p < 0,05) kürzere Touren baut |
| POMO | b ist das Mittel über die Touren aller n Startorte derselben Instanz |
| Leave-one-out | b_i ist das Mittel der anderen B − 1 Längen; nur im exakten Schritt, nicht in der Streuungsmessung am Netz und nicht im Training |
| Adam | Schrittregel des Trainings (Lernrate 0,001, Norm des Gradienten auf 1 begrenzt) |
| Held-Karp | exakte kürzeste Rundtour durch Teilmengen-Rekursion; liefert die Wahrheit für die Lücke |
| Nächster Nachbar, 2-opt | klassische Vergleichsverfahren (Vergleichsverfahren, nicht Referenzwert): immer zum nächsten freien Ort; 2-opt tauscht Kantenpaare, solange es hilft (einfachste Fassung ohne Nachbarlisten) |
| Wilcoxon-Test, Holm, Friedman | Rangtests über Zufalls-Seeds: exakter Vorzeichen-Rang-Test je Paar, Holm gegen die Zahl der Vergleiche, Friedman über alle vier Formen (einschließlich „keiner“) |
| Lücke | mittlere Abweichung (L / L_opt − 1) in Prozent über die Instanzen, Greedy-Tour ab Ort 0 |
"""

KERNFRAGEN = [
    "Was ändert ein Referenzwert am Erwartungswert und an der Streuung eines Lernschritts?",
    "Lernt das Netz ohne Referenzwert?",
    "Welcher der drei Referenzwerte reicht?",
    "Ist „keiner“ nur falsch eingestellt?",
    "Was kostet das Lernen je Referenzwert?",
    "Woher kommt die Streuung des Endergebnisses?",
]

ASSUMPTIONS_MARKDOWN = """
1. **Nur n = 10 und n = 20, gleichverteilte Orte im Einheitsquadrat, Fahrzeug die erzeugte Rundtour.** Das Netz ist einfach (d = 32, eine Schicht, ein Kopf, keine Normalisierung, kein Glimpse), die Auswertung greedy ab Ort 0, ohne Augmentierung.
2. **Die Rangfolge der drei Referenzwerte gilt für diese Netzgröße, diesen Optimierer (Adam, Norm des Gradienten auf 1 begrenzt) und die Lernrate der jeweiligen Spalte.** Zu n = 50 liegt in dieser Demo keine Messung vor.
3. **Die Zahl der Zufalls-Seeds begrenzt, was ein Test zeigen kann:** zehn gepaarte Läufe (n = 10) geben im exakten Wilcoxon-Test als kleinsten p-Wert 2/1024 und nach Holm über sechs Vergleiche 0,0117; sechs gepaarte Läufe (n = 20) geben 2/64 und 0,1875, dort kann also kein paarweiser Vergleich unter 0,05 kommen, auch ein großer Unterschied nicht. Feinere Aussagen brauchen mehr Zufalls-Seeds.
4. **Der Rest der Quadratsumme in Experiment 6 enthält Wechselwirkung und Rauschen** und lässt sich ohne Wiederholungszellen nicht weiter zerlegen.
5. **Zeit steht nur als Steuerung da, nie als Befund.** Die Operationszahlen sind Annahmen (8 je Bewertung der klassischen Verfahren, Rückwärtspass doppelt) und auf Zehnerpotenzen belastbar.
6. **Die Laufzeit in der Streamlit Cloud ist nicht gemessen;** das Live-Training hat eine Zeitgrenze von 60 Sekunden und die Demo vorgerechnete Läufe als Rückfall.
7. **Nicht enthalten:** Bellos gleitendes Mittel, ein gelernter Kritiker und Leave-one-out im Training (Nachbarn: `actor-critic-demo`, `monte-carlo-gradient-demo`).
8. **Die Verzerrungsfaktoren (B − 1)/B und (N − 1)/N sind hergeleitet und aufgezählt, ihre Wirkung auf das Lernen ist nicht gemessen.**
9. **Mit Netzen dieser Größe (12 480 Parameter) sagt die Messung nichts über die großen Modelle der Literatur** (Kool u. a., POMO), die mit sehr viel mehr Instanzen und Parametern trainiert wurden.
"""

FORMELN = r"""
Gegeben: Instanz $x$ mit $n$ Orten, Netz mit Parametern $\theta$, Tour $\pi$ als Permutation, Länge $L(\pi,x)$. Gesucht: Parameter mit kleiner mittlerer Länge.

$$p_\theta(\pi \mid x) = \prod_{t=1}^{n-1} p_\theta\big(\pi_t \mid \pi_{<t}, x\big)$$

$$p_\theta(j \mid \pi_{<t}, x)=\frac{\exp u_j}{\sum_{k \notin \pi_{<t}} \exp u_k}\quad (j \notin \pi_{<t})$$

$$p_\theta(j \mid \pi_{<t}, x)=0\quad (j \in \pi_{<t})$$

$$J(\theta) = \mathbb{E}_{x}\,\mathbb{E}_{\pi \sim p_\theta(\cdot\mid x)}\big[L(\pi,x)\big]$$

$$\nabla_\theta J = \mathbb{E}_{x}\,\mathbb{E}_{\pi}\Big[\big(L(\pi,x) - b(x)\big)\,\nabla_\theta \log p_\theta(\pi\mid x)\Big]\quad\text{für jedes } b \text{ ohne Abhängigkeit von } \pi$$

Beweis in einer Zeile: $\mathbb{E}_\pi[\nabla \log p_\theta(\pi\mid x)] = \sum_\pi \nabla p_\theta(\pi\mid x) = \nabla 1 = 0$, also verschwindet der Term mit $b$.

$$\hat g = \frac{1}{B}\sum_{i=1}^{B}\big(L_i - b_i\big)\,\nabla_\theta \log p_\theta(\pi_i \mid x_i),\qquad L_i = L(\pi_i,x_i)$$

Die vier Formen:

$$b_i^{\text{keiner}} = 0,\qquad b_i^{\text{Mittel}} = \bar L = \frac{1}{B}\sum_{j=1}^{B} L_j$$

$$b_i^{\text{Rollout}} = L\big(\pi^{\text{greedy}}_{\bar\theta}(x_i),x_i\big),\qquad b_i^{\text{POMO}} = \frac{1}{N}\sum_{s=1}^{N} L\big(\pi_i^{(s)},x_i\big)$$

Bei POMO startet die Tour $\pi_i^{(s)}$ in Ort $s$ und $N=n$. Der Rollout nutzt ein Referenznetz $\bar\theta$, das alle 100 Schritte ersetzt wird, wenn das aktuelle Netz auf 500 festen Instanzen im gepaarten einseitigen t-Test (Schwelle $t<-t_{0{,}95;499}=-1{,}6479$, $p<0{,}05$) kürzere Greedy-Touren baut.

Verzerrung bei unabhängig gezogenen Paaren $(x_i,\pi_i)$:

$$\mathbb{E}[\hat g^{\text{keiner}}] = \mathbb{E}[\hat g^{\text{Rollout}}] = \nabla_\theta J$$

$$\mathbb{E}[\hat g^{\text{Mittel}}] = \Big(1-\frac{1}{B}\Big)\nabla_\theta J,\qquad \mathbb{E}[\hat g^{\text{POMO}}\mid x] = \Big(1-\frac{1}{N}\Big)\nabla_\theta J_x$$

Begründung für das Mittel: für $j\neq i$ ist $\mathbb{E}[L_j \nabla\log p(\pi_i)]=\mathbb{E}[L_j]\cdot 0 = 0$, nur $j=i$ bleibt, mit dem Faktor $1/B$ im Mittel. Dasselbe gilt je Instanz für POMO mit $J_x = \frac{1}{N}\sum_s \mathbb{E}[L(\pi^{(s)},x)]$.

$$\hat g^{\text{Mittel}} = \frac{B-1}{B}\,\hat g^{\text{Leave-one-out}}$$

$$b_i^{\text{LOO}} = \frac{1}{B-1}\sum_{j\ne i} L_j,\qquad L_i - b_i^{\text{LOO}} = \frac{B}{B-1}\,(L_i-\bar L)$$

Varianzoptimaler konstanter Referenzwert für eine Instanz (Gesamtvarianz mit $s=\nabla\log p$):

$$b^\star = \frac{\mathbb{E}\big[L\,\lVert s\rVert^2\big]}{\mathbb{E}\big[\lVert s\rVert^2\big]}$$

Posten des Trainings bei Annahme Rückwärtspass doppelt so teuer wie Vorwärtspass ($E$ Operationen des Kodierers, $T$ des Dekodierers je Tour, $I$ Instanzen je Schritt, $B=120$):

$$\text{Ops}_{\text{Schritt}} = 3\,\big(I\,E + B\,T\big) \;+\; \underbrace{B\,(E+T)+\tfrac{2\cdot 500\,(E+T)}{100}}_{\text{nur Rollout}}$$

Lücke und Zerlegung:

$$\text{Lücke} = \frac{100}{M}\sum_{m=1}^{M}\Big(\frac{L_m}{L^\star_m}-1\Big)$$

$$SS_{\text{ges}} = SS_{\text{Gewichte}}+SS_{\text{Daten}}+SS_{\text{Rest}}$$

Exakte Wahrheit (Teilmengen-DP, Held-Karp):

$$D(S,j) = \min_{i\in S\setminus\{j\}}\Big(D\big(S\setminus\{j\},i\big)+d_{ij}\Big)$$

$$L^\star = \min_{j}\big(D(\{1,\dots,n-1\},j)+d_{j0}\big)$$

Die Gleichungen sind ein vollständiges Modell des Lernschritts, bis auf Adam, die Begrenzung der Gradientennorm und den Aufbau des Netzes (`zeigernetz-demo`).
"""
