# Paper Follow-Up: Repo Status And Next Steps

Stand: 2026-03-05

## Bereits vorhanden

- Benchmark-faehige Auto-Labeling-Pipeline fuer `wdc`, `abt-buy`, `amazon-google`, `dblp-acm`, `dblp-scholar` und `walmart-amazon`.
- Profil-Logik fuer `small`, `medium`, `large` und `all` inklusive Ditto-Export.
- Ditto-Baseline-Training ueber mehrere Benchmarks.
- Erste `autolabel_v1`-Ergebnisse fuer mehrere Benchmarks und Profile.
- WDC-spezifischer Multi-Agent-/Committee-Pfad inklusive Konsens- und Mehrheitslogik.
- Batch-Eval-Skript fuer direktes `gpt-5.2`-Matching auf Benchmark-Testsets.
- Aeltere Skripte fuer Fehleranalyse und Validierung im `scripts/old`-Bereich.

## Was fuer die Mail des Professors noch fehlt

- Systematische Analyse `generated vs original splits` auf Paar-, Entitaets- und Merkmalsebene.
- Saubere Antwort auf die Frage, welche Paare in den generierten Sets neu, wiederverwendet oder potentiell problematisch sind.
- Einheitliche Fehleranalyse fuer originale und generierte Datensaetze.
- Breiter Downstream-Vergleich jenseits von Ditto.
- Konsistentes Logging fuer Kosten, Laufzeiten und Trainingsaufwand.
- Reproduzierbare Ablage aller Run-Artefakte; derzeit liegen einige Ergebnisse nur als Zusammenfassungen vor.

## Konkrete To-dos

1. Experimentmatrix festziehen: Benchmarks, generierte Sets, Profile, Downstream-Modelle, Testsets.
2. Trainingsset-Profiling bauen:
   - Overlap zu offiziellen Train/Valid/Test-Splits
   - Anteil wirklich neuer Paare
   - Label-Verteilung
   - Entity-Coverage links/rechts
   - Paare pro Entitaet
   - Laengen- und Aehnlichkeitsverteilungen
3. Fehleranalyse fuer generierte und originale Sets definieren:
   - automatische Checks
   - LLM-unterstuetzte Review
   - kleine manuelle Stichprobe nach Ralphs Schema
4. Committee-Experimente in den allgemeinen Benchmark-Workflow integrieren oder explizit als WDC-Sonderpfad behandeln.
5. Ditto-Vergleich fuer alle verfuegbaren generierten Sets reproduzierbar abschliessen.
6. Guenstige Downstream-Modelle ergaenzen:
   - klassische EM-Baselines wie Magellan oder `py_entitymatching`
   - danach teurere Modelle
7. `gpt-5.2` out-of-the-box ueber alle gewuenschten Testsets wirklich ausfuehren und versioniert ablegen.
8. Pruefen, ob `gpt-5.2 mini` Fine-Tuning im aktuellen Setup ueberhaupt geplant und verfuegbar ist.
9. Llama oder aehnliche Modelle nur aufnehmen, wenn GPU-/Fine-Tuning-Infrastruktur und Datenformat feststehen.
10. Ergebnisaggregation erweitern:
    - Performance
    - Aufwand
    - Fehlertypen
    - Datensatzcharakteristika

## Repo-spezifische Beobachtungen

- Lokale Auto-Label-Rohdaten sind derzeit nur teilweise vorhanden.
- Fuer Ditto-Baselines liegen die Summaries im Repo, aber nicht alle referenzierten Roh-Run-Verzeichnisse.
- Fuer die naechsten Paper-Schritte sollte zuerst eine belastbare Profilierung der vorhandenen Auto-Label-Sets aufgebaut werden, bevor weitere Downstream-Modelle nachgezogen werden.
