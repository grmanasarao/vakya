# वाक्य · Vākya

**A free, offline-first Sanskrit decoder**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Rule-based prototype live](https://img.shields.io/badge/status-prototype--live-brightgreen)]()
[![Neural upgrade: spec complete](https://img.shields.io/badge/neural%20upgrade-spec%20complete-blue)](docs/TECHNICAL_SPECIFICATION.md)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange)](CONTRIBUTING.md)

---

## What this is

Vākya takes raw Sanskrit text in Devanagari and gives back four things at once:

1. **Sandhi-resolved segmentation** — the fused word-string split back into its real words
2. **Word-by-word morphological analysis** — case, number, gender, tense, root (dhātu), grammar tags
3. **A fluent English translation**
4. **A fluent Kannada translation**

It runs as a **single HTML file with zero dependencies** — no server, no API key, no signup, no subscription, no internet connection required after the page loads once. Open `index.html` in any browser and it works. That is not a limitation of an early prototype; it is the permanent, non-negotiable design floor of this project (see [Why offline-first](#why-offline-first-is-not-a-compromise) below).

---

## Why this exists

Sanskrit decoding tools that do parts of this well already exist — [ByT5-Sanskrit / Dharmamitra](https://dharmamitra.org) for segmentation, [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) for translation, the [Sanskrit Heritage Site](https://sanskrit.inria.fr) for morphology. All of them assume an Indologist at a terminal with a stable connection.

---

## What works right now

The current build (`index.html`, ~90KB, 1,212 lines) is a **rule-based system** — no AI, no model weights, no external calls:

| Component | What it does | Coverage |
|---|---|---|
| Dictionary | 392 hand-curated Sanskrit–English–Kannada entries, sourced from Monier-Williams (1899) and Apte (1890), both public domain | Core Upaniṣadic, Gītā, Yoga Sūtra, and Vedic vocabulary |
| Transliterator | Full Devanagari → IAST conversion | Complete Unicode Devanagari |
| Suffix stripper | 73 Pāṇinian suffix rules (a-stem, ā-stem, i-stem, u-stem, an-stem, as-stem nominals + common verb endings) | High-frequency inflections |
| Sandhi reversal | Heuristic vowel-junction splitting (a+a→ā, a+i→e, a+u→o, etc.), checked against the dictionary | Common vowel sandhi |

Try it on the seven built-in sample verses (Māṇḍūkya, Bhagavad Gītā 2.47, Yoga Sūtra 1.1, the Gāyatrī Mantra, Īśa Upaniṣad, Kena Upaniṣad) and it works cleanly. Try it on a word outside the dictionary and it says so honestly — `[unknown: ...]` — rather than guessing.

**That honest failure mode is the whole reason the next phase exists.**

---

## What's being built next

A **neural translation layer** that replaces the closed-vocabulary dictionary lookup with a model that generalizes to text it has never seen — while keeping the offline dictionary permanently as a zero-connectivity fallback (Tier 0).

The full technical specification — architecture, training data sources, model sizing, evaluation framework, solid HPC training plan, budget, and funding targets — is documented in **[`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md)**. Read that before opening an issue about "why not just use GPT-4 for this" — it's answered there, in detail, with reasoning.

The short version:

- **Fine-tune, don't train from scratch.** [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (Apache 2.0, AI4Bharat/IIT Madras) already speaks Sanskrit and Kannada. [ByT5-Sanskrit](https://huggingface.co/buddhist-nlp) already segments Sanskrit well. We fine-tune both rather than reinventing them.
- **Target model size: ~50M parameters** for the eventual from-scratch research model — small enough to quantize to ~50MB and run *inside the browser* via WebAssembly. No server, ever, is the end goal.
- **Training data is free and already identified**: the [Digital Corpus of Sanskrit](https://github.com/OliverHellwig/sanskrit) (560K morphologically-annotated sentences), [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar), [GRETIL](http://gretil.sub.uni-goettingen.de/), and public-domain 19th-century translations (Müller, Griffith, Ryder) from archive.org.
- **Kannada is phase 2, explicitly**, because the parallel data for it (mainly the H.P. Venkatrao 36-volume Kannada Rigveda) needs a digitization effort first. We're not going to pretend both languages ship at equal quality on the same day.

---

## Why offline-first is not a compromise

This is worth stating plainly because it will come up in every design discussion: **the neural model is an enhancement layer, not a replacement.** Tier 0 (the existing dictionary) stays forever. A device with zero connectivity should always get a working, if less capable, decoder. Relaxing this for engineering convenience is the one thing this project explicitly refuses to do.

```
Tier 0 — Offline dictionary     → runs always, zero cost, zero connectivity needed
Tier 1 — In-browser neural      → quantized ~50MB model, WebAssembly, one-time download
Tier 2 — Hosted API             → HuggingFace Spaces free-tier GPU, needs connectivity
```

---

## Repository structure

```
vakya/
├── index.html                          # The working app — open this in any browser
├── README.md                           # You are here
├── CONTRIBUTING.md                     # How to help, by skill set
├── LICENSE                             # MIT
├── docs/
│   └── TECHNICAL_SPECIFICATION.md      # Full architecture, training plan, budget, funding targets
├── data/
│   └── SOURCES.md                      # Every data source, licence, and access instructions
└── src/
    └── (neural pipeline code lands here as it's built — see CONTRIBUTING.md)
```

---

## Deployment

This is a single static HTML file. To host your own copy:

- **GitHub Pages**: Settings → Pages → deploy from `main` branch, root. Free, forever.
- **Netlify / Vercel**: drag-and-drop `index.html` or connect the repo. Free tier is sufficient.
- **Local / fully offline**: just open `index.html` directly in a browser. No build step, no `npm install`, nothing.

---

## Contributing

This project needs, in roughly this order of urgency:

1. **Sanskrit scholars** willing to review the 50-verse gold evaluation set (Māṇḍūkya, Gītā ch.2, Yoga Sūtra ch.1) once the neural model has a first checkpoint
2. **ML engineers** comfortable with HuggingFace `transformers` for the fine-tuning work (see the technical spec, Section 4 and Appendix A for ready-to-adapt Slurm scripts)
3. **Kannada-Sanskrit bilingual contributors**, especially anyone with a connection to the Kannada Sahitya Parishat or the Kannada and Culture Department (Government of Karnataka) — the Venkatrao volume digitization is the single highest-leverage contribution possible for Phase 2
4. **Dictionary contributors** — the current 392-entry dictionary in `index.html` is trivially extensible; see `CONTRIBUTING.md` for the exact format
5. **Frontend/UX contributors** — the interface can always be more accessible on low-end devices and slow connections

See **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for specifics on each.

---

## Licensing & attribution

- This project's code: **MIT License** (see [`LICENSE`](LICENSE))
- Monier-Williams (1899) and Apte (1890) dictionaries: public domain
- Digital Corpus of Sanskrit: CC-BY-SA
- IndicTrans2: Apache 2.0
- Full licensing detail for every data source used or planned: [`data/SOURCES.md`](data/SOURCES.md)

Model weights and any derived training data this project has rights to redistribute will be released under a permissive open license (Apache 2.0 or CC-BY-SA) — never held proprietary. This is a founding commitment, not a marketing line.

---

## Acknowledgment

No part of this pipeline was invented from nothing. Every stage stands on open research: [ByT5-Sanskrit / Dharmamitra](https://dharmamitra.org) (Nehrdich & Hellwig), the [Sanskrit Heritage Site](https://sanskrit.inria.fr) (Gérard Huet, INRIA), [Samsādhanī](https://sanskrit.uohyd.ac.in/scl/) (University of Hyderabad), [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (AI4Bharat, IIT Madras), and the [Digital Corpus of Sanskrit](https://github.com/OliverHellwig/sanskrit) (University of Cologne). Vākya's contribution is the integration of these into a single, free, offline-capable tool for students — not a novel algorithm at any individual stage. See `docs/TECHNICAL_SPECIFICATION.md` Section 3 for the full honest accounting of prior art.

---

*ॐ शान्तिः शान्तिः शान्तिः*
