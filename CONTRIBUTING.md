# Contributing to Vākya

Thank you for considering this. Vākya only works as a free tool if more than one person carries it — this document tells you exactly where to start based on what you know.

Before anything else, read the [Technical Specification](docs/TECHNICAL_SPECIFICATION.md) — most "why not just do X" questions are answered there with the actual reasoning, not just a decision.

---

## If you're a Sanskrit scholar or serious student

The highest-leverage thing you can do right now, before any neural model exists, is help stress-test the **existing rule-based dictionary** in `index.html`.

- Try the app on text outside the 7 built-in sample verses. Note every word that comes back `[unknown: ...]`.
- Open an issue tagged `dictionary` with: the Devanagari word, its IAST transliteration, part of speech, grammatical form, root, English gloss, and Kannada gloss if you have it. See the exact format below.
- Once a first neural checkpoint exists, you're needed for the **50-verse gold evaluation pass** (Māṇḍūkya Upaniṣad complete, Bhagavad Gītā chapter 2, Yoga Sūtra chapter 1) — comparing model output against Śaṅkara's Bhāṣya, Sāyaṇa's commentary, and Swami Gambhirananda's translation, and flagging anything that reads as hallucinated rather than translated. This is a **hard release gate** in the evaluation framework (spec, Section 8.2) — your review is not a formality, it blocks ship if something is wrong.

### Dictionary entry format

Entries live in the `DICT` object inside `index.html`. Each entry:

```javascript
'iast_key': ['pos', 'grammar', 'root', 'english meaning', 'kannada meaning'],
```

Example:

```javascript
'ātman': ['noun', 'nom.sg.m.', 'ātman', 'the Self, Ātman, inner essence', 'ಆತ್ಮ'],
```

Open a pull request adding entries directly, or open an issue with the data and someone will add it.

---

## If you're an ML engineer

Start with **Section 4 (The Four-Stage Neural Pipeline)** and **Section 7 (Training Plan)** of the technical spec. The short version of what's needed:

| Stage | What it needs | Where to start |
|---|---|---|
| 1 — Segmentation | Fine-tune [ByT5-Sanskrit](https://huggingface.co/buddhist-nlp) on Vedic-specific sandhi pairs from [DCS](https://github.com/OliverHellwig/sanskrit) | `src/stage1_segmentation/` |
| 2 — Morphology | Train a small sequence-labelling model on DCS's gold morphological tags | `src/stage2_morphology/` |
| 3 — Translation | Fine-tune [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) on curated Sanskrit–English (and later Kannada) parallel data | `src/stage3_translation/` |
| 4 — Rendering/routing | Wire model outputs into the existing `index.html` render functions, with graceful fallback to Tier 0 | `src/stage4_integration/` |

Ready-to-adapt Slurm job templates for any Slurm-based HPC cluster are in the spec's Appendix A. Swap `<PROJECTNAME>` and the paths and they should run largely as-is.

**Do not propose training a large model from scratch as a first move.** The spec (Section 6.2) explains exactly why: training data ceiling (~300–500K parallel pairs), the deployment target (must eventually run on a browser/edge device), and Sanskrit's own grammatical systematicity all argue for starting small and starting from existing checkpoints. If you disagree, open an issue and make the case — the spec is a living document, not scripture.

---

## If you're a Kannada-Sanskrit bilingual speaker

The single highest-leverage contribution to Phase 2 is help accessing or digitizing the **H.P. Venkatrao 36-volume Kannada Rigveda translation** (commissioned by the Mysore Maharaja, 1950s, currently under reprint by the Kannada and Culture Department, Government of Karnataka). This is the highest-authority Sanskrit–Kannada parallel text in existence and currently exists only in print form.

If you have any connection to the Kannada Sahitya Parishat or the Kannada and Culture Department, or experience with OCR/digitization projects for Kannada-script text, please open an issue tagged `kannada-data` — this is explicitly gated as its own fundable sub-project (spec, Section 13) and could unlock a grant application on its own.

Short of that: reviewing and correcting the Kannada glosses already in `index.html`'s `DICT` object is immediately useful and needs no special access.

---

## If you're a frontend/UX person

The interface (`index.html`) is intentionally a single dependency-free file. Contributions here should preserve that constraint — no build step, no framework, no external CDN dependency beyond what's already there (Google Fonts, loaded with graceful fallback to system fonts).

Priorities:
- Accessibility on low-end Android devices and slow/2G connections
- Clarity of the "not found" / low-confidence states once the neural layer introduces uncertainty that the current all-or-nothing dictionary lookup doesn't have
- Visual/interaction polish that doesn't compromise load time or offline capability

---

*ॐ शान्तिः शान्तिः शान्तिः*
