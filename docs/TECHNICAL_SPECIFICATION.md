# VĀKYA — Technical Specification

**A Neural Machine Translation System for Sanskrit**
*Word Segmentation · Morphological Analysis · English & Kannada Translation*

**Author:** Manasa Rao G R
Senior UX Researcher · Independent Researcher, Computational Sanskrit · Bengaluru, India · July 2026

---

## Contents

1. [System Overview — What Vākya Does, End to End](#1-system-overview--what-vākya-does-end-to-end)
2. [The Existing Prototype — Architecture & Limitations](#2-the-existing-prototype--architecture--limitations)
3. [Prior Art & Positioning](#3-prior-art--positioning)
4. [The Four-Stage Neural Pipeline](#4-the-four-stage-neural-pipeline)
5. [Training Data — Sourcing, Cleaning, Licensing](#5-training-data--sourcing-cleaning-licensing)
6. [Model Architecture & Size](#6-model-architecture--size)
7. [Evaluation Framework](#8-evaluation-framework)
8. [Deployment Architecture](#9-deployment-architecture)
9. [Risk Register](#10-risk-register)
10. [Timeline & Milestones](#10-timeline--milestones)

---

## Executive Summary

Vākya is a proposed neural machine translation system for Sanskrit, designed to take raw Devanagari text as input and produce four synchronized outputs: (1) sandhi-resolved word segmentation, (2) full morphological analysis of every word (case, number, gender, tense, root/dhātu), (3) a fluent English translation, and (4) a fluent Kannada translation. The system is built specifically to run offline-first on low-cost devices, with the neural model as an online enhancement layer over a zero-dependency dictionary fallback that already exists as a working prototype (`index.html` / `vakya_offline.html`).

This document has two purposes. First, it is a complete technical specification of what Vākya does today and what the neural upgrade path requires — intended as project documentation for the author's own reference and for onboarding collaborators. Second, it is a training and evaluation pipeline for building this from scratch on an HPC service.

> **The core claim.** No existing open tool combines live sandhi-splitting, full morphological tagging, and bilingual (English + Kannada) neural translation in a single free, offline-capable interface built for students rather than Indologists. Component technologies exist separately (ByT5-Sanskrit / Dharmamitra for segmentation, IndicTrans2 for translation, the Sanskrit Heritage Site for morphology) but nothing unifies them for this audience. Vākya's contribution is the integration and the accessibility architecture, not a single novel algorithm.

### What exists today

The current build (`index.html`, ~90KB single file) is a rule-based system: a hand-curated 392-entry Sanskrit-English-Kannada dictionary, a 73-rule Pāṇinian suffix-stripping table, a full Devanagari→IAST transliterator, and a heuristic sandhi-reversal function. It runs entirely client-side in a browser with zero network calls, zero API keys, and zero recurring cost. It correctly handles known vocabulary from the Upaniṣads, Bhagavad Gītā, and Yoga Sūtras used as demonstration texts, but fails — visibly, with a "not found" label — on any word outside its dictionary. This is the ceiling of what a rule-based system can do without becoming an unbounded engineering project of manually encoding Pāṇini's ~4,000 grammatical rules.

The neural upgrade replaces the dictionary lookup with a trained sequence model that generalizes to unseen text, while keeping the offline dictionary as a fallback layer for zero-connectivity environments. This is the system specified in this document.

---

## 1. System Overview — What Vākya Does, End to End

A person enters raw Sanskrit text in Devanagari script — anything from a single word to a full verse. Vākya returns, in one pass:

- The **sandhi-resolved segmentation**: the input string split into its constituent words, reversing the phonetic fusion (sandhi) that Sanskrit applies at word boundaries.
- A **morphological analysis card** for every segmented word: part of speech, case/number/gender (for nominals) or person/number/tense/voice (for verbs), the dhātu (verbal root) or prātipadika (nominal stem), and compound decomposition where relevant (tatpuruṣa, bahuvrīhi, dvandva, avyayībhāva).
- A **fluent English translation** of the full input, not a word-for-word gloss stitched together — the two are shown separately so the person can see both the mechanical decomposition and the readable sentence.
- A **fluent Kannada translation**, generated with the same standard.
- **Source-text identification** where possible (e.g. recognizing an input as Māṇḍūkya Upaniṣad 2.7 or Bhagavad Gītā 2.47), with a link to the relevant Monier-Williams dictionary entries for further reading.

### 1.1 The pedagogical design constraint

This is not a generic MT system repurposed for Sanskrit. The design target is a specific person: a bright Indian student with an entry-level Android phone, intermittent or no mobile data, and no prior formal Sanskrit training beyond what school or family exposure has given them. Every architectural decision in this document is filtered through that constraint first, state-of-the-art quality second. This is why the offline dictionary fallback is treated as a permanent, non-negotiable layer of the system rather than a stopgap to be deleted once the neural model ships.

### 1.2 What "decode" means here, precisely

The word "decode" as used throughout this project has a specific technical meaning worth fixing before anything else: it is the reversal of three independent kinds of compression that Sanskrit applies to written text.

- **Sandhi compression** — adjacent words fuse their boundary sounds according to fixed phonological rules. *sa + ayam* becomes *so'yam*. Reversing this is segmentation.
- **Morphological compression** — a single inflected word encodes case, number, and gender (for nominals) or person, number, tense, mood, and voice (for verbs) in its ending, with no separator. *devasya* packs "deva" + genitive + singular + masculine into one string. Reversing this is morphological analysis.
- **Semantic compression** — Sanskrit philosophical vocabulary carries dense, context-dependent meaning that does not map one-to-one onto English or Kannada words (*akṣara* means both "syllable" and "the imperishable" simultaneously in Māṇḍūkya 1). Reversing this is translation, and it is the hardest of the three because it requires judgment, not just rule application.

A rule-based system, however large its dictionary, can approach the first two reliably and the third only for text it has seen before. A trained neural model is necessary specifically because it can generalize the third kind of decompression to text it has never seen — this is the entire justification for the project moving beyond the existing prototype.

---

## 2. The Existing Prototype — Architecture & Limitations

`index.html` is a single 1,212-line file with no build step and no external runtime dependency. It was built as a proof of concept and as the permanent offline fallback layer. Understanding its architecture precisely is necessary before specifying what the neural layer must add.

### 2.1 Components of the current build

| Component | Implementation | Coverage |
|---|---|---|
| Dictionary | 392 hand-entered lemmas, format `[pos, grammar, root, english, kannada]` | Core Upaniṣadic, Gītā, Yoga Sūtra, Vedic vocabulary |
| Transliterator | Full Devanagari→IAST character mapping incl. virāma, anusvāra, visarga, chandrabindu | Complete for Unicode Devanagari |
| Suffix stripper | 73 rules covering a-stem, ā-stem, i-stem, u-stem, an-stem, as-stem nominal endings + common verbal endings | High-frequency inflections only |
| Sandhi reversal | Heuristic vowel-junction candidate generation (a+a→ā, a+i→e, a+u→o etc.), checked against dictionary | Common vowel sandhi only; no consonant or visarga sandhi |
| Source ID | String-matching against 7 known sample texts | Only the 7 hardcoded demonstration verses |
| Translation | Concatenation of found dictionary glosses with em-dash separators | Not real translation — a gloss chain |

### 2.2 What breaks, and why it must break

Every word not in the 392-entry dictionary, and not reachable by stripping one of the 73 suffixes down to a dictionary entry, returns "not found" with the raw IAST shown instead. This is not a bug to patch — it is the fundamental ceiling of a closed-vocabulary rule system. Classical Sanskrit has an estimated active vocabulary in the hundreds of thousands of lemmas once compounding is accounted for (Sanskrit compounds are productive, meaning new ones are formed at read-time and cannot all be pre-listed). No dictionary, however large, closes this gap; only a model that has learned the productive rules of compounding and morphology can generalize to unseen forms.

Similarly, the "translation" in the current build is not translation in the linguistic sense — it is a chain of independently-looked-up word glosses joined with dashes. This is useful for word-level pedagogy (which is the correct design for that specific feature) but is not a fluent sentence, and does not attempt to resolve syntax, word order differences between Sanskrit's free word order and English/Kannada's more fixed order, or the semantic composition that turns a word list into a meaningful clause.

> **Design decision to preserve.** The offline dictionary is not being replaced or deprecated by the neural model — it is being kept permanently as Tier 0 of a tiered system (see Section 9, Deployment Architecture). Any device with zero connectivity still gets a working, if less capable, decoder. This is the single most important architectural commitment in this document and should not be relaxed for engineering convenience later.

---

## 3. Prior Art & Positioning

A fair account of what exists is essential both intellectually (to avoid re-deriving solved problems) and strategically (funding bodies will ask this question directly). The honest position is: every individual pipeline stage has a credible open precedent; the specific integration and audience does not.

### 3.1 Segmentation & morphology

| System | Institution | Method | Status |
|---|---|---|---|
| ByT5-Sanskrit / Dharmamitra | Independent (Nehrdich, Hellwig) | Byte-level T5 transformer, trained on Buddhist + classical corpora | Active, 2024–25, best-in-class segmentation |
| Sanskrit Heritage Site | INRIA, France | Pāṇinian finite-state morphological analyzer | Mature since ~2005, gold-standard rules |
| Samsādhanī | Univ. of Hyderabad | Rule-based sandhi splitter + morphological analyzer, public API | Active, used in Vākya's earlier online-mode prototype |
| sanskrit_parser | Open source (community) | Python library, dictionary-checked candidate segmentation | Actively maintained, MIT licence |
| BuddhaNexus joint tagger | Independent (Hellwig) | Joint splitter–stemmer–tagger | Active, ~100 character input cap |

### 3.2 Translation

| System | Institution | Coverage | Licence |
|---|---|---|---|
| IndicTrans2 | AI4Bharat, IIT Madras | 22 Indian languages incl. Sanskrit, English bidirectional | Apache 2.0 — open, fine-tunable |
| NLLB-200 | Meta AI | 200 languages incl. Sanskrit, general-purpose | MIT/CC-BY-NC hybrid depending on component |
| Dharmamitra translation layer | Independent | Sanskrit/Pāli/Tibetan/Chinese, Buddhist-text focused | Active, primarily academic use |

### 3.3 Word-by-word pedagogical translation (non-computational)

A static precedent for the exact content Vākya produces for the Māṇḍūkya Upaniṣad exists: Michael Douglas Neely's word-for-word Sanskrit-English translation with grammatical notes, deliberately modelled on Winthrop Sargeant's word-by-word Bhagavad Gītā format, published on academia.edu and last updated 2022. This confirms the pedagogical format itself (word-by-word grid, grammar tags, then fluent translation) has established scholarly precedent and demand — it validates the UI/UX approach, not the technical pipeline.

### 3.4 The specific gap Vākya fills

> **Positioning statement.** Vākya is not proposing a novel segmentation algorithm, a novel morphological tagger, or a novel translation architecture. It is proposing the first integration of research-grade Sanskrit NLP into a zero-cost, offline-capable, bilingual (English + Kannada) tool designed for students rather than Indologists — deployed as a single file that runs on infrastructure a rural Indian school can actually access. Every existing tool listed above requires either institutional access, a terminal, a stable internet connection, or fluency in English only. None combines full pipeline, Kannada output, and offline operation.

---

## 4. The Four-Stage Neural Pipeline

The system is deliberately kept as four separable stages rather than one monolithic end-to-end model. This is a considered choice, not a default: each stage has different training data availability, different evaluation criteria, and different failure modes, and a person debugging a bad output needs to be able to isolate which stage is responsible. It also allows Stage 1 and Stage 2 to run on CPU at negligible cost, reserving GPU time for Stage 3 where it is actually needed.

### 4.1 Stage 1 — Sandhi Segmentation

**Input:** raw Devanagari string. **Output:** list of segmented word tokens.

Recommended approach: fine-tune ByT5-Sanskrit rather than build from scratch. It is byte-level (not subword-level), which sidesteps a whole category of Devanagari tokenization edge cases, and it already achieves strong published segmentation accuracy on classical and Vedic Sanskrit. Fine-tuning on Vedic-specific text (which is under-represented relative to the classical/Buddhist text ByT5-Sanskrit was primarily trained on) is the main value-add work at this stage.

Fallback for zero-connectivity operation: the existing heuristic sandhi-reversal function in `index.html`, kept exactly as is. This is Tier 0 — see Section 9.

- **Training data need:** sandhi-annotated sentence pairs (joined form → segmented form). The Digital Corpus of Sanskrit (DCS) provides this natively for 560,000 sentences.
- **Evaluation:** segmentation accuracy against the SandhiKosh benchmark corpus (IIT Delhi), which was purpose-built for evaluating exactly this task across multiple existing splitter tools.

### 4.2 Stage 2 — Morphological Analysis

**Input:** segmented word list from Stage 1. **Output:** per-word grammar object (POS, case/number/gender or person/tense/voice, dhātu/prātipadika, compound type if applicable).

Recommended approach: a lightweight sequence-labelling head (either a small transformer encoder or a fine-tuned BERT-style model on Sanskrit, such as adapting muril-base or building a dedicated encoder) trained on the DCS's existing gold-standard morphological annotations — every one of the 560,000 DCS sentences already has this labelling done by scholars, which makes this the best-resourced stage of the entire pipeline.

This stage is the most tractable to get to high accuracy quickly because Sanskrit's morphology, while complex, is close to fully systematic (Pāṇini's grammar is famously a near-complete formal description) and the DCS training signal is clean, scholar-verified gold data rather than noisy web text.

- **Evaluation:** per-tag accuracy (case, number, gender, tense separately) plus full-tag-set exact match, evaluated against a held-out DCS split.

### 4.3 Stage 3 — Neural Translation (English + Kannada)

**Input:** original Sanskrit text plus Stage 1/2 outputs as auxiliary context. **Output:** fluent English sentence, fluent Kannada sentence.

This is the stage requiring the most compute and the most careful data curation, and the one where the "start with fine-tuning, don't train from scratch" recommendation applies most strongly.

- **Recommended Path A (fast, low-risk):** fine-tune IndicTrans2 (200M parameters, Apache 2.0, already handles Sanskrit-English and has a Kannada arm since it is one of the 22 target languages) on a curated Vedic/Upaniṣadic parallel corpus. This can be running end-to-end within weeks and gives immediate deployable results to validate the rest of the pipeline against.
- **Path B (research-grade, slower):** train a dedicated encoder-decoder transformer from scratch with a tokenizer designed around Sanskrit morphology (see Section 6). This is the path that produces a genuinely novel contribution suitable for a research paper, but should be pursued as a second phase after Path A has validated data quality and evaluation infrastructure.

Crucially, feeding the Stage 2 morphological tags into Stage 3 as auxiliary input (not just raw Sanskrit text) is expected to materially improve translation quality — this is a standard technique in low-resource MT called "morphologically-informed translation," and it is one of the few genuinely novel engineering contributions this project can claim, since general-purpose models like NLLB and IndicTrans2 do not have Stage 2's grammar output available to condition on.

### 4.4 Stage 4 — Alignment, Rendering & Fallback Logic

**Input:** all outputs from Stages 1-3. **Output:** the rendered Vākya interface state — word cards, translation tabs, sandhi flow display.

This stage is pure software engineering, not machine learning, and largely already exists in `index.html`'s rendering functions (`renderCards`, `renderTable`, `renderTranslation`). The work here is: (a) routing — deciding whether to call the neural API or fall back to the offline dictionary based on connectivity and confidence scores, (b) confidence display — showing the person when a word's analysis came from the high-confidence dictionary versus the neural model's best guess, and (c) graceful degradation — if Stage 3 translation fails but Stages 1-2 succeed, still show the word-by-word breakdown even without a fluent sentence.

> **Why four stages and not one end-to-end model.** A single sequence-to-sequence model trained to go directly from raw Devanagari to English/Kannada translation, skipping explicit segmentation and morphology, is architecturally simpler and is what most production MT systems for well-resourced languages actually do. It is deliberately rejected here for three reasons: (1) it would produce zero pedagogical value — the entire point of Vākya is showing the decomposition, not hiding it; (2) with limited parallel training data, an explicitly staged pipeline is a strong inductive bias that improves data efficiency, since Stage 2 can borrow the DCS's 560K sentences of gold morphological signal even though only a fraction of those have English/Kannada translations; (3) it keeps debugging tractable — when output is wrong, staged outputs show which stage is at fault.

---

## 5. Training Data — Sourcing, Cleaning, Licensing

### 5.1 Monolingual Sanskrit (pre-training / Stage 1-2 signal)

| Source | Size | Content | Licence |
|---|---|---|---|
| Digital Corpus of Sanskrit (DCS) | ~560,000 sentences, fully annotated | Morphology gold-standard: lemma, stem, case, number, gender for every word | CC-BY-SA, open |
| GRETIL (Göttingen) | ~1.5 billion tokens | Vedic through Classical Sanskrit, largest open raw corpus | Mixed, mostly open for research |
| AI4Bharat IndicCorp (Sanskrit subset) | Several hundred million tokens | Web-crawled modern + classical Sanskrit | Open, research use |
| Sanskrit Wikipedia | ~14,000 articles | Modern Sanskrit prose | CC-BY-SA |

### 5.2 Parallel Sanskrit–English (Stage 3 training)

| Source | Size (pairs) | Notes |
|---|---|---|
| Samanantar corpus (IIT Madras) | ~100,000 | Largest existing open parallel set for this pair |
| Public-domain translations, sentence-aligned to DCS source | ~150,000–200,000 (after alignment work) | Max Müller (Upaniṣads, 1879–1910), Ralph Griffith (Rigveda, Rāmāyaṇa, 1870s-90s), Arthur Ryder (Pañcatantra) — all archive.org / sacred-texts.com, fully public domain, but require careful sentence-level alignment since translations were done at a paragraph/verse level historically |
| FLORES-200 Sanskrit devtest | 1,012 | Professional-quality, gold-standard — reserved entirely for evaluation, never training |
| IndicTrans2's own training data (inherited via fine-tuning, not directly accessed) | Unknown, large | Not used as raw data; benefit comes via starting from their pretrained weights |

Realistic usable total after cleaning and deduplication: **300,000–500,000 Sanskrit–English sentence pairs.** This is a workable fine-tuning set for a 200M parameter model starting from IndicTrans2's pretrained weights; it would be inadequate for training a model of that size entirely from scratch, which is the core argument for the fine-tuning-first strategy in Section 4.3.

### 5.3 Parallel Sanskrit–Kannada

This is the harder half of the data problem and should be budgeted for accordingly. Realistic parallel pair count from existing digitized sources: **20,000–50,000**, substantially smaller than the English side.

- **Primary source:** the H. P. Venkatrao 36-volume Rigveda Kannada translation, commissioned by the Mysore Maharaja in the 1950s and currently being reprinted by the Kannada and Culture Department, Government of Karnataka. This is the single highest-authority Sanskrit-Kannada parallel resource in existence for Vedic text but exists in print/reprint form, not digitized parallel corpus form — digitization and alignment is a discrete, fundable sub-project in itself (see Section 12, resourcing).
- **Secondary source:** sanskritdocuments.org's Kannada-script section, which has partial padaccheda (word-split) Rigveda text already in digital form.
- **Tertiary source:** traditional Kannada commentaries on the Upaniṣads and Gītā from Karnataka maṭhas and publishing houses, where digitized.

> **Recommendation on Kannada scope.** Given the data asymmetry, the realistic phased plan is: ship high-quality Sanskrit→English translation first (sufficient data exists now), and treat Sanskrit→Kannada as a Phase 2 deliverable explicitly contingent on a digitization sub-project for the Venkatrao volumes. Promising both at equal quality on the same timeline risks either delaying the whole launch or shipping a visibly weaker Kannada output that undermines trust in the tool. This should be stated plainly in any funding application rather than glossed over.

### 5.4 Data cleaning pipeline

- Unicode normalization (NFC), removing OCR artifacts common in digitized 19th/20th-century texts
- Sentence-boundary alignment between source Sanskrit and target translation using a bitext aligner (e.g. Bleualign or an embedding-similarity aligner such as LaBSE), since historical translations are rarely at 1:1 sentence granularity with the source
- Deduplication against the FLORES-200 and any other held-out evaluation sets, to prevent test-set leakage into training data — this step is non-negotiable for credible evaluation numbers
- Length-ratio filtering to catch clear misalignments (a 3-word Sanskrit line aligned to a 40-word English paragraph is almost certainly a bad alignment)

---

## 6. Model Architecture & Size

The core architectural question raised at the outset — how big should this model be — has a specific answer once the deployment constraint (must eventually run on modest hardware, ideally browser/edge-capable) is taken seriously alongside the quality target.

### 6.1 Recommended architecture per stage

| Stage | Architecture | Parameters | Rationale |
|---|---|---|---|
| 1. Segmentation | Fine-tuned ByT5-Sanskrit (byte-level T5) | ~300M (frozen mostly; light fine-tune) | Byte-level sidesteps Devanagari tokenization edge cases; strong existing checkpoint |
| 2. Morphology | Small transformer encoder, sequence-labelling head | ~50-80M | Task is tag prediction per token, not generation — does not need decoder-scale capacity |
| 3a. Translation (Path A) | Fine-tuned IndicTrans2 encoder-decoder | 200M (En/Kn arm) | Inherits strong multilingual Indian-language priors; fastest path to usable quality |
| 3b. Translation (Path B, later) | Encoder-decoder transformer from scratch, Sanskrit-aware tokenizer | 50-100M target (deployable), scale up only if justified | Purpose-built vocabulary and morphology-conditioned decoding; research contribution |

### 6.2 Why small, specifically

The instinct to reach for a large model is understandable but wrong for this problem, for three concrete reasons. First, the deployment target is explicitly a low-end Android device with unreliable connectivity — a multi-billion parameter model cannot ever run there, full stop, regardless of quantization. Second, the parallel training data ceiling (300-500K pairs for English, far less for Kannada) cannot productively train a very large model; beyond a certain size a model this data-starved will overfit or simply fail to converge usefully. Third, Sanskrit's grammar is exceptionally systematic (a near-complete formal description exists in Pāṇini's Aṣṭādhyāyī, 3,959 sūtras) — this means a well-designed smaller model with the right inductive biases (Section 4.4's staged pipeline, morphology-conditioned decoding) can plausibly match a much larger undifferentiated model's quality on this specific language pair.

### 6.3 Tokenizer & embedding space

A shared SentencePiece BPE vocabulary of 32,000 subword tokens across Sanskrit, English, and Kannada is recommended for the from-scratch Path B model (IndicTrans2 already ships its own vocabulary for Path A).

- 32,000 vocabulary size at 512-dimensional embeddings gives an embedding table of 16.4 million parameters — roughly a third of a 50M-parameter model's total budget, which is normal and expected for a model this size (embedding tables are proportionally larger in smaller models).
- Training the BPE vocabulary jointly across all three languages (rather than three separate vocabularies) lets shared subwords emerge naturally where they exist — Sanskrit-derived Kannada vocabulary in particular shares substantial surface form with Sanskrit, which the joint tokenizer can exploit.
- Sanskrit's morphological regularity (Pāṇinian systematicity) means the embedding space should learn approximately consistent offset vectors for grammatical operations — e.g. the vector difference between *ātmā* (nominative) and *ātmanaḥ* (genitive) should be close to the same offset as between *devaḥ* and *devasya*. This is directly testable during evaluation (Section 8) as a diagnostic of whether the model has learned real morphological structure versus surface memorization.

### 6.4 Full parameter budget, target model (Path B, small)

| Component | Parameters | Share |
|---|---|---|
| Embedding table (32K × 512) | 16.4M | 33% |
| Encoder (6 layers, d=512, 8 heads, FFN=2048) | ~15M | 30% |
| Decoder (6 layers, d=512, 8 heads, FFN=2048) | ~15M | 30% |
| Output projection + norms/biases | ~3.6M | 7% |
| **TOTAL** | **~50M** | **100%** |

At this size: ~200MB in fp32, ~50MB after INT8 post-training quantization — small enough to bundle directly into a browser via WebAssembly (ONNX Runtime Web) for genuinely offline neural inference, which is the eventual target state for Tier 1 of the deployment architecture in Section 9.
The HPC Training plan is yet to be formalised, identifying GPU resources available and a concrete job plan.

---


## 7. Evaluation Framework

### 7.1 Automated metrics

| Metric | Applied to | Why | Target threshold |
|---|---|---|---|
| chrF (character n-gram F-score) | Stage 3 translation, both languages | Rewards partial morphological matches; more forgiving and appropriate than BLEU for a highly inflected source language | ≥35 (En), ≥25 (Kn, given data scarcity) |
| COMET | Stage 3 translation | Neural metric trained on human quality judgements; specifically penalises hallucination, which is unacceptable for sacred/philosophical text | ≥0.60 |
| Segmentation accuracy | Stage 1 | Exact-match against SandhiKosh benchmark corpus (IIT Delhi) | ≥85% on classical Sanskrit; Vedic text is harder — track separately |
| Per-tag morphology accuracy | Stage 2 | Separate accuracy for case, number, gender, tense against DCS gold labels | ≥90% per-tag, ≥75% full-tag-set exact match |

### 7.2 Human evaluation protocol

Automated metrics alone are insufficient for a tool whose core promise is philosophical and grammatical fidelity. The evaluation protocol therefore includes a fixed human-review pass before any release:

- A held-out set of 50 verses spanning the Māṇḍūkya Upaniṣad (all 12), Bhagavad Gītā chapter 2, and Yoga Sūtra chapter 1 — chosen because high-quality scholarly reference translations exist for direct comparison (Śaṅkara's Bhāṣya, Swami Gambhirananda's translation, Sāyaṇa's commentary).
- Every model output is compared line-by-line against the reference and every divergence is categorised: wrong case reading, wrong compound split, wrong dhātu identification, register/tone mismatch, or outright hallucination (content not supported by the source).
- **Hallucination-category errors are treated as release-blocking** regardless of how favourable the aggregate automated score is — this is a hard gate, not a soft target, given the subject matter.
- Where budget allows, a second independent Sanskrit-literate reviewer cross-checks a sample of the error categorisation to control for single-annotator bias.

---

## 8. Deployment Architecture

The deployment model is explicitly tiered, preserving the zero-cost offline guarantee as a permanent floor rather than a temporary stopgap.

| Tier | Trigger | What runs | Cost to operate |
|---|---|---|---|
| Tier 0 — Offline dictionary | No connectivity, or as instant fallback | Existing `index.html` rule-based pipeline, entirely client-side | Zero, forever |
| Tier 1 — In-browser neural (future) | Connectivity present, or model cached | INT8-quantised 50M model via ONNX Runtime Web / WebAssembly, runs client-side after first download | Zero recurring; one-time ~50MB download |
| Tier 2 — Hosted API | Connectivity present, first use / uncached | Full model served from HuggingFace Spaces free-tier T4 GPU | Zero on free tier, up to its rate limits |

Tier 2 (HuggingFace Spaces, free T4) is the realistic near-term deployment target once Phase 1 training completes — it requires no infrastructure spend and can serve an estimated 500-1,000 concurrent users at roughly 50ms inference latency per the size-4 model class specified in Section 6. Tier 1 (true in-browser inference) is the long-term target and should be pursued once the Path B small model (Section 6.4) is trained and validated, since it removes the dependency on any server entirely — the final form of "works everywhere, costs nothing, needs no one's permission."

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kannada parallel data proves too sparse for usable quality | High | High | Phase the roadmap explicitly (Section 5.3); ship English first; treat Venkatrao digitisation as its own fundable sub-project |
| Model hallucinates philosophically incorrect content on sacred text | Medium | Very high (reputational, cultural) | Hard hallucination gate in evaluation (Section 8.2); always show Stage 1/2 mechanical breakdown alongside Stage 3 translation so the person can sanity-check |
| Sandhi/morphology accuracy on Vedic (vs. Classical) Sanskrit lower than expected, since most tools are Classical-tuned | Medium-high | Medium | Budget explicit fine-tuning time on Vedic-specific DCS/GRETIL subsets; track Vedic and Classical accuracy as separate metrics throughout, not one blended number |
| Scope creep toward "general Sanskrit AI" delays a shippable v1 | Medium | Medium | Freeze v1 scope to the 4-stage pipeline in Section 4 with the 7 demonstration texts already in the prototype; explicitly defer broader corpus coverage to v2 |

---

## 10. Timeline & Milestones

| Phase | Duration | Key milestones |
|---|---|---|
| Phase 0 — Setup | Weeks 1–2 | Formalise HPC options; download & clean DCS, Samanantar, GRETIL subsets; stand up evaluation harness (chrF, COMET, SandhiKosh scoring) |
| Phase 1 — Path A (fine-tune) | Weeks 3–8 | Fine-tune ByT5-Sanskrit (Stage 1) and IndicTrans2 (Stage 3a) on curated Vedic/Upaniṣadic data; train Stage 2 morphology tagger on DCS; run full evaluation suite |
| Phase 1 release | Week 9–10 | Wire Stage 1-3 outputs into `index.html` as Tier 2 hosted-API mode via HuggingFace Spaces; human evaluation pass on the 50-verse gold set; public English-only release |
| Phase 2 — Kannada | Weeks 11–20 | Venkatrao volume digitisation sub-project; Sanskrit-Kannada alignment; Stage 3 Kannada arm fine-tune; release bilingual v1 |
| Phase 3 — Path B (research) | Months 6–14 | From-scratch Sanskrit-aware tokenizer and encoder-decoder (Section 6.4); morphology-conditioned decoding; target Tier 1 in-browser deployment; write up as a paper |

---

*ॐ शान्तिः शान्तिः शान्तिः*

**End of document.**
