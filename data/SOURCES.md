# Data Sources

Every dataset identified for Vākya's current and planned pipeline, with licensing status and access instructions. See the [Technical Specification, Section 5](../docs/TECHNICAL_SPECIFICATION.md#5-training-data--sourcing-cleaning-licensing) for the full sourcing and cleaning methodology.

## Currently used (in `index.html`)

| Source | Content | License | Notes |
|---|---|---|---|
| Monier-Williams, *A Sanskrit-English Dictionary* (1899) | Dictionary definitions underlying the 392-entry `DICT` object | Public domain | Oxford, Clarendon Press |
| Apte, *Practical Sanskrit-English Dictionary* (1890) | Secondary cross-reference for definitions | Public domain | Pune |
| Traditional Kannada commentaries (Sāyaṇa, Ānandagiri, Śaṅkara's Bhāṣya) | Kannada gloss sourcing | Traditional/public domain | Informal sourcing; formal citation trail to be built out as dictionary grows |

## Identified for neural pipeline (not yet integrated)

### Monolingual Sanskrit

| Source | Size | License | Access |
|---|---|---|---|
| [Digital Corpus of Sanskrit (DCS)](https://github.com/OliverHellwig/sanskrit) | ~560,000 morphologically-annotated sentences | CC-BY-SA | `git clone` — no signup needed |
| [GRETIL](http://gretil.sub.uni-goettingen.de/) | ~1.5 billion tokens | Mixed, mostly open for research | Direct download from Göttingen |
| [AI4Bharat IndicCorp](https://ai4bharat.iitm.ac.in/corpora) (Sanskrit subset) | Several hundred million tokens | Open, research use | Via AI4Bharat portal |
| [Sanskrit Wikipedia](https://sa.wikipedia.org) | ~14,000 articles | CC-BY-SA | Wikipedia dump download |

### Parallel Sanskrit–English

| Source | Size (pairs) | License | Access |
|---|---|---|---|
| [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) | ~100,000 | Open | `datasets.load_dataset('ai4bharat/samanantar', 'sa')` — no key needed |
| Müller, *Sacred Books of the East* (1879–1910) | Contributes to ~150-200K after alignment | Public domain | [archive.org](https://archive.org) — search by volume title |
| Griffith, *Rigveda* / *Rāmāyaṇa* translations (1870s–90s) | (as above) | Public domain | archive.org |
| Ryder, *Pañcatantra* translation | (as above) | Public domain | archive.org |
| [FLORES-200](https://huggingface.co/datasets/facebook/flores) Sanskrit devtest | 1,012 | Open | **Evaluation only — never train on this set** |

### Parallel Sanskrit–Kannada

| Source | Size (pairs) | License | Access |
|---|---|---|---|
| H.P. Venkatrao 36-volume Kannada Rigveda (1950s, Mysore Maharaja commission) | Est. bulk of usable Sanskrit-Kannada parallel text | Status TBD — reprint held by Kannada and Culture Dept., Govt. of Karnataka | **Requires direct outreach; not yet digitized as parallel corpus.** See CONTRIBUTING.md |
| [sanskritdocuments.org](https://sanskritdocuments.org) Kannada-script section | Partial padaccheda Rigveda text | Open | Direct download |
| Traditional Karnataka maṭha commentaries | Varies, undigitized | Case-by-case | Requires case-by-case outreach |

### Models (for fine-tuning, not training from scratch)

| Model | Purpose | License | Source |
|---|---|---|---|
| [ByT5-Sanskrit](https://huggingface.co/buddhist-nlp) | Stage 1 — sandhi segmentation | Check individual model card | HuggingFace |
| [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | Stage 3 — translation (English + Kannada arm) | Apache 2.0 | GitHub / HuggingFace |

### Benchmarks (evaluation only)

| Benchmark | Purpose | Source |
|---|---|---|
| [SandhiKosh](https://github.com/iitd) (IIT Delhi) | Segmentation accuracy scoring | IIT Delhi |
| FLORES-200 Sanskrit devtest | Translation quality (chrF, COMET) | Meta AI |

---

## A note on why this list matters

Every source above is either fully public domain, openly licensed for research, or requires a specific, named, trackable outreach step (the Venkatrao volumes). Nothing in this pipeline depends on scraped, unlicensed, or ambiguously-sourced data. This is a deliberate choice consistent with the project's commitment to releasing its own outputs openly — see the [Technical Specification, Section 15](../docs/TECHNICAL_SPECIFICATION.md#15-ethical-cultural--licensing-considerations) for the full ethical and licensing framework.
