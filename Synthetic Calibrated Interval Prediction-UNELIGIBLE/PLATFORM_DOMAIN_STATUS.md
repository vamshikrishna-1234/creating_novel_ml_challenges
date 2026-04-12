# Platform domain eligibility — **NOT ELIGIBLE** (as currently designed)

## What this task actually is

- **Tabular supervised learning** with a **continuous float** label (`target`) and **three float outputs** per test row (`point_estimate`, `lower_90`, `upper_90`).
- Substance: **regression + uncertainty-style bands**, not a distinct platform domain.

## Check against your enabled domains

| Domain | Fits? | Reason |
|--------|-------|--------|
| NLP | No | No text; only numeric / integer-coded columns. |
| Computer Vision | No | No images or spatial inputs. |
| Object Detection | No | No images or boxes. |
| Recommendation | No | Not user–item interactions or ranking. |
| Sequence to Sequence | No | No input/output sequences as primary structure. |
| Prompt Engineering | No | No prompts or LLM I/O. |
| RAG | No | No retrieval corpus or evidence documents. |
| Fine-Tuning | No | Not updating weights of a pretrained model; tabular prediction only. |
| LLM Evaluation | No | Not scoring model-generated text or LLM behavior. |

**Conclusion:** This challenge **does not genuinely** belong to any of the listed domains. Automated classifiers correctly label it **Regression**, which your platform currently rejects.

## What would be needed to align with a listed domain

You would need a **structural** change (e.g. text-in-the-loop for NLP/RAG, sequences for seq2seq, user–item logs for rec, etc.), not only rewording the problem description.

## Status

- **Marked:** `INELIGIBLE` for submission under the current domain list + regression gate.
- **Folder / pipeline:** Kept as a reference implementation; do not expect platform acceptance until the task is redesigned or Regression opens.
