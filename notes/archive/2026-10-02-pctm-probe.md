# Pairwise probe vs generative recommenders, Amazon 2023 Instruments (2026-10-02)

Script: `scripts/pctm.py`, a re-implementation of PCTM from Petrov et al., RecSys 2026, arXiv 2608.19833, Algorithms 1-2.
Output: `runs/amazon2023_Musical_Instruments_pctm.json`.
- Data: the same leave-one-out split as our TIGER, with 57,439 users and 24,587 items. Ranking is over the full catalog, with history items excluded (targets never repeat in this split).
- Tuning: 60 random configurations, then a refinement around the best. Selection is on the valid target.
- Best configuration: gamma_d = 1/sqrt(d), W = 20, tau = 50,000, head-tail kernel H(10, 0.8, 0.7), no popularity term. Results plateau for W from 20 to 100 and tau from 20k to 200k.

| model | R@10 | N@10 | source |
|---|---|---|---|
| popularity | 0.0253 | 0.0129 | ours |
| first-order Markov (last item) | 0.0307 | 0.0192 | ours |
| TIGER (ours, 4.8M) | 0.0501 | 0.0260 | ours |
| TIGER (published) | 0.0537-0.0566 | | EPIC / SPRINT / Latte tables |
| **PCTM, counts from train** | **0.0558** | **0.0321** | ours |
| **PCTM, counts from train + valid** | **0.0578** | **0.0332** | ours |
| ActionPiece | 0.0609 | 0.0318 | SPRINT Table 2 |
| LLaDA-Rec | 0.0623 | 0.0337 | SPRINT Table 2 |
| EPIC | 0.0640 | 0.0352 | EPIC |
| DiffGRM | 0.0659 | 0.0356 | SPRINT Table 2 |
| SPRINT | 0.0713 | 0.0386 | SPRINT Table 2 |

## Reading
- PCTM is a recency-weighted sum of pairwise transition counts, with no embeddings and no content. It runs in seconds on CPU.
  - It beats our TIGER (+15% R@10, +28% N@10) and published TIGER.
  - It matches ActionPiece on N@10 and is within about 1.5% of LLaDA-Rec on N@10.
- Only the newest non-autoregressive models clearly exceed it: SPRINT by +23% R@10 and +16% N@10, DiffGRM by +14% / +7%, EPIC by +11% / +6%.
- **Confound:** generative recommenders use item text through their SIDs; PCTM uses no content at all. A margin over PCTM can therefore come from content sharing (sparse and tail items) rather than from higher-order sequence modelling. Any principled measure must separate three sources: pairwise transitions, item content, and genuine higher-order interactions.
- **Caveats:**
  - Published numbers are paper-to-paper; we have not verified that preprocessing is identical.
  - SeqRules was not run.
  - This is a single dataset with a single split.

## Content-aware pairwise probe (`scripts/pctm_content.py`, 2026-10-03)
The probe stays pairwise; every term is a weighted sum over single history items. Two content additions:
- **Source smoothing:** C~ = C + beta * S C. A rare source item borrows the transitions of its text neighbours.
- **Content expert:** a term mu * sum_j w_j S[h_j, a] that scores a candidate by text similarity to each history item.

S is the 10-NN cosine graph of the Sentence-T5 item embeddings. Best on valid: k 10, beta 0.1, mu 0.3. Output: `runs/amazon2023_Musical_Instruments_pctm_content.json`.

| model | R@10 | N@10 |
|---|---|---|
| PCTM (counts from train / train + valid) | 0.0558 / 0.0578 | 0.0321 / 0.0332 |
| **PCTM + content** (counts from train / train + valid) | **0.0606 / 0.0624** | **0.0344 / 0.0353** |
| ActionPiece | 0.0609 | 0.0318 |
| LLaDA-Rec | 0.0623 | 0.0337 |
| EPIC | 0.0640 | 0.0352 |
| DiffGRM | 0.0659 | 0.0356 |
| SPRINT | 0.0713 | 0.0386 |

Test R@10 by target popularity (train count), PCTM -> PCTM + content:

| target popularity | n | R@10 |
|---|---|---|
| tail<=5 | 8,693 | 0.0044 -> 0.0060 |
| 6-25 | 20,144 | 0.0111 -> 0.0143 |
| 26-100 | 15,803 | 0.0273 -> 0.0297 |
| head>100 | 12,799 | 0.1963 -> 0.2085 |

**Reading:**
- Content adds about 8-9% to the pairwise probe.
- The content-aware probe matches or beats ActionPiece and LLaDA-Rec. It is about level with EPIC on N@10 and within about 1% of DiffGRM on N@10.
- SPRINT keeps a margin: +14% R@10 and +9% N@10 over the probe with train + valid counts.
- Recall on tail targets is near zero for the probe (0.6%). If SPRINT's margin sits on tail items, it comes from SID sharing, not from higher-order modelling. A per-group comparison needs SPRINT run on this split.
