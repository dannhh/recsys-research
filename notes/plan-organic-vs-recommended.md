# Plan: what do next-event benchmarks measure? Organic replay vs recommender-driven events (2026-10-04)

## Claim to test
Logged next events mix generating processes with very different predictability:
- organic replay of the user's own albums and playlists, which is highly predictable;
- recommender-driven events (the logging policy's choices), which are nearly unpredictable from history;
- genuine discovery.

Next-event accuracy on mixed logs is dominated by replay order. Benchmarks may therefore rank models for the wrong task.

## Evidence so far
Details are in `2026-10-03-yambda-organic-pilot.md`.
- Yambda-50M with the `is_organic` flag: first-order Markov HR@10 is 0.35 on organic targets and 0.013 on recommender-driven ones. The gap holds within repeats (17x) and within new items (25x).
- Prior work only infers this indirectly and has no event-level labels:
  - Woolridge et al. 2021 (MovieLens pseudo-sequences);
  - Klenitskiy et al. TORS 2026 (shuffling, uses Yambda but not the flag);
  - Petrov et al. RecSys 2026 (pairwise probes).
- Repeat vs explore (Li et al. SIGIR 2023) is a different axis.

## Steps
1. **Decisive pilot (1-2 days, 1 rented A100).** Models: popularity, PCTM, item-kNN, GRU4Rec, SASRec+ (full-softmax CE) and possibly a small GR, on Yambda-50M.
   - Global temporal split with several test windows.
   - Evaluate per group: organic-repeat, organic-new, recommended-repeat, recommended-new.
   - **Go:** model rankings change across groups, or replay explains most of SASRec's metric. **Stop:** rankings are identical in every group.
2. **Second dataset (in parallel, desk research).** Find public data with event-source labels or random exposure, for example KuaiRand's random-exposure flag. Check what each label means.
3. **Quantify and protocol.** Measure how much of each model's metric comes from replay order (truncation and shuffling within each group). Propose a group-wise evaluation protocol and re-benchmark.
4. **Theory.** A mixture model of the user replay process and the logging policy: when MLE training and mixed-log evaluation distort model rankings, and what the organic flag identifies.
5. **Write-up.** Target: RecSys / SIGIR full paper. Post the preprint on arXiv before PhD application deadlines.

## Risks
- **"So what":** an exploring policy is unpredictable by design. Answering this needs step 1 to show consequences.
- **One platform:** the flag definition is Yandex's. Step 2 addresses this.
- **Scoop:** Yandex lists "organic vs algorithmic interaction analysis" as its own future work; the Spotify and Klenitskiy groups work nearby.

## Open inputs from the user
- Access details for the rented A100 server, when needed for step 1.
- Where the project should live: this repo, or a new one.

## Second-dataset search (2026-10-04)
| dataset | domain | label separating user-driven from policy-driven events | access | verdict |
|---|---|---|---|---|
| [KuaiSAR](https://kuaisar.github.io/) (CIKM 2023) | short video (Kuaishou), 25.9k users, 19 days, 5M search + 14M recommendation actions | each action is tagged search (S) or recommendation (R); transitions between the two are logged | Zenodo, CC BY-NC-SA 4.0 | **best candidate**; downloading |
| [FINN.no Slates](https://github.com/finn-no/recsys_slates_dataset) (RecSys 2021) | marketplace, 2.3M users, 37.4M interactions, 30 days | `interaction_type`: search slate (69.7%) or recommendation slate (30.3%); full exposure logged | pip package, Apache-2.0 | good third candidate; at most 20 slates per user |
| [ContentWise Impressions](https://contentwise.github.io/contentwise-impressions/) (CIKM 2020) | video on demand | interactions directly linked to a recommendation list vs not | download link (form optional) | candidate; meaning of "not linked" needs checking |
| [Spotify MSSD](https://arxiv.org/abs/1901.09851) (WWW 2019) | music, 160M sessions | `context_type`: user collection / radio / personalised / editorial / charts | **no longer downloadable** (AIcrowd notice, July 2024); request access from Spotify Research | the closest analogue to Yambda, but unavailable |
| [Qilin](https://arxiv.org/abs/2503.00501) (SIGIR 2025) | Xiaohongshu search + recommendation | search vs recommendation sessions | GitHub / Hugging Face | small (about 100k samples) |

Caveat: "search" is user-initiated but not the same as Yambda's "organic" (which includes library and playlist replay). The analogue is user-driven vs policy-driven exposure.

## KuaiSAR pilot, count models (`scripts/kuaisar_pilot.py`, 2026-10-04)
**Data:** feed plays (click = 1) and search clicks (click_cnt > 0), merged per user by time: 7.8M events, 25.8k users, 2.54M items, 19 days.
- Search is 7.5% of events.
- Repeat rate: search 7.0%, feed 0.1%. The feed almost never repeats.

**Protocol:** the last 3 days are test, with a stratified sample of 5,000 search and 5,000 feed events. A leakage bug in the train cutoff of `run_counts` was fixed: the cutoff now follows the test window. The Yambda runs used a 1-day window and were not affected.

| model | search (user-driven) HR@10 / HR@100 | feed (policy-driven) HR@10 / HR@100 | search repeat HR@10 | search new HR@10 | feed new HR@10 |
|---|---|---|---|---|---|
| Markov-1 | 0.0062 / 0.0074 | 0.0004 / 0.0018 | 0.048 | 0.0013 | 0.0004 |
| pairwise-5 | **0.0172 / 0.0242** | 0.0016 / 0.0048 | 0.132 | 0.0038 | 0.0016 |

**Reading:**
- **Same direction as Yambda:** user-driven events are about 11x more predictable than policy-driven ones (pairwise HR@10), and policy-driven feed events sit near the floor.
- The gap is smaller within new items (2.4x vs 25x on Yambda), and most of the search advantage comes from repeats.
- Absolute levels are tiny (2.5M items, novelty-driven short-video feed). KuaiSAR therefore supports the asymmetry, but is weak ground for a ranking-flip test.
- Next candidate: FINN.no (search vs recommendation slates, with exposure logged).

## KuaiRand-Pure pilot, count models (`scripts/kuairand_pilot.py`, 2026-10-04)
**Data:**
- 2.62M exposures, of which 45% are random (`is_rand`). Click rate: policy 0.46, random 0.18.
- Kept: 869k clicks, 27.1k users, 7,579 items. Policy-exposed clicks are 76%.
- Random exposures are 99% on tab 1.

**Protocol:** the last 3 days are test, stratified 5,000 policy-exposed and 5,000 random-exposed clicks.

| model | policy-exposed click HR@10 / HR@100 | random-exposed click HR@10 / HR@100 |
|---|---|---|
| popularity | 0.038 / 0.182 | 0.003 / 0.024 |
| Markov-1 | 0.030 / 0.151 | 0.002 / 0.022 |
| pairwise-5 | **0.048 / 0.227** | 0.003 / 0.027 |

Chance level at HR@100 is 100 / 7,579 = 0.013.

**Reading:**
- **Here the policy-driven events are the predictable ones** (17x over random exposure). Popularity alone gets 0.038 on policy clicks: the Kuaishou policy is imitable from history plus popularity, which is the Woolridge direction.
- Clicks on random exposures are near chance for history-based models. They are the closest thing to unbiased preference given exposure; KuaiRand was built for exactly this purpose.

## Cross-dataset picture (count models)
| dataset | user-driven | policy-driven | random |
|---|---|---|---|
| Yambda (music) | organic replay: **most** predictable (0.35) | recommender stream: near floor (0.016) | n/a |
| KuaiSAR (short video) | search: more predictable (0.017) | feed: near floor (0.0016) | n/a |
| KuaiRand (short video) | n/a | policy feed: predictable (0.048), largely popularity | random: near chance (0.003) |

Values are HR@10, pairwise or Markov.

**Unifying claim (revised):** next-event metrics measure a mixture of
1. user replay of self-ordered sequences,
2. imitation of the logging policy, which is imitable for some platforms (Kuaishou) and not for others (Yandex's stream),
3. preference-driven choice.

The mixture weights depend on the platform and the logging policy, so benchmark conclusions do not transfer. A natural theory: next-event likelihood decomposes over these generating processes. Event-level labels (organic, search/feed, random) identify the components. This also connects directly to exposure-bias evaluation, which is AISIA's topic.

## FINN.no
The Google Drive download is blocked for scripts (gdown refused). It needs a browser download of `data.npz` (file id 1XHqyk01qi9qnvBTfWWwqgDzrdjv1eBVV) and `ind2val.json` into `data/raw/finn/`.

## KuaiRand SASRec (2026-10-04)
- **Training:** 26k windows, d = 64. Validation HR@10 peaked at epoch 1 (0.0157) and then fell (0.012, 0.011), so early stopping ended training at epoch 3. The model is roughly a popularity model.
- **Test HR@10 / HR@100:**
  - policy clicks 0.035 / 0.213;
  - random clicks 0.003 / 0.024, at chance.
- **The same asymmetry holds, but there is no ranking information:** on random-exposure clicks every model sits at chance under full-catalogue next-item ranking.

**Protocol change needed for KuaiRand.** Use exposure-conditioned evaluation, the standard in the debiasing literature:
- score each exposed item given the user's history;
- compute click AUC (or NDCG within the exposures of a session) separately on policy exposures and on random exposures;
- compare model rankings across the two.

This tests whether models that look good on policy-logged data also rank random exposures well, and it plugs directly into exposure-bias work.

## KuaiRand exposure-conditioned AUC (`scripts/kuairand_exposure_eval.py`, 2026-10-04)
**Setup:**
- Test exposures from the last 3 days: 46.6k policy and 46.6k random. Click rate is 0.44 on policy exposures and 0.18 on random ones.
- Exposures of never-clicked items are dropped, the same for every model.
- The score uses the click history before each exposure.

| model | policy AUC | policy GAUC | random AUC | random GAUC |
|---|---|---|---|---|
| click popularity | **0.596** | **0.551** | **0.610** | **0.628** |
| exposure popularity (policy imitation) | 0.542 | 0.514 | 0.571 | 0.580 |
| Markov-1 | 0.593 | 0.541 | 0.609 | 0.625 |
| pairwise-5 | 0.578 | 0.539 | 0.590 | 0.618 |
| SASRec (stopped at epoch 3, ~popularity) | 0.586 | 0.552 | 0.601 | 0.620 |

**Reading: no ranking flip on KuaiRand.**
- All models are weak (AUC about 0.6; they are next-item models, not CTR models with features), and the ranking is nearly the same on policy and random exposures.
- GAUC is lower on policy exposures for every model, a selection effect: the policy has already filtered to plausible items.
- KuaiRand supports the predictability asymmetry (previous section) but not model-ranking changes.

**Status of the evidence:**
- The asymmetry between event sources is shown on 3 datasets.
- **A ranking change is shown only on Yambda so far.** The generality of the flip claim is the main open risk.
- Options:
  - Spotify MSSD (needs an access request);
  - FINN.no (needs a browser download);
  - the robust Yambda runs (several models, seeds and windows on a rented A100).

## More second-dataset candidates (2026-10-04)
- **FINN.no:** about 8 citations (OpenAlex). Rarely used; dropped.
- **Spotify MSSD:** about 79 citations, but no longer downloadable; access must be requested.
- **Zvuk** (KDD 2024, music, 245M plays): fields are user, session, datetime, track, duration. **No source field.**
- **VK-LSVD** (WWW 2026, short video, 40B exposures, Apache 2.0, ready-made subsamples): has `place` with 24 ids (feed/search/group/...), but the ids are **anonymised with no mapping**. Semantics would have to be inferred (the dominant id is presumably the feed). Usable as a heterogeneity-across-surfaces check, not as a clean user/policy label.
- **T-ECD:** synthetic data; no.
- **New related work to cite:** "Follow the guides: disentangling human and algorithmic curation in online music consumption" ([arXiv 2109.03915](https://arxiv.org/abs/2109.03915), Deezer). It separates organic from algorithmic and editorial listening on proprietary data and is descriptive.

**Conclusion:** Yambda appears to be the only large *public* dataset with clean event-level organic vs recommendation labels in music; MSSD is by request only. Frame Yambda's uniqueness explicitly, and verify that claim in related work. Use KuaiRand and KuaiSAR (and optionally VK-LSVD place ids) for generality of the predictability asymmetry.

## Deep dataset search, round 2 (2026-10-05)
**A. Direct event-source labels (user-driven vs policy-driven):**
- **Yambda:** the only clean one in music.
- **KuaiSAR:** search vs feed.
- **VK-LSVD:** `place` with 24 ids, anonymised and unmapped.
- **ContentWise Impressions** (CIKM 2020, VOD): interactions carry an impression id except when they came from provider-added rows. Whether any interactions are truly organic needs checking.
- **EB-NeRD** (RecSys Challenge 2024, news): front page vs article page, both curated or recommended surfaces. Partial at best.
- **JDsearch** (SIGIR 2023): search queries plus multi-channel interactions; the README could not be fetched, so it is unverified.

**B. Exposure randomisation (logged policy vs unbiased):**
- **KuaiRand** (done; ~132 citations).
- **KuaiRec** (CIKM 2022, ~163 citations): a fully observed small matrix (1,411 users x 3,327 videos, 99.6% density) plus a policy-logged big matrix for the same users and items. **Best candidate for a second ranking-flip test:** rank models on held-out logged interactions vs on the fully observed matrix, which is the ground-truth preference.
- **Yahoo! R3** (music ratings, self-selected vs random songs) and **Coat:** classic MNAR test sets, non-sequential. Optional, for the theory.
- **Open Bandit Dataset** (ZOZOTOWN): uniform-random vs Bernoulli-TS logging policies, 80 items, non-sequential.

**C. Music without source labels (replay-mechanism generality only):**
- **Deezer listening events** (RecSys 2024, 700M events, 3.4M users, 50k songs, CC BY-NC, 7.8 GB): no documented source field.
- **Zvuk** (KDD 2024).
- **Last.fm 1K / LFM-2b.**

These can show that replay and album-order structure dominates sequential predictability in music generally, but not the organic vs policy split.

**Plan:**
- Add **KuaiRec** as the second ranking-flip dataset (cheap, reputable, matches the exposure-bias framing).
- Optionally add Deezer or Zvuk to test the replay mechanism.
- Verify ContentWise semantics before use.

## KuaiRec: logged vs fully observed evaluation (`scripts/kuairec_pilot.py`, 2026-10-05)
**Setup:**
- Positives are watches with watch_ratio >= 2: 7.5% of the big matrix, 4.6% of the small matrix.
- Models train on big-matrix positives before the last 7 days.
- **Logged evaluation:** the next positive after the cutoff (10k sampled events), ranked over the full 10.7k catalogue.
- **Observed evaluation:** for each of the 1,411 small-matrix users, all 3,327 small-matrix videos are ranked from the big-matrix history before the cutoff. Ground truth is watch_ratio >= 2 in the fully observed matrix.
- SASRec stopped at epoch 15 (validation HR@10 0.091).

| model | logged HR@10 | logged NDCG@10 | logged HR@100 | observed NDCG@10 | observed NDCG@50 | observed Recall@50 |
|---|---|---|---|---|---|---|
| popularity | 0.003 (4th) | 0.001 | 0.036 | **0.508 (1st)** | **0.423** | **0.222** |
| Markov-1 | 0.035 (2nd) | **0.021** | 0.105 | 0.301 (3rd) | 0.280 | 0.154 |
| pairwise-5 | **0.036 (1st)** | 0.019 | **0.176** | 0.381 (2nd) | 0.277 | 0.137 |
| SASRec | 0.011 (3rd) | 0.006 | 0.086 | 0.297 (4th) | 0.242 | 0.123 |

**Reading: a full ranking reversal.**
- Logged next-item evaluation crowns the transition models and puts popularity last.
- Evaluation against fully observed preferences puts popularity first by a wide margin, and the transition models and SASRec below it.
- Logged next-item metrics reward continuing logged sequences (consistent with imitating the logging process). They do not reward predicting what users actually like among exposed items.

**Caveats:**
1. The candidate sets differ: the full catalogue for logged evaluation, the 3,327 small-matrix videos for observed evaluation. The small matrix was built from popular videos, which favours popularity.
2. The KuaiRec paper itself reports that conclusions on sparse logged data can differ from the fully observed matrix (for CF models). **Positioning must be checked against its findings.** Our addition is sequential next-item models and the transition-imitation mechanism.
3. Single seed and small SASRec.

**Next robustness check:** run the logged evaluation restricted to small-matrix videos for non-small users, so the candidate sets match.

### KuaiRec robustness checks (2026-10-05)
Logged evaluation restricted to the **same 3,327 small-matrix candidates**, on non-small users, 10k events, with **seen items filtered**:

| model | HR@10 | NDCG@10 | HR@100 |
|---|---|---|---|
| popularity | 0.011 (4th) | 0.004 | 0.093 |
| Markov-1 | 0.069 | 0.044 | 0.194 |
| pairwise-5 | **0.085 (1st)** | **0.050** | **0.331** |
| SASRec | 0.021 | 0.011 | 0.262 |

Without seen filtering the same ordering holds (popularity 0.004, pairwise 0.065). The reversal therefore holds on identical candidates and with or without seen filtering.

Remaining caveat: the user populations differ (non-small users for logged evaluation, the 1,411 small-matrix users for observed evaluation).

### Positioning vs the KuaiRec paper ([Gao et al., CIKM 2022](https://arxiv.org/abs/2202.10842))
Its three insights:
1. exposure biases change the performance and rankings of models;
2. even uniformly random exposure at different densities gives inconsistent results;
3. matrix completion only partly helps.

Those experiments **simulate** exposure strategies (uniform, positivity-oriented, popularity-oriented) by subsampling the small matrix, and evaluate **conversational recommenders** (multi-round CRS policies).

**Our result differs:**
- it uses the **real policy-logged big matrix**, not simulated exposure;
- it evaluates **sequential next-item models**, the standard benchmark task;
- it identifies a mechanism: logged next-item metrics favour transition imitation.

Cite the paper as foundational for "exposure changes rankings".

### PRIOR WORK FOUND: Parajuli, Vaez Barenji, Ekstrand, "On the Convergent Validity of Offline Evaluation Designs for Recommender Systems", RecSys 2026 ([arXiv 2607.25097](https://arxiv.org/abs/2607.25097))
- They correlate model rankings under sparse logged evaluation with rankings against dense ground truth on MovieLens and KuaiRec. The design choices varied are filtering threshold, candidate set (full, U-1000, P-1000), list length and relevance threshold.
- **On KuaiRec, "sparse evaluation inverts the dense ground-truth ranking regardless of any design choice"** (negative Kendall tau). MovieLens shows weak positive correlation.
- Models: collaborative filtering (fsSLIM variants, explicit vs implicit training). **No sequential models, no transition or Markov baselines, no Yambda or organic analysis.**
- They write that the MovieLens/KuaiRec contrast "could be partially attributed to their underlying characteristics, although we did not [investigate]". **The WHY is left open.**

**Consequence for us:**
- The KuaiRec ranking reversal is **not novel as a phenomenon**. It corroborates Parajuli et al. and extends them to sequential next-item models.
- Our contribution must be the **mechanism**: logged next-item metrics reward continuing logged sequences (organic replay, transition or policy imitation). Yambda's event-level labels are what make this mechanism testable.
- Reframe:
  - KuaiRec: "consistent with Parajuli et al., and our transition-imitation account explains which models gain".
  - Yambda: the main, novel evidence, because it labels the generating process.

### KuaiRec control results (2026-10-05): the reversal is driven by the popular head
**Same users:** logged evaluation on the 1,411 small-matrix users (4,940 events, full catalogue) gives the same ordering as before: pairwise 0.039 > Markov 0.026 > SASRec 0.013 > popularity 0.000 (HR@10).

**Excluding the most popular small-matrix videos** from candidates and ground truth:

| setting | eval | popularity | Markov-1 | pairwise-5 | SASRec |
|---|---|---|---|---|---|
| all 3,327 | observed NDCG@10 | **0.508** | 0.301 | 0.381 | 0.294 |
| drop top 20% (2,662 kept) | observed NDCG@10 | 0.053 (4th) | 0.078 | **0.097** | 0.082 |
| drop top 20% | observed Recall@50 | 0.040 | 0.041 | 0.054 | **0.067** |
| drop top 50% (1,664 kept) | observed NDCG@10 | 0.035 (4th) | 0.048 | **0.068** | 0.066 |
| drop top 50% | observed Recall@50 | 0.036 | 0.043 | 0.073 | **0.105** |
| drop top 50% | logged-on-small HR@10 | 0.011 | 0.071 | **0.112** | 0.031 |

**Reading:**
- The full reversal on KuaiRec comes from the **popular head** of the small matrix: everyone watches popular videos with ratio >= 2, so popularity dominates dense ground truth.
- Once the top 20% is removed, logged and observed rankings **largely agree** (pairwise at or near the top, popularity last). One exception: SASRec moves from 3rd on logged to 1st or 2nd on observed at deeper cutoffs (Recall@50).

**Implications:**
1. KuaiRec is **weak** evidence for our mechanism. It does not show transition imitation winning on logs but losing on true preference beyond the head.
2. A possible small contribution: **the inversion reported by Parajuli et al. is concentrated in the popular head.** Check whether they analysed this.
3. The ranking-flip claim rests on **Yambda** alone. This makes the A100 robustness runs on Yambda decisive.
