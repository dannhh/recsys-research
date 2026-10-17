# Pilot: organic vs recommendation-driven listens in Yambda-50M (2026-10-03)

Script: `scripts/yambda_organic_pilot.py`.
Data: Yambda-50M listens (Apache 2.0), downloaded to `data/raw/yambda/`.
- 46.5M listens, 9,238 users, 877k tracks.
- Keep listen+ events (played_ratio >= 50%): 29.4M events, 631k tracks, 48% organic.
- Global temporal split: the last day is test (158k events).
- Models are fitted on the training days; 3,000 test events are sampled.

| target group | n | first-order Markov HR@10 / HR@100 | pairwise over last 5 (PCTM-style) HR@10 / HR@100 | popularity HR@100 |
|---|---|---|---|---|
| all | 3,000 | 0.174 / 0.257 | 0.169 / 0.297 | 0.028 |
| organic target | 1,424 | **0.351 / 0.484** | 0.343 / 0.530 | 0.035 |
| recommendation-driven target | 1,576 | **0.013 / 0.052** | 0.011 / 0.086 | 0.022 |

Repeat and album structure:

| events | flag | seen before by user | same album as previous | same track as previous |
|---|---|---|---|---|
| all events | organic | 0.791 | 0.113 | 0.048 |
| all events | recommended | 0.603 | 0.006 | 0.005 |
| test day | organic | 0.855 | 0.229 | 0.170 |
| test day | recommended | 0.615 | 0.054 | 0.053 |

## Reading
- **The hypothesis was wrong in direction.** I expected recommender-driven events (autoplay chains) to carry the pairwise structure. Instead, the predictable transitions are **organic**: users replay their own library, albums and playlists in a fixed order.
- Recommendation-driven listens are half the data. They are also mostly repeats (60%), yet a transition model almost never predicts them (HR@10 1.3%): the recommender picks familiar tracks in an order that history does not reveal.
- **Consequence for evaluation:** next-event accuracy on mixed logs is dominated by organic replay order, that is, memorizing the user's own playlists. It says little about what a new recommender should serve. The recommendation-driven half is the logging policy's choice, a counterfactual target.

## Caveats
- One-day test window, 3,000 sampled events, first-order and pairwise models only. No neural sequential model has been run yet.
- The meaning of `is_organic` is as documented by Yandex: an event counts as recommendation-driven if it came from a personalised stream or a recommended playlist.

## Control for repetition (10,000 test events)
| target | n | Markov HR@10 / HR@100 | pairwise-5 HR@10 / HR@100 |
|---|---|---|---|
| organic, repeat | 3,938 | 0.388 / 0.512 | 0.364 / 0.571 |
| recommended, repeat | 3,263 | 0.023 / 0.069 | 0.023 / 0.105 |
| organic, new | 673 | 0.143 / 0.216 | 0.129 / 0.285 |
| recommended, new | 2,126 | 0.006 / 0.042 | 0.007 / 0.061 |

- The organic vs recommended gap holds within repeats (about 17x on HR@10) and within new items (about 25x). It is therefore **not** the known repeat-vs-explore effect.
- Sessions stay in one mode: org->org and rec->rec cover about 97% of consecutive pairs.

## Overlap check (2026-10-03)
- **Repeat vs explore:** Li, Vardasbi, Yates, de Rijke, [SIGIR 2023](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf). Repeat items carry most of the accuracy, and average accuracy hides explore users. This is known, and our gap is orthogonal to it.
- **Organic vs algorithmic consumption, descriptive, proprietary data:**
  - Anderson et al., [WWW 2020](https://www.cs.toronto.edu/~ashton/pubs/alg-effects-spotify-www2020.pdf): algorithmic listening is less diverse.
  - [ICWSM 2022](https://www.cs.toronto.edu/~ashton/pubs/spotify-exploration-dynamics-icwsm2022.pdf): exploration dynamics.
  - [ISMIR 2021](https://arxiv.org/html/2109.03538.pdf): adoption by affordance.
  - None evaluates models.
- **Discovery vs familiar listening as separate tasks:** [Amazon 2026](https://www.amazon.science/publications/why-one-size-doesnt-fit-all-improving-music-discovery-and-familiar-listening-with-specialized-models). This is the repeat axis again.
- **Klenitskiy et al., [TORS 2026](https://dl.acm.org/doi/10.1145/3787969):** shuffling tests on 19 datasets; cites Yambda. The full text is behind a 403, so it is **unresolved** whether the paper analyses `is_organic`.
- **Yambda citing papers (24, via Semantic Scholar):** none mentions organic, feedback loops or logging in its abstract.
- **Verdict (confidence: medium):** the model-evaluation asymmetry between organic and recommendation-driven events, not explained by repetition, looks open on public data.

## Full-text read: Klenitskiy et al., TORS 2026 (2026-10-04)
- **How they use Yambda:** Yambda-50M is one of 19 datasets, with a single event type. Preprocessing is 5-core filtering plus removal of consecutive repeats, leaving 44.4M interactions. The split is a global temporal split at 90% of interactions, with the last interaction of each test user as the target.
- **No use of `is_organic` anywhere.** The flag, organic vs recommended events and the logging policy are never mentioned for Yambda.
- **What they find for Yambda:**
  - Very strong sequential structure: SASRec+ NDCG@10 drops 60% after shuffling, GRU4Rec 66-69%, and 2-gram rules drop 100%.
  - Yambda is grouped under "complex sequential patterns" (Fig. 8e).
  - Repetition rate is 71.7%; filtering seen items lowers NDCG by 35%.
- **The closest idea in the paper (Sec. 6.2):** MovieLens sequences may be "pseudo-sequences shaped by the platform's recommendation flow". The paper credits this to Woolridge et al. and argues it qualitatively, without any test. Our pilot gives the first quantitative test of this with event-level labels, and in music the direction is the opposite: the recommender-driven half is the unpredictable part, while the strong sequential structure lives in organic replay.
- **Consequence for us:** dataset-level sequentiality measures (shuffling, rules, Spotify's pairwise envelope) average over two generating processes. On Yambda, "strong complex sequential patterns" reflect organic replay order, not recommendation-relevant dynamics. **Not taken; directly complementary.** The paper's future work: "conclusions about relative performance may change after selecting more appropriate datasets".
- **To read next:** Woolridge et al., "Sequence or Pseudo-Sequence? An Analysis of Sequential Recommendation Datasets" (Perspectives @ RecSys 2021), the source of the pseudo-sequence argument.

## Full-text read: Woolridge, Wilner, Glick, "Sequence or Pseudo-Sequence?", Perspectives @ RecSys 2021 ([CEUR Vol-2955](https://ceur-ws.org/Vol-2955/paper8.pdf), CC BY 4.0)
**Data and argument:** ML-1M/25M, Amazon Beauty, Video Games 2014/2018 and Steam. Timestamps are not a meaningful order: in ML-1M, 59% of users rate everything on one day and 53% of intervals are 0 seconds.

**Experiments, SASRec with sampled-100 metrics:**
- Shuffling training sequences costs ML-1M 30% NDCG@10 but other datasets only 0-4%. Beauty even improves when shuffled.
- With rating prediction on ML-1M, which removes "which item gets shown", the shuffle gap falls to 1-2%.
- **Inference:** much of SASRec's ML-1M performance comes from modelling MovieLens' internal recommendation engine, which constrains which items can appear next.

**What it lacks:**
- No event-level labels, so the recommender-driven share is inferred indirectly, through shuffling and the change of task.
- MovieLens is a rating site, not consumption; there is no implicit-feedback streaming data and no music.
- Sampled metrics.

**Relation to our pilot:** this paper is the origin of the "recommender-shaped pseudo-sequences" hypothesis, from which Klenitskiy et al. 2026 quote. Yambda's `is_organic` allows the first direct test.
- In music streaming, the recommender-driven half is the **unpredictable** part (HR@10 0.013-0.023).
- The predictable sequential structure is **organic** replay.

So the naive extension of their hypothesis ("sequential signal = the logging recommender") does not hold here. The real point is that next-event metrics mix generating processes whose predictability differs by more than an order of magnitude.

## Neural pilot, first run (`scripts/yambda_seq_pilot.py`, 2026-10-04): UNDERTRAINED
- Same 10,000 test events.
- SASRec and GRU4Rec: d = 64, sampled softmax with 2,048 negatives, logQ correction, 8 epochs.
- **Validation HR@10 was still rising steeply at epoch 8** (SASRec 0.058, GRU4Rec 0.056; no plateau). Neural numbers below are therefore lower bounds.
- GRU4Rec epochs 7-8 slowed from 136 s to 1,877 s and 2,994 s (machine contention).

| model | all | organic repeat | organic new | rec. repeat | rec. new |
|---|---|---|---|---|---|
| Markov-1 | 0.171 | **0.388** | **0.143** | **0.023** | 0.006 |
| pairwise-5 | 0.161 | 0.364 | 0.129 | **0.023** | 0.007 |
| SASRec (8 ep) | 0.106 | 0.239 | 0.040 | 0.019 | **0.013** |
| GRU4Rec (8 ep) | 0.101 | 0.228 | 0.042 | 0.019 | 0.012 |

HR@10 values. At HR@100 on recommended targets: SASRec 0.123, pairwise 0.088, Markov 0.058.

Signals, pending convergence:
- The organic vs recommended gap also holds for neural models (0.21 vs 0.017).
- **The ranking already differs by group.** Count models win clearly on organic replay; SASRec and GRU4Rec win on recommended-new (about 2x over Markov) and on recommended at HR@100.
- This must be rechecked with converged models: SASRec will improve on every group, and the open question is whether it overtakes Markov on organic replay.

## SASRec retrained with early stopping (2026-10-04)
- The run stopped at epoch 14: best validation HR@10 was 0.0897 at epoch 12, then 0.0883 and 0.0890 (patience 2, validation on 3,000 events, so noisy). It is not certain the model has converged.
- Several epochs took 20-35 minutes instead of about 110 s, again because of machine contention.

| model | metric | all | organic repeat | organic new | rec. repeat | rec. new | organic | recommended |
|---|---|---|---|---|---|---|---|---|
| Markov-1 | HR@10 | **0.171** | **0.388** | **0.143** | 0.023 | 0.006 | **0.352** | 0.016 |
| pairwise-5 | HR@10 | 0.161 | 0.364 | 0.129 | 0.023 | 0.007 | 0.330 | 0.017 |
| SASRec (14 ep) | HR@10 | 0.123 | 0.276 | 0.048 | **0.024** | **0.015** | 0.243 | **0.020** |
| Markov-1 | HR@100 | 0.247 | 0.512 | 0.216 | 0.069 | 0.042 | 0.468 | 0.058 |
| pairwise-5 | HR@100 | 0.291 | **0.571** | **0.285** | 0.105 | 0.061 | **0.529** | 0.088 |
| SASRec (14 ep) | HR@100 | **0.305** | 0.487 | 0.223 | **0.238** | **0.095** | 0.448 | **0.182** |

**Pilot verdict: GO, with caveats.**
- **Rankings depend on the event group and on the cutoff.**
  - Count models win organic replay at both cutoffs.
  - SASRec wins recommender-driven events: 2.6x Markov on rec-new HR@10, and about 2x pairwise on recommended HR@100 (0.182 vs 0.088).
  - Aggregated over all events, HR@10 crowns Markov while HR@100 crowns SASRec. The aggregate verdict is set by the organic/recommended mix and the cutoff.
- **Caveats:**
  - A single small SASRec (d = 64, sampled softmax, noisy early stopping).
  - No tuning, one seed, one test day.
  - A stronger SASRec might close the gap on organic replay.
- **Next, on rented A100:**
  - tuned and larger SASRec, plus BERT4Rec and a small GR;
  - several test windows and seeds;
  - the replay-share decomposition.
