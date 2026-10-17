# Data checks, 2026-10-01

Goal: confirm the data can answer each evaluation criterion of the conformal-GR pilot before training anything.
Scripts: `scripts/check_amazon.py`, `scripts/check_yambda.py`. Raw numbers: `data/processed/*_checks.json`.

## Amazon Reviews 2014 Beauty (5-core)

- 22,363 users, 12,101 items, 198,502 interactions, 2002-06 to 2014-07.
- **Human categories:** 100% of items have a category path of depth ≥ 2. Depth distribution {2: 104, 3: 1,629, 4: 7,566, 5: 2,451, 6: 351}. Distinct nodes: 8 at depth 2, 42 at depth 3, 159 at depth 4. Enough for a "SID tree vs human tree" comparison.
- **Unseen items:** leave-one-out 0.23% (essentially none). Global temporal split at the 90% time quantile (2014-05-13): 5,517 test users with history, 517 unseen targets (9.4%). Usable but small: at 90% coverage the standard error of a coverage estimate on 517 users is about 1.3 points.
  - This differs from the cold-item paper's ~95% cold ratio on Beauty; they use different windows and filtering. Not a contradiction to chase now.
- **User activity:** history length quantiles 10/25/50/75/90/99% = 4/4/5/8/13/42. Groups: 4–5 (11,383), 6–10 (7,433), 11–20 (2,528), 21+ (1,019). Narrow range, as expected.
- **Target popularity:** tail ≤ 5 (4,446), mid 6–20 (8,891), head > 20 (9,026).

## Yambda-50M (listens only)

- 46.5M listens, 9,238 users, 877,168 items, **301 days**.
- **Timestamp unit:** seconds rounded to multiples of 5 (README says "binned into 5s units"). A first run treated values as 5-second bins and got a 1,505-day span; fixed and rerun. Median gap between listens 145 s vs median track length 200 s confirms seconds.
- **Audio embeddings:** 7,721,749 tracks overall; cover 93.4% of 50M items and 96.9% of listen events. Unseen test targets with audio: 97.8%.
- **Split used for the check:** last 7 days = test, previous 7 days = calibration, the rest = train. First event per user after the cut is the target.
  - Calibration: 8,194 users; re-listen of an item the same user heard before 60.1%; item never in train 6.4% (523 users).
  - Test: 8,026 users; re-listen 51.4%; item never in train 10.1% (811 users); 17,823 distinct new items appear in the test week.
- **User activity:** train history quantiles 10/25/50/75/90% = 293/1,036/3,312/7,400/13,026. Very wide, good for activity groups.
- **Target popularity (test):** unseen 811, tail 1–10 (442), mid 11–100 (1,020), head > 100 (5,753).
- **Drift:** 15–17K first-appearing items per week at the end of the period (~1.5M events per week).

## Verdict per criterion

| Criterion | Amazon Beauty | Yambda-50M |
|---|---|---|
| Overall coverage | yes | yes |
| Coverage by item popularity | yes | yes (tail group only 442) |
| Coverage by user activity | weak (narrow range) | yes |
| Seen vs unseen items | temporal split only; 517 unseen | yes; 811 unseen in test, 523 in calibration |
| SID tree vs human category tree | yes (100% categorised) | no (no categories) |
| Coverage over time (drift) | weak (sparse windows) | yes (43 weeks) |

## Open points

- Yambda's popularity-tail group (442) and Amazon's unseen group (517) are small; report intervals, or pool several test windows.
- "Re-listen" here means the same user heard the item before (51–60%); the Meta Verifiers paper reports >70% under its own definition, not directly comparable.
- The Meta paper lists 0.63M items for Yambda-50M; the raw listens file has 877K. Their preprocessing is not known; ours applies no filtering yet.
