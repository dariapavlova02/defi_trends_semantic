# Methodology

## Research question

Does protocol context known before a DeFi security incident improve the ranking of incidents by
severe financial loss? The portfolio experiment deliberately narrows the question to information
that can be shown to have existed at prediction time.

## Profiles

| Profile | Purpose | Threshold | Graph-feature policy |
|---|---|---|---|
| `demo` | End-to-end smoke test on synthetic data | Training period | Timestamp required |
| `portfolio` | Corrected primary evaluation | Training period | Timestamp required |
| `conference` | Archival paper reproduction | Full dataset | Retrospective facts allowed and labelled |

The `conference` profile does not silently substitute synthetic data. It requires a separately
prepared dataset because the original research graph is not redistributed here.

## Target and split

Rows are sorted by `incident_date` and split 75/25. For the primary portfolio profile, the severe
loss threshold is the 75th percentile of training-period losses and remains fixed for the test
period. Five expanding-window splits estimate temporal stability. No shuffled cross-validation is
used in the corrected evaluation.

## Feature isolation

The baseline contains temporal and categorical incident features. The graph-enriched model begins
with the same columns and adds only eligible graph features. This makes the comparison attributable
to graph context rather than to different conventional inputs.

An input graph value is eligible only when:

1. it has a non-null `*_available_at` timestamp;
2. that timestamp is no later than the incident timestamp; and
3. the feature contains at least one observed value.

The current corrected dataset admits `protocol_past_events_count`, computed using strictly earlier
incidents for the same protocol. Multi-chain count, fork status, and fork-child count are excluded
because the available source export does not establish when those facts became known.

## Model and metrics

Both feature sets use the same deterministic LightGBM configuration and preprocessing. Numeric
missing values are median-imputed. Categorical values are imputed and one-hot encoded with unknown
categories tolerated at inference time. The evaluation reports ROC AUC, F1, precision, recall, and
confusion matrices at a 0.5 decision threshold.

## Interpretation

The conference values and corrected values answer different questions and must not be mixed. The
corrected holdout does not demonstrate improvement from the currently timestamp-safe graph feature.
Temporal CV suggests a small mean improvement, but its variance and threshold sensitivity preclude a
strong predictive claim.

