# Mathematical decisions behind the implementation

## Ratios and scale

Ratios such as net profit / total assets remove absolute currency scale but do not remove differences in accounting definitions or populations. A zero denominator makes a ratio undefined; treating it as zero would change its meaning. Signed-log maps `x` to `sign(x) log(1 + abs(x))`, retaining negative values while reducing the effect of extreme magnitudes. Its parameters are fixed; imputation and standardization are fitted only on training rows.

## Logistic and neural classification

The linear baseline computes a logit `z = wᵀx + b` and score `sigmoid(z) = 1 / (1 + exp(-z))`. Binary cross entropy is `-y log(p) - (1-y) log(1-p)`. PyTorch's BCEWithLogitsLoss evaluates the equivalent expression stably from logits. AdamW updates the neural weights; weight decay and dropout limit the fitted model's flexibility.

Class weighting changes the optimized objective. Under idealized weighted log loss, weights `w1`, `w0` transform a population probability `p` to `w1*p / (w1*p + w0*(1-p))`. Therefore a balanced-loss output is not automatically a calibrated population probability. The experiment compares weighted/unweighted fits and evaluates calibration separately.

## Ranking and calibration

Average precision summarizes precision along the recall curve and is the primary selection metric for this imbalanced benchmark. ROC-AUC measures pairwise ranking; neither metric establishes probability calibration. Brier score averages `(p-y)²`; log loss penalizes confidently wrong estimates. A sigmoid mapping is learned from out-of-fold training scores, and complete procedures are evaluated outside outer training. Calibration within a historical source does not prove correctness under a different country's distribution.

The top-10% review metric sorts each held-out fold separately, then measures the fraction of its positive outcomes in the selected tenth. It describes a review-capacity experiment, not the effect of taking action on those companies.

## Cash conservation

For month `t`, `closing[t] = opening[t] + receipts[t] + funding[t] - payments[t] - one_off[t]`, and `opening[t+1] = closing[t]`. Monetary entries are rounded to cents using Decimal, with half-up rounding. Delayed collections move to later months; collections outside the horizon are separately tracked. A negative balance denotes financing required under the assumptions, not an observed legal bankruptcy event.

The UI's FX scenario multiplies changed payments by `1 + exposed_share * fx_change`. No sales elasticity, causal effect, or distribution of future macro shocks is learned.
