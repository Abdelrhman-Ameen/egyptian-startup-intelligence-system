export type Metric = {
  schema: string;
  family: string;
  average_precision: number;
  average_precision_sd: number;
  roc_auc: number;
  brier: number;
  recall_at_10pct: number;
};
export type Metadata = {
  version: string;
  results: Metric[];
  released_models: Record<
    string,
    { family: string; calibration_used: boolean }
  >;
  source: { rows: number; positive: number; source_sha256: string };
  limitations: string[];
  stress_summary?: {
    schema: string;
    fraction_randomly_masked: number;
    average_precision: number;
    brier: number;
  }[];
};
export type Example = {
  id: string;
  title: string;
  outcome: number;
  fold: number;
};
export type Prediction = {
  schema: string;
  score: number;
  calibrated: boolean;
  model_family: string;
  observed_outcome: number | null;
  missing_fields: string[];
  outside_reference_range: string[];
  evaluation_mode: string;
  important_features: {
    feature: string;
    label: string;
    value: number | null;
    mean_heldout_ap_decrease: number;
  }[];
  features: Record<string, number | null>;
  scope: string;
  explanation_note: string;
};
export type Month = {
  month: number;
  opening_cash: number;
  receipts: number;
  payments: number;
  one_off_cost: number;
  funding: number;
  closing_cash: number;
};
export type Schedule = {
  months: Month[];
  closing_cash: number;
  first_below_floor: number | null;
  first_nonpositive: number | null;
  minimum_cash: number;
  financing_gap_to_zero: number;
  deferred_receipts_beyond_horizon: number;
};
export type Simulation = {
  currency: string;
  cash_floor: number;
  baseline: Schedule;
  scenario: Schedule;
  closing_cash_difference: number;
  assumptions: string[];
};
