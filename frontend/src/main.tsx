import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  ExternalLink,
  FlaskConical,
  Github,
  LineChart as ChartIcon,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Wallet,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "@fontsource/manrope/latin-600.css";
import "@fontsource/manrope/latin-700.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "./styles.css";
import type { Example, Metadata, Prediction, Simulation } from "./types";

const API =
  (import.meta as unknown as { env: { VITE_API_URL?: string } }).env
    .VITE_API_URL || "";
const repo =
  "https://github.com/Abdelrhman-Ameen/egyptian-startup-intelligence-system";
const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const familyName = (v: string) =>
  ({
    logistic: "Logistic regression",
    hgb: "Gradient boosting",
    mlp: "PyTorch residual MLP",
    nested_selection: "Nested model selection",
    training_prevalence: "Prevalence baseline",
  })[v] || v;

async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(API + url, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : `Request failed (${response.status}). Check the values and try again.`,
    );
  }
  return response.json();
}

function download(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [page, setPage] = useState<"cash" | "lab" | "method">("cash");
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const connect = () => {
    setConnectionError("");
    api<Metadata>("/v1/model-info")
      .then(setMeta)
      .catch((e) => setConnectionError(e.message));
  };
  useEffect(connect, []);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("cash");
          }}
        >
          <span className="brand-icon">
            <Activity size={23} />
          </span>
          <span>
            VentureGuard<small>FINANCIAL WORKSPACE</small>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          <button
            className={page === "cash" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("cash")}
          >
            <Wallet size={19} />
            Cash planner
            <ChevronRight size={15} />
          </button>
          <button
            className={page === "lab" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("lab")}
          >
            <FlaskConical size={19} />
            Financial ML lab
            <ChevronRight size={15} />
          </button>
          <button
            className={page === "method" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("method")}
          >
            <BookOpen size={19} />
            Methodology
            <ChevronRight size={15} />
          </button>
        </nav>
        <div className="sidebar-note">
          <span className="tiny-label">BUILT TO BE INSPECTED</span>
          <p>
            Trace the assumptions.
            <br />
            Explore the evidence.
          </p>
          <a href={repo} target="_blank" rel="noreferrer">
            <Github size={16} />
            View the source
            <ArrowUpRight size={15} />
          </a>
        </div>
        <div className="sidebar-bottom">
          <span className="status-dot" />
          Research demo <span>v{meta?.version || "1.0.0"}</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            Egyptian Startup Intelligence System <ChevronRight size={14} />
            <strong>
              {page === "cash"
                ? "Cash planner"
                : page === "lab"
                  ? "Financial ML lab"
                  : "Methodology"}
            </strong>
          </span>
          <a href={`${API}/docs`} target="_blank" rel="noreferrer">
            API docs
            <ExternalLink size={14} />
          </a>
        </header>
        <main>
          {connectionError && (
            <div className="error-box" role="alert">
              The research service is unavailable. It may need a moment to
              start. <button onClick={connect}>Reconnect</button>
            </div>
          )}
          {page === "cash" ? (
            <CashPlanner />
          ) : page === "lab" ? (
            <ResearchLab meta={meta} />
          ) : (
            <Methodology meta={meta} />
          )}
        </main>
        <footer>
          <span>VentureGuard · DEPI project</span>
          <span>
            Monthly scenarios are conditional calculations. Research scores are
            source-domain estimates.
          </span>
        </footer>
      </div>
    </div>
  );
}

const defaults = {
  opening: 1000000,
  receipts: 200000,
  payments: 300000,
  floor: 200000,
  cut: 15,
  delay: 0,
  fx: 0,
  share: 40,
  start: 1,
  oneoff: 0,
};
function CashPlanner() {
  const [values, setValues] = useState(defaults);
  const [result, setResult] = useState<Simulation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stale, setStale] = useState(false);
  const [details, setDetails] = useState(false);
  const update = (key: keyof typeof defaults, value: number) => {
    setValues((v) => ({ ...v, [key]: value }));
    setStale(true);
  };
  async function calculate(v = values) {
    setBusy(true);
    setError("");
    try {
      const response = await api<Simulation>("/v1/simulate", {
        currency: "EGP",
        opening_cash: v.opening,
        cash_floor: v.floor,
        receipts: Array(12).fill(v.receipts),
        payments: Array(12).fill(v.payments),
        scenario: {
          expense_change_pct: -v.cut / 100,
          collection_delay_months: v.delay,
          fx_change_pct: v.fx / 100,
          fx_payment_share: v.share / 100,
          start_month: v.start,
          one_off_cost: v.oneoff,
        },
      });
      setResult(response);
      setStale(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    void calculate(defaults);
  }, []);
  const chart = result
    ? [
        {
          month: "Now",
          baseline: result.baseline.months[0].opening_cash,
          scenario: result.scenario.months[0].opening_cash,
        },
        ...result.baseline.months.map((r, i) => ({
          month: `M${r.month}`,
          baseline: r.closing_cash,
          scenario: result.scenario.months[i].closing_cash,
        })),
      ]
    : [];
  const displayFloor = result?.cash_floor ?? values.floor;
  const floorMonth = (value: number | null | undefined) =>
    value == null
      ? "Beyond 12 months"
      : value === 0
        ? "At opening"
        : `Month ${value}`;
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span />
            CASH-FLOW PLANNING
          </div>
          <h1>
            Give your next decision
            <br className="desktop-break" /> a clearer runway.
          </h1>
          <p>Change the assumptions. See when the cash picture changes.</p>
        </div>
        <button
          className="button secondary"
          onClick={() => {
            setValues(defaults);
            void calculate(defaults);
          }}
        >
          <RotateCcw size={15} />
          Load example
        </button>
      </div>
      <div className="planner-layout">
        <section className="panel input-panel" aria-label="Cash inputs">
          <div className="panel-heading">
            <span className="icon-label">
              <SlidersHorizontal size={18} />
              Your starting point
            </span>
            <span className="unit-tag">EGP / MONTH</span>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void calculate();
            }}
          >
            <MoneyInput
              label="Opening cash"
              name="opening"
              value={values.opening}
              onChange={(v) => update("opening", v)}
            />
            <MoneyInput
              label="Monthly cash receipts"
              name="receipts"
              value={values.receipts}
              onChange={(v) => update("receipts", v)}
            />
            <MoneyInput
              label="Monthly cash payments"
              name="payments"
              value={values.payments}
              onChange={(v) => update("payments", v)}
            />
            <MoneyInput
              label="Minimum cash buffer"
              name="floor"
              value={values.floor}
              onChange={(v) => update("floor", v)}
            />
            <div className="scenario-section">
              <div className="section-label">
                <span className="scenario-mark" />
                SCENARIO CHANGES
              </div>
              <label className="range-label" htmlFor="cut">
                Reduce payments<strong>{values.cut}%</strong>
              </label>
              <input
                id="cut"
                aria-label="Reduce payments"
                type="range"
                min="0"
                max="50"
                step="1"
                value={values.cut}
                onChange={(e) => update("cut", +e.target.value)}
              />
              <div className="range-ticks">
                <span>No reduction</span>
                <span>50%</span>
              </div>
              <label className="field" htmlFor="delay">
                Collection delay
                <select
                  id="delay"
                  value={values.delay}
                  onChange={(e) => update("delay", +e.target.value)}
                >
                  <option value="0">No delay</option>
                  <option value="1">1 month</option>
                  <option value="2">2 months</option>
                  <option value="3">3 months</option>
                </select>
              </label>
              <button
                type="button"
                className="text-button advanced-toggle"
                onClick={() => setDetails(!details)}
                aria-expanded={details}
              >
                {details ? "Hide" : "Show"} timing & currency assumptions
                <ChevronRight size={14} />
              </button>
              {details && (
                <div className="advanced-fields">
                  <label className="field">
                    Changes start in month
                    <input
                      aria-label="Scenario start month"
                      type="number"
                      min="1"
                      max="12"
                      value={values.start}
                      onChange={(e) => update("start", +e.target.value)}
                    />
                  </label>
                  <MoneyInput
                    label="One-off implementation cost"
                    name="oneoff"
                    value={values.oneoff}
                    onChange={(v) => update("oneoff", v)}
                  />
                  <label className="field">
                    FX cost increase (%)
                    <input
                      type="number"
                      min="0"
                      max="300"
                      value={values.fx}
                      onChange={(e) => update("fx", +e.target.value)}
                    />
                  </label>
                  <label className="field">
                    Payment share exposed to FX (%)
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={values.share}
                      onChange={(e) => update("share", +e.target.value)}
                    />
                  </label>
                </div>
              )}
            </div>
            <button
              className="button primary calculate"
              disabled={busy}
              type="submit"
            >
              {busy ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <ChartIcon size={17} />
              )}
              Compare cash paths
              <ArrowRight size={16} />
            </button>
            <p className="form-note">
              Example values are illustrative. Inputs are not saved.
            </p>
          </form>
        </section>
        <div className="results-column" aria-live="polite">
          {error && (
            <div className="error-box" role="alert">
              {error}
            </div>
          )}
          {stale && (
            <div className="stale-note">
              Inputs changed. Select “Compare cash paths” to refresh the
              results.
            </div>
          )}
          <div className="metric-grid">
            <div className="metric-card">
              <span>Cash after 12 months</span>
              <strong>
                {result ? number.format(result.scenario.closing_cash) : "—"}
                <small>EGP</small>
              </strong>
              <div className="metric-foot">
                <span className="scenario-dot" />
                Your scenario
              </div>
            </div>
            <div className="metric-card">
              <span>First month below buffer</span>
              <strong className="text-metric">
                {result ? floorMonth(result.scenario.first_below_floor) : "—"}
              </strong>
              <div className="metric-foot">
                Baseline:{" "}
                {result ? floorMonth(result.baseline.first_below_floor) : "—"}
              </div>
            </div>
            <div className="metric-card">
              <span>Difference at month 12</span>
              <strong
                className={
                  result && result.closing_cash_difference < 0
                    ? "negative"
                    : "positive"
                }
              >
                {result
                  ? `${result.closing_cash_difference >= 0 ? "+" : ""}${number.format(result.closing_cash_difference)}`
                  : "—"}
                <small>EGP</small>
              </strong>
              <div className="metric-foot">Compared with unchanged inputs</div>
            </div>
          </div>
          <section className="panel chart-panel">
            <div className="chart-header">
              <div>
                <h2>Two paths. One cash position.</h2>
                <p>Monthly closing cash over a 12-month horizon</p>
              </div>
              <div className="legend">
                <span>
                  <i className="baseline-line" />
                  Baseline
                </span>
                <span>
                  <i className="scenario-line" />
                  Scenario
                </span>
              </div>
            </div>
            <div
              className="chart-wrap"
              role="img"
              aria-label="Line chart comparing baseline and scenario monthly cash balances"
            >
              {result ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chart}
                    margin={{ top: 18, right: 14, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid stroke="#e8edf3" vertical={false} />
                    <XAxis
                      dataKey="month"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#748396", fontSize: 11 }}
                      minTickGap={14}
                    />
                    <YAxis
                      width={60}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#748396", fontSize: 11 }}
                      tickFormatter={(v) =>
                        `${v < 0 ? "−" : ""}${Math.abs(v) / 1000}k`
                      }
                    />
                    <Tooltip
                      formatter={(v) => `${number.format(Number(v))} EGP`}
                      contentStyle={{
                        border: "1px solid #dce3eb",
                        borderRadius: 8,
                        fontSize: 13,
                      }}
                    />
                    <ReferenceLine
                      y={displayFloor}
                      stroke="#b56c11"
                      strokeDasharray="4 5"
                    />
                    <ReferenceLine y={0} stroke="#a9b5c4" />
                    <Line
                      name="Baseline"
                      type="linear"
                      dataKey="baseline"
                      stroke="#9aabba"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      name="Scenario"
                      type="linear"
                      dataKey="scenario"
                      stroke="#5065c9"
                      strokeWidth={3}
                      dot={{ r: 3, fill: "#fff", strokeWidth: 2 }}
                      activeDot={{ r: 5 }}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="loading-placeholder">
                  <LoaderCircle className="spin" />
                  Calculating the example…
                </div>
              )}
            </div>
            <div className="buffer-caption">
              <span className="buffer-line" />
              Cash buffer: {number.format(displayFloor)} EGP
              <span>Balances shown at month end</span>
            </div>
          </section>
          <section className="runway-strip">
            <div className="runway-label">
              <span>SCENARIO BUFFER</span>
              <small>Each segment is a month</small>
            </div>
            <div className="runway-segments">
              {result?.scenario.months.map((m) => (
                <div
                  key={m.month}
                  className={
                    m.closing_cash <= 0
                      ? "segment gap"
                      : m.closing_cash < displayFloor
                        ? "segment caution"
                        : "segment healthy"
                  }
                  title={`Month ${m.month}: ${number.format(m.closing_cash)} EGP`}
                >
                  <span>{m.month}</span>
                </div>
              ))}
            </div>
            <span className="strip-key">
              <span className="status-dot" />
              Above buffer
            </span>
          </section>
          <div className="assumption-note">
            <CircleHelp size={18} />
            <div>
              <strong>The assumptions are part of the answer.</strong>
              <p>
                Receipts stay fixed unless delayed. Payment cuts do not
                automatically change sales. A cash gap is a financing need, not
                a bankruptcy prediction.
              </p>
            </div>
          </div>
          {result && (
            <details className="monthly-details">
              <summary>
                Inspect monthly cash movements
                <span>
                  12 months
                  <ChevronRight size={15} />
                </span>
              </summary>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Month</th>
                      <th>Opening</th>
                      <th>Receipts</th>
                      <th>Payments + one-off</th>
                      <th>Closing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.scenario.months.map((m) => (
                      <tr key={m.month}>
                        <td>{m.month}</td>
                        <td>{number.format(m.opening_cash)}</td>
                        <td>{number.format(m.receipts)}</td>
                        <td>{number.format(m.payments + m.one_off_cost)}</td>
                        <td>{number.format(m.closing_cash)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="detail-note">
                Deferred beyond this horizon:{" "}
                {number.format(
                  result.scenario.deferred_receipts_beyond_horizon,
                )}{" "}
                EGP. Negative balances require additional financing.
              </p>
              <button
                className="text-button"
                onClick={() =>
                  download("ventureguard-cash-scenario.json", result)
                }
              >
                <Download size={15} />
                Download calculation
              </button>
            </details>
          )}
        </div>
      </div>
    </>
  );
}

function MoneyInput({
  label,
  name,
  value,
  onChange,
}: {
  label: string;
  name: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="field" htmlFor={name}>
      {label}
      <div className="money-input">
        <span>EGP</span>
        <input
          id={name}
          name={name}
          type="number"
          min="0"
          max="1000000000000"
          step="any"
          required
          value={value}
          onChange={(e) => onChange(+e.target.value)}
        />
      </div>
    </label>
  );
}

function ResearchLab({ meta }: { meta: Metadata | null }) {
  const [examples, setExamples] = useState<Example[]>([]);
  const [selected, setSelected] = useState("");
  const [schema, setSchema] = useState("core13");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    api<Example[]>("/v1/examples")
      .then((v) => {
        setExamples(v);
        setSelected(v[0]?.id || "");
      })
      .catch((e) => setError(e.message));
  }, []);
  async function evaluate() {
    setBusy(true);
    setError("");
    setPrediction(null);
    try {
      setPrediction(
        await api<Prediction>("/v1/benchmark/predict", {
          schema_name: schema,
          example_id: selected,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const metrics = meta?.results.find(
    (r) => r.schema === schema && r.family === "nested_selection",
  );
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span />
            FINANCIAL MACHINE LEARNING
          </div>
          <h1>
            Inspect the score.
            <br />
            Understand the limits.
          </h1>
          <p>
            A reproducible experiment on historical Polish financial statements.
          </p>
        </div>
        <span className="research-tag">
          <FlaskConical size={15} />
          Research benchmark
        </span>
      </div>
      <div className="scope-note">
        <ShieldCheck size={20} />
        <p>
          <strong>Source-domain research only.</strong> These models have not
          been validated for Egyptian startups. Historical examples are scored
          by an outer-fold model that excluded the case from fitting and
          selection.
        </p>
      </div>
      <div className="lab-layout">
        <section className="panel lab-controls">
          <div className="panel-heading">
            <span className="icon-label">
              <Database size={18} />
              Choose a historical case
            </span>
          </div>
          <label className="field">
            Example
            <select
              aria-label="Historical example"
              value={selected}
              onChange={(e) => {
                setSelected(e.target.value);
                setPrediction(null);
              }}
            >
              {examples.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.title} · {v.id}
                </option>
              ))}
            </select>
          </label>
          <div className="field">Financial information</div>
          <div className="schema-switch">
            <button
              className={schema === "core13" ? "selected" : ""}
              onClick={() => {
                setSchema("core13");
                setPrediction(null);
              }}
            >
              <strong>13 ratios</strong>
              <small>Compact schema</small>
            </button>
            <button
              className={schema === "full64" ? "selected" : ""}
              onClick={() => {
                setSchema("full64");
                setPrediction(null);
              }}
            >
              <strong>64 ratios</strong>
              <small>Full schema</small>
            </button>
          </div>
          <p className="help-text">
            Compare information availability. Ratio count does not directly
            measure data-entry effort.
          </p>
          <button
            className="button primary"
            disabled={busy || !selected}
            onClick={() => void evaluate()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <FlaskConical size={17} />
            )}
            Evaluate example
            <ArrowRight size={16} />
          </button>
          <div className="mini-method">
            <span className="tiny-label">EVALUATION DESIGN</span>
            <p>
              <Check size={14} />5 outer / 3 inner folds
            </p>
            <p>
              <Check size={14} />
              Exact-vector groups kept together
            </p>
            <p>
              <Check size={14} />
              Calibration selected inside training
            </p>
          </div>
        </section>
        <div className="results-column" aria-live="polite">
          {error && (
            <div className="error-box" role="alert">
              {error}
            </div>
          )}
          <div className="metric-grid">
            <div className="metric-card">
              <span>Average precision</span>
              <strong>{metrics?.average_precision.toFixed(3) || "—"}</strong>
              <div className="metric-foot">
                Outer-fold mean · higher is better
              </div>
            </div>
            <div className="metric-card">
              <span>Brier score</span>
              <strong>{metrics?.brier.toFixed(4) || "—"}</strong>
              <div className="metric-foot">
                Outer-fold mean · lower is better
              </div>
            </div>
            <div className="metric-card">
              <span>Recall in top 10%</span>
              <strong>{metrics ? pct(metrics.recall_at_10pct) : "—"}</strong>
              <div className="metric-foot">Within each held-out fold</div>
            </div>
          </div>
          {prediction ? (
            <>
              <section className="panel prediction-panel">
                <div>
                  <span className="tiny-label">HISTORICAL MODEL SCORE</span>
                  <div className="score-number">
                    {prediction.score < 0.001
                      ? "<0.001"
                      : prediction.score > 0.999
                        ? ">0.999"
                        : prediction.score.toFixed(3)}
                  </div>
                  <span className="score-scale">
                    Scale 0–1 ·{" "}
                    {prediction.calibrated
                      ? "sigmoid-calibrated within source"
                      : "uncalibrated score"}
                  </span>
                  <div className="score-meter">
                    <span style={{ width: `${prediction.score * 100}%` }} />
                  </div>
                </div>
                <div className="prediction-facts">
                  <div>
                    <span>Observed historical outcome</span>
                    <strong>
                      {prediction.observed_outcome === 1
                        ? "Bankruptcy recorded"
                        : "No bankruptcy recorded"}
                    </strong>
                  </div>
                  <div>
                    <span>Selected model</span>
                    <strong>{familyName(prediction.model_family)}</strong>
                  </div>
                  <div>
                    <span>Input diagnostics</span>
                    <strong>
                      {prediction.missing_fields.length} missing ·{" "}
                      {prediction.outside_reference_range.length} outside
                      reference range
                    </strong>
                  </div>
                </div>
                <p className="prediction-disclaimer">
                  This score describes the historical source task. It is not a
                  personal bankruptcy probability or a confidence estimate.
                </p>
              </section>
              <section className="panel feature-panel">
                <div className="panel-heading">
                  <span>Which information mattered globally?</span>
                </div>
                <p className="help-text">
                  Held-out permutation relevance, averaged across folds. These
                  are global features, not local reasons for this case.
                </p>
                {prediction.important_features.map((v) => (
                  <div className="feature-row" key={v.feature}>
                    <span>
                      <strong>{v.label}</strong>
                      <small>
                        {v.feature} · case value:{" "}
                        {v.value == null ? "Missing" : v.value.toPrecision(4)}
                      </small>
                    </span>
                    <div className="feature-bar">
                      <i
                        style={{
                          width: `${Math.min(100, Math.max(2, v.mean_heldout_ap_decrease * 500))}%`,
                        }}
                      />
                    </div>
                    <code>{v.mean_heldout_ap_decrease.toFixed(3)}</code>
                  </div>
                ))}
                <p className="detail-note">
                  AP decrease after permutation. Correlated ratios can share
                  relevance. No causal effects are established.
                </p>
                <button
                  className="text-button"
                  onClick={() =>
                    download("ventureguard-research-result.json", prediction)
                  }
                >
                  <Download size={15} />
                  Download result and inputs
                </button>
              </section>
            </>
          ) : (
            <section className="panel empty-state">
              <div className="empty-icon">
                <FlaskConical size={30} />
              </div>
              <h2>A case, not a claim.</h2>
              <p>
                Select a historical example to inspect its score, observed
                outcome and input diagnostics.
              </p>
              <span>The outcome is revealed after evaluation.</span>
            </section>
          )}
        </div>
      </div>
    </>
  );
}

function Methodology({ meta }: { meta: Metadata | null }) {
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span />
            METHODS & EVIDENCE
          </div>
          <h1>The work behind the workspace.</h1>
          <p>What was trained, what was measured, and what remains unproven.</p>
        </div>
      </div>
      <div className="method-overview">
        <section className="panel">
          <Database size={24} />
          <h2>The data</h2>
          <p>
            UCI Polish Companies Bankruptcy, fifth file:{" "}
            {meta ? number.format(meta.source.rows) : "5,910"} financial
            statements and {meta?.source.positive || 410} recorded bankruptcy
            outcomes after one year.
          </p>
          <a
            className="text-button"
            href="https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data"
            target="_blank"
            rel="noreferrer"
          >
            Original source
            <ExternalLink size={14} />
          </a>
        </section>
        <section className="panel">
          <FlaskConical size={24} />
          <h2>The experiment</h2>
          <p>
            Logistic regression, gradient boosting and a PyTorch residual MLP.
            Matched grouped folds, train-only transformations,
            weighted/unweighted candidates and calibration comparisons.
          </p>
        </section>
        <section className="panel">
          <CircleHelp size={24} />
          <h2>The boundary</h2>
          <p>
            Internal evaluation after prior exploration. No verified
            company/time split, independent country test or validated Egyptian
            startup probabilities. Cash scenarios are conditional accounting.
          </p>
        </section>
      </div>
      <section className="panel comparison-panel">
        <div className="panel-heading">
          <span>Outer-fold model comparison</span>
          <span className="unit-tag">MEANS · 5 FOLDS</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Schema / experiment</th>
                <th>Model</th>
                <th>AP ↑</th>
                <th>AP SD</th>
                <th>ROC-AUC ↑</th>
                <th>Brier ↓</th>
              </tr>
            </thead>
            <tbody>
              {meta?.results.map((r) => (
                <tr
                  key={`${r.schema}-${r.family}`}
                  className={
                    r.family === "nested_selection" ? "selection-row" : ""
                  }
                >
                  <td>
                    {{
                      core13: "13 ratios",
                      full64: "64 ratios",
                      missingness_only: "Missingness only",
                      full64_without_explicit_indicators:
                        "64 ratios · no explicit masks",
                    }[r.schema] || r.schema}
                  </td>
                  <td>{familyName(r.family)}</td>
                  <td>{r.average_precision.toFixed(3)}</td>
                  <td>{r.average_precision_sd.toFixed(3)}</td>
                  <td>{r.roc_auc.toFixed(3)}</td>
                  <td>{r.brier.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="detail-note">
          SD is variability across folds, not a confidence interval. Nested
          selection chooses the family using inner-fold AP. Removing explicit
          masks does not guarantee removal of missingness signals.
        </p>
      </section>
      <section className="panel stress-panel">
        <div className="panel-heading">
          <span>What happens when more inputs are missing?</span>
          <span className="unit-tag">HELD-OUT STRESS TEST</span>
        </div>
        <p className="help-text">
          The full model performs better on source-like inputs, but artificial
          extra missingness sharply reduces its performance. A higher source
          benchmark score does not establish robustness.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Schema</th>
                <th>Added random masking</th>
                <th>Mean AP ↑</th>
                <th>Mean Brier ↓</th>
              </tr>
            </thead>
            <tbody>
              {meta?.stress_summary?.map((r) => (
                <tr key={`${r.schema}-${r.fraction_randomly_masked}`}>
                  <td>{r.schema === "core13" ? "13 ratios" : "64 ratios"}</td>
                  <td>{pct(r.fraction_randomly_masked)}</td>
                  <td>{r.average_precision.toFixed(3)}</td>
                  <td>{r.brier.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="detail-note">
          Independent artificial masking without refitting. This does not
          reproduce all real missing-data mechanisms or establish external
          validity.
        </p>
      </section>
      <div className="method-bottom">
        <div>
          <h2>Reproduce. Inspect. Extend.</h2>
          <p>
            The repository includes the training protocol, out-of-fold
            predictions, PyTorch export checks, cash accounting tests and
            deployment instructions. RAG integration is deferred to a future
            release.
          </p>
        </div>
        <a
          className="button secondary"
          href={repo}
          target="_blank"
          rel="noreferrer"
        >
          <Github size={17} />
          Open repository
          <ArrowUpRight size={16} />
        </a>
      </div>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
