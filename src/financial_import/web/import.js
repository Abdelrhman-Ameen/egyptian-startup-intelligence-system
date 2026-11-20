"use strict";

const $ = (id) => document.getElementById(id);
const api = "/api/financial-statements";
let fields = {};
let sheets = [];
let staged = [];
let mappings = [];
let previewName = "";
let previewRun = 0;

function node(tag, value = "") {
  const elem = document.createElement(tag);
  elem.textContent = value;
  return elem;
}

function status(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle("error", error);
}

function safeCell(value) {
  return value && typeof value === "object" ? value.value : value;
}

function asLabel(s) {
  return String(s ?? "").normalize("NFKC").toLowerCase()
    .replace(/[أإآ]/g, "ا").replace(/[\u064B-\u065F\u0640]/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

function suggestedField(label) {
  const normalized = asLabel(label);
  return Object.keys(fields).find((key) =>
    [key, fields[key].label, ...fields[key].aliases].some((alias) => asLabel(alias) === normalized)
  ) ?? "";
}

async function request(path, options) {
  const response = await fetch(api + path, options);
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof result?.detail === "string" ? result.detail : "الطلب ماكملش. جرّب مرة تانية.");
  }
  return result;
}

function choice(select, items) {
  select.replaceChildren(...items.map(([value, name]) => {
    const option = node("option", name);
    option.value = value;
    return option;
  }));
}

function letter(index) {
  return index < 26 ? String.fromCharCode(65 + index) : "A" + letter(index - 26);
}

function selectedRows() {
  return sheets[Number($("sheet").value)]?.rows ?? [];
}

function configureAxes() {
  const rows = selectedRows();
  const byRows = $("orientation").value === "rows";
  const width = rows.reduce((max, row) => Math.max(max, row.length), 0);
  const axis = byRows ? width : rows.length;
  const axisChoices = Array.from({length: axis}, (_, index) => [
    index,
    byRows ? letter(index) : String(index + 1),
  ]);
  choice($("label-axis"), axisChoices);
  choice($("value-axis"), axisChoices);
  $("value-axis").value = axis > 1 ? "1" : "0";
  $("first").value = "2";
  $("first").max = byRows ? rows.length : width;
  $("labels-label").textContent = byRows ? "عمود أسماء البنود" : "صف أسماء البنود";
  $("values-label").textContent = byRows ? "عمود مبالغ الفترة المختارة" : "صف مبالغ الفترة المختارة";
  $("first-label").textContent = byRows ? "أول صف للبنود" : "رقم أول عمود للبنود";
  showGrid(rows, width);
  showCandidates();
}

function showGrid(rows, width) {
  const table = node("table");
  const head = node("tr");
  head.append(node("th", "الصف"));
  for (let col = 0; col < width; col++) head.append(node("th", letter(col)));
  const thead = node("thead");
  thead.append(head);
  table.append(thead);
  const body = node("tbody");
  rows.slice(0, 8).forEach((row, index) => {
    const tr = node("tr");
    tr.append(node("td", index + 1));
    for (let col = 0; col < width; col++) tr.append(node("td", String(safeCell(row[col]) ?? "")));
    body.append(tr);
  });
  table.append(body);
  $("grid").replaceChildren(table);
}

function showCandidates() {
  const rows = selectedRows();
  const byRows = $("orientation").value === "rows";
  const labels = Number($("label-axis").value);
  const values = Number($("value-axis").value);
  const first = Number($("first").value) - 1;
  mappings = [];
  $("candidates").replaceChildren();
  $("stage").disabled = true;
  if (labels === values || !Number.isInteger(first) || first < 0) {
    status("map-status", "اختار مكانين مختلفين لأسماء البنود والمبالغ، وحدد أول بند.", true);
    return;
  }

  const size = byRows ? rows.length : Math.max(rows[labels]?.length ?? 0, rows[values]?.length ?? 0);
  for (let i = first; i < size; i++) {
    const label = String(safeCell(byRows ? rows[i]?.[labels] : rows[labels]?.[i]) ?? "").trim();
    if (!label) continue;
    const cell = byRows ? rows[i]?.[values] : rows[values]?.[i];
    const amount = String(safeCell(cell) ?? "").trim();
    const field = suggestedField(label);
    const address = byRows ? letter(values) + (i + 1) : letter(i) + (values + 1);
    const tr = node("tr");
    const source = node("td", label);
    source.append(node("small", address));
    const raw = node("td", amount || "—");
    if (cell?.status === "formula_cached") raw.append(node("small", "قيمة معادلة محفوظة"));
    if (cell?.status === "formula_missing") raw.append(node("small", "المعادلة محتاجة إعادة حساب وحفظ في Excel"));
    if (cell?.status === "error") raw.append(node("small", "فيه خطأ في الخلية. راجع الملف"));
    const fieldCell = node("td");
    const select = node("select");
    select.setAttribute("aria-label", "البند المقابل لـ" + label);
    choice(select, [["", "تخطّي البند"], ...Object.entries(fields).map(([key, spec]) => [key, spec.label])]);
    const eligible = Boolean(amount) && cell?.status !== "error";
    select.disabled = !eligible;
    if (eligible) {
      select.value = field;
      mappings.push({select, amount, address, label});
    }
    fieldCell.append(select);
    tr.append(source, raw, fieldCell);
    $("candidates").append(tr);
  }
  $("stage").disabled = !mappings.length;
  status("map-status", mappings.length ? "" : "مفيش مبالغ قابلة للاختيار هنا. راجع مكان البنود والأرقام.");
}

$("file").addEventListener("change", () => {
  previewRun++;
  $("separator-control").hidden = !/\.csv$/i.test($("file").files[0]?.name ?? "");
  $("mapping").hidden = true;
  $("preview-button").disabled = false;
  status("upload-status", "");
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  const format = file.name.split(".").at(-1).toLowerCase();
  if (!["csv", "xlsx"].includes(format) || file.size > 2 * 1024 * 1024) {
    status("upload-status", "اختار XLSX أو CSV بحجم لحد 2 MB.", true);
    return;
  }
  const run = ++previewRun;
  $("preview-button").disabled = true;
  $("mapping").hidden = true;
  status("upload-status", "جاري قراءة الملف…");
  try {
    if (!Object.keys(fields).length) fields = await request("/fields");
    const query = new URLSearchParams({format});
    if (format === "csv" && $("separator").value) query.set("separator", $("separator").value);
    const preview = await request("/preview?" + query, {
      method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file,
    });
    if (run !== previewRun) return;
    sheets = preview.sheets;
    previewName = file.name;
    choice($("sheet"), sheets.map((sheet, index) => [index, sheet.name]));
    $("orientation").value = "rows";
    $("scale").value = "";
    $("mapping").hidden = false;
    configureAxes();
    status("upload-status", "اتقرى الملف. اختار الورقة والفترة اللي عايز تستخدمها.");
  } catch (error) {
    if (run === previewRun) status("upload-status", error.message, true);
  } finally {
    if (run === previewRun) $("preview-button").disabled = false;
  }
});

$("sheet").addEventListener("change", () => {
  $("scale").value = "";
  configureAxes();
});
$("orientation").addEventListener("change", configureAxes);
for (const id of ["label-axis", "value-axis", "first"]) $(id).addEventListener("change", showCandidates);

function invalidateResult() {
  $("results").hidden = true;
  status("confirm-status", "");
}

function showReview() {
  $("empty").hidden = staged.length > 0;
  $("review-list").hidden = staged.length === 0;
  $("confirm-form").hidden = staged.length === 0;
  $("agreed").checked = false;
  invalidateResult();
  $("review").replaceChildren(...staged.map((item, index) => {
    const row = node("tr");
    const value = node("td");
    const amount = node("bdi", item.amount);
    amount.className = "number";
    value.append(amount, node("small", {1: "وحدة العملة", 1000: "بالآلاف", 1000000: "بالملايين"}[item.scale]));
    const source = node("td", item.source);
    source.append(node("small", item.label));
    const actions = node("td");
    const remove = node("button", "إزالة");
    remove.type = "button";
    remove.className = "link-button";
    remove.setAttribute("aria-label", "إزالة " + fields[item.field].label);
    remove.addEventListener("click", () => { staged.splice(index, 1); showReview(); });
    actions.append(remove);
    row.append(node("td", fields[item.field].label), value, source, actions);
    return row;
  }));
}

$("stage").addEventListener("click", () => {
  const scale = Number($("scale").value);
  if (!scale) {
    status("map-status", "اختار إذا كانت الأرقام بالوحدة أو بالآلاف أو بالملايين قبل إضافتها.", true);
    return;
  }
  const additions = mappings.filter((row) => row.select.value).map((row) => ({
    field: row.select.value, amount: row.amount, scale,
    source: [previewName, sheets[Number($("sheet").value)].name, row.address].join(" · "),
    label: row.label,
  }));
  if (!additions.length) {
    status("map-status", "اختار بند مالي واحد على الأقل من القوائم.", true);
    return;
  }
  const used = new Set(staged.map((item) => item.field));
  for (const item of additions) {
    if (used.has(item.field)) {
      status("map-status", "فيه قيمتين لبند " + fields[item.field].label + ". اختار قيمة واحدة.", true);
      return;
    }
    used.add(item.field);
  }
  staged.push(...additions);
  showReview();
  status("map-status", "اتضافت البنود للمراجعة. تقدر تختار ورقة تانية أو ترفع ملف تاني.");
});

for (const id of ["currency", "period-start", "period-end"]) {
  $(id).addEventListener("input", () => { $("agreed").checked = false; invalidateResult(); });
}
$("agreed").addEventListener("change", invalidateResult);

$("confirm-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  invalidateResult();
  const statement = () => ({
    currency: $("currency").value.toUpperCase(),
    period_start: $("period-start").value,
    period_end: $("period-end").value,
    items: staged.map(({field, amount, scale}) => ({field, amount, scale})),
  });
  const payload = JSON.stringify(statement());
  $("confirm-button").disabled = true;
  try {
    const review = await request("/confirm", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: payload,
    });
    if (payload !== JSON.stringify(statement()) || !$("agreed").checked) return;
    $("period-summary").textContent = review.currency + " · " + review.period_start + " إلى " + review.period_end;
    $("warnings").hidden = !review.warnings.length;
    $("warnings").replaceChildren(...review.warnings.map((warning) => node("li", warning)));
    $("ratios").replaceChildren(...Object.entries(review.features).map(([key, entry]) => {
      const row = node("tr");
      const title = node("td", entry.label);
      title.append(node("small", key));
      const value = node("td", entry.value === null ? "—" : Number(entry.value.toPrecision(6)).toString());
      value.className = "number";
      const note = entry.zero_denominator
        ? "المقام صفر: " + fields[entry.zero_denominator].label
        : entry.missing_fields.length ? "ناقص: " + entry.missing_fields.map((field) => fields[field].label).join("، ") : "اتحسب من البنود المراجعة";
      row.append(title, value, node("td", note));
      return row;
    }));
    $("results").hidden = false;
    status("confirm-status", "اتحسبت النسب المتاحة من البنود المؤكدة.");
  } catch (error) {
    status("confirm-status", error.message, true);
  } finally {
    $("confirm-button").disabled = false;
  }
});
