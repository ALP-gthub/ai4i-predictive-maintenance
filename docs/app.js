// Page logic for the AI4I 2020 demo. The model itself is in model.js / model.json.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fields = ["air", "process", "speed", "torque", "wear"];

  // Example inputs. Each one triggers the named documented condition.
  const PRESETS = {
    normal: { type: "L", air: 300.0, process: 310.0, speed: 1500, torque: 40.0, wear: 100 },
    heat: { type: "L", air: 302.0, process: 310.2, speed: 1300, torque: 55.0, wear: 20 },
    strain: { type: "M", air: 300.0, process: 310.0, speed: 1400, torque: 60.0, wear: 210 },
    power: { type: "H", air: 300.0, process: 310.0, speed: 2800, torque: 10.0, wear: 50 },
  };

  let model = null;

  const fmt = (x, d) =>
    Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x) => fmt(x * 100, 1) + " %";

  function setInputs(p) {
    $("type").value = p.type;
    fields.forEach((f) => ($(f).value = p[f]));
  }

  function readInputs() {
    const input = { type: $("type").value };
    for (const f of fields) {
      const raw = $(f).value.trim();
      const v = Number(raw);
      if (raw === "" || !Number.isFinite(v)) return null;
      input[f] = v;
    }
    return input;
  }

  function statusCell(triggered) {
    return triggered
      ? '<span class="status trig">✕ Triggered</span>'
      : '<span class="status ok">✓ Not met</span>';
  }

  function predict(event) {
    if (event) event.preventDefault();
    const input = readInputs();
    const error = $("error");
    if (!input) {
      error.textContent = "Please fill in every field with a number.";
      error.hidden = false;
      return;
    }
    error.hidden = true;

    const p = AI4I.predictProbability(model, input);
    const c = AI4I.documentedChecks(input);
    const fail = p >= 0.5;

    $("prob").textContent = pct(p);
    $("bar").style.width = Math.min(100, p * 100) + "%";
    $("bar").style.background = fail ? "var(--bad)" : "var(--accent)";
    const badge = $("badge");
    badge.textContent = fail ? "Failure expected" : "No failure expected";
    badge.className = fail ? "fail" : "ok";

    $("checks").innerHTML = [
      ["Heat dissipation (HDF)", `ΔT ${fmt(c.e.tempDiff, 1)} K, ${fmt(input.speed, 0)} rpm`, "ΔT &lt; 8.6 K and speed &lt; 1,380 rpm", c.hdf],
      ["Power (PWF)", `${fmt(c.e.power, 0)} W`, "below 3,500 W or above 9,000 W", c.pwf],
      ["Overstrain (OSF)", `${fmt(c.e.strain, 0)} min·Nm`, `above ${fmt(c.osfLimit, 0)} min·Nm for type ${input.type}`, c.osf],
    ]
      .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${statusCell(r[3])}</td></tr>`)
      .join("");
  }

  const NAMES = {
    logistic_raw: "Logistic regression, raw measurements",
    boosting_raw: "Gradient boosting, raw measurements",
    boosting_engineered: "Gradient boosting, raw + engineered features (this page)",
    documented_rules: "Documented physical rules (reference, no learning)",
  };

  function renderStudy(metrics, check) {
    const s = metrics.split;
    $("split").textContent =
      `Held-out test set of ${fmt(s.test_rows, 0)} rows (${s.test_failures} actual failures), ` +
      `not used in training. Stratified split, seed ${s.seed}.`;
    $("results").innerHTML = Object.keys(NAMES)
      .map((k) => {
        const t = metrics.test[k];
        const cls = k === "boosting_engineered" ? ' class="best"' : "";
        return `<tr${cls}><td>${NAMES[k]}</td><td>${pct(t.precision)}</td><td>${pct(t.recall)}</td>` +
          `<td>${fmt(t.f1, 3)}</td><td>${t.fn}</td><td>${t.fp}</td></tr>`;
      })
      .join("");
    $("webcheck").textContent =
      `Browser check: this page was compared with the Python model on ${fmt(check.dataset_rows_compared, 0)} dataset rows ` +
      `and ${fmt(check.random_rows_compared, 0)} random inputs. Decision differences: ${check.decision_mismatches_at_0_5}. ` +
      `Largest probability difference: ${check.max_abs_probability_difference.toExponential(1)}.`;
  }

  async function init() {
    const [m, metrics, check] = await Promise.all(
      ["model.json", "metrics.json", "web_export_check.json"].map((u) => fetch(u).then((r) => r.json()))
    );
    model = m;
    renderStudy(metrics, check);
    document.querySelectorAll("[data-preset]").forEach((b) =>
      b.addEventListener("click", () => {
        setInputs(PRESETS[b.dataset.preset]);
        predict();
      })
    );
    $("form").addEventListener("submit", predict);
    setInputs(PRESETS.normal);
    predict();
  }

  init().catch((e) => {
    $("error").textContent = "The model could not be loaded: " + e.message;
    $("error").hidden = false;
  });
})();
