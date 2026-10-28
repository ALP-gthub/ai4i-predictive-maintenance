// Page logic for the AI4I 2020 demo. The model itself is in model.js / model.json.
// The page is bilingual (English / Turkish). Temperatures can be entered in °C
// or K; the model and the dataset use kelvin, so °C values are converted.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---------- Text ----------
  const TEXT = {
    en: {
      doc_title: "AI4I 2020 Failure Prediction",
      site_title: "AI4I 2020 Failure Prediction",
      site_sub: "Predictive maintenance • Gradient boosting • Engineering features",
      source: "Source on GitHub",
      kicker: "MACHINE LEARNING THROUGH AN ENGINEER'S EYES",
      hero_h1: "Same data. Same algorithm. The difference was the question.",
      hero_p: "A failure prediction model was trained twice on the same data. The second time, it also received three values an experienced maintenance engineer checks first: temperature difference, mechanical power and tool load. Nothing else changed.",
      stat_missed: "Missed failures",
      stat_false: "False alarms",
      stat_phys: "Failures with a physical cause, caught",
      stats_note: (s) => `Held-out test set of ${s.rows} records the model never saw, with ${s.failures} real failures.`,
      try_h2: "Try it yourself",
      try_p: "Enter the conditions of a milling process. The model estimates the failure probability, and the documented physical failure conditions are checked next to it. Everything runs in your browser.",
      examples: "Examples:",
      p_normal: "Normal",
      p_heat: "Heat dissipation",
      p_strain: "Overstrain",
      p_power: "Low power",
      l_type: "Product quality type",
      t_L: "L (low)",
      t_M: "M (medium)",
      t_H: "H (high)",
      l_air: "Air temperature",
      l_process: "Process temperature",
      l_speed: "Rotational speed",
      l_torque: "Torque",
      l_wear: "Tool wear",
      unit_label: "Temperature unit",
      hint_k: (k) => `= ${k} K, the unit the model uses`,
      hint_c: (c) => `= ${c} °C`,
      predict: "PREDICT",
      result_kicker: "RESULT",
      result_h2: "Model estimate",
      waiting: "Waiting for input",
      fail: "Failure expected",
      ok: "No failure expected",
      prob_label: "Failure probability",
      threshold: "Decision threshold 50 %",
      check_h3: "Engineering check",
      th_mode: "Failure mode",
      th_value: "Value",
      th_cond: "Documented condition",
      th_status: "Status",
      m_hdf: "Heat dissipation (HDF)",
      m_pwf: "Power (PWF)",
      m_osf: "Overstrain (OSF)",
      c_hdf: "ΔT &lt; 8.6 K and speed &lt; 1,380 rpm",
      c_pwf: "below 3,500 W or above 9,000 W",
      c_osf: (lim, t) => `above ${lim} min·Nm for type ${t}`,
      triggered: "✕ Triggered",
      not_met: "✓ Not met",
      random_note: "Tool wear failures (TWF) and random failures (RNF) are assigned at random in the dataset, so neither the model nor a rule can predict them.",
      eval_h2: "How the model was evaluated",
      split: (s) => `Held-out test set of ${s.rows} rows (${s.failures} actual failures), not used in training. Stratified split, seed ${s.seed}.`,
      th_approach: "Approach",
      th_precision: "Precision",
      th_recall: "Recall",
      th_missed: "Missed failures",
      th_false: "False alarms",
      a_logistic_raw: "Logistic regression, raw measurements",
      a_boosting_raw: "Gradient boosting, raw measurements",
      a_boosting_engineered: "Gradient boosting, raw + engineered features (this page)",
      a_documented_rules: "Documented physical rules (reference, no learning)",
      webcheck: (c) => `Browser check: this page was compared with the Python model on ${c.rows} dataset rows and ${c.random} random inputs. Decision differences: ${c.mismatch}. Largest probability difference: ${c.diff}.`,
      scope: "<b>Scope:</b> the AI4I 2020 dataset is synthetic and its failures follow fixed rules. The model and the thresholds shown here belong to that dataset, not to any real machine. This page is for learning and demonstration.",
      footer: 'Data: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, <a href="https://doi.org/10.24432/C5HS5C">doi.org/10.24432/C5HS5C</a>, CC BY 4.0. Study and page: ALP ATA. Code: <a href="https://github.com/ALP-gthub/ai4i-predictive-maintenance">github.com/ALP-gthub/ai4i-predictive-maintenance</a>',
      err_fill: "Please fill in every field with a number.",
      err_load: "The model could not be loaded: ",
    },
    tr: {
      doc_title: "AI4I 2020 Arıza Tahmini",
      site_title: "AI4I 2020 Arıza Tahmini",
      site_sub: "Kestirimci bakım • Gradient boosting • Mühendislik özellikleri",
      source: "GitHub'da kaynak kod",
      kicker: "MÜHENDİS GÖZÜYLE MAKİNE ÖĞRENMESİ",
      hero_h1: "Aynı veri. Aynı algoritma. Farkı yaratan soruydu.",
      hero_p: "Bir arıza tahmin modeli aynı veriyle iki kez eğitildi. İkincisinde, deneyimli bir bakım mühendisinin ilk baktığı üç değer de modele verildi: sıcaklık farkı, mekanik güç ve takım yükü. Başka hiçbir şey değişmedi.",
      stat_missed: "Kaçırılan arıza",
      stat_false: "Yanlış alarm",
      stat_phys: "Fiziksel nedenli arızalardan yakalanan",
      stats_note: (s) => `Modelin hiç görmediği ${s.rows} kayıtlık test seti, ${s.failures} gerçek arıza.`,
      try_h2: "Kendiniz deneyin",
      try_p: "Bir freze prosesinin koşullarını girin. Model arıza olasılığını tahmin eder; yanında veri setinde belgelenen fiziksel arıza koşulları kontrol edilir. Tüm hesaplama tarayıcınızda yapılır.",
      examples: "Örnekler:",
      p_normal: "Normal",
      p_heat: "Isı dağılımı",
      p_strain: "Aşırı zorlanma",
      p_power: "Düşük güç",
      l_type: "Ürün kalite tipi",
      t_L: "L (düşük)",
      t_M: "M (orta)",
      t_H: "H (yüksek)",
      l_air: "Ortam havası sıcaklığı",
      l_process: "Proses sıcaklığı",
      l_speed: "Devir hızı",
      l_torque: "Tork",
      l_wear: "Takım aşınması (kullanım süresi)",
      unit_label: "Sıcaklık birimi",
      hint_k: (k) => `= ${k} K, modelin kullandığı birim`,
      hint_c: (c) => `= ${c} °C`,
      predict: "TAHMİN ET",
      result_kicker: "SONUÇ",
      result_h2: "Model tahmini",
      waiting: "Girdi bekleniyor",
      fail: "Arıza bekleniyor",
      ok: "Arıza beklenmiyor",
      prob_label: "Arıza olasılığı",
      threshold: "Karar eşiği %50",
      check_h3: "Mühendislik kontrolü",
      th_mode: "Arıza türü",
      th_value: "Değer",
      th_cond: "Belgelenen koşul",
      th_status: "Durum",
      m_hdf: "Isı dağılımı (HDF)",
      m_pwf: "Güç (PWF)",
      m_osf: "Aşırı zorlanma (OSF)",
      c_hdf: "ΔT &lt; 8,6 K ve devir &lt; 1.380 rpm",
      c_pwf: "3.500 W altı veya 9.000 W üstü",
      c_osf: (lim, t) => `${t} tipi için ${lim} min·Nm üstü`,
      triggered: "✕ Tetiklendi",
      not_met: "✓ Sağlanmadı",
      random_note: "Takım aşınması (TWF) ve rastgele arızalar (RNF) veri setinde rastgele atanır; bu yüzden ne model ne de bir kural bunları tahmin edebilir.",
      eval_h2: "Model nasıl değerlendirildi",
      split: (s) => `Eğitimde kullanılmayan ${s.rows} satırlık test seti (${s.failures} gerçek arıza). Tabakalı bölme, tohum değeri ${s.seed}.`,
      th_approach: "Yaklaşım",
      th_precision: "Kesinlik",
      th_recall: "Duyarlılık",
      th_missed: "Kaçırılan arıza",
      th_false: "Yanlış alarm",
      a_logistic_raw: "Lojistik regresyon, ham ölçümler",
      a_boosting_raw: "Gradient boosting, ham ölçümler",
      a_boosting_engineered: "Gradient boosting, ham ölçümler + mühendislik özellikleri (bu sayfa)",
      a_documented_rules: "Belgelenen fiziksel kurallar (referans, öğrenme yok)",
      webcheck: (c) => `Tarayıcı kontrolü: Bu sayfa, Python modeliyle ${c.rows} veri satırında ve ${c.random} rastgele girdide karşılaştırıldı. Karar farkı: ${c.mismatch}. En büyük olasılık farkı: ${c.diff}.`,
      scope: "<b>Kapsam:</b> AI4I 2020 veri seti sentetiktir ve arızaları sabit kurallarla üretilmiştir. Burada gösterilen model ve eşikler bu veri setine aittir, gerçek bir makineye ait değildir. Bu sayfa öğrenme ve tanıtım amaçlıdır.",
      footer: 'Veri: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, <a href="https://doi.org/10.24432/C5HS5C">doi.org/10.24432/C5HS5C</a>, CC BY 4.0. Çalışma ve sayfa: ALP ATA. Kod: <a href="https://github.com/ALP-gthub/ai4i-predictive-maintenance">github.com/ALP-gthub/ai4i-predictive-maintenance</a>',
      err_fill: "Lütfen her alana bir sayı girin.",
      err_load: "Model yüklenemedi: ",
    },
  };

  let lang = "en";
  const t = (key) => TEXT[lang][key];
  const locale = () => (lang === "tr" ? "tr-TR" : "en-US");
  const fmt = (x, d) =>
    Number(x).toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x) => (lang === "tr" ? "%" + fmt(x * 100, 1) : fmt(x * 100, 1) + " %");

  // ---------- Temperature: °C on screen, kelvin for the model ----------
  const toK = AI4I.toK;
  const toC = AI4I.toC;

  // Example inputs, stored in kelvin as in the dataset. Each one triggers the named documented condition.
  const PRESETS = {
    normal: { type: "L", air: 300.0, process: 310.0, speed: 1500, torque: 40.0, wear: 100 },
    heat: { type: "L", air: 302.0, process: 310.2, speed: 1300, torque: 55.0, wear: 20 },
    strain: { type: "M", air: 300.0, process: 310.0, speed: 1400, torque: 60.0, wear: 210 },
    power: { type: "H", air: 300.0, process: 310.0, speed: 2800, torque: 10.0, wear: 50 },
  };

  let model = null;
  let metrics = null;
  let check = null;
  let lastInput = null;
  let unit = "C"; // temperature unit shown in the form: "C" or "K"

  function setInputs(p) {
    $("type").value = p.type;
    $("airT").value = unit === "C" ? toC(p.air) : p.air;
    $("processT").value = unit === "C" ? toC(p.process) : p.process;
    ["speed", "torque", "wear"].forEach((f) => ($(f).value = p[f]));
    updateKHints();
  }

  function num(id) {
    const raw = $(id).value.trim();
    const v = Number(raw);
    return raw === "" || !Number.isFinite(v) ? null : v;
  }

  function readInputs() {
    const vals = { air: num("airT"), process: num("processT"), speed: num("speed"), torque: num("torque"), wear: num("wear") };
    if (Object.values(vals).some((v) => v === null)) return null;
    const kelvin = (v) => (unit === "C" ? toK(v) : v);
    return {
      type: $("type").value,
      air: kelvin(vals.air),
      process: kelvin(vals.process),
      speed: vals.speed,
      torque: vals.torque,
      wear: vals.wear,
    };
  }

  // Under each temperature field: the same value in the other unit.
  function updateKHints() {
    [["airT", "airHint"], ["processT", "processHint"]].forEach(([inp, out]) => {
      const v = num(inp);
      if (v === null) $(out).textContent = "";
      else if (unit === "C") $(out).textContent = t("hint_k")(fmt(toK(v), 2));
      else $(out).textContent = t("hint_c")(fmt(toC(v), 2));
    });
  }

  // Switching the unit converts the values already in the fields.
  function setUnit(next, convert) {
    if (convert && next !== unit) {
      ["airT", "processT"].forEach((id) => {
        const v = num(id);
        if (v !== null) $(id).value = next === "C" ? toC(v) : toK(v);
      });
    }
    unit = next;
    document.querySelectorAll("[data-unit]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.unit === unit)));
    document.querySelectorAll(".tunit").forEach((el) => (el.textContent = unit === "C" ? "°C" : "K"));
    try {
      localStorage.setItem("ai4i-unit", unit);
    } catch (e) {
      /* storage unavailable: the choice is simply not remembered */
    }
    updateKHints();
  }

  const status = (on) =>
    on ? `<span class="status trig">${t("triggered")}</span>` : `<span class="status ok">${t("not_met")}</span>`;

  function render(input) {
    const badge = $("badge");
    if (!input) {
      badge.textContent = t("waiting");
      badge.className = "";
      return;
    }
    const p = AI4I.predictProbability(model, input);
    const c = AI4I.documentedChecks(input);
    const fail = p >= 0.5;

    $("prob").textContent = pct(p);
    $("bar").style.width = Math.min(100, p * 100) + "%";
    $("bar").style.background = fail ? "var(--bad)" : "var(--accent)";
    badge.textContent = fail ? t("fail") : t("ok");
    badge.className = fail ? "fail" : "ok";

    const speed = fmt(input.speed, 0);
    $("checks").innerHTML = [
      [t("m_hdf"), `ΔT ${fmt(c.e.tempDiff, 1)} K, ${speed} rpm`, t("c_hdf"), c.hdf],
      [t("m_pwf"), `${fmt(c.e.power, 0)} W`, t("c_pwf"), c.pwf],
      [t("m_osf"), `${fmt(c.e.strain, 0)} min·Nm`, t("c_osf")(fmt(c.osfLimit, 0), input.type), c.osf],
    ]
      .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${status(r[3])}</td></tr>`)
      .join("");
  }

  function predict(event) {
    if (event) event.preventDefault();
    updateKHints();
    const input = readInputs();
    const error = $("error");
    if (!input) {
      error.textContent = t("err_fill");
      error.hidden = false;
      return;
    }
    error.hidden = true;
    lastInput = input;
    render(input);
  }

  // Failures whose mode flags include HDF, PWF or OSF: the ones with a physical cause.
  function physical(groups) {
    let caught = 0;
    let total = 0;
    Object.keys(groups).forEach((g) => {
      if (/HDF|PWF|OSF/.test(g)) {
        caught += groups[g].caught;
        total += groups[g].failures;
      }
    });
    return { caught, total };
  }

  const APPROACHES = ["logistic_raw", "boosting_raw", "boosting_engineered", "documented_rules"];

  function renderStudy() {
    const s = metrics.split;
    const raw = metrics.test.boosting_raw;
    const eng = metrics.test.boosting_engineered;
    const phys = physical(metrics.test_caught_by_failure_mode.boosting_engineered);
    const sizes = { rows: fmt(s.test_rows, 0), failures: s.test_failures, seed: s.seed };

    $("missedRaw").textContent = raw.fn;
    $("missedEng").textContent = eng.fn;
    $("fpRaw").textContent = raw.fp;
    $("fpEng").textContent = eng.fp;
    $("physCaught").textContent = phys.caught;
    $("physTotal").textContent = phys.total;
    $("statsNote").textContent = t("stats_note")(sizes);
    $("split").textContent = t("split")(sizes);

    $("results").innerHTML = APPROACHES.map((k) => {
      const r = metrics.test[k];
      const cls = k === "boosting_engineered" ? ' class="best"' : "";
      return `<tr${cls}><td>${t("a_" + k)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td>` +
        `<td>${fmt(r.f1, 3)}</td><td>${r.fn}</td><td>${r.fp}</td></tr>`;
    }).join("");

    $("webcheck").textContent = t("webcheck")({
      rows: fmt(check.dataset_rows_compared, 0),
      random: fmt(check.random_rows_compared, 0),
      mismatch: check.decision_mismatches_at_0_5,
      diff: check.max_abs_probability_difference.toExponential(1).replace(".", lang === "tr" ? "," : "."),
    });
  }

  function applyLanguage(next) {
    lang = next;
    document.documentElement.lang = lang;
    document.title = t("doc_title");
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-html]").forEach((el) => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    document.querySelectorAll("[data-lang]").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
    });
    try {
      localStorage.setItem("ai4i-lang", lang);
    } catch (e) {
      /* storage unavailable: the choice is simply not remembered */
    }
    updateKHints();
    if (metrics) renderStudy();
    if (model) render(lastInput);
  }

  function initialLanguage() {
    try {
      const saved = localStorage.getItem("ai4i-lang");
      if (saved === "tr" || saved === "en") return saved;
    } catch (e) {
      /* ignore */
    }
    return (navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en";
  }

  async function init() {
    applyLanguage(initialLanguage());
    document.querySelectorAll("[data-lang]").forEach((b) =>
      b.addEventListener("click", () => applyLanguage(b.dataset.lang))
    );
    ["airT", "processT"].forEach((id) => $(id).addEventListener("input", updateKHints));
    let savedUnit = "C";
    try {
      if (localStorage.getItem("ai4i-unit") === "K") savedUnit = "K";
    } catch (e) {
      /* ignore */
    }
    setUnit(savedUnit, false);
    document.querySelectorAll("[data-unit]").forEach((b) =>
      b.addEventListener("click", () => {
        setUnit(b.dataset.unit, true);
        if (model) predict();
      })
    );

    [model, metrics, check] = await Promise.all(
      ["model.json", "metrics.json", "web_export_check.json"].map((u) => fetch(u).then((r) => r.json()))
    );
    renderStudy();
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
    $("error").textContent = t("err_load") + e.message;
    $("error").hidden = false;
  });
})();
