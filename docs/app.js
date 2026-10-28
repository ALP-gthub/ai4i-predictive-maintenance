// Page logic for the AI4I 2020 demo. The model itself is in model.js / model.json.
// The page is bilingual (English / Turkish). Temperatures can be entered in °C
// or K; the model and the dataset use kelvin, so °C values are converted.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Turkish case endings for a number, chosen from its last spoken word:
  // ablative ("21'den") and dative ("12'ye").
  const TR_WORDS = {
    units: ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"],
    tens: ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"],
  };
  function trLastWord(n) {
    n = Math.abs(Math.trunc(n));
    if (n === 0) return "sıfır";
    if (n % 10) return TR_WORDS.units[n % 10];
    if (n % 100) return TR_WORDS.tens[(n % 100) / 10];
    if (n % 1000) return "yüz";
    return "bin";
  }
  function trVowel(word) {
    const v = word.match(/[aeıioöuü](?=[^aeıioöuü]*$)/);
    return v ? v[0] : "e";
  }
  const trBack = (v) => "aıou".includes(v);
  const HARD = "fstkçşhp";
  function trAbl(n) {
    const w = trLastWord(n);
    const d = HARD.includes(w[w.length - 1]) ? "t" : "d";
    return "'" + d + (trBack(trVowel(w)) ? "an" : "en");
  }
  function trDat(n) {
    const w = trLastWord(n);
    const vowelEnd = "aeıioöuü".includes(w[w.length - 1]);
    return "'" + (vowelEnd ? "y" : "") + (trBack(trVowel(w)) ? "a" : "e");
  }

  // ---------- Text ----------
  const TEXT = {
    en: {
      doc_title: "Failure Prediction: field knowledge meets machine learning",
      site_title: "Failure Prediction",
      site_sub: "Field knowledge meets machine learning",
      source: "Source on GitHub",
      kicker: "MACHINE LEARNING THROUGH AN ENGINEER'S EYES",
      hero_title: (raw, eng) => `Three values a maintenance engineer checks first cut missed failures from ${raw} to ${eng}.`,
      hero_lede: "Same data, same algorithm. The difference was the question.",
      cmp_metric: "Result",
      cmp_raw: "Raw sensor data only",
      cmp_eng: "+ 3 engineering values",
      cmp_change: "Change",
      fewer: (p) => `${p} fewer`,
      phys_line: (caught, total) => `Of the ${total} failures with a physical cause, ${caught} were caught.`,
      hero_p: "A failure prediction model was trained twice on the same data. The second time, it also received three values an experienced maintenance engineer checks first: temperature difference, mechanical power and tool load. Nothing else changed.",
      stat_missed: "Missed failures",
      stat_false: "False alarms",
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
      cost_h2: "Decision by cost",
      cost_p: "A probability alone does not tell you what to do. Enter what an unplanned failure and an inspection cost you. The page finds the alarm threshold with the lowest total cost and shows what it would have meant on the test records.",
      c_fail: "Cost of one unplanned failure",
      c_inspect: "Cost of one inspection",
      c_hint: "Example ratio. Enter your own values; any currency works as long as both use the same one.",
      o_none: "React after failure, no model",
      o_half: "Model, default 50 % threshold",
      o_best: (t) => `Model, cost-based threshold (${t})`,
      o_best_none: "Cost-based choice: no alarms",
      d_counts: (alarms, missed, fa) => `${alarms} alarms, ${missed} missed failures, ${fa} false alarms`,
      d_none: (missed) => `${missed} failures, all unplanned`,
      saving: (amount, pct) => `Cost-based threshold compared with reacting after failure: ${amount} lower total cost (${pct}).`,
      no_saving: "With these costs, inspections are not worth it: reacting after failure is the cheapest option.",
      raw_line: (cost, cmp) => `Same costs, model without engineering features at its own cost-based threshold: ${cost} (${cmp}).`,
      cmp_more: (d) => `${d} more`,
      cmp_less: (d) => `${d} less`,
      cmp_same: "the same",
      method: (r) => `Assumptions: every alarm leads to one inspection; a failure caught by an alarm is prevented by that inspection; every missed failure costs one unplanned failure. The threshold is chosen on the training data only (five-fold out-of-fold predictions). Costs are measured on ${r.rows} test records with ${r.failures} real failures that played no part in the choice. The calculation was checked against an independent Python implementation in ${r.cases} cases with ${r.diff} differences.`,
      err_cost: "Please enter two non-negative numbers.",
      err_fill: "Please fill in every field with a number.",
      err_load: "The model could not be loaded: ",
    },
    tr: {
      doc_title: "Arıza Tahmini: saha bilgisi ile makine öğrenmesi",
      site_title: "Arıza Tahmini",
      site_sub: "Saha bilgisi ile makine öğrenmesi",
      source: "GitHub'da kaynak kod",
      kicker: "MÜHENDİS GÖZÜYLE MAKİNE ÖĞRENMESİ",
      hero_title: (raw, eng) => `Bir bakım mühendisinin baktığı üç değer, kaçan arızaları ${raw}${trAbl(raw)} ${eng}${trDat(eng)} indirdi.`,
      hero_lede: "Aynı veri, aynı algoritma. Farkı yaratan soruydu.",
      cmp_metric: "Sonuç",
      cmp_raw: "Yalnızca ham sensör verisi",
      cmp_eng: "+ 3 mühendislik değeri",
      cmp_change: "Değişim",
      fewer: (p) => `${p} daha az`,
      phys_line: (caught, total) => `Fiziksel nedeni olan ${total} arızadan ${caught} tanesi yakalandı.`,
      hero_p: "Bir arıza tahmin modeli aynı veriyle iki kez eğitildi. İkincisinde, deneyimli bir bakım mühendisinin ilk baktığı üç değer de modele verildi: sıcaklık farkı, mekanik güç ve takım yükü. Başka hiçbir şey değişmedi.",
      stat_missed: "Kaçırılan arıza",
      stat_false: "Yanlış alarm",
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
      cost_h2: "Maliyet açısından karar",
      cost_p: "Olasılık tek başına ne yapılacağını söylemez. Plansız bir arızanın ve bir kontrolün size maliyetini girin. Sayfa, toplam maliyeti en düşük alarm eşiğini bulur ve bunun test kayıtlarında ne anlama geleceğini gösterir.",
      c_fail: "Bir plansız arızanın maliyeti",
      c_inspect: "Bir kontrolün maliyeti",
      c_hint: "Örnek oran. Kendi değerlerinizi girin; iki değer aynı para biriminde olduğu sürece birim önemli değil.",
      o_none: "Arızadan sonra müdahale, model yok",
      o_half: "Model, varsayılan %50 eşik",
      o_best: (t) => `Model, maliyete dayalı eşik (${t})`,
      o_best_none: "Maliyete dayalı seçim: alarm yok",
      d_counts: (alarms, missed, fa) => `${alarms} alarm, ${missed} kaçan arıza, ${fa} yanlış alarm`,
      d_none: (missed) => `${missed} arıza, hepsi plansız`,
      saving: (amount, pct) => `Maliyete dayalı eşik, arızadan sonra müdahaleye göre toplam maliyeti ${amount} düşürüyor (${pct}).`,
      no_saving: "Bu maliyetlerle kontrol yapmak kârlı değil: En ucuz seçenek arızadan sonra müdahale.",
      raw_line: (cost, cmp) => `Aynı maliyetlerle, mühendislik özellikleri olmayan model kendi maliyete dayalı eşiğinde: ${cost} (${cmp}).`,
      cmp_more: (d) => `${d} daha fazla`,
      cmp_less: (d) => `${d} daha az`,
      cmp_same: "aynı",
      method: (r) => `Varsayımlar: Her alarm bir kontrole yol açar; alarmın yakaladığı arıza o kontrolle önlenir; kaçan her arıza bir plansız arıza maliyeti doğurur. Eşik yalnızca eğitim verisinden seçilir (beş katlı çapraz doğrulama tahminleri). Maliyetler, seçimde hiç kullanılmayan ${r.rows} test kaydında (${r.failures} gerçek arıza) ölçülür. Hesap, bağımsız bir Python hesabıyla ${r.cases} senaryoda karşılaştırıldı; fark sayısı: ${r.diff}.`,
      err_cost: "Lütfen negatif olmayan iki sayı girin.",
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
  let trainPreds = null;
  let testPreds = null;
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

    $("heroTitle").textContent = t("hero_title")(raw.fn, eng.fn);
    $("missedRaw").textContent = raw.fn;
    $("missedEng").textContent = eng.fn;
    $("fpRaw").textContent = raw.fp;
    $("fpEng").textContent = eng.fp;
    const drop = (a, b) => (lang === "tr" ? "%" + fmt(((a - b) / a) * 100, 0) : fmt(((a - b) / a) * 100, 0) + " %");
    $("missedDelta").textContent = t("fewer")(drop(raw.fn, eng.fn));
    $("fpDelta").textContent = t("fewer")(drop(raw.fp, eng.fp));
    $("physLine").textContent = t("phys_line")(phys.caught, phys.total);
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

  // ---------- Decision by cost ----------
  const money = (x) => fmt(x, Number.isInteger(x) ? 0 : 2);
  const share = (x) => (lang === "tr" ? "%" + fmt(x * 100, 0) : fmt(x * 100, 0) + " %");

  function decide(name, failCost, inspectCost) {
    return AI4I.costDecision(
      { label: trainPreds.label, probs: trainPreds[name] },
      { label: testPreds.label, probs: testPreds[name] },
      failCost,
      inspectCost
    );
  }

  function renderCost() {
    if (!trainPreds || !testPreds) return;
    const A = num("costFail");
    const B = num("costInspect");
    const err = $("costError");
    if (A === null || B === null || A < 0 || B < 0) {
      err.textContent = t("err_cost");
      err.hidden = false;
      return;
    }
    err.hidden = true;

    const eng = decide("boosting_engineered", A, B);
    const raw = decide("boosting_raw", A, B);

    $("cNone").textContent = money(eng.noMonitoring.cost);
    $("cNoneD").textContent = t("d_none")(eng.noMonitoring.fn);
    $("cHalf").textContent = money(eng.atHalf.cost);
    $("cHalfD").textContent = t("d_counts")(eng.atHalf.alarms, eng.atHalf.fn, eng.atHalf.fp);

    const noAlarm = eng.threshold === Infinity;
    $("cBestLabel").textContent = noAlarm ? t("o_best_none") : t("o_best")(pctThreshold(eng.threshold));
    $("cBest").textContent = money(eng.chosen.cost);
    $("cBestD").textContent = t("d_counts")(eng.chosen.alarms, eng.chosen.fn, eng.chosen.fp);

    const saved = eng.noMonitoring.cost - eng.chosen.cost;
    $("cSaving").textContent =
      saved > 0 ? t("saving")(money(saved), share(saved / eng.noMonitoring.cost)) : t("no_saving");

    const diff = raw.chosen.cost - eng.chosen.cost;
    const cmp = diff > 0 ? t("cmp_more")(money(diff)) : diff < 0 ? t("cmp_less")(money(-diff)) : t("cmp_same");
    $("cRaw").textContent = t("raw_line")(money(raw.chosen.cost), cmp);

    $("cMethod").textContent = t("method")({
      rows: fmt(testPreds.label.length, 0),
      failures: eng.noMonitoring.fn,
      cases: check.cost_cases_compared,
      diff: check.cost_differences,
    });
  }

  // Thresholds can be very small, so show enough digits to be meaningful.
  function pctThreshold(x) {
    const v = x * 100;
    const d = v >= 10 ? 1 : v >= 1 ? 2 : 3;
    return lang === "tr" ? "%" + fmt(v, d) : fmt(v, d) + " %";
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
    renderCost();
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

    [model, metrics, check, trainPreds, testPreds] = await Promise.all(
      ["model.json", "metrics.json", "web_export_check.json", "train_oof_predictions.json", "test_predictions.json"].map((u) =>
        fetch(u).then((r) => r.json())
      )
    );
    renderStudy();
    ["costFail", "costInspect"].forEach((id) => $(id).addEventListener("input", renderCost));
    renderCost();
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
