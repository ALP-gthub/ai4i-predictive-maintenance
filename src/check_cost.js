// Used by src/export_web.py: runs AI4I.costDecision from docs/model.js for
// each cost pair and both gradient boosting models, and prints the results.
// Usage: node src/check_cost.js <docs-dir> <cases.json>

const fs = require("fs");
const path = require("path");

const [docsDir, casesPath] = process.argv.slice(2);
const AI4I = require(path.resolve(docsDir, "model.js"));
const read = (f) => JSON.parse(fs.readFileSync(path.join(docsDir, f), "utf8"));
const train = read("train_oof_predictions.json");
const test = read("test_predictions.json");
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));

const out = [];
for (const [failCost, inspectCost] of cases) {
  for (const name of ["boosting_raw", "boosting_engineered"]) {
    const r = AI4I.costDecision(
      { label: train.label, probs: train[name] },
      { label: test.label, probs: test[name] },
      failCost,
      inspectCost
    );
    out.push({ name, failCost, inspectCost, threshold: r.threshold === Infinity ? null : r.threshold, chosen: r.chosen, atHalf: r.atHalf });
  }
}
process.stdout.write(JSON.stringify(out));
