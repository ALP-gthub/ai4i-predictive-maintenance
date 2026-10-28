// Used by src/export_web.py: runs docs/model.js on every dataset row and
// prints the predicted probabilities, and checks the page's °C path: each
// temperature converted to °C and back must be unchanged and give the same probability.
// Usage: node src/check_web_model.js <rows.json> <docs-dir>

const fs = require("fs");
const path = require("path");

const [rowsPath, docsDir] = process.argv.slice(2);
const AI4I = require(path.resolve(docsDir, "model.js"));
const model = JSON.parse(fs.readFileSync(path.join(docsDir, "model.json"), "utf8"));
const { rows } = JSON.parse(fs.readFileSync(rowsPath, "utf8"));

const toInput = (r, air, proc) => ({
  type: r["Type"],
  air: air,
  process: proc,
  speed: r["Rotational speed [rpm]"],
  torque: r["Torque [Nm]"],
  wear: r["Tool wear [min]"],
});

// Probability with the kelvin values as given.
const probabilities = rows.map((r) =>
  AI4I.predictProbability(model, toInput(r, r["Air temperature [K]"], r["Process temperature [K]"]))
);

// The page's path: kelvin shown as °C, typed back, converted to kelvin.
let roundtripMismatches = 0;
let celsiusPathDifferences = 0;
rows.forEach((r, i) => {
  const air = AI4I.toK(AI4I.toC(r["Air temperature [K]"]));
  const proc = AI4I.toK(AI4I.toC(r["Process temperature [K]"]));
  if (air !== r["Air temperature [K]"] || proc !== r["Process temperature [K]"]) roundtripMismatches++;
  if (AI4I.predictProbability(model, toInput(r, air, proc)) !== probabilities[i]) celsiusPathDifferences++;
});

process.stdout.write(JSON.stringify({ probabilities, roundtripMismatches, celsiusPathDifferences }));
