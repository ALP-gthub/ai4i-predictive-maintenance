// Used by src/export_web.py: runs docs/model.js on every dataset row and
// prints the predicted probabilities as a JSON array.
// Usage: node src/check_web_model.js <rows.json> <docs-dir>

const fs = require("fs");
const path = require("path");

const [rowsPath, docsDir] = process.argv.slice(2);
const AI4I = require(path.resolve(docsDir, "model.js"));
const model = JSON.parse(fs.readFileSync(path.join(docsDir, "model.json"), "utf8"));
const { rows } = JSON.parse(fs.readFileSync(rowsPath, "utf8"));

const out = rows.map((r) =>
  AI4I.predictProbability(model, {
    type: r["Type"],
    air: r["Air temperature [K]"],
    process: r["Process temperature [K]"],
    speed: r["Rotational speed [rpm]"],
    torque: r["Torque [Nm]"],
    wear: r["Tool wear [min]"],
  })
);
process.stdout.write(JSON.stringify(out));
