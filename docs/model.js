// Browser predictor for the AI4I 2020 failure model.
// Mirrors src/features.py and the scikit-learn tree traversal so that the
// page gives the same probability as model/model.joblib. This is checked on
// all 10,000 dataset rows by src/export_web.py.

(function (root) {
  "use strict";

  const TYPE_CODES = { L: 0, M: 1, H: 2 };
  const OSF_LIMIT = { L: 11000, M: 12000, H: 13000 };

  // Same quantities and order of operations as src/features.py.
  function engineered(input) {
    const tempDiff = input.process - input.air;
    const angularSpeed = (input.speed * 2 * Math.PI) / 60;
    return {
      tempDiff: tempDiff,
      power: input.torque * angularSpeed,
      strain: input.wear * input.torque,
    };
  }

  // Feature order must match model.json "features".
  function featureVector(input) {
    const e = engineered(input);
    return [
      input.air,
      input.process,
      input.speed,
      input.torque,
      input.wear,
      TYPE_CODES[input.type],
      e.tempDiff,
      e.power,
      e.strain,
    ];
  }

  function treeValue(tree, x) {
    let node = 0;
    while (!tree.leaf[node]) {
      const v = x[tree.feature[node]];
      let goLeft;
      if (Number.isNaN(v)) {
        goLeft = tree.missing_left[node] === 1;
      } else {
        goLeft = v <= tree.threshold[node];
      }
      node = goLeft ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
  }

  function predictProbability(model, input) {
    const x = featureVector(input);
    let raw = model.baseline;
    for (const tree of model.trees) {
      raw += treeValue(tree, x);
    }
    return 1 / (1 + Math.exp(-raw));
  }

  // Failure conditions as stated in the UCI dataset description.
  function documentedChecks(input) {
    const e = engineered(input);
    return {
      e: e,
      hdf: e.tempDiff < 8.6 && input.speed < 1380,
      pwf: e.power < 3500 || e.power > 9000,
      osf: e.strain > OSF_LIMIT[input.type],
      osfLimit: OSF_LIMIT[input.type],
    };
  }

  // The page takes temperatures in °C; the model and the dataset use kelvin.
  // Rounding to 0.01 K removes floating point residue, so a dataset value
  // converted to °C and back is unchanged. src/export_web.py checks this on
  // every dataset temperature.
  function toK(celsius) {
    return Math.round((celsius + 273.15) * 100) / 100;
  }

  function toC(kelvin) {
    return Math.round((kelvin - 273.15) * 100) / 100;
  }

  // Cost of acting on the model's alarms, over the held-out test set.
  // Assumptions, stated on the page:
  //   every alarm triggers one inspection (inspectCost),
  //   a failure caught by an alarm is prevented by that inspection,
  //   every missed failure costs one unplanned failure (failCost).
  // An alarm is raised when probability >= threshold. The sweep tries every
  // distinct probability as the threshold and keeps the cheapest; on a tie it
  // keeps the higher threshold (fewer alarms); "no alarm at all" is the
  // starting point. src/export_web.py checks costDecision against an
  // independent Python implementation.
  function outcome(labels, probs, threshold, failCost, inspectCost) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < labels.length; i++) {
      const alarm = probs[i] >= threshold;
      if (labels[i] === 1 && alarm) tp++;
      else if (labels[i] === 1) fn++;
      else if (alarm) fp++;
    }
    return { threshold, tp, fp, fn, alarms: tp + fp, cost: fn * failCost + (tp + fp) * inspectCost };
  }

  function costSweep(labels, probs, failCost, inspectCost) {
    const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
    const failures = labels.reduce((s, v) => s + v, 0);
    let tp = 0;
    let fp = 0;
    let best = { threshold: Infinity, tp: 0, fp: 0, fn: failures, alarms: 0, cost: failures * failCost };
    let k = 0;
    while (k < order.length) {
      const t = probs[order[k]];
      while (k < order.length && probs[order[k]] === t) {
        if (labels[order[k]] === 1) tp++;
        else fp++;
        k++;
      }
      const fn = failures - tp;
      const cost = fn * failCost + (tp + fp) * inspectCost;
      if (cost < best.cost) best = { threshold: t, tp, fp, fn, alarms: tp + fp, cost };
    }
    return {
      best: best,
      atHalf: outcome(labels, probs, 0.5, failCost, inspectCost),
      noMonitoring: { fn: failures, cost: failures * failCost },
    };
  }

  // The threshold is chosen on out-of-fold predictions for the training set
  // (train), then applied to the held-out test set (test), which played no
  // part in the choice. Each of train and test is { label: [...], probs: [...] }.
  function costDecision(train, test, failCost, inspectCost) {
    const chosen = costSweep(train.label, train.probs, failCost, inspectCost).best.threshold;
    const failures = test.label.reduce((s, v) => s + v, 0);
    return {
      threshold: chosen,
      chosen: outcome(test.label, test.probs, chosen, failCost, inspectCost),
      atHalf: outcome(test.label, test.probs, 0.5, failCost, inspectCost),
      noMonitoring: { fn: failures, alarms: 0, cost: failures * failCost },
    };
  }

  const api = { engineered, featureVector, predictProbability, documentedChecks, toK, toC, costSweep, costDecision };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.AI4I = api;
  }
})(typeof window !== "undefined" ? window : this);
