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

  const api = { engineered, featureVector, predictProbability, documentedChecks, toK, toC };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.AI4I = api;
  }
})(typeof window !== "undefined" ? window : this);
