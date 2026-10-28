---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
tags:
- predictive-maintenance
- tabular-classification
- scikit-learn
---

# AI4I 2020 failure prediction model

Gradient boosting classifier (scikit-learn `HistGradientBoostingClassifier`) that predicts machine failure from the process conditions of a milling operation. It is trained on the public AI4I 2020 Predictive Maintenance Dataset.

* Source code, data, method and full results: https://github.com/ALP-gthub/ai4i-predictive-maintenance
* Browser demo: https://huggingface.co/spaces/ALP-hface/ai4i-failure-prediction

## Inputs

Product quality type (L, M, H), air temperature [K], process temperature [K], rotational speed [rpm], torque [Nm] and tool wear [min]. The pipeline adds three physical quantities itself: process minus air temperature [K], mechanical power [W] and tool wear × torque [min·Nm].

## Results on the held-out test set

2,000 rows not used in training, 68 actual failures, stratified split, seed 42.

| Metric | Value |
|---|---|
| Precision | 94.9% |
| Recall | 82.4% |
| F1 | 0.882 |
| PR-AUC | 0.901 |
| Missed failures | 12 |
| False alarms | 3 |

Five-fold cross-validation on the training set: F1 0.853 ± 0.016.

## Use

```python
import joblib, pandas as pd
model = joblib.load("model.joblib")  # needs src/features.py from this repository on the import path
row = pd.DataFrame([{"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                     "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}])
print(model.predict_proba(row)[0, 1])
```

Saved with scikit-learn 1.9.1 under Python 3.12.3. Load it with the same scikit-learn version.

## Limitations

The dataset is synthetic and its failures follow fixed rules. The scores above do not carry over to real machine data, where limits are not sharp and sensors are noisy. Tool wear failures and random failures are assigned at random in the dataset and cannot be predicted.

## Data and citation

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C. License: CC BY 4.0.

## License

Model and code: MIT License. Training data: CC BY 4.0 (UCI, see above).

## Author

ALP ATA
