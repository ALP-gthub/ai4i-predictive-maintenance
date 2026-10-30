---
title: Failure Prediction with Engineering Knowledge
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Engineering features for failure prediction, TR/EN
---

# Failure Prediction with Engineering Knowledge

Example application: milling machine, AI4I 2020 dataset (UCI, CC BY 4.0).

Browser tool, in Turkish and English, built around a gradient boosting model that also uses three values a maintenance engineer checks first: temperature difference, mechanical power and tool load. Tabs: prediction with the documented physical failure conditions next to it, a cost-based alarm threshold, model evaluation, and a step-by-step explanation of how it works.

The page runs entirely in the browser. Its predictions were checked against the original scikit-learn model on all 10,000 dataset rows and 20,000 random inputs.

Source code, data, method and full results: https://github.com/ALP-gthub/ai4i-predictive-maintenance

License: MIT for the code; the dataset is CC BY 4.0 (UCI).

Author: ALP ATA
