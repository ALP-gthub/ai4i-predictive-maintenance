"""Feature definitions for the AI4I 2020 predictive maintenance study.

Raw features are the six process measurements available before a failure.
Engineered features express the physical quantities that the dataset
documentation names as failure mechanisms:

  * temperature difference between process and air  -> heat dissipation
  * mechanical power, torque x angular speed           -> power failure
  * tool wear x torque                                 -> overstrain

The five failure-mode flags (TWF, HDF, PWF, OSF, RNF) are labels, not inputs.
Using them as inputs would leak the answer into the model.
"""

import numpy as np
import pandas as pd

AIR = "Air temperature [K]"
PROCESS = "Process temperature [K]"
SPEED = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
WEAR = "Tool wear [min]"
TYPE = "Type"

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_NUMERIC = [AIR, PROCESS, SPEED, TORQUE, WEAR]
ENGINEERED = ["temp_diff_K", "power_W", "strain_minNm"]

TYPE_CODES = {"L": 0, "M": 1, "H": 2}


def load_raw(path):
    """Read the original CSV. The file starts with a UTF-8 byte order mark."""
    return pd.read_csv(path, encoding="utf-8-sig")


def add_engineered(df):
    out = df.copy()
    out["temp_diff_K"] = out[PROCESS] - out[AIR]
    angular_speed = out[SPEED] * 2 * np.pi / 60  # rpm -> rad/s
    out["power_W"] = out[TORQUE] * angular_speed
    out["strain_minNm"] = out[WEAR] * out[TORQUE]
    return out


def build_matrix(df, engineered):
    """Return the model input table. Type is encoded L=0, M=1, H=2."""
    x = df[RAW_NUMERIC].copy()
    x["type_code"] = df[TYPE].map(TYPE_CODES)
    if engineered:
        x = pd.concat([x, add_engineered(df)[ENGINEERED]], axis=1)
    return x


def raw_matrix(df):
    """Model input without engineered features. Kept at module level so a saved model can be reloaded."""
    return build_matrix(df, engineered=False)


def engineered_matrix(df):
    """Model input with engineered features. Kept at module level so a saved model can be reloaded."""
    return build_matrix(df, engineered=True)


def documented_rules(df):
    """Failure conditions exactly as stated in the UCI dataset description.

    TWF (random tool change between 200 and 240 min) and RNF (0.1 % random
    chance) are random by design and cannot be written as a rule.
    """
    e = add_engineered(df)
    threshold = df[TYPE].map({"L": 11000, "M": 12000, "H": 13000})
    return pd.DataFrame(
        {
            "HDF": (e["temp_diff_K"] < 8.6) & (df[SPEED] < 1380),
            "PWF": (e["power_W"] < 3500) | (e["power_W"] > 9000),
            "OSF": e["strain_minNm"] > threshold,
        }
    ).astype(int)
