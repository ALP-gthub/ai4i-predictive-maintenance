"""Gradio demo for the AI4I 2020 failure prediction model."""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from src import features as F

MODEL = joblib.load(Path(__file__).parent / "model" / "model.joblib")
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def predict(product_type, air_k, process_k, speed_rpm, torque_nm, wear_min):
    row = pd.DataFrame(
        [
            {
                F.TYPE: product_type,
                F.AIR: air_k,
                F.PROCESS: process_k,
                F.SPEED: speed_rpm,
                F.TORQUE: torque_nm,
                F.WEAR: wear_min,
            }
        ]
    )
    probability = float(MODEL.predict_proba(row)[0, 1])
    e = F.add_engineered(row).iloc[0]

    checks = pd.DataFrame(
        [
            [
                "Heat dissipation (HDF)",
                f"ΔT = {e['temp_diff_K']:.1f} K, speed = {speed_rpm:.0f} rpm",
                "ΔT < 8.6 K and speed < 1380 rpm",
                "TRIGGERED" if (e["temp_diff_K"] < 8.6 and speed_rpm < 1380) else "ok",
            ],
            [
                "Power (PWF)",
                f"P = {e['power_W']:.0f} W",
                "P < 3500 W or P > 9000 W",
                "TRIGGERED" if (e["power_W"] < 3500 or e["power_W"] > 9000) else "ok",
            ],
            [
                "Overstrain (OSF)",
                f"wear × torque = {e['strain_minNm']:.0f} min·Nm",
                f"> {OSF_LIMIT[product_type]:,} min·Nm for type {product_type}",
                "TRIGGERED" if e["strain_minNm"] > OSF_LIMIT[product_type] else "ok",
            ],
        ],
        columns=["Failure mode", "Value", "Documented condition", "Status"],
    )
    verdict = "Failure expected" if probability >= 0.5 else "No failure expected"
    return f"{verdict} (model probability {probability:.1%})", checks


with gr.Blocks(title="AI4I 2020 Failure Prediction") as demo:
    gr.Markdown(
        "# Machine failure prediction (AI4I 2020)\n"
        "Enter process conditions. The model estimates the failure probability, "
        "and the table checks the physical conditions documented for the dataset.\n\n"
        "**Scope:** trained on a synthetic public dataset for learning and demonstration. "
        "The thresholds belong to that dataset, not to any real machine."
    )
    with gr.Row():
        with gr.Column():
            product_type = gr.Radio(["L", "M", "H"], value="L", label="Product quality type")
            air = gr.Number(value=300.0, label="Air temperature [K]")
            process = gr.Number(value=310.0, label="Process temperature [K]")
            speed = gr.Number(value=1500, label="Rotational speed [rpm]")
            torque = gr.Number(value=40.0, label="Torque [Nm]")
            wear = gr.Number(value=100, label="Tool wear [min]")
            button = gr.Button("Predict", variant="primary")
        with gr.Column():
            verdict = gr.Textbox(label="Result")
            table = gr.Dataframe(label="Engineering check", interactive=False)
    button.click(
        predict,
        inputs=[product_type, air, process, speed, torque, wear],
        outputs=[verdict, table],
    )

if __name__ == "__main__":
    demo.launch()
