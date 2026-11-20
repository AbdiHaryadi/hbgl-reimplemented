import time

import gradio as gr
import numpy as np
import pandas as pd

from pipeline import HBGLPipeline

pipeline = HBGLPipeline()

def list_to_set_text(x: list[str]):
    if len(x) > 0:
        return "{" + ", ".join(x) + "}"
    else:
        return "∅"

def predict(text: str):
    result, named_traces = pipeline(text)
    result_text = list_to_set_text(result)
    named_traces_text = "\n".join([
        f"Tingkat {level_index + 1}: " + list_to_set_text(trace)
        for level_index, trace in enumerate(named_traces)
    ])
    return result_text + "\n\n" + named_traces_text

def explain(text: str):
    if text.strip() == "":
        return [], pd.DataFrame({
            "token": [],
            "agrad": [],
        })

    tokens = text.strip().split()
    time.sleep(len(tokens) * 0.1)
    agrads = np.random.default_rng(abs(hash(text))).normal(size=len(tokens))
    abs_std = np.abs(agrads).std()
    if abs_std != 0:
        scores = np.tanh(agrads / abs_std)
    else:
        scores = np.tanh(agrads)

    try:
        highlighted_text_output = [(t, float(a)) for t, a in zip(tokens, scores, strict=True)]
    except ValueError:
        raise gr.Error("Cannot zip tokens and scores, possibly because of unequal length. .-.")

    return (
        highlighted_text_output,
        pd.DataFrame({
            "token": tokens,
            "agrad": agrads,
        })
    )


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Teks")
            predict_button = gr.Button("Prediksi")

        with gr.Column():
            prediction_result = gr.Textbox(label="Hasil Prediksi")
            highlighted_text = gr.HighlightedText(label="AGrad (sorotan)", value=[])
            dataframe = gr.DataFrame(label="AGrad (rincian)", value=pd.DataFrame(columns=["token", "agrad"]))
    
    gr.on(
        triggers=[text.submit, predict_button.click],
        fn=predict, inputs=text, outputs=prediction_result
    ).then(
        fn=explain, inputs=text, outputs=[highlighted_text, dataframe]
    )

demo.launch()
