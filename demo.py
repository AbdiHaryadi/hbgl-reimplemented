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
    result, named_traces, tokens, agrads_result = pipeline(text)
    result_text = list_to_set_text(result)
    named_traces_text = "\n".join([
        f"Tingkat {level_index + 1}: " + list_to_set_text(trace)
        for level_index, trace in enumerate(named_traces)
    ])
    return result_text + "\n\n" + named_traces_text, tokens, [x.tolist() for x in agrads_result]

def new_predict(text: str):
    result, named_traces, tokens, agrads_result = pipeline(text)
    return format_for_summary(result, named_traces), named_traces, tokens, agrads_result

def format_for_summary(result: list[str], named_traces: list[list[str]]):
    result_text = list_to_set_text(result)
    named_traces_text = "\n".join([
        f"Tingkat {level_index + 1}: " + list_to_set_text(trace)
        for level_index, trace in enumerate(named_traces)
    ])
    return result_text + "\n\n" + named_traces_text

def visualize_agrad(tokens: list[str], agrads_result: list[list[float]], level: int):
    level_index = level - 1
    agrads = np.array(agrads_result[level_index][1:1+len(tokens)])
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
        highlighted_text_output
    )

def show_prediction_per_level(named_traces: list[list[str]], level: int):
    level_index = level - 1
    return list_to_set_text(named_traces[level_index])

def show_dataframe(tokens: list[str], agrads_result: list[list[float]]):
    return pd.DataFrame({
        "token": tokens,
    } | {
        f"agrad_{level_index + 1}": agrads[1:1+len(tokens)] for level_index, agrads in enumerate(agrads_result)
    })

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Teks")
            predict_button = gr.Button("Prediksi")

        with gr.Column():
            prediction_result = gr.Textbox(label="Hasil Prediksi Keseluruhan", value=format_for_summary([], [[] for _ in range(5)]))
            level_slider = gr.Slider(
                minimum=1, maximum=5, step=1, value=1, label="Tingkat"
            )
            level_prediction_result = gr.Textbox(label="Hasil Prediksi Spesifik Tingkat", value=list_to_set_text([]))
            highlighted_text = gr.HighlightedText(label="AGrad (sorotan)", value=[])
            dataframe = gr.DataFrame(label="AGrad (rincian)", value=pd.DataFrame(columns=["token"] + [
                f"agrad_{level_index + 1}" for level_index in range(5)
            ]))

    named_traces = gr.State([[] for _ in range(5)])
    tokens = gr.State([])
    agrads_result = gr.State([[] for _ in range(5)])
    gr.on(
        triggers=[text.submit, predict_button.click],
        fn=new_predict, inputs=text, outputs=[prediction_result, named_traces, tokens, agrads_result]
    )
    gr.on(
        triggers=[tokens.change, agrads_result.change, level_slider.change],
        fn=visualize_agrad, inputs=[tokens, agrads_result, level_slider],
        outputs=highlighted_text
    )
    gr.on(
        triggers=[tokens.change, agrads_result.change],
        fn=show_dataframe, inputs=[tokens, agrads_result], outputs=dataframe
    )
    gr.on(
        triggers=[named_traces.change, level_slider.change],
        fn=show_prediction_per_level, inputs=[named_traces, level_slider], outputs=level_prediction_result
    )

demo.launch()
