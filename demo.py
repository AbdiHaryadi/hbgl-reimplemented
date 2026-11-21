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
    return format_for_summary(result, named_traces), named_traces, tokens, agrads_result

def format_for_summary(result: list[str], named_traces: list[list[str]]):
    result_text = list_to_set_text(result)
    named_traces_text = "\n".join([
        f"Tingkat {level_index + 1}: " + list_to_set_text(trace)
        for level_index, trace in enumerate(named_traces)
    ])
    return result_text + "\n\n" + named_traces_text

def visualize_agrad(tokens: list[str], agrads_result: list[list[float]], level: int, is_global_scale: bool):
    level_index = level - 1
    if is_global_scale:
        abs_std = np.abs(np.asarray([y for x in agrads_result for y in x])).std()
    else:
        abs_std = np.abs(np.asarray(agrads_result[level_index])).std()
    
    if abs_std != 0:
        scores = np.tanh(np.asarray(agrads_result[level_index]) / abs_std)
    else:
        scores = np.tanh(np.asarray(agrads_result[level_index]))

    try:
        highlighted_text_output = (
            [("[CLS]", float(scores[0]))]
            + [(t, float(a)) for t, a in zip(tokens, scores[1:1+len(tokens)], strict=True)]
            + [("[SEP]", float(scores[1+len(tokens)]))]
            + [(f"[u{prev_level_index + 1}]", scores[2+len(tokens)+prev_level_index]) for prev_level_index in range(level_index)]
            + [("[MASK]", float(scores[2+len(tokens)+level_index]))]
        )
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
            prediction_result = gr.Textbox(
                label="Hasil Prediksi Keseluruhan",
                value=format_for_summary([], [[] for _ in range(pipeline.max_tgt_length)])
            )

    with gr.Row():
        level_slider = gr.Slider(
            minimum=1, maximum=5, step=1, value=1, label="Tingkat"
        )
        level_prediction_result = gr.Textbox(label="Hasil Prediksi Spesifik Tingkat", value=list_to_set_text([]))

    highlighted_text = gr.HighlightedText(label="AGrad (sorotan)", value=[])
    is_global_scale = gr.Checkbox(label="Skala warna global", value=False)
    dataframe = gr.DataFrame(label="AGrad (rincian)", value=pd.DataFrame(columns=["token"] + [
        f"agrad_{level_index + 1}" for level_index in range(pipeline.max_tgt_length)
    ]))

    named_traces = gr.State([[] for _ in range(pipeline.max_tgt_length)])
    tokens = gr.State([])
    agrads_result = gr.State([[] for _ in range(pipeline.max_tgt_length)])
    gr.on(
        triggers=[text.submit, predict_button.click],
        fn=predict, inputs=text, outputs=[prediction_result, named_traces, tokens, agrads_result]
    )
    gr.on(
        triggers=[tokens.change, agrads_result.change, level_slider.change, is_global_scale.change],
        fn=visualize_agrad, inputs=[tokens, agrads_result, level_slider, is_global_scale],
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
