"""Gradio layout and event wiring for the voice studio."""

import gradio as gr

from .config import HARDWARE, STEP_RECOMMENDATION
from .speech_models import FULL_MODEL, MODEL_CHOICES
from .generation import (
    cancel_generation, generate_named_take, model_description, preview_estimate, switch_model,
)
from .voices import (
    SCRIPT_SAMPLES, add_permanent_voice, save_generated_voice, styles_for_model, voices_for_model,
)


def lock_for_generation():
    return (
        gr.update(value="Generating…", interactive=False),
        gr.update(interactive=False),
        "**Preparing audio…**",
        gr.update(interactive=True),
    )


def unlock_after_generation():
    return (
        gr.update(value="Generate audio", interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def lock_for_voice_save():
    """
    Disable important actions while saving a voice.
    """

    return (
        gr.Button(
            value="Generate audio",
            interactive=False,
        ),

        gr.Button(
            value="Saving…",
            interactive=False,
        ),

        "⏳ **Saving permanent voice...**",
    )


def unlock_after_voice_save():
    return (
        gr.Button(
            value="Generate audio",
            interactive=True,
        ),

        gr.Button(
            value="Save voice",
            interactive=True,
        ),
    )


STUDIO_CSS = """
.gradio-container { width: 100% !important; max-width: 1100px !important; box-sizing: border-box;
    margin: auto; padding: 28px 24px !important;
    --block-label-background-fill: transparent; --block-label-text-color: var(--body-text-color);
    --block-title-background-fill: transparent; --block-title-text-color: var(--body-text-color); }
#studio-header { padding: 8px 0 26px; }
.studio-heading { display: flex; align-items: center; gap: 14px; }
.studio-mark { background: #0f766e; color: white; border-radius: 16px; width: 52px; height: 52px;
    display: flex; align-items: center; justify-content: center; font-size: 26px; }
.studio-heading h1 { margin: 0 !important; font-size: 27px !important; letter-spacing: -0.8px; }
.studio-heading p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 14px; }
#studio-tabs > .tab-nav { border-bottom: 1px solid var(--border-color-primary); margin-bottom: 22px; gap: 8px; }
#studio-tabs > .tab-nav button { padding: 12px 24px; font-size: 14px; }
#studio-tabs > .tabitem { padding: 0; border: none; background: transparent; }
.studio-card { border: 1px solid var(--border-color-primary) !important; border-radius: 20px !important;
    padding: 24px !important; background: var(--block-background-fill) !important; }
.studio-card h2 { font-size: 19px !important; margin: 0 0 4px !important; letter-spacing: -0.3px; }
.studio-card .section-intro p { color: var(--body-text-color-subdued); font-size: 13px; margin: 0 0 8px; }
#script-box textarea { font-size: 18px; line-height: 1.9; min-height: 240px; }
#script-box { border-radius: 14px; }
#generate-action { min-height: 50px; border-radius: 12px; font-size: 15px; }
#progress-card { padding: 12px 14px; border-radius: 12px; background: var(--background-fill-secondary); }
#progress-card p { margin: 0; font-size: 13px; }
#progress-card h3 { margin: 0 0 6px; font-size: 15px; }
#timing-preview { color: var(--body-text-color-subdued); font-size: 12px; }
#timing-preview p { margin: 0; }
#runtime-card { color: var(--body-text-color-subdued); font-size: 13px; }
#preview-card { gap: 16px; }
#audio-preview { min-height: 140px; border-radius: 14px; }
@media (max-width: 640px) {
    .gradio-container { padding: 16px 12px !important; }
    .studio-card { padding: 18px !important; }
    .studio-heading h1 { font-size: 23px !important; }
    #studio-tabs > .tab-nav button { padding: 10px 16px; }
}
"""


def build_app():
    """Build the interface without launching a server or loading a speech model."""
    with gr.Blocks(title="Voice Studio") as app:
        gr.HTML(
            '<div class="studio-heading"><div class="studio-mark" aria-hidden="true">ក</div>'
            '<div><h1>Voice Studio</h1><p>Khmer & English. Your words, your voice.</p></div></div>',
            elem_id="studio-header",
        )
        active_model = gr.State(FULL_MODEL)
        generated_take = gr.State(None)

        with gr.Tabs(elem_id="studio-tabs"):
            with gr.Tab("Create", id="create"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=320, elem_classes="studio-card"):
                        gr.Markdown("## Your script")
                        with gr.Row():
                            selected_voice = gr.Dropdown(
                                choices=voices_for_model(FULL_MODEL), value=voices_for_model(FULL_MODEL)[0],
                                label="Voice", filterable=True, scale=2,
                                info="Saved voices reuse a speaker. Design presets can change between takes.",
                            )
                            selected_style = gr.Dropdown(
                                choices=styles_for_model(FULL_MODEL), value="Natural", label="Style", scale=1,
                            )
                        text_input = gr.Textbox(
                            label="Script", show_label=False,
                            placeholder="សរសេរអត្ថបទខ្មែររបស់អ្នកនៅទីនេះ…",
                            lines=8,
                            value=SCRIPT_SAMPLES["Khmer"],
                            elem_id="script-box",
                        )
                        with gr.Accordion("Voice options", open=False):
                            custom_style = gr.Textbox(
                                label="Extra direction", placeholder="Warm and conversational…",
                                lines=2,
                            )
                            temporary_reference = gr.Audio(
                                label="Reference voice", sources=["upload", "microphone"],
                                type="filepath", format="wav",
                            )
                            gr.Markdown("A clean 10–20s clip overrides the selected voice.", elem_classes="section-intro")
                    with gr.Column(scale=2, min_width=300, elem_id="preview-card", elem_classes="studio-card"):
                        gr.Markdown("## Audio preview")
                        gr.Markdown("Create a take, then listen or download.", elem_classes="section-intro")
                        audio_output = gr.Audio(
                            label="Audio preview", show_label=False, type="filepath", format="wav",
                            interactive=False, elem_id="audio-preview",
                        )
                        generation_status = gr.Markdown(
                            "Ready when you are.", elem_id="progress-card", sanitize_html=True,
                        )
                        timing_preview = gr.Markdown(
                            "Estimate: **Run a sample first**", elem_id="timing-preview",
                        )
                        with gr.Row():
                            generate_button = gr.Button(
                                "Generate audio", variant="primary", size="lg", scale=3,
                                elem_id="generate-action",
                            )
                            cancel_button = gr.Button("Stop", interactive=False, scale=1, min_width=70)
                        download_button = gr.DownloadButton(label="Download WAV", variant="secondary")
                        with gr.Accordion("Keep this speaker", open=False):
                            gr.Markdown("Like this voice? Save it once, then choose its name for future scripts. A short, clear sample works best.")
                            generated_voice_name = gr.Textbox(label="Speaker name", placeholder="Dara, Sophea…")
                            save_generated_button = gr.Button("Save this speaker")
                            generated_voice_status = gr.Markdown()

            with gr.Tab("Voices", id="voices"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                        gr.Markdown("## Save a voice")
                        gr.Markdown("Record once. Reuse in any script.", elem_classes="section-intro")
                        new_voice_audio = gr.Audio(
                            label="Reference recording", sources=["upload", "microphone"],
                            type="filepath", format="wav",
                        )
                        gr.Markdown("Use a clean 10–20s clip of one speaker.", elem_classes="section-intro")
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                        gr.Markdown("## Voice details")
                        new_voice_language = gr.Dropdown(choices=list(SCRIPT_SAMPLES), value="Khmer", label="Language")
                        new_voice_name = gr.Textbox(label="Name", placeholder="Dara, Sophea…")
                        new_voice_description = gr.Textbox(
                            label="Description", placeholder="Warm, clear speaking voice", lines=3,
                        )
                        save_voice_button = gr.Button("Save voice", variant="primary")
                        permanent_voice_status = gr.Markdown("Saved voices appear in the Create tab.", elem_classes="section-intro")

            with gr.Tab("Settings", id="settings"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                        gr.Markdown("## Speech model")
                        model_choice = gr.Dropdown(
                            choices=MODEL_CHOICES, value=FULL_MODEL, label="Model", show_label=False,
                        )
                        model_info = gr.Markdown(model_description(FULL_MODEL), elem_classes="section-intro")
                        switch_model_button = gr.Button("Apply model", variant="secondary")
                        gr.Markdown(f"Device: **{HARDWARE['accelerator']}**", elem_id="runtime-card")
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                        gr.Markdown("## Generation")
                        inference_steps = gr.Slider(
                            minimum=4, maximum=30, value=STEP_RECOMMENDATION["default"], step=1,
                            label="Detail steps",
                            info=f"Lower is faster · Recommended: {STEP_RECOMMENDATION['range_min']}–{STEP_RECOMMENDATION['range_max']}",
                        )
                        with gr.Row():
                            quick_test_button = gr.Button("Fast · 4 steps", size="sm")
                            recommended_button = gr.Button("Recommended", size="sm")
                        with gr.Accordion("Advanced", open=False):
                            cfg_value = gr.Slider(
                                minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                                label="Voice guidance", info="Default: 2.0",
                            )
                with gr.Accordion("About timing & models", open=False):
                    gr.Markdown(
                        "Estimates learn from completed takes and exclude model loading. "
                        "Stop cancels at the next computation checkpoint.\n\n"
                        "MMS Khmer and English each have one voice, without cloning or style controls. "
                        "[CC-BY-NC 4.0](https://huggingface.co/facebook/mms-tts-khm) · Noncommercial only."
                    )

        quick_test_button.click(fn=lambda: 4, outputs=inference_steps, queue=False)
        recommended_button.click(
            fn=lambda: STEP_RECOMMENDATION["default"], outputs=inference_steps, queue=False,
        )

        switch_model_button.click(
            fn=switch_model, inputs=[model_choice, text_input],
            outputs=[
                active_model, model_info, selected_voice, selected_style, custom_style,
                temporary_reference, inference_steps, cfg_value, quick_test_button,
                recommended_button, generation_status, text_input,
            ],
            queue=False,
        )
        cancel_button.click(fn=cancel_generation, outputs=generation_status, queue=False)
        save_generated_button.click(
            fn=save_generated_voice,
            inputs=[generated_voice_name, generated_take, active_model],
            outputs=[selected_voice, generated_voice_name, generated_voice_status, temporary_reference],
        )
        estimate_inputs = [
            text_input, selected_voice, selected_style, custom_style,
            temporary_reference, inference_steps, active_model,
        ]
        for control in estimate_inputs:
            control.change(fn=preview_estimate, inputs=estimate_inputs, outputs=timing_preview, queue=False)

        save_voice_button.click(
            fn=lock_for_voice_save,
            inputs=[],
            outputs=[
                generate_button,
                save_voice_button,
                permanent_voice_status,
            ],
            queue=False,
        ).then(
            fn=add_permanent_voice,
            inputs=[
                new_voice_name,
                new_voice_description,
                new_voice_audio,
                new_voice_language,
                active_model,
            ],
            outputs=[
                selected_voice,
                new_voice_name,
                new_voice_description,
                new_voice_audio,
                permanent_voice_status,
            ],
            show_progress="full",
        ).then(
            fn=unlock_after_voice_save,
            inputs=[],
            outputs=[
                generate_button,
                save_voice_button,
            ],
            queue=False,
        )

        # ========================================================
        # GENERATE AUDIO
        #
        # 1. Lock buttons
        # 2. Generate
        # 3. Unlock buttons
        # ========================================================

        generate_button.click(
            fn=lock_for_generation,
            inputs=[],
            outputs=[
                generate_button,
                save_voice_button,
                generation_status,
                cancel_button,
            ],
            queue=False,
        ).then(
            fn=generate_named_take,
            inputs=[
                text_input,
                selected_voice,
                selected_style,
                custom_style,
                temporary_reference,
                cfg_value,
                inference_steps,
                active_model,
            ],
            outputs=[
                audio_output,
                download_button,
                generation_status,
                generated_take,
            ],
            show_progress="full",
        ).then(
            fn=unlock_after_generation,
            inputs=[],
            outputs=[
                generate_button,
                save_voice_button,
                cancel_button,
            ],
            queue=False,
        ).then(
            fn=preview_estimate, inputs=estimate_inputs, outputs=timing_preview, queue=False,
        )
    return app
