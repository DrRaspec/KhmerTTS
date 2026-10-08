"""Launch the Khmer and English voice studio."""

import gradio as gr

from studio.config import HARDWARE, RUNTIME_DEVICE
from studio.ui import STUDIO_CSS, build_app


def main():
    print(f"Hardware: {HARDWARE['accelerator']} ({RUNTIME_DEVICE})")
    print("Studio ready. Speech models load only when you generate audio.")
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        max_file_size="100mb",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=STUDIO_CSS,
    )


if __name__ == "__main__":
    main()
