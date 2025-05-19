import gradio as gr
from four import do
import time

def search_and_format_markdown(input_text, use_svd):
    start_time = time.time()

    results = do(input_text, use_svd)
    elapsed = time.time() - start_time

    if not results:
        return f"### 😔 Brak wyników lub zbyt niskie podobieństwo.\n\n⏱️ Czas wyszukiwania: `{elapsed:.3f}` sekundy"

    output_lines = ["### 🔍 Wyniki wyszukiwania:"]
    for i, r in enumerate(results):
        doc = r['document_identifier']
        sim = r['similarity']

        if doc.startswith("http://") or doc.startswith("https://"):
            link = f"[{doc}]({doc})"
        else:
            link = f"{doc}"

        output_lines.append(f"**{i+1}.** {link}\nPodobieństwo: `{sim:.4f}`\n")

    output_lines.append(f"⏱️ **Czas wyszukiwania:** `{elapsed:.3f}` sekundy")

    return "\n".join(output_lines)

iface = gr.Interface(
    fn=search_and_format_markdown,
    inputs=[
        gr.Textbox(label="Wpisz zapytanie tekstowe"),
        gr.Checkbox(label="Użyj SVD (redukcja wymiarów)", value=False)
    ],
    outputs=gr.Markdown(label="Wyniki wyszukiwania"),
    title="Wyszukiwarka dokumentów",
    description="Znajdź dokumenty dopasowane do Twojego zapytania"
)

iface.launch()