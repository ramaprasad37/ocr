# OCR Pipeline

Python utilities for extracting multilingual text from scanned images or PDF files
using either Google Gemini or Google Document AI. The tools can optionally clean
and enhance images, return layout-aware JSON, generate human-friendly Unicode HTML,
and (when supported by the backend) translate the recognized text into a secondary language.

> Requires access to the Google Generative AI (Gemini) API (with key) or Google Document AI (with processor credentials).

---

## Features

- Image OCR with optional preprocessing (auto-contrast, sharpening, resizing).
- PDF-to-image conversion (`pdf2image`) with page-by-page OCR.
- Anti-hallucination QA steps: Strict visual extraction only - no guessing, inferring, or substituting words. Unreadable text is marked with `[...]` instead of being invented.
- Rich prompt guiding the model to detect language, ignore decorative text, supply bounding boxes, and report confidence scores.
- Structured JSON with per-block metadata, optional translations, usage stats, and cost estimates.
- Clean Unicode HTML output mirroring the document structure.
- Choice of backends: Google Gemini LLM or Google Document AI (processor-based OCR).
- CLI options to control preprocessing, translation, MIME type overrides, DPI, page limits, pricing, and output destinations.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

For optional cost estimation, supply model pricing (USD per million tokens) via CLI flags.

### (Optional) Google Document AI

To run with the Document AI backend, configure credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Create or reuse a Document AI processor ID (OCR or Form Parser), and note the **project ID**, **location**, and **processor ID**—you'll pass them to the CLI.

### Google API Key

1. Obtain a Gemini API key and place it in `config.txt` at the project root.
2. The file can contain either `api_key=YOUR_KEY` or simply `YOUR_KEY`.

---

## Usage

### Image OCR (`ocr_jpg.py`)

Single image:

```bash
python ocr_jpg.py path/to/image.jpg \
  --language Malayalam \
  --translate-to English \
  --preprocess \
  --output out.json \
  --html-output out.html
```

**Process a folder:** pass a directory path; each image is processed and written as `<filename>.html` in the same folder (content-only HTML, no metadata). Optional `--output <dir>` writes `<stem>.json` for each image into that directory.

```bash
python ocr_jpg.py path/to/images/ --language Malayalam
# Writes path/to/images/photo1.html, path/to/images/photo2.html, ...
```

Key options:

- `path`: image file or folder of images. For a folder, each image is written as `<stem>.html` (content-only) in the same folder.
- `--language`: target language name (default `Malayalam`).
- `--translate-to`: optional translation language for each block.
- `--preprocess/--no-preprocess`: toggle image cleanup (default on).
- `--resize-long-edge`: max pixels for longest edge when preprocessing (default 2048).
- `--prompt`: custom instruction override.
- `--mime-type`: explicit MIME type when inference is unreliable.
- `--output`: JSON path (file for single image; directory for folder, writes `<stem>.json` per image).
- `--html-output`: HTML path (single image only). Folder mode always writes content-only `<stem>.html` next to each image.
- `--raw-output`: print raw Gemini response even if JSON parsing succeeds.

> Need Document AI for single images? Use `ocr.py` with `--mode image --engine docai` instead of `ocr_jpg.py`.

### PDF OCR (`ocr.py`)

```bash
python ocr.py path/to/document.pdf \
  --mode pdf \
  --language Malayalam \
  --engine gemini \
  --translate-to English \
  --preprocess \
  --pdf-dpi 300 \
  --page-limit 5 \
  --pricing-input-per-million 0.07 \
  --pricing-output-per-million 0.21 \
  --output document.json \
  --html-output document.html
```

To switch to Google Document AI (no translation, no token pricing needed), provide processor details:

```bash
python ocr.py path/to/document.pdf \
  --mode pdf \
  --engine docai \
  --docai-project YOUR_PROJECT_ID \
  --docai-location us \
  --docai-processor YOUR_PROCESSOR_ID \
  --preprocess \
  --output document_docai.json \
  --html-output document_docai.html
```

Document AI currently returns layout metadata and text only; translation and token-based cost estimation are not available in this mode.

Important CLI flags:

- `input_path`: image or PDF file.
- `--mode`: `auto` (default), `image`, or `pdf` to force processing.
- `--translate-to`: optional translation target language.
- `--preprocess/--no-preprocess`: enable or disable preprocessing.
- `--resize-long-edge`: resize threshold for preprocessing.
- `--pdf-dpi`: DPI used when rasterizing PDF pages (default 300).
- `--page-limit`: process only the first *N* pages of a PDF.
- `--output`: JSON file capturing layout metadata.
- `--html-output`: clean Unicode HTML reconstruction of the document.
- `--pricing-input-per-million` / `--pricing-output-per-million`: supply pricing to record per-page and total cost estimates.
- `--engine`: choose the OCR backend (`gemini` or `docai`).
- `--docai-project`, `--docai-location`, `--docai-processor`: required when using Document AI.

Outputs contain:

- JSON: list of pages with per-block text, bounding boxes, direction, language detection, confidence, optional translations (Gemini only), usage metadata, and cost estimates (when pricing is provided).
- HTML: styled, readable text in Unicode with original paragraph ordering and optional translation sections.

---

## Tips & Notes

- Install `poppler` or equivalent PDF rendering backend required by `pdf2image`.
- Preprocessing needs Pillow; if absent, the scripts fall back to the original bytes.
- `google.genai.types.Part` is imported at runtime; ensure the SDK is properly installed to avoid IDE warnings.
- Document AI requires a processor and service-account credentials; translation and token-cost estimates are not available in this mode.
- You can add or remove prompt steps for specific use cases—see `extract_text_from_image`.
- For batch processing or automation, wrap the CLI in your own scripts or invoke the functions directly.

---

## Troubleshooting

- **Missing dependencies**: run `pip install -r requirements.txt` to install all required packages.
- **Document AI import errors**: verify `GOOGLE_APPLICATION_CREDENTIALS` (or other auth) is configured.
- **pdf2image errors**: ensure `poppler` is installed (`brew install poppler` on macOS, `apt install poppler-utils` on Debian/Ubuntu).
- **API errors**: verify `config.txt` contains a valid Gemini API key and network access is available.
- **Raw JSON in HTML**: ensure you’re using the latest script version; current HTML output composes clean paragraphs without embedding JSON.

---

## License

MIT License unless otherwise noted. Adjust this section to match your project’s license.


Example:

python ocr.py LSMalayalam.pdf --language Malayalam --preprocess  --html-output Lsmalayalam.html  --output lsmalayalam.json --page-limit 10 --engine gemini --pricing-input-per-million 0.15 --pricing-output-per-million 0.6
