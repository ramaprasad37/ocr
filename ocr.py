import argparse
import html
import io
import json
import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai.types import Part
from pdf2image import convert_from_path

try:
    from google.cloud import documentai_v1 as documentai

    HAS_DOCUMENT_AI = True
except ImportError:
    documentai = None
    HAS_DOCUMENT_AI = False

ENGINE_GEMINI = "gemini"
ENGINE_DOCAI = "docai"


# 1. Setup: Read API key from config file
def load_api_key(config_file: str = "config.txt") -> str:
    """
    Reads the API key from a config file.
    Expected format: api_key=YOUR_ACTUAL_API_KEY
    Or simply: YOUR_ACTUAL_API_KEY
    """
    try:
        with open(config_file, "r") as file:
            content = file.read().strip()
            # Check if it's in key=value format
            if "=" in content:
                api_key = content.split("=")[1].strip()
            else:
                api_key = content
            return api_key
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        print("Please create a config.txt file with your API key.")
        raise
    except Exception as e:
        print(f"Error reading config file: {e}")
        raise


# Load API key from config file
api_key = load_api_key("config.txt")
client = genai.Client(api_key=api_key) 


def _read_image_bytes(image_path: str) -> Tuple[bytes, str]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    return image_bytes, mime_type


def preprocess_image_path(
    image_path: str,
    resize_long_edge: Optional[int] = 2048,
) -> Tuple[bytes, str]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # type: ignore
    except ImportError:
        print("Pillow is not installed. Skipping preprocessing.")
        return _read_image_bytes(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")

        if resize_long_edge:
            long_edge = max(img.size)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / long_edge
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.LANCZOS)

        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"


def preprocess_image_bytes(
    image_bytes: bytes,
    resize_long_edge: Optional[int] = 2048,
) -> Tuple[bytes, str]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # type: ignore
    except ImportError:
        print("Pillow is not installed. Skipping preprocessing.")
        return image_bytes, "image/png"

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")

        if resize_long_edge:
            long_edge = max(img.size)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / long_edge
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.LANCZOS)

        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"


def _ensure_docai_config(docai_config: Optional[Dict[str, str]]) -> Dict[str, str]:
    if docai_config is None:
        raise ValueError("Document AI configuration is required when engine='docai'.")
    if not HAS_DOCUMENT_AI:
        raise ImportError(
            "google-cloud-documentai is not installed. "
            "Install it with `pip install google-cloud-documentai`."
        )
    required_keys = ("project_id", "location", "processor_id")
    missing = [key for key in required_keys if not docai_config.get(key)]
    if missing:
        raise ValueError(
            "Missing Document AI configuration values: " + ", ".join(missing)
        )
    return {
        "project_id": docai_config["project_id"],
        "location": docai_config["location"],
        "processor_id": docai_config["processor_id"],
    }


def _get_document_ai_client(docai_config: Dict[str, str]):
    client = documentai.DocumentProcessorServiceClient()
    processor_name = client.processor_path(
        docai_config["project_id"],
        docai_config["location"],
        docai_config["processor_id"],
    )
    return client, processor_name


def _process_document_ai_bytes(
    data: bytes,
    mime_type: str,
    docai_config: Dict[str, str],
):
    client, processor_name = _get_document_ai_client(docai_config)
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=data,
            mime_type=mime_type,
        ),
    )
    result = client.process_document(request=request)
    return result.document


def _document_ai_layout_text(layout: Any, full_text: str) -> str:
    if not layout or not getattr(layout, "text_anchor", None):
        return ""
    text_anchor = layout.text_anchor
    segments = getattr(text_anchor, "text_segments", None)
    if not segments:
        return ""
    pieces: List[str] = []
    for segment in segments:
        start_index = getattr(segment, "start_index", 0) or 0
        end_index = getattr(segment, "end_index", 0) or 0
        if end_index < start_index:
            continue
        pieces.append(full_text[start_index:end_index])
    return "".join(pieces)


def _document_ai_orientation(layout: Any) -> str:
    orientation = getattr(layout, "orientation", None)
    if not HAS_DOCUMENT_AI or orientation is None:
        return "unknown"

    mapping = {
        documentai.Document.Page.Layout.Orientation.PAGE_UP: "horizontal",
        documentai.Document.Page.Layout.Orientation.PAGE_DOWN: "horizontal",
        documentai.Document.Page.Layout.Orientation.PAGE_RIGHT: "vertical",
        documentai.Document.Page.Layout.Orientation.PAGE_LEFT: "vertical",
    }
    return mapping.get(orientation, "unknown")


def _document_ai_bounding_box(layout: Any, width: float, height: float) -> Optional[List[int]]:
    bounding_poly = getattr(layout, "bounding_poly", None)
    if not bounding_poly:
        return None

    def _from_vertices(vertices, scale_x: float = 1.0, scale_y: float = 1.0):
        xs = [v.x * scale_x for v in vertices if getattr(v, "x", None) is not None]
        ys = [v.y * scale_y for v in vertices if getattr(v, "y", None) is not None]
        if not xs or not ys:
            return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        left = int(round(min_x))
        top = int(round(min_y))
        width_box = int(round(max(max_x - min_x, 0)))
        height_box = int(round(max(max_y - min_y, 0)))
        return [left, top, width_box, height_box]

    normalized_vertices = getattr(bounding_poly, "normalized_vertices", None)
    if normalized_vertices:
        scale_x = width or 1.0
        scale_y = height or 1.0
        bbox = _from_vertices(normalized_vertices, scale_x, scale_y)
        if bbox:
            return bbox

    vertices = getattr(bounding_poly, "vertices", None)
    if vertices:
        return _from_vertices(vertices, 1.0, 1.0)

    return None


def _document_ai_detected_language(page: Any) -> Optional[str]:
    detected_languages = getattr(page, "detected_languages", None) or []
    if not detected_languages:
        return None
    best = max(
        detected_languages,
        key=lambda item: getattr(item, "confidence", 0.0) or 0.0,
    )
    return getattr(best, "language_code", None)


def _document_ai_blocks_from_page(page: Any, full_text: str, start_order: int) -> Tuple[List[Dict[str, Any]], int]:
    width = float(getattr(getattr(page, "dimension", None), "width", 0.0)) or 1.0
    height = float(getattr(getattr(page, "dimension", None), "height", 0.0)) or 1.0
    elements = (
        getattr(page, "paragraphs", None)
        or getattr(page, "lines", None)
        or getattr(page, "tokens", None)
        or []
    )

    blocks: List[Dict[str, Any]] = []
    order = start_order
    page_language = _document_ai_detected_language(page) or "unknown"

    for element in elements:
        layout = getattr(element, "layout", None)
        if not layout:
            continue
        text = _document_ai_layout_text(layout, full_text).strip()
        if not text:
            continue

        language_code = getattr(layout, "language_code", None) or page_language
        block = {
            "order": order,
            "bounding_box": _document_ai_bounding_box(layout, width, height),
            "text_direction": _document_ai_orientation(layout),
            "text_language": language_code or "unknown",
            "text": text,
        }
        confidence = getattr(layout, "confidence", None)
        if confidence is not None:
            block["confidence"] = round(float(confidence), 4)
        blocks.append(block)
        order += 1

    return blocks, order


def _document_ai_to_structured(document_obj: Any) -> Dict[str, Any]:
    full_text = getattr(document_obj, "text", "") or ""
    blocks: List[Dict[str, Any]] = []
    order = 1
    language_weights: defaultdict[str, float] = defaultdict(float)

    for page in getattr(document_obj, "pages", []):
        page_blocks, order = _document_ai_blocks_from_page(page, full_text, order)
        for block in page_blocks:
            text_length = len(block.get("text", ""))
            language = block.get("text_language", "unknown")
            language_weights[language] += text_length
        blocks.extend(page_blocks)

    language_detected = "unknown"
    if language_weights:
        language_detected = max(language_weights.items(), key=lambda item: item[1])[0]

    return {
        "language_detected": language_detected,
        "blocks": blocks,
    }


def _dict_items_to_list(data: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key, value in data.items():
        if value is None:
            continue
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return lines


def extract_text_from_image(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    language: str = "Malayalam",
    prompt: Optional[str] = None,
    mime_type: Optional[str] = None,
    translate_to: Optional[str] = None,
    preprocess: bool = False,
    resize_long_edge: Optional[int] = 2048,
    pricing_input_per_million: Optional[float] = None,
    pricing_output_per_million: Optional[float] = None,
    engine: str = ENGINE_GEMINI,
    docai_config: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts text from an image file or bytes and returns structured layout metadata.

    Args:
        image_path: Path to an image file on disk.
        image_bytes: Raw image bytes (used for in-memory images like PDF pages).
        language: Target language for the extracted Unicode text.
        prompt: Optional custom instruction for the model. Overrides the default.
        mime_type: Optional MIME type. Required when providing image_bytes without preprocessing.
        translate_to: Optional language to include per-block translation (Gemini engine only).
        preprocess: Whether to apply basic image cleanup before OCR.
        resize_long_edge: Optional maximum size for the longest edge during preprocessing.
        pricing_input_per_million: Optional USD price per million prompt tokens for cost estimation.
        pricing_output_per_million: Optional USD price per million completion tokens for cost estimation.
        engine: OCR backend to use ("gemini" or "docai").
        docai_config: Configuration dictionary required when engine='docai'.
    
    Returns:
        Dictionary containing the extracted content, raw text, usage, cost, and metadata.
        Returns None if an unrecoverable error occurs.
    """

    if not image_path and image_bytes is None:
        raise ValueError("Either image_path or image_bytes must be provided.")
    if image_path and image_bytes is not None:
        raise ValueError("Provide only one of image_path or image_bytes.")

    # 2. Load or preprocess the Image Data
    print("Preparing image data...")
    try:
        if image_path:
            if preprocess:
                image_bytes_local, inferred_mime = preprocess_image_path(
                    image_path,
                    resize_long_edge=resize_long_edge,
                )
            else:
                image_bytes_local, inferred_mime = _read_image_bytes(image_path)
        else:
            if preprocess:
                image_bytes_local, inferred_mime = preprocess_image_bytes(
                    image_bytes,
                    resize_long_edge=resize_long_edge,
                )
            else:
                if not mime_type:
                    inferred_mime = "image/png"
                else:
                    inferred_mime = mime_type
                image_bytes_local = image_bytes
    except FileNotFoundError:
        if image_path:
            print(f"Error: File not found at {image_path}")
        else:
            print("Error: Image bytes could not be processed.")
        return None
    except Exception as exc:
        print(f"Unexpected error while preparing image data: {exc}")
        return None

    final_mime_type = mime_type or inferred_mime

    if engine == ENGINE_DOCAI:
        try:
            validated_config = _ensure_docai_config(docai_config)
            if translate_to:
                print("Warning: Document AI backend does not support translation; 'translate_to' will be ignored.")
            print("Sending request to Document AI...")
            document_obj = _process_document_ai_bytes(
                data=image_bytes_local,
                mime_type=final_mime_type,
                docai_config=validated_config,
            )
            structured_content = _document_ai_to_structured(document_obj)
            print(f"\n--- Extracted {language} Layout-Aware Text (Document AI) ---")
            for block in structured_content.get("blocks", []):
                print(block.get("text", ""))
            print("--------------------------------------")
            return {
                "content": structured_content,
                "raw_text": getattr(document_obj, "text", ""),
                "usage": None,
                "cost": None,
                "engine": ENGINE_DOCAI,
            }
        except Exception as exc:
            print(f"\nAn error occurred during the Document AI call: {exc}")
            return {
                "content": None,
                "raw_text": None,
                "usage": None,
                "cost": None,
                "engine": ENGINE_DOCAI,
                "error": str(exc),
            }

    # 3. Create the Image Part for the API
    image_part = Part.from_bytes(
        data=image_bytes_local,
        mime_type=final_mime_type,
    )

    # 4. Define the Prompt (Your Instruction)
    if prompt:
        instruction = prompt
    else:
        steps: List[str] = [
            "1. Language Detection: Identify the primary language of the text visible in the image.",
            "2. Ignore Background/Design: Exclude any text that is part of background design, watermarks, or decorative elements. Focus only on the main article/body text.",
            "3. CRITICAL - Strict Visual Extraction Only: Extract ONLY text that is clearly visible and readable in the image. DO NOT infer, guess, or substitute words. DO NOT use context to fill in missing words. DO NOT change the meaning or use alternate words. If any character, word, or phrase is unclear, blurry, partially obscured, or unreadable, replace it EXACTLY with [...]. Never invent text that is not clearly visible.",
            "4. Layout Metadata: Segment the document into logical blocks (paragraphs, headings, captions). For each block capture the bounding box as [x, y, width, height] in pixels relative to the original image, the reading order, text direction (horizontal or vertical), and a confidence score between 0 and 1.",
            "5. Quality Assurance Check: Before finalizing each block, verify that every character in the extracted text is actually visible in the image. If you are uncertain about any character, replace the entire uncertain word or phrase with [...]. Set confidence score to 0.5 or lower if any part of the block contains [...], indicating uncertainty.",
        ]
        steps.append(
            f"{len(steps) + 1}. Output Text: Present the extracted text as high-quality Unicode in {language}, maintaining original headings and paragraphs where possible. Use [...] for any unreadable portions - do not attempt to guess or infer missing text."
        )

        translation_structure_line = ""
        if translate_to:
            steps.append(
                f"{len(steps) + 1}. Translation: For each block include a translation field rendered in {translate_to} while preserving meaning and tone. Use null if translation is not possible. If the original text contains [...], the translation should also indicate uncertainty."
            )
            translation_structure_line = '      "translation": "<translated text>",\n'

        steps.append(
            f"{len(steps) + 1}. Output Format: Return a strict JSON object with the following structure:"
        )

        instruction = (
            "You are a strict, accuracy-focused OCR system. Your ONLY job is to extract text that is CLEARLY VISIBLE in the image. "
            "CRITICAL RULES:\n"
            "- Extract ONLY what you can see clearly. Do NOT guess, infer, or substitute words.\n"
            "- If text is unclear, blurry, or partially visible, replace it with [...]\n"
            "- Do NOT use context clues to fill in missing words.\n"
            "- Do NOT change the meaning or use alternate words.\n"
            "- Do NOT hallucinate text that is not present in the image.\n\n"
            "Analyze the provided image strictly and follow these steps:\n"
            + "\n".join(steps)
            + "\n{\n"
            '  "language_detected": "<ISO 639-1 or descriptive language name>",\n'
            '  "blocks": [\n'
            "    {\n"
            '      "order": <integer reading order>,\n'
            '      "bounding_box": [x, y, width, height],\n'
            '      "text_direction": "horizontal" | "vertical" | "mixed",\n'
            '      "text_language": "<language of the block>",\n'
            '      "text": "<extracted Unicode text - use [...] for unreadable portions>",\n'
            f"{translation_structure_line}"
            '      "confidence": <float between 0 and 1 - use lower scores if [...] is present>\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Ensure the JSON is valid and escape newlines within strings as \\n."
        )

    # 5. Generate Content (The API Call)
    print("Sending request to Gemini API...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                instruction,
                image_part,
            ],
        )

        # 6. Output the Result
        print(f"\n--- Extracted {language} Layout-Aware Text ---")
        raw_output = response.text or ""
        parsed_output: Optional[Union[str, Dict[str, Any]]]

        try:
            parsed_output = json.loads(raw_output)
            print(json.dumps(parsed_output, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            parsed_output = raw_output
            print(raw_output)

        usage_metadata = getattr(response, "usage_metadata", None)
        usage_dict: Optional[Dict[str, Any]] = None
        if usage_metadata:
            usage_dict = {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(usage_metadata, "total_token_count", None),
                "input_tokens": getattr(usage_metadata, "input_token_count", None),
                "output_tokens": getattr(usage_metadata, "output_token_count", None),
            }
            usage_dict = {
                key: value
                for key, value in usage_dict.items()
                if value is not None
            } or None

        cost_summary: Optional[Dict[str, float]] = None
        if usage_dict and (pricing_input_per_million or pricing_output_per_million):
            prompt_tokens = (
                usage_dict.get("prompt_tokens")
                or usage_dict.get("input_tokens")
                or 0
            ) or 0
            completion_tokens = (
                usage_dict.get("completion_tokens")
                or usage_dict.get("output_tokens")
                or 0
            ) or 0

            input_cost = (
                (prompt_tokens / 1_000_000.0) * pricing_input_per_million
                if pricing_input_per_million is not None
                else 0.0
            )
            output_cost = (
                (completion_tokens / 1_000_000.0) * pricing_output_per_million
                if pricing_output_per_million is not None
                else 0.0
            )
            total_cost = input_cost + output_cost
            cost_summary = {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
            }

        if usage_dict:
            print("Usage metadata:", json.dumps(usage_dict, indent=2))
        if cost_summary:
            print("Estimated cost (USD):", json.dumps(cost_summary, indent=2))

        print("--------------------------------------")
        return {
            "content": parsed_output,
            "raw_text": raw_output,
            "usage": usage_dict,
            "cost": cost_summary,
            "engine": ENGINE_GEMINI,
        }

    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")
        return {
            "content": None,
            "raw_text": None,
            "usage": None,
            "cost": None,
            "engine": ENGINE_GEMINI,
            "error": str(e),
        }


def pdf_to_layout_json(
    pdf_path: str,
    language: str = "Malayalam",
    prompt: Optional[str] = None,
    translate_to: Optional[str] = None,
    preprocess: bool = False,
    resize_long_edge: Optional[int] = 2048,
    dpi: int = 300,
    page_limit: Optional[int] = None,
    pricing_input_per_million: Optional[float] = None,
    pricing_output_per_million: Optional[float] = None,
    engine: str = ENGINE_GEMINI,
    docai_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Converts a PDF file page by page to structured OCR output with layout metadata.

    Args:
        pdf_path: The file path to the PDF.
        language: Target language for extraction.
        prompt: Optional custom instruction override.
        translate_to: Optional translation target (Gemini engine only).
        preprocess: Whether to apply image cleanup to each rendered page.
        resize_long_edge: Maximum size for the longest edge when preprocessing.
        dpi: Rendering DPI for PDF rasterisation.
        page_limit: Limit the number of pages processed.
        pricing_input_per_million: Prompt token pricing (USD per million).
        pricing_output_per_million: Completion token pricing (USD per million).
        engine: OCR backend to use ("gemini" or "docai").
        docai_config: Configuration dictionary required when engine='docai'.
    """

    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    print(f"Processing PDF: {pdf_path}")
    print("Converting PDF pages to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    total_pages = len(pages)
    print(f"Found {total_pages} pages in the PDF.")

    results: List[Dict[str, Any]] = []
    usage_totals: defaultdict[str, float] = defaultdict(float)
    cost_totals: defaultdict[str, float] = defaultdict(float)
    pages_processed = 0

    for page_index, page in enumerate(pages, start=1):
        if page_limit is not None and page_index > page_limit:
            break

        print(f"\n--- Processing page {page_index}/{total_pages} ---")
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        page_payload = extract_text_from_image(
            image_bytes=image_bytes,
            language=language,
            prompt=prompt,
            mime_type="image/png",
            translate_to=translate_to,
            preprocess=preprocess,
            resize_long_edge=resize_long_edge,
            pricing_input_per_million=pricing_input_per_million,
            pricing_output_per_million=pricing_output_per_million,
        )

        if not isinstance(page_payload, dict):
            page_payload = {
                "content": page_payload,
                "raw_text": None,
                "usage": None,
                "cost": None,
            }

        content = page_payload.get("content")
        usage = page_payload.get("usage")
        cost = page_payload.get("cost")

        results.append(
            {
                "page": page_index,
                "result": content,
                "usage": usage,
                "cost": cost,
                "raw_text": page_payload.get("raw_text"),
                "error": page_payload.get("error"),
                "engine": page_payload.get("engine", engine),
            }
        )
        pages_processed += 1

        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] += value

        if isinstance(cost, dict):
            for key, value in cost.items():
                if isinstance(value, (int, float)):
                    cost_totals[key] += value

    return {
        "pdf": str(pdf_path_obj.resolve()),
        "language": language,
        "translate_to": translate_to,
        "total_pages": total_pages,
        "page_limit": page_limit,
        "pages_processed": pages_processed,
        "pages": results,
        "usage_summary": dict(usage_totals) if usage_totals else None,
        "cost_summary": {k: round(v, 6) for k, v in cost_totals.items()} if cost_totals else None,
        "engine": engine,
    }


def _normalize_paragraphs(text: str) -> List[str]:
    raw_paragraphs = [para.strip() for para in text.split("\n\n")]
    return [para for para in raw_paragraphs if para]


def _blocks_to_paragraphs(blocks: List[Dict[str, Any]]) -> List[str]:
    ordered_blocks = sorted(
        [
            block
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ],
        key=lambda b: b.get("order", 0),
    )
    paragraphs: List[str] = []
    for block in ordered_blocks:
        text = block.get("text", "")
        if text:
            paragraphs.extend(_normalize_paragraphs(text))
    return paragraphs


def _blocks_to_translation_paragraphs(blocks: List[Dict[str, Any]]) -> List[str]:
    translated_blocks = [
        block
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("translation"), str)
    ]
    if not translated_blocks:
        return []

    ordered = sorted(
        translated_blocks,
        key=lambda b: b.get("order", 0),
    )
    paragraphs: List[str] = []
    for block in ordered:
        text = block.get("translation", "")
        if text:
            paragraphs.extend(_normalize_paragraphs(text))
    return paragraphs


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _coerce_structured_result(
    data: Union[str, List[Any], Dict[str, Any]]
) -> Union[str, Dict[str, Any]]:
    if isinstance(data, str):
        text = _strip_code_fences(data)
        try:
            parsed = json.loads(text)
            data = parsed
        except json.JSONDecodeError:
            return text

    if isinstance(data, list):
        return {"language_detected": "unknown", "blocks": data}

    return data


def _extract_paragraphs_from_string(text: str) -> List[str]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []

    paragraphs: List[str] = []

    pattern = re.compile(r'"text"\s*:\s*"(.*?)"', re.DOTALL)
    matches = pattern.findall(cleaned)
    for match in matches:
        try:
            decoded = json.loads(f'"{match}"')
        except json.JSONDecodeError:
            decoded = match
        decoded = decoded.strip()
        if decoded:
            paragraphs.extend(_normalize_paragraphs(decoded))

    if not paragraphs:
        paragraphs = _normalize_paragraphs(cleaned)

    return paragraphs


def generate_html_document(result: Union[str, Dict[str, Any]]) -> str:
    title = "OCR Output"
    structured_result = _coerce_structured_result(result)
    translate_to = None
    engine_label: Optional[str] = None
    if isinstance(structured_result, dict):
        engine_label = structured_result.get("engine")
    usage_override: Optional[Dict[str, Any]] = None
    cost_override: Optional[Dict[str, Any]] = None

    if isinstance(structured_result, dict) and "content" in structured_result and "pages" not in structured_result:
        usage_override = structured_result.get("usage")
        cost_override = structured_result.get("cost")
        structured_result = _coerce_structured_result(structured_result.get("content"))

    if isinstance(structured_result, dict):
        translate_to = structured_result.get("translate_to")

    stylesheet = """
        body {
            font-family: "Noto Sans", "Helvetica Neue", Arial, sans-serif;
            max-width: 840px;
            margin: 0 auto;
            padding: 32px 20px;
            background-color: #f5f5f5;
            color: #222;
        }
        h1 {
            text-align: center;
        }
        .document-meta {
            margin-bottom: 24px;
            color: #555;
        }
        .stats-heading {
            margin-top: 16px;
            font-weight: bold;
            color: #444;
        }
        .stats-list {
            margin: 6px 0 0 22px;
            padding: 0;
            list-style: disc;
            font-size: 0.9rem;
            color: #444;
        }
        .page {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        }
        .page-meta {
            margin-top: 12px;
            font-size: 0.9rem;
            color: #555;
        }
        .page-title {
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 12px;
            color: #333;
        }
        .page-language {
            font-style: italic;
            color: #666;
            margin-bottom: 20px;
        }
        .page-content p {
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1.2em;
            white-space: pre-wrap;
        }
        .page-content p:last-child {
            margin-bottom: 0;
        }
        .no-data {
            color: #888;
            font-style: italic;
        }
        .translation-title {
            margin-top: 20px;
            font-weight: bold;
        }
        .translation-content p {
            font-size: 0.95rem;
            line-height: 1.6;
            background-color: #eef6ff;
            border-left: 3px solid #2a7ae4;
            padding: 14px 18px;
            border-radius: 4px;
            margin-bottom: 1em;
        }
        .translation-content p:last-child {
            margin-bottom: 0;
        }
        footer {
            margin-top: 32px;
            font-size: 0.85rem;
            color: #777;
            text-align: center;
        }
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "    <meta charset=\"UTF-8\">",
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        f"    <title>{title}</title>",
        "    <style>",
        stylesheet,
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>{title}</h1>",
    ]

    if isinstance(structured_result, dict) and "pages" in structured_result:
        language = structured_result.get("language", "unknown")
        translate_note = f" Translation target: {translate_to}." if translate_to else ""
        engine_note = (
            f"<br>Engine: {html.escape(str(engine_label))}"
            if engine_label
            else ""
        )
        html_parts.append(
            f"    <div class=\"document-meta\">Source: {html.escape(structured_result.get('pdf', ''))}<br>"
            f"    Primary language requested: {html.escape(str(language))}.{translate_note}{engine_note}</div>"
        )

        usage_summary = structured_result.get("usage_summary")
        cost_summary = structured_result.get("cost_summary")
        if usage_summary:
            html_parts.append("    <div class=\"stats-heading\">Usage Summary</div>")
            html_parts.append("    <ul class=\"stats-list\">")
            for line in _dict_items_to_list(usage_summary):
                html_parts.append(f"        <li>{html.escape(line)}</li>")
            html_parts.append("    </ul>")
        if cost_summary:
            html_parts.append("    <div class=\"stats-heading\">Cost Summary</div>")
            html_parts.append("    <ul class=\"stats-list\">")
            for line in _dict_items_to_list(cost_summary):
                html_parts.append(f"        <li>{html.escape(line)}</li>")
            html_parts.append("    </ul>")

        for page_entry in structured_result.get("pages", []):
            page_number = page_entry.get("page", "?")
            page_result = page_entry.get("result")
            if page_result in (None, [], {}):
                raw_text = page_entry.get("raw_text")
                if isinstance(raw_text, str) and raw_text.strip():
                    page_result = raw_text
            page_result = _coerce_structured_result(page_result)
            html_parts.append("    <section class=\"page\">")
            html_parts.append(f"        <div class=\"page-title\">Page {page_number}</div>")

            if isinstance(page_result, dict):
                detected_language = page_result.get("language_detected", "unknown")
                html_parts.append(
                    f"        <div class=\"page-language\">Detected language: {html.escape(str(detected_language))}</div>"
                )
                paragraphs = _blocks_to_paragraphs(page_result.get("blocks", []))
                if paragraphs:
                    html_parts.append("        <div class=\"page-content\">")
                    for paragraph in paragraphs:
                        html_parts.append(f"            <p>{html.escape(paragraph)}</p>")
                    html_parts.append("        </div>")
                else:
                    html_parts.append("        <div class=\"no-data\">No blocks returned.</div>")

                translation_paragraphs = _blocks_to_translation_paragraphs(page_result.get("blocks", []))
                if translation_paragraphs:
                    html_parts.append('        <div class="translation-title">Translation</div>')
                    html_parts.append('        <div class="translation-content">')
                    for paragraph in translation_paragraphs:
                        html_parts.append(f'            <p>{html.escape(paragraph)}</p>')
                    html_parts.append("        </div>")
            elif isinstance(page_result, str):
                paragraphs = _extract_paragraphs_from_string(page_result)
                if paragraphs:
                    html_parts.append("        <div class=\"page-content\">")
                    for paragraph in paragraphs:
                        html_parts.append(f"            <p>{html.escape(paragraph)}</p>")
                    html_parts.append("        </div>")
                else:
                    html_parts.append("        <div class=\"no-data\">No recognizable text found.</div>")
            else:
                html_parts.append("        <div class=\"no-data\">No data returned for this page.</div>")

            page_usage = page_entry.get("usage")
            if page_usage:
                html_parts.append("        <div class=\"page-meta\">Usage</div>")
                html_parts.append("        <ul class=\"stats-list\">")
                for line in _dict_items_to_list(page_usage):
                    html_parts.append(f"            <li>{html.escape(line)}</li>")
                html_parts.append("        </ul>")

            page_cost = page_entry.get("cost")
            if page_cost:
                html_parts.append("        <div class=\"page-meta\">Cost</div>")
                html_parts.append("        <ul class=\"stats-list\">")
                for line in _dict_items_to_list(page_cost):
                    html_parts.append(f"            <li>{html.escape(line)}</li>")
                html_parts.append("        </ul>")

            page_error = page_entry.get("error")
            if page_error:
                html_parts.append(
                    f"        <div class=\"no-data\">Error: {html.escape(str(page_error))}</div>"
                )

            html_parts.append("    </section>")
    else:
        if isinstance(structured_result, str):
            paragraphs = _extract_paragraphs_from_string(structured_result)
        else:
            paragraphs = []
        html_parts.append("    <section class=\"page\">")
        html_parts.append("        <div class=\"page-title\">Image Result</div>")
        if isinstance(structured_result, dict):
            detected_language = structured_result.get("language_detected", "unknown")
            html_parts.append(
                f"        <div class=\"page-language\">Detected language: {html.escape(str(detected_language))}</div>"
            )
            if engine_label:
                html_parts.append(
                    f"        <div class=\"page-meta\">Engine: {html.escape(str(engine_label))}</div>"
                )
            paragraphs = _blocks_to_paragraphs(structured_result.get("blocks", []))
            if paragraphs:
                html_parts.append("        <div class=\"page-content\">")
                for paragraph in paragraphs:
                    html_parts.append(f"            <p>{html.escape(paragraph)}</p>")
                html_parts.append("        </div>")
            else:
                html_parts.append("        <div class=\"no-data\">No blocks returned.</div>")

            translation_paragraphs = _blocks_to_translation_paragraphs(structured_result.get("blocks", []))
            if translation_paragraphs:
                html_parts.append('        <div class="translation-title">Translation</div>')
                html_parts.append('        <div class="translation-content">')
                for paragraph in translation_paragraphs:
                    html_parts.append(f'            <p>{html.escape(paragraph)}</p>')
                html_parts.append("        </div>")
        elif isinstance(structured_result, str):
            if paragraphs:
                html_parts.append("        <div class=\"page-content\">")
                for paragraph in paragraphs:
                    html_parts.append(f"            <p>{html.escape(paragraph)}</p>")
                html_parts.append("        </div>")
            else:
                html_parts.append("        <div class=\"page-language\">Unstructured text response</div>")
                html_parts.append(
                    "        <div class=\"page-content\">"
                    f"            <p>{html.escape(structured_result)}</p>"
                    "        </div>"
                )
            if engine_label:
                html_parts.append(
                    f"        <div class=\"page-meta\">Engine: {html.escape(str(engine_label))}</div>"
                )
        else:
            html_parts.append("        <div class=\"no-data\">No data returned.</div>")

        if usage_override:
            html_parts.append("        <div class=\"page-meta\">Usage</div>")
            html_parts.append("        <ul class=\"stats-list\">")
            for line in _dict_items_to_list(usage_override):
                html_parts.append(f"            <li>{html.escape(line)}</li>")
            html_parts.append("        </ul>")

        if cost_override:
            html_parts.append("        <div class=\"page-meta\">Cost</div>")
            html_parts.append("        <ul class=\"stats-list\">")
            for line in _dict_items_to_list(cost_override):
                html_parts.append(f"            <li>{html.escape(line)}</li>")
            html_parts.append("        </ul>")

        html_parts.append("    </section>")

    html_parts.append("    <footer>Generated by OCR pipeline</footer>")
    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract multilingual OCR with layout metadata from images or PDFs using Google Gemini."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input image or PDF file.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "image", "pdf"],
        default="auto",
        help="Force processing mode (default: auto-detect by file extension).",
    )
    parser.add_argument(
        "--engine",
        choices=[ENGINE_GEMINI, ENGINE_DOCAI],
        default=ENGINE_GEMINI,
        help="OCR backend: 'gemini' (default) or 'docai' (Google Document AI).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Malayalam",
        help="Target language for the extracted text (default: Malayalam).",
    )
    parser.add_argument(
        "--translate-to",
        type=str,
        default=None,
        help="Optional language to translate each block into.",
    )
    parser.add_argument(
        "--preprocess/--no-preprocess",
        dest="preprocess",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable image preprocessing (default: enabled).",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=2048,
        help="Maximum size for the longest image edge during preprocessing (default: 2048).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt override for the OCR instruction.",
    )
    parser.add_argument(
        "--mime-type",
        type=str,
        default=None,
        help="Explicit MIME type for the image (overrides inference/preprocessing).",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="Rendering DPI when converting PDFs to images (default: 300).",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=None,
        help="Maximum number of PDF pages to process (default: all pages).",
    )
    parser.add_argument(
        "--pricing-input-per-million",
        type=float,
        default=None,
        help="Prompt/input token price in USD per million tokens for cost estimation.",
    )
    parser.add_argument(
        "--pricing-output-per-million",
        type=float,
        default=None,
        help="Completion/output token price in USD per million tokens for cost estimation.",
    )
    parser.add_argument(
        "--docai-project",
        type=str,
        default=None,
        help="Google Cloud project ID for the Document AI processor (engine='docai').",
    )
    parser.add_argument(
        "--docai-location",
        type=str,
        default=None,
        help="Google Cloud location of the Document AI processor (engine='docai').",
    )
    parser.add_argument(
        "--docai-processor",
        type=str,
        default=None,
        help="Document AI processor ID (engine='docai').",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print raw model response even if JSON parsing succeeds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the aggregated results as JSON.",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Optional path to write the Unicode HTML rendition of the results.",
    )
    return parser


def write_json_output(output_path: Path, data: Union[str, Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if isinstance(data, dict):
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(data))
    print(f"Results written to {output_path}")


def write_html_output(output_path: Path, html_content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    print(f"Unicode HTML written to {output_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if args.mode == "auto":
        if input_path.suffix.lower() == ".pdf":
            mode = "pdf"
        else:
            mode = "image"
    else:
        mode = args.mode

    engine = args.engine

    docai_config: Optional[Dict[str, str]] = None
    if engine == ENGINE_DOCAI:
        raw_docai_config = {
            "project_id": args.docai_project,
            "location": args.docai_location,
            "processor_id": args.docai_processor,
        }
        try:
            docai_config = _ensure_docai_config(raw_docai_config)
        except Exception as exc:
            raise SystemExit(f"Document AI configuration error: {exc}") from exc

    result: Optional[Dict[str, Any]] = None
    html_input: Optional[Union[str, Dict[str, Any]]] = None

    if mode == "image":
        if not input_path.exists():
            raise FileNotFoundError(f"Image file not found at {input_path}")

        result = extract_text_from_image(
            image_path=str(input_path),
            language=args.language,
            prompt=args.prompt,
            mime_type=args.mime_type,
            translate_to=args.translate_to,
            preprocess=args.preprocess,
            resize_long_edge=args.resize_long_edge,
            pricing_input_per_million=args.pricing_input_per_million,
            pricing_output_per_million=args.pricing_output_per_million,
            engine=engine,
            docai_config=docai_config,
        )

        if args.raw_output and result is not None:
            print("\n--- Raw JSON Output ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        html_input = result

    elif mode == "pdf":
        result = pdf_to_layout_json(
            pdf_path=str(input_path),
            language=args.language,
            prompt=args.prompt,
            translate_to=args.translate_to,
            preprocess=args.preprocess,
            resize_long_edge=args.resize_long_edge,
            dpi=args.pdf_dpi,
            page_limit=args.page_limit,
            pricing_input_per_million=args.pricing_input_per_million,
            pricing_output_per_million=args.pricing_output_per_million,
            engine=engine,
            docai_config=docai_config,
        )

        if args.raw_output:
            print("\n--- Raw JSON Output ---")
            print(json.dumps(result, ensure_ascii=False))

        html_input = result

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if args.output:
        write_json_output(Path(args.output), result if result is not None else {})

    if args.html_output and html_input is not None:
        html_doc = generate_html_document(html_input)
        write_html_output(Path(args.html_output), html_doc)


if __name__ == "__main__":
    main()