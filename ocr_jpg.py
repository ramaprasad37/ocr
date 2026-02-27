import argparse
import io
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ocr import generate_html_document_content_only

from google import genai
from google.genai.types import Part

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


def preprocess_image(
    image_path: str,
    resize_long_edge: Optional[int] = 2048,
) -> Tuple[bytes, str]:
    """
    Applies light-weight preprocessing to improve OCR quality.
    Returns the processed image bytes and MIME type.
    """
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


def extract_text_from_image(
    image_path: str,
    language: str = "Malayalam",
    prompt: Optional[str] = None,
    mime_type: Optional[str] = None,
    translate_to: Optional[str] = None,
    preprocess: bool = False,
    resize_long_edge: Optional[int] = 2048,
) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Extracts text from an image file and returns it in Unicode.

    Args:
        image_path: The file path to the image (e.g., 'document_page_01.jpg').
        language: Target language for the extracted Unicode text.
        prompt: Optional custom instruction for the model. If provided, it overrides the default.
        mime_type: Optional MIME type. If not provided, it is inferred or derived from preprocessing.
        translate_to: Optional language code/name to include per-block translation.
        preprocess: Whether to apply basic image cleanup before OCR.
        resize_long_edge: Optional maximum size for the longest edge during preprocessing.

    Returns:
        Parsed JSON response (dict) if the model returns structured output;
        otherwise, the raw text string. Returns None if an error occurs.
    """

    # 2. Load the Image Data
    print(f"Loading image: {image_path}...")
    try:
        if preprocess:
            image_bytes, inferred_mime = preprocess_image(
                image_path,
                resize_long_edge=resize_long_edge,
            )
        else:
            image_bytes, inferred_mime = _read_image_bytes(image_path)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as exc:
        print(f"Unexpected error while loading image: {exc}")
        return None

    # Infer MIME type if not provided
    if not mime_type:
        mime_type = inferred_mime

    # 3. Create the Image Part for the API
    image_part = Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,  # Adjust mime_type if your file is PNG, etc.
    )

    # 4. Define the Prompt (Your Instruction)
    if prompt:
        instruction = prompt
    else:
        steps = [
            "1. Language Detection: Identify the primary language of the text visible in the image.",
            "2. Ignore Background/Design: Exclude any text that is part of background design, watermarks, or decorative elements. Focus only on the main article/body text.",
            "3. CRITICAL - Strict Visual Extraction Only: Extract ONLY text that is clearly visible and readable in the image. DO NOT infer, guess, or substitute words. DO NOT use context to fill in missing words. DO NOT change the meaning or use alternate words. If any character, word, or phrase is unclear, blurry, partially obscured, or unreadable, replace it EXACTLY with [...]. Never invent text that is not clearly visible.",
            "4. Layout Metadata: Segment the document into logical blocks (paragraphs, headings, captions). For each block capture the bounding box as [x, y, width, height] in pixels relative to the original image, the reading order, text direction (horizontal or vertical), and a confidence score between 0 and 1.",
            "5. Quality Assurance Check: Before finalizing each block, verify that every character in the extracted text is actually visible in the image. If you are uncertain about any character, replace the entire uncertain word or phrase with [...]. Set confidence score to 0.5 or lower if any part of the block contains [...], indicating uncertainty.",
        ]
        steps.append(
            f"{len(steps) + 1}. Output Text: Present the extracted text as high-quality Unicode in {language}, maintaining original headings and paragraphs where possible. Use [...] for any unreadable portions - do not attempt to guess or infer missing text."
        )
        translation_step_number = len(steps) + 1
        translation_structure_line = ""
        if translate_to:
            steps.append(
                f"{translation_step_number}. Translation: For each block include a translation field rendered in {translate_to} while preserving meaning and tone. Use null if translation is not possible. If the original text contains [...], the translation should also indicate uncertainty."
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
                instruction,  # The text instruction
                image_part,  # The image data
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

        print("--------------------------------------")
        return parsed_output

    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract multilingual OCR text with layout metadata using Google Gemini."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to an image file or folder of images to process.",
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
        "--raw-output",
        action="store_true",
        help="Print raw model response even if JSON parsing succeeds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for JSON: file (single image) or directory (folder; writes <stem>.json per image).",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Path to write the HTML result (single image only). For a folder, each image gets <stem>.html in place.",
    )
    return parser


# Supported image extensions when processing a folder
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    if path.is_dir():
        # Process entire folder: each image -> <stem>.html (content-only) in same folder
        image_paths = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise SystemExit(f"No image files found in {path} (supported: {', '.join(IMAGE_EXTENSIONS)})")
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i + 1}/{len(image_paths)}] Processing {img_path.name}...")
            result = extract_text_from_image(
                image_path=str(img_path),
                language=args.language,
                prompt=args.prompt,
                mime_type=args.mime_type,
                translate_to=args.translate_to,
                preprocess=args.preprocess,
                resize_long_edge=args.resize_long_edge,
            )
            if result is not None:
                html_path = img_path.with_suffix(".html")
                html_doc = generate_html_document_content_only(result)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_doc)
                print(f"  Wrote {html_path}")
                if args.output:
                    out_dir = Path(args.output)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    json_path = out_dir / f"{img_path.stem}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        if isinstance(result, dict):
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        else:
                            f.write(str(result))
                    print(f"  Wrote {json_path}")
        return

    # Single file
    result = extract_text_from_image(
        image_path=str(path),
        language=args.language,
        prompt=args.prompt,
        mime_type=args.mime_type,
        translate_to=args.translate_to,
        preprocess=args.preprocess,
        resize_long_edge=args.resize_long_edge,
    )

    if args.raw_output and isinstance(result, dict):
        print("\n--- Raw JSON Output ---")
        print(json.dumps(result, ensure_ascii=False))

    if args.output and result is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(result, dict):
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(result))
        print(f"Results written to {output_path}")

    if args.html_output and result is not None:
        html_doc = generate_html_document_content_only(result)
        html_path = Path(args.html_output)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"Unicode HTML written to {html_path}")


if __name__ == "__main__":
    main()