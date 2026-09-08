from typing import Any

def extract_text_from_content(content: Any) -> str:
    """
    Extracts a clean string representation from an LLM response content block.
    Handles raw strings, lists of string segments, or lists of dict blocks (e.g. [{'type': 'text', 'text': '...'}]).
    Preserves all text blocks by joining them with newline characters.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    text_parts.append(item["text"])
                elif item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
        return "\n".join(text_parts)

    if content is None:
        return ""

    return str(content)
