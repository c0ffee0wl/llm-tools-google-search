"""
LLM tools for Google Search using Vertex/Gemini with Google Search grounding.

Provides a search_google tool that leverages Vertex AI or Gemini's google_search
option to perform web searches, making search available to any LLM model.
"""
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

import llm
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue


# Model priority: Vertex first (enterprise), then standard Gemini API
SEARCH_MODELS = [
    "vertex/gemini-2.5-flash",
    "gemini-2.5-flash",
]


# Configuration error patterns (model installed but not configured)
CONFIG_ERROR_PATTERNS = [
    'project',           # "No GCP project ID found"
    'api key',           # "API key not found"
    'credentials',       # "credentials not found"
    'not configured',    # Generic
    'authentication',    # Auth failures
    'google_cloud',      # Vertex env var errors
]


def _is_config_error(error_msg: str) -> bool:
    """Check if error indicates missing configuration vs actual API failure."""
    error_lower = error_msg.lower()
    return any(pattern in error_lower for pattern in CONFIG_ERROR_PATTERNS)


def _get_language_name(code: str) -> Optional[str]:
    """Convert ISO 639-1 language code to full language name.

    Args:
        code: ISO 639-1 language code (e.g., "en", "de", "fr")

    Returns:
        Full language name (e.g., "English", "German", "French"),
        or None if the code is invalid.
    """
    if not code or not code.strip():
        return None
    try:
        return Lang(code.lower().strip()).name
    except InvalidLanguageValue:
        return None


USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'


def _resolve_redirect(uri: str, timeout: float = 3.0) -> str:
    """
    Resolve a Vertex AI Search redirect URL to its final destination.

    Args:
        uri: The URL to resolve
        timeout: Request timeout in seconds

    Returns:
        The final resolved URL, or original URL if resolution fails
    """
    if not uri or 'vertexaisearch.cloud.google.com' not in uri:
        return uri

    try:
        request = urllib.request.Request(uri, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.geturl()
    except Exception:
        return uri


def _resolve_sources(sources: list, timeout: float = 3.0, max_workers: int = 5) -> list:
    """
    Resolve redirect URLs in parallel.

    Args:
        sources: List of {'uri': str} dictionaries
        timeout: Per-URL timeout in seconds
        max_workers: Maximum parallel requests

    Returns:
        List of resolved URL strings
    """
    if not sources:
        return []

    uris = [s.get('uri', '') for s in sources]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda u: _resolve_redirect(u, timeout), uris))


def _byte_to_char_position(text: str, byte_index: int) -> int:
    """
    Convert a UTF-8 byte offset to a character position.

    Gemini API returns byte positions for grounding segments, not character
    positions. This function converts byte offsets to character indices for
    correct text manipulation.

    Args:
        text: The original Unicode string
        byte_index: Position in the UTF-8 byte representation

    Returns:
        Corresponding character position in the string

    Raises:
        UnicodeDecodeError: If byte_index falls mid-character
    """
    text_bytes = text.encode('utf-8')
    return len(text_bytes[:byte_index].decode('utf-8'))


def _insert_inline_citations(
    text: str,
    grounding_supports: list,
    chunk_to_source: dict
) -> str:
    """
    Insert inline citation markers into text based on grounding supports.

    Args:
        text: The response text from Gemini
        grounding_supports: List of grounding support objects with segment info
        chunk_to_source: Maps groundingChunk indices to source numbers (1-indexed)

    Returns:
        Text with inline citation markers inserted (e.g., "claim text[1][2]")
    """
    if not grounding_supports:
        return text

    # Build list of (char_position, citation_string) tuples
    insertions = []

    for support in grounding_supports:
        segment = support.get('segment', {})
        chunk_indices = support.get('groundingChunkIndices', [])

        if not chunk_indices:
            continue

        end_byte = segment.get('endIndex', 0)
        if end_byte <= 0:
            continue

        # Map chunk indices to source numbers
        source_nums = []
        for idx in chunk_indices:
            if idx in chunk_to_source:
                source_nums.append(chunk_to_source[idx])

        if not source_nums:
            continue

        # Build citation string with deduplicated, sorted source numbers
        source_nums = sorted(set(source_nums))
        citation = ' ' + ''.join(f'[{n}]' for n in source_nums)

        # Convert byte position to character position
        try:
            char_pos = _byte_to_char_position(text, end_byte)
        except UnicodeDecodeError:
            # Skip if byte index falls mid-character
            continue

        # Validate position is within bounds
        if char_pos < 0 or char_pos > len(text):
            continue

        insertions.append((char_pos, citation))

    if not insertions:
        return text

    # Sort by position descending (insert from end to preserve indices)
    insertions.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate insertions at same position (keep first occurrence)
    seen_positions = set()
    unique_insertions = []
    for pos, citation in insertions:
        if pos not in seen_positions:
            seen_positions.add(pos)
            unique_insertions.append((pos, citation))

    # Insert citations from end to start
    result = text
    for pos, citation in unique_insertions:
        result = result[:pos] + citation + result[pos:]

    return result


WEB_CITATION_RULES = """#### Note

IMPORTANT: You MUST include the "#### Sources" section above in your response to the user.
The text contains inline citations [1], [2], etc. that reference these sources."""


def _format_sources_markdown(resolved_urls: list) -> str:
    """
    Format resolved URLs as a simple numbered list with citation instructions.

    Args:
        resolved_urls: List of resolved URL strings

    Returns:
        Markdown formatted string with sources as [n] URL, followed by note
    """
    if not resolved_urls:
        return ""

    lines = ["#### Sources", ""]
    for i, url in enumerate(resolved_urls, start=1):
        if url:
            lines.append(f"[{i}] {url}")
            lines.append("")  # Blank line after each entry

    lines.append(WEB_CITATION_RULES)

    return "\n".join(lines)


def search_google(query: str, language: str, max_results: int = 7) -> str:
    """
    Search the web using Google Search via Vertex/Gemini grounding.

    Performs a live web search and synthesizes results into a coherent answer with
    source citations. Requires Vertex AI or Gemini API to be configured. The search
    uses Google's grounding feature to retrieve current information from the web.

    Args:
        query: The search query - be specific for better results
        language: ISO 639-1 language code for the response (e.g., "en", "de", "fr")
        max_results: Maximum number of source URLs to return (default: 7)

    Returns:
        Markdown formatted response with synthesized answer followed by a Sources
        section containing links to the web pages used for grounding.
    """
    language_name = _get_language_name(language)
    if not language_name:
        return f"Error: invalid language code '{language}'. Use ISO 639-1 codes (e.g., 'en', 'de', 'fr')"
    today = date.today()

    # Craft prompt that encourages grounded search results
    search_prompt = f"""Today's date: {today.isoformat()}

Search the web for: {query}

Provide a factual, comprehensive answer based on current web search results.
Include specific facts, numbers, and dates where relevant.
When citing information, mention the source name inline.
Remember it is {today.year} this year.

You MUST respond in {language_name}."""

    last_error = None
    tried_models = []

    # Try each model in priority order, falling back on configuration errors
    for model_id in SEARCH_MODELS:
        # Check if plugin is installed
        try:
            model = llm.get_model(model_id)
        except llm.UnknownModelError:
            continue  # Plugin not installed, try next

        tried_models.append(model_id)

        try:
            # Attempt the actual search
            response = model.prompt(
                search_prompt,
                google_search=True
            )

            result_text = response.text()

            # Extract grounding metadata if available
            sources = []
            chunk_to_source = {}
            grounding_supports = []
            try:
                response_json = response.response_json
                if response_json:
                    for candidate in response_json.get('candidates', []):
                        grounding = candidate.get('groundingMetadata', {})
                        # Extract from groundingChunks and build chunk-to-source mapping
                        for chunk_idx, chunk in enumerate(grounding.get('groundingChunks', [])):
                            web_info = chunk.get('web', {})
                            if web_info:
                                source = {
                                    'title': web_info.get('title', ''),
                                    'uri': web_info.get('uri', '')
                                }
                                if source not in sources:
                                    sources.append(source)
                                    chunk_to_source[chunk_idx] = len(sources)  # 1-indexed
                                else:
                                    # Duplicate source - map to existing
                                    chunk_to_source[chunk_idx] = sources.index(source) + 1
                        # Extract groundingSupports for inline citations
                        grounding_supports.extend(grounding.get('groundingSupports', []))
                        # Check searchEntryPoint for search suggestions
                        search_entry = grounding.get('searchEntryPoint', {})
                        if search_entry and not sources:
                            # Fallback: use rendered content URL if no other sources
                            rendered = search_entry.get('renderedContent', '')
                            if rendered:
                                sources.append({
                                    'title': 'Google Search',
                                    'uri': f'https://www.google.com/search?q={query.replace(" ", "+")}'
                                })
            except Exception:
                # If we can't extract grounding metadata, continue without it
                pass

            # Apply max_results limit and filter mapping
            sources = sources[:max_results]
            chunk_to_source = {k: v for k, v in chunk_to_source.items() if v <= max_results}

            # Insert inline citations into result text
            text_with_citations = _insert_inline_citations(
                result_text, grounding_supports, chunk_to_source
            )

            # Resolve redirect URLs before returning
            resolved_sources = _resolve_sources(sources)

            # Format as markdown with sources
            sources_md = _format_sources_markdown(resolved_sources)
            if sources_md:
                return f"{text_with_citations}\n\n{sources_md}"
            return text_with_citations

        except Exception as e:
            error_msg = str(e)
            if _is_config_error(error_msg):
                # Configuration error - try next model
                last_error = f"{model_id}: {error_msg}"
                continue
            else:
                # Actual API/search error - return it
                return f"Error: {error_msg}"

    # All models failed
    if not tried_models:
        return "Error: No search provider installed. Run: install-llm-tools.sh --gemini"
    else:
        return f"Error: No configured provider. Tried: {', '.join(tried_models)}. Last error: {last_error}"


@llm.hookimpl
def register_tools(register):
    """Register Google Search tool."""
    register(search_google)
