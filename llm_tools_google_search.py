"""
LLM tools for Google Search using Vertex/Gemini with Google Search grounding.

Provides a search_google tool that leverages Vertex AI or Gemini's google_search
option to perform web searches, making search available to any LLM model.
"""
import urllib.error
import urllib.request
from datetime import date
from typing import Optional, Tuple

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


def _resolve_redirect_url(redirect_url: str, timeout: float = 5.0) -> Tuple[str, Optional[str]]:
    """
    Resolve a Vertex AI Search redirect URL to its final destination.

    Args:
        redirect_url: The redirect URL to resolve
        timeout: Request timeout in seconds

    Returns:
        Tuple of (resolved_url, error_message)
        - On success: (final_url, None)
        - On failure: (original_url, error_description)
    """
    # Skip non-redirect URLs (already resolved or not from Vertex)
    if 'vertexaisearch.cloud.google.com' not in redirect_url:
        return (redirect_url, None)

    try:
        request = urllib.request.Request(
            redirect_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (response.geturl(), None)
    except urllib.error.HTTPError as e:
        return (redirect_url, f"HTTP {e.code}")
    except urllib.error.URLError as e:
        return (redirect_url, f"URL error: {e.reason}")
    except TimeoutError:
        return (redirect_url, "timeout")
    except Exception as e:
        return (redirect_url, str(e))


def _resolve_sources(sources: list, timeout: float = 5.0) -> list:
    """
    Resolve redirect URLs in a list of source dictionaries.

    Args:
        sources: List of {'title': str, 'uri': str} dictionaries
        timeout: Per-URL timeout in seconds

    Returns:
        List of sources with 'uri' resolved and 'resolved'/'error' fields
    """
    resolved_sources = []
    for source in sources:
        uri = source.get('uri', '')
        title = source.get('title', '')

        resolved_uri, error = _resolve_redirect_url(uri, timeout)

        result = {
            'title': title,
            'uri': resolved_uri,
        }

        if error:
            result['resolved'] = False
            result['error'] = error
        else:
            result['resolved'] = True

        resolved_sources.append(result)

    return resolved_sources


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
        citation = ''.join(f'[{n}]' for n in source_nums)

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

WEB CITATION RULES:
1. START with [1] and increment sequentially ([1], [2], [3], etc.) with NO gaps
2. Cite ONLY when introducing new factual claims, statistics, or direct quotes from the search results
3. After every cited claim, place the corresponding citation immediately after the sentence ("The study found X [1]")
4. End with '#### Sources' and provide definitions EXACTLY in this format: [n] [Short Title](URL)

IMPORTANT: Each source definition must follow this exact pattern:
- Start with [n] (where n is the citation number)
- Follow with [Title](URL) where Title is SHORT (2-5 words) and wrapped in square brackets
- Example: [1] [Paul Graham Essay](https://paulgraham.com/wealth.html)
- DO NOT write long descriptions - keep titles concise"""


def _format_sources_markdown(sources: list) -> str:
    """
    Format resolved sources as markdown citation definitions with citation instructions.

    Args:
        sources: List of source dicts with 'title', 'uri', and optional 'error'

    Returns:
        Markdown formatted string with sources in the format [n] [Title](URL),
        followed by citation rules
    """
    if not sources:
        return ""

    lines = ["#### Sources"]
    for i, source in enumerate(sources, start=1):
        title = source.get('title') or 'Untitled'
        uri = source.get('uri', '')
        if uri:
            lines.append(f"[{i}] [{title}]({uri})")
        elif title:
            lines.append(f"[{i}] {title}")

    lines.append("")
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
