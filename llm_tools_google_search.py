"""
LLM tools for Google Search using Vertex/Gemini with Google Search grounding.

Provides a search_google tool that leverages Vertex AI or Gemini's google_search
option to perform web searches, making search available to any LLM model.
"""
import urllib.error
import urllib.request
from typing import Optional, Tuple

import llm


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


def _format_sources_markdown(sources: list) -> str:
    """
    Format resolved sources as markdown links.

    Args:
        sources: List of source dicts with 'title', 'uri', and optional 'error'

    Returns:
        Markdown formatted string with sources as bullet points
    """
    if not sources:
        return ""

    lines = ["Sources:"]
    for source in sources:
        title = source.get('title') or 'Untitled'
        uri = source.get('uri', '')
        if uri:
            lines.append(f"- [{title}]({uri})")
        elif title:
            lines.append(f"- {title}")

    return "\n".join(lines)


def search_google(query: str, max_results: int = 7) -> str:
    """
    Search the web using Google Search via Vertex/Gemini grounding.

    Performs a live web search and synthesizes results into a coherent answer with
    source citations. Requires Vertex AI or Gemini API to be configured. The search
    uses Google's grounding feature to retrieve current information from the web.

    Args:
        query: The search query - be specific for better results
        max_results: Maximum number of source URLs to return (default: 7)

    Returns:
        Markdown formatted response with synthesized answer followed by a Sources
        section containing links to the web pages used for grounding.
    """
    # Craft prompt that encourages grounded search results
    search_prompt = f"""Search the web for: {query}

Provide a factual, comprehensive answer based on current web search results.
Include specific facts, numbers, and dates where relevant.
When citing information, mention the source name inline."""

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
            try:
                response_json = response.response_json
                if response_json:
                    for candidate in response_json.get('candidates', []):
                        grounding = candidate.get('groundingMetadata', {})
                        # Extract from groundingChunks
                        for chunk in grounding.get('groundingChunks', []):
                            web_info = chunk.get('web', {})
                            if web_info:
                                source = {
                                    'title': web_info.get('title', ''),
                                    'uri': web_info.get('uri', '')
                                }
                                if source not in sources:
                                    sources.append(source)
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

            # Resolve redirect URLs before returning
            resolved_sources = _resolve_sources(sources[:max_results])

            # Format as markdown with sources
            sources_md = _format_sources_markdown(resolved_sources)
            if sources_md:
                return f"{result_text}\n\n{sources_md}"
            return result_text

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
