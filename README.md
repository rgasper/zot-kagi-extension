# zot-kagi-extension

A [zot](https://github.com/patriceckhart/zot) extension that gives the AI agent two
LLM-callable tools backed by the [Kagi API](https://kagi.com/api):

| Tool | What it does |
|---|---|
| `kagi_search` | Web search — returns ranked results with titles, URLs, and snippets |
| `kagi_extract` | Content extraction — fetches full markdown content from up to 10 URLs |

## Prerequisites

- Python 3.10+
- A Kagi API key — get one at <https://kagi.com/api>
- `uv` or `pip`
- `zot` installed and on `$PATH`

## Installation

### 1. Install the Kagi Python SDK

```bash
pip install git+https://github.com/kagisearch/kagi-openapi-python.git
# or with uv:
uv pip install git+https://github.com/kagisearch/kagi-openapi-python.git
```

### 2. Set your API key

Add this to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):

```bash
export KAGI_API_KEY=your_api_key_here
```

### 3. Make the script executable

```bash
chmod +x main.py
```

### 4. Load the extension

**For a single zot session (no install needed — great for development):**

```bash
zot --ext /path/to/zot-kagi-extension
```

**To install globally:**

```bash
zot ext install /path/to/zot-kagi-extension
```

## Tools

### `kagi_search`

Search the web and get back ranked results.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | The search query |
| `limit` | integer | | Max results (default 10, max 20) |

### `kagi_extract`

Extract full markdown content from one or more URLs.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `urls` | string[] | ✅ | List of HTTPS URLs to extract (max 10) |

## Logs

```bash
zot ext logs kagi        # print the extension's stderr log
zot ext logs kagi -f     # tail it live
```

## Development

Edit `main.py`, then reload without restarting zot:

```
/reload-ext
```
