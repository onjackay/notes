# Personal Notes Repository

This repository contains my personal notes organized with MkDocs.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd notes
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

To view the notes locally with live reloading:
```bash
uv run mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

To build the static site:
```bash
uv run mkdocs build
```

The static site will be generated in the `site/` directory.

## Adding New Notes

1. Create a new markdown file in the `docs/` directory
2. Add it to the navigation in `mkdocs.yml` if you want it to appear in the sidebar
3. The changes will automatically reload if you're running `mkdocs serve`

## Deployment

To deploy to GitHub Pages:
```bash
uv run mkdocs gh-deploy
```