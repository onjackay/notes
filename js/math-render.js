// Small script to auto-render math using KaTeX's auto-render
// This file is referenced from mkdocs.yml via `extra_javascript`.
// It waits for the DOM and then renders math delimiters.

document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== 'function') {
    // KaTeX auto-render script not loaded
    console.warn('KaTeX auto-render not available; math will not be rendered.');
    return;
  }

  renderMathInElement(document.body, {
    // Delimiters to support $...$ and $$...$$ as well as LaTeX \(...\) / \[...\]
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
    // Don't throw on invalid LaTeX; just show the source
    throwOnError: false,
  });
});
