/* 
 * Replace the contents of the Chainlit watermark/footer
 */
(function () {
  const CUSTOM_FOOTER_HTML = `
    <div class="text-xs text-muted-foreground text-center">
      <span>
        <em>
          <strong>Disclaimer:</strong>
          Our chatbot uses AI to assist you.
          Responses are generated automatically and may not always be accurate.
          Do not share sensitive, personal or confidential information.
          For more information, please click on the “Readme” icon at the top-right of this window.
        </em>
      </span>
    </div>
  `.trim();

  const WATERMARK_SELECTOR = 'a.watermark';
  const APPLIED_ATTR = 'data-custom-watermark';

  function replaceFooterContents(root = document) {
    const nodes = root instanceof Element
      ? root.querySelectorAll(WATERMARK_SELECTOR)
      : document.querySelectorAll(WATERMARK_SELECTOR);

    nodes.forEach((el) => {
      if (!(el instanceof HTMLElement)) return;
      if (el.getAttribute(APPLIED_ATTR) === '1') return;

      el.innerHTML = CUSTOM_FOOTER_HTML;

      // disable the link behaviour
      el.removeAttribute('href');
      el.removeAttribute('target');
      el.style.pointerEvents = 'none';

      el.setAttribute(APPLIED_ATTR, '1');
    });
  }

  // Initial run (in case the element is already present).
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => replaceFooterContents(document));
  } else {
    replaceFooterContents(document);
  }

  // Re-apply on future UI updates (SPA re-renders).
  const mo = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node instanceof Element) {
          // If the watermark itself is added or its parent subtree changes, update.
          if (node.matches?.(WATERMARK_SELECTOR) || node.querySelector?.(WATERMARK_SELECTOR)) {
            replaceFooterContents(node);
          }
        }
      }
    }
  });

  mo.observe(document.documentElement, { childList: true, subtree: true });
})();
