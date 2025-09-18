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
  const STYLE_ID = 'custom-watermark-style';
  const SIBLING_ATTR = 'data-custom-watermark-sibling';

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      a.watermark {
        display: none !important;
      }
    `;
    document.head.appendChild(style);
  }

  function ensureSiblingAfterWatermark(el) {
    if (!(el instanceof HTMLElement)) return;

    // If the immediate next sibling is our disclaimer, update it; otherwise, insert one.
    const nextEl = el.nextElementSibling;
    if (nextEl && nextEl.getAttribute(SIBLING_ATTR) === '1') {
      if (nextEl.innerHTML.trim() !== CUSTOM_FOOTER_HTML) {
        nextEl.innerHTML = CUSTOM_FOOTER_HTML;
      }
      return;
    }

    const container = document.createElement('div');
    container.setAttribute(SIBLING_ATTR, '1');
    container.style.margin = '0';
    container.style.pointerEvents = 'auto';
    container.setAttribute('aria-live', 'polite');
    container.innerHTML = CUSTOM_FOOTER_HTML;

    // Insert directly after the watermark anchor
    el.insertAdjacentElement('afterend', container);
  }

  function applyAll(root = document) {
    const nodes = root instanceof Element
      ? root.querySelectorAll(WATERMARK_SELECTOR)
      : document.querySelectorAll(WATERMARK_SELECTOR);

    nodes.forEach((el) => ensureSiblingAfterWatermark(el));
  }

  function init() {
    injectStyles();
    applyAll(document);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Re-apply on future UI updates
  const mo = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'childList') {
        // Re-ensure CSS and siblings if the UI changes.
        if (!document.getElementById(STYLE_ID)) injectStyles();
        applyAll(document);
        break;
      }
    }
  });

  mo.observe(document.documentElement, { childList: true, subtree: true });
})();
