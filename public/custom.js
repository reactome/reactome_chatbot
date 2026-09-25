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

/*
 * Continue in chat (spec 013): claim a handoff carried in the URL fragment.
 *
 * The website opens /chat/guest/#handoff=<id> in a new tab. The fragment
 * never reaches a server, so the ID stays out of logs and Referer headers.
 * Chainlit forwards every window message to the server over this tab's own
 * socket, which binds the handoff to this tab -- a cookie would be shared by
 * every tab and could seed the wrong one.
 *
 * Chainlit drops a message posted before its socket is up, so this retries
 * until the server acknowledges, and gives up after a while rather than
 * posting forever.
 */
(function () {
  // A first-time visitor goes through the captcha page on the way here, and
  // its form POST drops the fragment; that page stashes it in sessionStorage
  // (per tab, never sent to a server). Put it back in the URL so a reload
  // keeps working, and clear the stash so it cannot be claimed twice.
  const STASH = 'reactome-handoff-fragment';
  let fragment = window.location.hash;
  try {
    const stashed = sessionStorage.getItem(STASH);
    sessionStorage.removeItem(STASH);
    if (stashed && !/(^|[#&])handoff=/.test(fragment)) {
      fragment = stashed;
      history.replaceState(null, '', window.location.pathname + window.location.search + stashed);
    }
  } catch (e) {
    // Storage unavailable (private mode, blocked): the direct path still works.
  }

  const match = /(?:^|&)handoff=([A-Za-z0-9_-]{22,128})(?:&|$)/.exec(fragment.slice(1));
  if (!match) return;
  const id = match[1];

  let acknowledged = false;
  window.addEventListener('message', function (event) {
    const data = event.data;
    if (data && data.type === 'reactome-handoff-ack' && data.id === id) {
      acknowledged = true;
    }
  });

  let attempts = 0;
  const timer = setInterval(function () {
    if (acknowledged || ++attempts > 40) {
      clearInterval(timer);
      return;
    }
    window.postMessage({ type: 'reactome-handoff', id: id }, window.location.origin);
  }, 500);
})();
