// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The interface description of the local service, rendered by Swagger UI without network.  The
// icon opens this page with the administration token in the fragment of its address, which the
// browser never sends: the token authorizes the administration routes, and leaves the address.
window.addEventListener('load', function () {
  // The link back, in the language of the application: the one chosen there, else the browser's.
  const back = {
    en: 'Back to the application', fr: "Retour à l'application", de: 'Zurück zur Anwendung',
    it: "Torna all'applicazione", es: 'Volver a la aplicación'
  };
  let language = null;
  try { language = localStorage.getItem('pseudalis.language'); } catch (e) { /* storage unavailable */ }
  if (!back[language]) {
    language = 'en';
    const wanted = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language];
    for (let i = 0; i < wanted.length; ++i) {
      const code = String(wanted[i] || '').slice(0, 2).toLowerCase();
      if (back[code]) { language = code; break; }
    }
  }
  document.documentElement.lang = language;
  const link = document.getElementById('back');
  if (link) link.textContent = back[language];

  let admin = '';
  const m = /(?:^#|&)admin=([0-9A-Za-z_-]+)/.exec(location.hash || '');
  try {
    if (m) sessionStorage.setItem('pseudalis.admin', m[1]);
    admin = sessionStorage.getItem('pseudalis.admin') || '';
  } catch (e) {
    admin = m ? m[1] : '';
  }
  if (m) history.replaceState(null, '', location.pathname + location.search);
  window.ui = SwaggerUIBundle({
    url: '../openapi.json',
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis],
    layout: 'BaseLayout',
    validatorUrl: null,
    displayRequestDuration: true,
    defaultModelsExpandDepth: 0,
    persistAuthorization: false,
    onComplete: function () {
      if (admin) window.ui.preauthorizeApiKey('admin', admin);
    }
  });
});
