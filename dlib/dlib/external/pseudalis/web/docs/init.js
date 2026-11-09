// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The interface description of the local service, rendered by Swagger UI without network and in
// English whatever the language of the application.  The icon opens this page with the
// administration token in the fragment of its address, which the browser never sends: the token
// authorizes the administration routes, and leaves the address.
window.addEventListener('load', function () {
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
