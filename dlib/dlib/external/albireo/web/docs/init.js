// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The interface description of the local service, rendered by Swagger UI without network.
window.addEventListener('load', function () {
  window.ui = SwaggerUIBundle({
    url: '../openapi.json',
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis],
    layout: 'BaseLayout',
    validatorUrl: null,
    displayRequestDuration: true,
    defaultModelsExpandDepth: 0,
    persistAuthorization: false
  });
});
