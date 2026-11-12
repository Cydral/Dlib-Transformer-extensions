TradingView Lightweight Charts bundled with dlib
===============================================

Version   : 5.3.0
Origin    : npm package lightweight-charts 5.3.0,
            https://registry.npmjs.org/lightweight-charts/-/lightweight-charts-5.3.0.tgz
            (integrity sha512-w16fu2/FteUkIJdgc9G2tOCKs1fwFSqnqF8+kMgHGv0uGIfcfpEJhziSaQbML1yrkbeg4nQXV3msjmhBzhLNhw==),
            built from https://github.com/tradingview/lightweight-charts at commit
            0bad38a070220e6315aa779bb80483fe97cf6d1b
SHA-256   : 0b6d1ed409283a0f9a493a99dcf11c650e111d0c61619cd922198975fb82fa99
            (lightweight-charts.standalone.production.mjs)
Files     : lightweight-charts.standalone.production.mjs, kept byte for byte as released;
            LICENSE from the package; NOTICE from the tag v5.3.0 of the repository.
License   : Apache License 2.0, see LICENSE and NOTICE in this folder

What it is
----------

Lightweight Charts draws financial charts (candlesticks, bars, lines, areas, histograms)
on HTML5 canvas.  The standalone production build is a single ES module with no
dependency, so a web application served by a dlib program imports it from its own origin:

   import { createChart, CandlestickSeries } from './vendor/lightweight-charts/lightweight-charts.standalone.production.mjs';

The module contains no eval() and no new Function(), and loads nothing from other
origins.

How to use it
-------------

A program copies this folder into the static files of its web application at build
time, for instance with CMake:

   add_custom_command(... COMMAND ${CMAKE_COMMAND} -E copy_directory
      <dlib>/dlib/external/lightweight-charts <output>/web/vendor/lightweight-charts)

The license requires naming TradingView as the creator of the library: the page that
shows the charts must display the attribution notice of the NOTICE file and a link to
https://www.tradingview.com/.  The attributionLogo option of the chart displays such a
link on the chart itself, but it adds an inline <style> element to the page, which a
Content-Security-Policy with style-src 'self' refuses.  A program with such a policy sets
attributionLogo to false and displays the notice and the link in its own markup, styled
by its own style sheet.

Updating
--------

Take the standalone production build and the LICENSE of a newer npm release, the NOTICE
of the matching tag, check the integrity of the archive against the registry, update the
version, origin and hash above, and check again that the module contains no eval() and
no new Function() and still adds no element but the attribution logo's <style>.
