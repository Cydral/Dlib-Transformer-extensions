# SVOX Pico

The speech synthesizer with which Albireo (`dlib/dlib/external/albireo`) reads its answers
aloud, compiled into its program `albireo-speech`.

## Origin

The SVOX Pico engine of the Android Open Source Project (`external/svox`, folders
`pico/lib` and `pico/lang`), as published with the patches of Debian in
https://github.com/naggety/picotts at commit `21089d2`.

License: Apache License 2.0, in `NOTICE`.

## Content

- `lib/`: the sources of the engine, as published, except `picopltf.h`, where Windows, which
  has no `endian.h` and runs on little-endian processors only, takes `ENDIANNESS_LITTLE`;
- `lang/`: the voices of six languages (`en-US`, `en-GB`, `fr-FR`, `de-DE`, `it-IT`,
  `es-ES`), each a text analysis file (`*_ta.bin`) and a signal generation file
  (`*_sg.bin`).
