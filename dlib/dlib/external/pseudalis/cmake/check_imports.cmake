# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Runs pseudalis-check-imports (CHECK) on FILES, a list separated by '|', through EMULATOR when a
# cross build sets one (cmake -P).  When a file imports another library than those of Windows, the
# file REMOVE, the installer just linked, is deleted, so that no build leaves an installer refused.

string(REPLACE "|" ";" files "${FILES}")
execute_process(COMMAND ${EMULATOR} ${CHECK} ${files} RESULT_VARIABLE rc)
if (NOT rc EQUAL 0)
   if (REMOVE)
      file(REMOVE "${REMOVE}")
   endif()
   message(FATAL_ERROR "a program imports another library than those of Windows (pseudalis-check-imports)")
endif()
