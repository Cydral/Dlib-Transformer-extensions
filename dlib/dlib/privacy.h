// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_
#define DLIB_PRIVACY_

#include "crypto.h"
#include "zip.h"
#include "documents.h"
#include "privacy/pii_types.h"
#include "privacy/date_formats.h"
#include "privacy/pseudonymizer.h"
#include "privacy/pii_detector.h"
#include "privacy/document_report.h"
#include "privacy/ooxml_anonymizer.h"
#include "privacy/pdf_anonymizer.h"
#include "privacy/document_anonymizer.h"
#include "privacy/pii_dataset.h"

/*
    The learned recognizers are kept out of this header, since they pull in the machine
    learning parts of dlib. Include them where they are used:

        #include <dlib/privacy/pii_sequence_tagger.h>   // structural SVM sequence labeler
        #include <dlib/privacy/pii_span_model.h>        // byte level transformer
*/

#endif // DLIB_PRIVACY_
