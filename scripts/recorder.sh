#!/bin/bash

if [$1 == 1]; then
    python3 logic1_scripts/recorder.py
else
    python3 logic2_scripts/recorder.py
fi
