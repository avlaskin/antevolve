#!/bin/bash
cd src
find . -type d -name "__pycache__" -exec rm -r {} +
cd ..
rm controller.log


