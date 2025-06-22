#!/bin/bash

DOC_DIR=./doc/docs
PACKAGE_DIR=./src

rm -rf $DOC_DIR/api $DOC_DIR/build

# Run live server with watch on source code directory
sphinx-autobuild -b html $DOC_DIR $DOC_DIR/build --port 38000 --watch $PACKAGE_DIR
