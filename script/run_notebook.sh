#!/bin/bash

./docker/run.sh --jupyter jupyter lab --no-browser --notebook-dir=notebook --ip=* --port=38888 notebook/main.ipynb
