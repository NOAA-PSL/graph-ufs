#!/bin/bash

source /opt/conda/bin/conda.sh
conda activate graphufs

python -c 'from calc_normalization import main ; main("day_progress", "ocean")'