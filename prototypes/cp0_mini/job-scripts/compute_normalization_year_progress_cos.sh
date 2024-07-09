#!/bin/bash

source /opt/conda/bin/conda.sh
conda activate graphufs

python -c 'from calc_normalization import main ; main("year_progress_cos", "ocean")'