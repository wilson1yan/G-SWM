#!/usr/bin/env bash
python scripts/dataset_maze/gen.py --grid_size 3 --obj_min 3 --obj_max 5 --inter 1 --render_resize_factor 20 --output_dir './data/MAZE' --seq_len 50 --shape 'circle'
