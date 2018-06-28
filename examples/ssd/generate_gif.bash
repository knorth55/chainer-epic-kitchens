#!/usr/bin/env bash

viz_dir=$1
id=$2

convert -loop 0 -delay 10 ./${viz_dir}/${id}_*.png ./${viz_dir}/${id}.gif
