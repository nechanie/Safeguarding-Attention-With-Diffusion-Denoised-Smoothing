#!/bin/bash

mkdir -p adv_images
cd adv_images/e_0.0_n_5_s_0.00784313725490196

# Loop through numbers from 0 to 999
for ((i=0; i<1000; i++))
do
  # Create directory
  dir_name=$(printf "$i")
  mkdir -p "$dir_name"

done
