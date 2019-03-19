#!/bin/bash

echo "The training is now beginning!"

python GANs_Gluon.py

echo "Now, the plots are now generated!"

python plots/animation.py

echo "All jobs done!"
