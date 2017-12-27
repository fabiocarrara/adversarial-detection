#!/bin/bash

nvidia-docker run -it --privileged -v /home/fabio/SLOW/ImageNet:/ImageNet \
    -v /home/fabio/SLOW/cleverhans/examples/nips17_adversarial_competition:/code \
    -w /code/adversarial_detection \
    adversarial_detection \
    bash run_detection.sh
