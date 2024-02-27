# Nvidia DALI demonstration

## Overview

This is a program to demonstrate the use of the [Nvidia DALI python package](https://github.com/NVIDIA/DALI/).

It reads a series of frames from a specified video, and optionally performs a simple colour space conversion 
from BT.709 to BT.2020, or a hue shifting operation, while playing back the video, or when on a paused frame.

## Installation

Clone this repository:

    git clone https://github.com/icaven/dali_demo.git
    cd dali_demo

[Create a virtual environment](https://docs.python.org/3/library/venv.html) to install the required python packages.  
The DALI package currently requires CUDA version 11 or 12; please follow the installation instructions from the above link.

Other additional packages are required:

    numpy
    cupy
    dearpygui

Use:

    pip install -r requirements.txt



## Sample videos

Sample videos are available from [https://github.com/NVIDIA/DALI_extra](https://github.com/NVIDIA/DALI_extra).
Please follow the installation instructions.  

If this repository is cloned into the `~/Videos` directory,
then this demonstration program will find the default test video. 
Alternatively, set the `DALI_EXTRA_PATH` environment variable to the full path to where the `DALI_extra` directory is.

