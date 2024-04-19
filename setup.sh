#!/bin/bash

sudo apt install -y python3 python3-venv python3-pip python3-dev python-is-python3 build-essential libssl-dev libffi-dev python3-setuptools libopenjp2-7 libtiff5 libjpeg62 libfreetype6 zlib1g liblcms2-2 libglvnd-dev python3-opencv  libbz2-dev libcurses-ocaml-dev lzma-dev python3-tk libsqlite3-dev nano ffmpeg build-essential git git-lfs wget curl

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install setuptools wheel gnureadline

pip install -r requirements.txt

pip install -e .

pip install communex

c serve vali.YOURVALI::YOURTAG

comx module register vali::project_management vali::project_management 5202.5 0 


