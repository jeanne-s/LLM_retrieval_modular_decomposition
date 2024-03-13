#!/bin/bash

sudo apt-get update
sudo apt install python3-pip
sudo apt install python3-virtualenv

virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
