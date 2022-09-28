########################################################
# This file holds special setup / install ation scripts
########################################################

# Setup OvationPyme
git clone https://github.com/lkilcommons/OvationPyme.git
cd OvationPyme
pip install .
rm -rf OvationPyme
# Setup additional dependencies
git clone https://github.com/lkilcommons/nasaomnireader.git
cd nasaomnireader
pip install .
rm -rf nasaomnireader
