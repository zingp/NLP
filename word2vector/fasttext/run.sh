#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
echo "active python2.7 environment"
source ~/venv-py2/bin/activate # Use python2
cd $baseDir
echo `python --version`
set -x
python fasttext.py

echo "deactivate python2.7 environment ..."
deactivate