#!/bin/bash
export PYTHONPATH=".."
cd tests
for t in splitproc
do
    echo "running $t"
    python $t"_tests.py"
done
