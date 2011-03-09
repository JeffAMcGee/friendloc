#!/bin/bash
export PYTHONPATH=".."
cd tests
for t in splitproc geocrawl
do
    echo "running $t"
    python $t"_tests.py"
done
