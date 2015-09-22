#!/bin/bash -e

TRAIN_LANG="AR CA HG MD UR" 
TEST_LANG="SW"

# Train multilingual system, test on unseen language
#./run.sh

# Adapt multilingual to probabilistic transcripts of test language
#./run-pt-text-G-map-2.sh  "${TRAIN_LANG}" "${TEST_LANG}"


./run_dnn_adapt_to_sbs.sh "${TRAIN_LANG}" "${TEST_LANG}"
