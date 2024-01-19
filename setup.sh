#!/bin/bash

function install_deps() {
    pip install openai;
    pip install tiktoken;
    pip install z3;
    pip install z3-solver;
    pip install nltk==3.8.1;
    conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia;
    pip install transformers==4.16.2;
    pip install datasets==1.18.3;
    pip install scikit-learn==1.2.1;
    pip install tenacity==8.2.2;
    # Please add the command if you add any package.
}


install_deps;