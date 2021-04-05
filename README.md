# topic_manifolds
Code to take objects assigned probabilistic labels (most directly through something like topic modelling where there is a mechanistic assumption about how the objects are generated) and localize them onto user-input simplicial manifolds.

# Installation
  0. Clone the github and go into that directory `cd topic_manifolds`
  1. Decide your conda environment path `export OUTPUT_ENV_PATH=[/insert/the/path/to/your/env/here]`
  1. Run `conda env create -p $OUTPUT_ENV_PATH -f env.yml`.
  2. Run `$OUTPUT_ENV_PATH/bin/pip install pygraphviz --install-option="--include-path=$OUTPUT_ENV_PATH/include/"
     --install-option="--library-path=$OUTPUT_ENV_PATH/lib/"

## Notes
  1. This requires graphviz, which may require a separate system install. You additionally need to install
     pygraphviz.
