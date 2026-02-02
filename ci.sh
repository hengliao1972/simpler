#!/bin/bash

# run all tests
pytest tests
# run all examples
python examples/scripts/run_example.py -k examples/host_build_graph_example/kernels -g examples/host_build_graph_example/golden.py
python examples/scripts/run_example.py -k examples/host_build_graph_sim_example/kernels -g examples/host_build_graph_sim_example/golden.py -p a2a3sim
