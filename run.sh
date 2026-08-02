#!/bin/bash
# SAMA: Sparse Adaptive Momentum Attack against Federated Recommender Systems
# Runs attack (Sum) and defense (SDCSum) experiments on ml-100k.

set -e

DATASET="ml-100k/"

echo "============================================"
echo "  SAMA Attack & SDC-Sum Defense Experiments"
echo "  Dataset: ${DATASET}"
echo "============================================"

mkdir -p logs/

echo ""
echo ">>> Running: SAMA attack (Sum aggregation, no defense) <<<"
python main.py --dataset "${DATASET}" --agg_type Sum 2>&1 | tee "logs/attack_sum.log"
echo ">>> Finished: attack_sum <<<"

echo ""
echo ">>> Running: SAMA attack + SDC-Sum defense <<<"
python main.py --dataset "${DATASET}" --agg_type SDCSum 2>&1 | tee "logs/defense_sdcsum.log"
echo ">>> Finished: defense_sdcsum <<<"

echo ""
echo "All experiments completed! Logs saved in logs/."
