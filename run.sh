#!/bin/bash
# SAMA: Sparse Adaptive Momentum Attack against Federated Recommender Systems
# Runs attack (Sum) and defense (InertiaDamp) experiments on ml-100k.

set -e

DATASET="ml-100k/"

echo "============================================"
echo "  SAMA Attack & InertiaDamp Defense Experiments"
echo "  Dataset: ${DATASET}"
echo "============================================"

mkdir -p logs/

echo ""
echo ">>> Running: SAMA attack (Sum aggregation, no defense) <<<"
python main.py --dataset "${DATASET}" --agg_type Sum 2>&1 | tee "logs/attack_sum.log"
echo ">>> Finished: attack_sum <<<"

echo ""
echo ">>> Running: SAMA attack + InertiaDamp defense <<<"
python main.py --dataset "${DATASET}" --agg_type InertiaDamp 2>&1 | tee "logs/defense_inertiadamp.log"
echo ">>> Finished: defense_inertiadamp <<<"

echo ""
echo "All experiments completed! Logs saved in logs/."
