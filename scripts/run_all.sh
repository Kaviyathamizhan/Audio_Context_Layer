#!/usr/bin/env bash
# Audio Context Layer (ACL) Modular Turnkey Execution Script for Linux / Colab
# Usage:
#   bash scripts/run_all.sh [step] [data_dir] [author_name]
# Steps:
#   clone_esc50, generate, test, train_ast, train_crnn, evaluate, edge, report, all

set -euo pipefail

STEP="${1:-all}"
DATA_DIR="${2:-dataset}"
AUTHOR="${3:-Researcher}"

echo "========================================================"
echo "Audio Context Layer Pipeline - Step: ${STEP}"
echo "Dataset Dir: ${DATA_DIR}"
echo "Author: ${AUTHOR}"
echo "========================================================"

do_clone() {
  if [ ! -d "ESC-50" ]; then
    echo "[1/8] Cloning ESC-50 dataset..."
    git clone --depth 1 https://github.com/karolpiczak/ESC-50.git ESC-50
    rm -rf ESC-50/.git
  else
    echo "ESC-50 already present."
  fi
}

do_generate() {
  echo "[2/8] Generating synthetic audio scenes and QA pairs..."
  # Medium scale (500 train / 80 val / 120 test scenes); for full scale pass --n_train 1600 --n_val 300 --n_test 500
  python -m acl.data_gen --esc50 ESC-50 --out "${DATA_DIR}" --n_train 500 --n_val 80 --n_test 120 --seed 42
}

do_test() {
  echo "[3/8] Running internal consistency and leakage unit tests..."
  python -m pytest -q tests
}

do_train_ast() {
  echo "[4/8] Training Main Model: AST (frozen AudioSet embeddings) + BiGRU head..."
  python -m acl.train_sed --data "${DATA_DIR}" --backend ast --out runs/ast --epochs 30 --bs 32 --seed 42
}

do_train_crnn() {
  echo "[5/8] Training Baseline: Lightweight Edge CRNN..."
  python -m acl.train_sed --data "${DATA_DIR}" --backend crnn --out runs/crnn --epochs 40 --bs 32 --seed 42
}

do_evaluate() {
  echo "[6/8] Evaluating Models (validation tuning, SED F1, QA accuracy, error breakdown)..."
  python -m acl.evaluate --data "${DATA_DIR}" --run runs/ast
  python -m acl.evaluate --data "${DATA_DIR}" --run runs/crnn
}

do_edge() {
  echo "[7/8] Measuring Edge Feasibility (FP32 latency, size, dynamic INT8 quantization)..."
  python -m acl.edge_report --data "${DATA_DIR}" --run runs/ast
  python -m acl.edge_report --data "${DATA_DIR}" --run runs/crnn
}

do_report() {
  echo "[8/8] Compiling Technical PDF Report..."
  python -m acl.make_report --data "${DATA_DIR}" --runs runs/ast runs/crnn --author "${AUTHOR}" --out report/ACL_technical_report.pdf
  echo "[REPORT CREATED] Output located at report/ACL_technical_report.pdf"
}

case "${STEP}" in
  clone_esc50) do_clone ;;
  generate) do_generate ;;
  test) do_test ;;
  train_ast) do_train_ast ;;
  train_crnn) do_train_crnn ;;
  evaluate) do_evaluate ;;
  edge) do_edge ;;
  report) do_report ;;
  all)
    do_clone
    do_generate
    do_test
    do_train_ast
    do_train_crnn
    do_evaluate
    do_edge
    do_report
    echo "[SUCCESS] Entire ACL pipeline completed successfully!"
    ;;
  *)
    echo "Unknown step '${STEP}'. Options: clone_esc50, generate, test, train_ast, train_crnn, evaluate, edge, report, all"
    exit 1
    ;;
esac
