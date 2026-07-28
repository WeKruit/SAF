#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
project_root="${script_dir:h:h}"
cd "$project_root"

if [[ ! -f "registries/factors/nfl_factor_lab_run_index_v2.json" ]]; then
  print -u2 "Factor Lab run index 尚未发布，拒绝打开空壳 Notebook。"
  print -u2 "缺少: ${project_root}/registries/factors/nfl_factor_lab_run_index_v2.json"
  exit 1
fi

exec uv run --locked --group research jupyter lab \
  "${script_dir}/NFL_Factor_Lab_Master.ipynb"
