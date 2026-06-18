#!/bin/bash
# Monitor a Slurm batch from manifest TSV: job_id, config_id, job_name, submitted_at
set -euo pipefail

REPO="${REPO:-$HOME/BED-LLM-Mod-strategyeig-20trial-20260517T172146}"
MANIFEST="${MANIFEST:-${REPO}/slurm_logs/comparison_jobs.tsv}"

append_note() {
  local current="$1"
  local piece="$2"
  if [[ -n "${current}" ]]; then
    printf '%s; %s' "${current}" "${piece}"
  else
    printf '%s' "${piece}"
  fi
}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "No manifest at ${MANIFEST}"
  exit 2
fi

running=0
done_ok=0
failed=0

echo "=== Cluster batch monitor ==="
echo "repo: ${REPO}"
echo "manifest: ${MANIFEST}"
echo "time: $(date -Is)"
echo ""
echo "JOB      CONFIG   NAME                   STATE      ELAPSED  ROUND        NOTES"
echo "-------------------------------------------------------------------------------"

while IFS=$'\t' read -r job_id config_id job_name submitted_at _rest; do
  [[ -z "${job_id}" || "${job_id}" == job_id ]] && continue

  log="${REPO}/slurm_logs/slurm-${job_id}.out"
  err="${REPO}/slurm_logs/slurm-${job_id}.err"
  state="UNKNOWN"
  elapsed="-"
  round="-"
  notes=""

  if squeue -j "${job_id}" -h 2>/dev/null | grep -q .; then
    state="$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null | head -1)"
    elapsed="$(squeue -j "${job_id}" -h -o '%M' 2>/dev/null | head -1)"
    running=$((running + 1))
  elif [[ -f "${log}" ]] && grep -q "END TIME:" "${log}"; then
    state="DONE"
    done_ok=$((done_ok + 1))
    if grep -q "Source RMSE:" "${log}"; then
      rmse_line="$(grep 'Source RMSE:' "${log}" | tail -1)"
      notes="$(append_note "${notes}" "${rmse_line#*Source RMSE: }")"
    fi
  elif [[ -f "${log}" ]]; then
    state="ENDED?"
    failed=$((failed + 1))
    notes="$(append_note "${notes}" "no END TIME in log")"
  else
    state="MISSING"
    failed=$((failed + 1))
    notes="$(append_note "${notes}" "log not found")"
  fi

  if [[ -f "${log}" ]]; then
    round="$(grep -oE 'round [0-9]+/[0-9]+' "${log}" 2>/dev/null | tail -1 || true)"
    if [[ -z "${round}" ]]; then
      round="$(grep -oE 'trial [0-9]+/[0-9]+' "${log}" 2>/dev/null | tail -1 || true)"
    fi
    retries="$(grep -cE 'attempt [123]/3|repair failed|could not parse|Invalid JSON' "${log}" 2>/dev/null || true)"
    if [[ "${retries}" -gt 0 && "${state}" != "DONE" ]]; then
      notes="$(append_note "${notes}" "retries/errors=${retries}")"
    fi
    if [[ -f "${err}" ]] && [[ -s "${err}" ]]; then
      err_tail="$(tail -1 "${err}" | tr -d '\n' | cut -c1-60)"
      notes="$(append_note "${notes}" "err: ${err_tail}")"
    fi
  fi

  run_dir="${REPO}/runs/${job_id}_config${config_id}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    notes="$(append_note "${notes}" "metrics=ok")"
  fi

  display_round="${round}"
  if [[ -z "${display_round}" ]]; then
    display_round="-"
  fi

  printf '%-8s %-8s %-22s %-10s %-8s %-12s %s\n' \
    "${job_id}" "${config_id}" "${job_name}" "${state}" "${elapsed}" "${display_round}" "${notes}"
done < "${MANIFEST}"

total=$((running + done_ok + failed))
echo ""
echo "summary: running=${running} done=${done_ok} problem=${failed} total=${total}"

if [[ "${running}" -gt 0 ]]; then
  exit 2
fi
if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
exit 0
