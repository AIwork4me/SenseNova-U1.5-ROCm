#!/usr/bin/env bash
# HSA runtime compatibility shim — Strix Halo (gfx1151) hardware profile — source this file to get hsa_fix_apply().
#
# Official PyTorch +rocm7.0 wheels bundle a ROCm 7.0 HSA runtime that
# segfaults on gfx1151 (Strix Halo) when the kernel driver is from ROCm
# >= 7.1: crash in GpuAgent::QueueCreate while freezing the first code
# object. Preloading the SYSTEM runtime fixes it, because the system
# runtime always matches the installed kernel driver.
#
# Only applied when the system runtime is NEWER than the bundled one, so
# machines where the wheel's runtime works (or is newer) are untouched.
#
# Exports: LD_PRELOAD (extended when the fix applies)
#
# shellcheck disable=SC2034  # $2 (REPO_ROOT) kept for the sourced interface
hsa_fix_apply() { # $1 = venv python, $2 = repo root
  local PY="$1" REPO_ROOT="$2"
  local sys_rt="/opt/rocm/lib/libhsa-runtime64.so"
  [ -f "$sys_rt" ] || return 0
  local sys_ver bundled_ver
  sys_ver="$(cat /opt/rocm/.info/version 2>/dev/null || echo 0)"
  bundled_ver="$("$PY" -c 'import torch; print(torch.version.hip or "0")' 2>/dev/null || echo 0)"
  # numeric major.minor.patch comparison; preload only when system is newer
  if awk -v a="$sys_ver" -v b="$bundled_ver" 'BEGIN {
      split(a, A, "[^0-9]"); split(b, B, "[^0-9]");
      for (i = 1; i <= 3; i++) { if (A[i] + 0 != B[i] + 0) exit (A[i] + 0 > B[i] + 0) ? 0 : 1 }
      exit 1
    }'; then
    case ":${LD_PRELOAD:-}:" in
      *":$sys_rt:"*) ;;  # already included
      *) export LD_PRELOAD="$sys_rt${LD_PRELOAD:+:${LD_PRELOAD}}"
         echo "[sensenova-rocm] applied HSA runtime fix: LD_PRELOAD=$sys_rt (system ROCm ${sys_ver} > bundled ${bundled_ver})"
         ;;
    esac
  fi
}
