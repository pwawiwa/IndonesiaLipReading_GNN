#!/bin/bash
# ============================================
# Test multiple speech mask scaling values
# ============================================

set -e

cd "$(dirname "$0")/.."

# Values to test (you mentioned "we done 10 mostly", so starting with higher values)
SPEECH_MASK_SCALES=(0.5 1.0 2.0 5.0 10.0)

# Log file
LOGS_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOGS_DIR}"
TEST_LOG="${LOGS_DIR}/speech_mask_scale_test_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${TEST_LOG}"
}

log "============================================"
log "TESTING SPEECH MASK SCALING VALUES"
log "============================================"
log "Values to test: ${SPEECH_MASK_SCALES[@]}"
log ""

for scale in "${SPEECH_MASK_SCALES[@]}"; do
    log ""
    log "============================================"
    log "Testing speech_mask_scale = ${scale}"
    log "============================================"
    
    # Run experiment with this scale value
    SPEECH_MASK_SCALE=${scale} ./scripts/experiment.sh
    
    log "Completed testing with speech_mask_scale = ${scale}"
    log ""
done

log "============================================"
log "ALL TESTS COMPLETE"
log "============================================"
log "Results saved in: ${RESULTS_DIR}"
log "Check individual run folders for best validation accuracy"
log ""

