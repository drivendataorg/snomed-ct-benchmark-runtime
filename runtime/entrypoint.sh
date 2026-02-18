#!/usr/bin/env bash

set -euxo pipefail

log() {
    set +x
    local level="$1"; shift
    printf '%s | %-5s | %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S.%3N')" \
        "$level" \
        "$*"
    set -x
}

main () {
    expected_filename=main.py

    cd /code_execution

    if [ "${IS_SMOKE_TEST:-0}" -eq 1 ]; then
        log INFO "This is a smoke test run."
    fi

    # Check that expected entrypoint script exists
    submission_files=$(unzip -Z1 ./submission/submission.zip)
    if ! grep -F -x -q -- "$expected_filename" <<<"$submission_files"; then
        log ERROR "Submission zip archive must include $expected_filename"
        return 1
    fi

    log INFO "Unpacking submission into src/..."
    unzip ./submission/submission.zip -d ./src

    log INFO "Showing current working directory contents:"
    ls -alh

    log INFO "Showing data/ directory contents:"
    ls -alh data/

    log INFO "Showing src/ directory contents:"
    find src/

    log INFO "Running submission..."

    if [ "${IS_SMOKE_TEST:-0}" -eq 1 ]; then
        uv run src/main.py
    else
        uv run src/main.py &> "/code_execution/submission/private_log.txt"
    fi

    log INFO "Exporting submission.csv result..."

    # Valid scripts must create a "submission.csv" file in the /code_execution directory
    if [ -f "submission.csv" ]; then
        log INFO "Script completed its run."
        cp submission.csv ./submission/submission.csv
    else
        log ERROR "Script did not produce a submission.csv file in the /code_execution directory."
        return 1
    fi
}

main |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

log INFO "Submission run completed with exit code: $exit_code" | tee -a "/code_execution/submission/log.txt" "/tmp/log"

exit $exit_code

# End of entrypoint
