#!/bin/bash
# ================================
# 멀티코어에서 커맨드 및 CPU 유틸리티 실행 스크립트
#
# 각 메인 커맨드 실행 시 cpu_util.py를 함께 실행하고,
# 메인 커맨드 종료 시 해당 cpu_util.py도 종료합니다.
# ================================

# 시스템의 총 CPU 코어 수 가져오기
NUM_CORES=$(nproc)
echo "사용 가능한 CPU 코어 수: $NUM_CORES"

# 사용자 정의 커맨드 목록
commands=(
    "python Wildfire_spread_graph.py --data_number 1"
    "python Wildfire_spread_graph.py --data_number 2"
    "python Wildfire_spread_graph.py --data_number 3"
    "python Wildfire_spread_graph.py --data_number 4"
    "python Wildfire_spread_graph.py --data_number 5"
    "python Wildfire_spread_graph.py --data_number 6"
    "python Wildfire_spread_graph.py --data_number 7"
)

# 실행 중인 프로세스 수를 추적하기 위한 카운터
running_jobs=0
# max_parallel=$NUM_CORES # 모든 코어 사용
max_parallel=2 # 예시로 동시에 2개만 실행 (테스트용, 필요에 따라 조절)
# max_parallel=0 # 이 값은 아래 로직에서 1 이상이어야 합니다. 0이면 루프에 진입하지 않습니다.
                # 0으로 설정하고 싶으시면, 아래 while 루프 조건을 수정해야 합니다.
                # 보통은 최소 1개 이상 병렬 실행을 원하므로 $NUM_CORES 또는 특정 숫자로 설정합니다.

if (( max_parallel == 0 )); then
    echo "[WARN] max_parallel is set to 0. Defaulting to 1 for this script's logic."
    max_parallel=1
fi


# 모든 백그라운드 cpu_util PID를 저장할 배열
declare -a cpu_pids=()

# 함수: 모든 백그라운드 cpu_util 프로세스 정리
cleanup_all_cpu_utils() {
    echo "스크립트 종료 중... 모든 cpu_util.py 프로세스를 정리합니다."
    if [ ${#cpu_pids[@]} -ne 0 ]; then
        for pid_to_kill in "${cpu_pids[@]}"; do
            # PID가 유효한 숫자인지, 그리고 실제로 실행 중인지 확인
            if [[ "$pid_to_kill" =~ ^[0-9]+$ ]] && ps -p "$pid_to_kill" > /dev/null; then
                echo "cpu_util.py (PID: $pid_to_kill) 종료 시도..."
                kill "$pid_to_kill"
                wait "$pid_to_kill" 2>/dev/null # 종료될 때까지 기다리되, 오류는 숨김
            fi
        done
    fi
    echo "모든 cpu_util.py 정리 완료."
}

# 스크립트 종료 또는 인터럽트 시 cleanup_all_cpu_utils 함수 실행하도록 설정
trap cleanup_all_cpu_utils EXIT SIGINT SIGTERM

# 함수: 특정 메인 프로세스 종료 후 해당 cpu_util 프로세스 종료
# $1: 메인 프로세스 (Wildfire_spread_graph.py) PID
# $2: 해당 cpu_util.py PID
# $3: 메인 커맨드 문자열 (로깅용)
manage_single_job() {
    local graph_pid=$1
    local cpu_pid_to_manage=$2
    local main_cmd=$3

    wait "$graph_pid" # 메인 프로세스 종료 대기
    local exit_status=$? # 메인 프로세스 종료 상태

    echo "완료: $main_cmd (종료 코드: $exit_status)"
    echo "cpu_util.py (PID: $cpu_pid_to_manage) 종료 중 (관련 작업: $main_cmd)..."
    if [[ "$cpu_pid_to_manage" =~ ^[0-9]+$ ]] && ps -p "$cpu_pid_to_manage" > /dev/null; then
        kill "$cpu_pid_to_manage"
        wait "$cpu_pid_to_manage" 2>/dev/null
        echo "cpu_util.py (PID: $cpu_pid_to_manage) 종료됨."
    else
        echo "cpu_util.py (PID: $cpu_pid_to_manage)는 이미 종료되었거나 찾을 수 없습니다."
    fi
    
    # cpu_pids 배열에서 해당 PID 제거 (선택적이지만, 더 깔끔한 관리를 위해)
    local temp_pids=()
    for p in "${cpu_pids[@]}"; do
        if [[ "$p" != "$cpu_pid_to_manage" ]]; then
            temp_pids+=("$p")
        fi
    done
    cpu_pids=("${temp_pids[@]}")

    return $exit_status # 메인 프로세스의 종료 코드를 반환
}


for cmd in "${commands[@]}"; do
    # 최대 동시 실행 프로세스 수에 도달하면 대기
    while (( running_jobs >= max_parallel )); do
        # 백그라운드 작업 중 하나가 완료되기를 기다립니다.
        # 이 wait -n은 manage_single_job 프로세스 중 하나의 완료를 기다립니다.
        # wait -n은 종료된 프로세스의 PID를 반환하지 않으므로,
        # running_jobs 카운트는 manage_single_job이 실제 종료될 때 (내부적으로 wait $graph_pid 이후)
        # 정확히 감소해야 하지만, 여기서는 일단 하나가 끝나면 감소시킵니다.
        # 더 정교한 제어를 위해서는 각 manage_single_job PID를 추적하고
        # 특정 PID의 종료를 기다려야 합니다. (예: wait -n $specific_manage_job_pid)
        # 여기서는 단순성을 위해 어떤 백그라운드 작업이든 완료되면 카운트를 줄입니다.
        wait -n 
        ((running_jobs--))
    done

    echo "----------------------------------------------------"
    echo "새 작업 시작 준비: $cmd"

    # 1. cpu_util.py 백그라운드 실행 및 PID 저장
    # python cpu_util.py & # cpu_util.py에 전달할 인자 (예: intensity_level)가 있다면 추가
    python cpu_util.py > /dev/null 2>&1 & # 출력을 숨기고 백그라운드 실행
    CPU_UTIL_PID=$!
    if ! [[ "$CPU_UTIL_PID" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] cpu_util.py 시작 실패!"
        # 이 경우 해당 cmd를 건너뛰거나 스크립트를 중단할 수 있습니다. 여기서는 일단 진행.
    else
        cpu_pids+=("$CPU_UTIL_PID") # 전체 정리용 배열에 추가
        echo "cpu_util.py 시작됨 (PID: $CPU_UTIL_PID) - 작업: $cmd"
    fi


    # 2. 메인 커맨드(cmd) 백그라운드 실행 및 PID 저장
    echo "메인 작업 실행: $cmd"
    eval "$cmd" &
    GRAPH_PID=$!
    if ! [[ "$GRAPH_PID" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 메인 작업 '$cmd' 시작 실패!"
        # cpu_util.py를 이미 시작했다면 여기서 정리해야 할 수 있습니다.
        if [[ "$CPU_UTIL_PID" =~ ^[0-9]+$ ]] && ps -p "$CPU_UTIL_PID" > /dev/null; then
             kill "$CPU_UTIL_PID"; wait "$CPU_UTIL_PID" 2>/dev/null; 
        fi
        continue # 다음 명령으로 넘어감
    fi
    echo "메인 작업 시작됨 (PID: $GRAPH_PID) - $cmd"

    # 3. 메인 커맨드 종료 후 cpu_util 정리하는 함수를 백그라운드로 실행
    # CPU_UTIL_PID가 유효할 때만 manage_single_job 실행
    if [[ "$CPU_UTIL_PID" =~ ^[0-9]+$ ]]; then
        manage_single_job "$GRAPH_PID" "$CPU_UTIL_PID" "$cmd" &
        ((running_jobs++)) # 실행 중인 'job 쌍' (graph + cpu_util_manager) 카운터 증가
        echo "현재 실행 중인 작업 쌍 수: $running_jobs / $max_parallel"
    else
        # GRAPH_PID만 실행된 경우, 이것도 어떤 방식으로든 카운트하거나 관리해야 할 수 있음
        # 여기서는 cpu_util이 실패하면 manage_single_job을 시작하지 않고, running_jobs도 증가시키지 않음
        # (또는 GRAPH_PID만이라도 관리하는 별도 로직 추가 가능)
        echo "[WARN] cpu_util.py가 시작되지 않아 manage_single_job을 실행하지 않습니다 for $cmd"
        # 이 경우 GRAPH_PID가 끝나기를 기다리는 로직이 별도로 필요할 수 있으나,
        # 여기서는 일단 넘어갑니다. trap EXIT가 GRAPH_PID를 직접 정리하진 않습니다.
        # 하지만 스크립트가 끝나면 어차피 종료됩니다.
    fi
    echo "----------------------------------------------------"
done

# 남은 모든 'manage_single_job' 백그라운드 프로세스가 완료될 때까지 대기
echo "모든 초기 명령어 제출 완료. 남은 작업들의 완료를 기다립니다..."
wait # 모든 자식 백그라운드 프로세스(manage_single_job들)의 종료를 기다림
echo "모든 작업이 최종적으로 완료되었습니다."
