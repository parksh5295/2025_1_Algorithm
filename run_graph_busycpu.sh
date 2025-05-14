#!/bin/bash
# ================================
# 멀티코어에서 커맨드 및 CPU 유틸리티 실행 스크립트 (안정화 버전)
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

if (( max_parallel == 0 )); then
    echo "[WARN] max_parallel is set to 0. Defaulting to 1 for this script's logic."
    max_parallel=1
fi

# 백그라운드 PID 관리용 배열 및 맵
declare -a ACTIVE_GRAPH_PIDS=() # 현재 실행 중인 Wildfire_spread_graph.py의 PID 목록
declare -a ALL_CPU_UTIL_PIDS=() # 실행된 모든 cpu_util.py PID (trap에서 전체 정리용)
declare -A GRAPH_TO_CPU_MAP=()  # GRAPH_PID를 키로, 해당 CPU_UTIL_PID를 값으로 매핑

# 함수: 모든 백그라운드 cpu_util 프로세스 정리
cleanup_all_cpu_utils() {
    echo "스크립트 종료 중... 모든 cpu_util.py 프로세스를 정리합니다."
    if [ ${#ALL_CPU_UTIL_PIDS[@]} -ne 0 ]; then
        for pid_to_kill in "${ALL_CPU_UTIL_PIDS[@]}"; do
            if [[ "$pid_to_kill" =~ ^[0-9]+$ ]] && ps -p "$pid_to_kill" > /dev/null; then
                echo "cpu_util.py (PID: $pid_to_kill) 종료 시도..."
                kill "$pid_to_kill"
                wait "$pid_to_kill" 2>/dev/null # 오류 메시지 숨김
            fi
        done
    fi
    # 남아있을 수 있는 메인 그래프 프로세스도 정리 (선택적이지만, 안전을 위해)
    echo "남아있는 메인 그래프 프로세스 확인 및 정리 시도..."
    if [ ${#ACTIVE_GRAPH_PIDS[@]} -ne 0 ]; then
        for graph_pid_to_kill in "${ACTIVE_GRAPH_PIDS[@]}"; do
             if [[ "$graph_pid_to_kill" =~ ^[0-9]+$ ]] && ps -p "$graph_pid_to_kill" > /dev/null; then
                echo "메인 그래프 프로세스 (PID: $graph_pid_to_kill) 종료 시도..."
                kill "$graph_pid_to_kill"
                wait "$graph_pid_to_kill" 2>/dev/null
            fi
        done
    fi
    echo "모든 정리 작업 시도 완료."
}

# 스크립트 종료 또는 인터럽트 시 cleanup_all_cpu_utils 함수 실행하도록 설정
trap cleanup_all_cpu_utils EXIT SIGINT SIGTERM

# --- 메인 루프 ---
command_idx=0
total_commands=${#commands[@]}

while (( command_idx < total_commands || running_jobs > 0 )); do

    # 새 작업 시작 조건: 실행할 명령이 남았고, 현재 실행 중인 작업 수가 max_parallel보다 적을 때
    while (( command_idx < total_commands && running_jobs < max_parallel )); do
        cmd_to_run="${commands[$command_idx]}"
        echo "----------------------------------------------------"
        echo "새 작업 시작 준비: $cmd_to_run (인덱스: $command_idx)"

        # 1. cpu_util.py 백그라운드 실행
        python cpu_util.py > /dev/null 2>&1 &
        current_cpu_util_pid=$!
        if ! [[ "$current_cpu_util_pid" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] cpu_util.py 시작 실패! (명령: $cmd_to_run)"
            ((command_idx++))
            continue # 다음 명령어로
        fi
        ALL_CPU_UTIL_PIDS+=("$current_cpu_util_pid")
        echo "cpu_util.py 시작됨 (PID: $current_cpu_util_pid) - 작업: $cmd_to_run"

        # 2. 메인 커맨드(Wildfire_spread_graph.py) 백그라운드 실행
        # 종료 코드 127 문제 방지 및 $GRAPH_PID가 현재 쉘의 자식이 되도록 bash -c 사용
        # 만약 가상환경(예: pipenv) 사용 시: bash -c "pipenv run $cmd_to_run" &
        bash -c "$cmd_to_run" &
        current_graph_pid=$!

        if ! [[ "$current_graph_pid" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] 메인 작업 '$cmd_to_run' 시작 실패!"
            # 이미 시작된 cpu_util.py 정리
            if ps -p "$current_cpu_util_pid" > /dev/null; then
                 echo "관련 cpu_util.py (PID: $current_cpu_util_pid) 정리 중..."
                 kill "$current_cpu_util_pid"; wait "$current_cpu_util_pid" 2>/dev/null
                 # ALL_CPU_UTIL_PIDS에서 제거하는 로직 추가 가능 (복잡도 증가로 일단 생략)
            fi
            ((command_idx++))
            continue # 다음 명령어로
        fi
        echo "메인 작업 시작됨 (PID: $current_graph_pid) - $cmd_to_run"

        # PID 정보 저장
        ACTIVE_GRAPH_PIDS+=("$current_graph_pid")
        GRAPH_TO_CPU_MAP["$current_graph_pid"]="$current_cpu_util_pid"
        
        ((running_jobs++))
        ((command_idx++))
        echo "현재 실행 중인 작업 수: $running_jobs / $max_parallel"
        echo "----------------------------------------------------"
    done

    # 실행 중인 작업이 있고, (새 작업을 시작할 수 없거나 || 모든 명령을 이미 시작한 경우)
    if (( running_jobs > 0 )); then
        # 하나 이상의 백그라운드 작업(메인 그래프 스크립트)이 완료되기를 기다림
        # wait -n 은 현재 쉘의 자식 프로세스 중 하나가 종료되면 반환
        # bash -c 로 실행했으므로 메인 그래프 PID는 현재 쉘의 자식임
        wait -n
        # 참고: wait -n은 종료된 PID를 직접 알려주지 않음. Bash 4.3+에서는 wait -p var -n 사용 가능.
        # 여기서는 ACTIVE_GRAPH_PIDS를 순회하며 종료된 것을 찾아야 함.

        temp_active_graph_pids=()
        processed_finished_job_this_iteration=false

        for graph_pid_check in "${ACTIVE_GRAPH_PIDS[@]}"; do
            if ! ps -p "$graph_pid_check" > /dev/null; then # 프로세스가 존재하지 않으면 종료된 것
                echo "----------------------------------------------------"
                echo "메인 작업 (PID: $graph_pid_check) 완료 감지."
                
                associated_cpu_pid="${GRAPH_TO_CPU_MAP[$graph_pid_check]}"
                if [[ -n "$associated_cpu_pid" ]] && ps -p "$associated_cpu_pid" > /dev/null; then
                    echo "관련 cpu_util.py (PID: $associated_cpu_pid) 종료 중..."
                    kill "$associated_cpu_pid"
                    wait "$associated_cpu_pid" 2>/dev/null
                    echo "cpu_util.py (PID: $associated_cpu_pid) 종료됨."
                else
                    echo "관련 cpu_util.py (PID: $associated_cpu_pid)는 이미 종료되었거나 찾을 수 없습니다."
                fi

                unset GRAPH_TO_CPU_MAP["$graph_pid_check"]
                ((running_jobs--))
                processed_finished_job_this_iteration=true
                # 참고: 동시에 여러 작업이 끝났을 수 있으므로, break하지 않고 계속 확인
            else
                temp_active_graph_pids+=("$graph_pid_check") # 아직 실행 중인 PID는 임시 배열에 유지
            fi
        done
        ACTIVE_GRAPH_PIDS=("${temp_active_graph_pids[@]}") # 실행 중인 PID로 배열 업데이트

        if ! $processed_finished_job_this_iteration && (( running_jobs > 0 )); then
            # wait -n이 반환되었지만, 우리가 추적하는 ACTIVE_GRAPH_PIDS 중 어느 것도 종료되지 않은 경우.
            # 이는 cpu_util.py가 먼저 (오류 등으로) 종료되었거나, 다른 예상치 못한 백그라운드 작업이 끝난 경우일 수 있음.
            # 이 경우 running_jobs를 강제로 줄여서 루프가 멈추는 것을 방지할 수 있으나, 원인 파악이 필요할 수 있음.
            # 여기서는 일단 그대로 두지만, 장시간 실행 시 이 부분을 모니터링할 필요가 있을 수 있음.
            echo "[DEBUG] wait -n 반환되었으나, 추적 중인 메인 작업의 종료를 감지하지 못함. 현재 running_jobs: $running_jobs"
        fi
        echo "현재 실행 중인 작업 수: $running_jobs / $max_parallel (점검 후)"
        echo "----------------------------------------------------"
    fi

    # 모든 명령이 시작되었고 실행 중인 작업도 없으면 루프 종료
    if (( command_idx >= total_commands && running_jobs == 0 )); then
        break
    fi

    sleep 0.1 # CPU 과부하 방지를 위한 짧은 대기 (루프 반복 속도 조절)
done

echo "모든 작업이 제출되고 완료되었습니다."
# trap EXIT 핸들러가 최종 정리를 수행합니다.
