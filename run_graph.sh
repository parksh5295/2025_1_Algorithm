#!/bin/bash
# ================================
# 멀티코어에서 커맨드 실행 스크립트
#
# 사용법:
# 1. 아래 commands 배열에 실행하고자 하는 커맨드를 원하는 대로 추가합니다.
#    예: "python code.py --argument1 true --argument2 False"
#
# 2. 스크립트에 실행 권한을 부여하고 실행합니다.
#    chmod +x run.sh
#    ./run.sh
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
running=0
#max_parallel=$NUM_CORES
max_parallel=0

for cmd in "${commands[@]}"; do
    # 최대 동시 실행 프로세스 수에 도달하면 대기
    while (( running >= max_parallel )); do
        wait -n # 가장 먼저 종료된 백그라운드 프로세스를 기다림
        ((running--)) # 실행 중인 프로세스 카운터 감소
    done

    # 현재 명령어 실행 (백그라운드에서)
    echo "실행 시작: $cmd"
    eval "$cmd" & # eval을 사용하여 명령어 문자열 실행, 백그라운드 실행

    ((running++)) # 실행 중인 프로세스 카운터 증가
done

# 남은 모든 프로세스가 완료될 때까지 대기
wait
echo "모든 작업이 완료되었습니다."
