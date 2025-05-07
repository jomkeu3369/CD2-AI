import json
import os
from rl_criteria_config import criteria as schema_criteria

TEST_WEIGHTS_FILE = "test_criterion_rl_weights.json"
criterion_rl_data = {}

def initialize_weights():
    global criterion_rl_data
    for category, items in schema_criteria.items():
        for kr_name, _ in items:
            criterion_key = f"{category}::{kr_name}"
            if criterion_key not in criterion_rl_data:
                criterion_rl_data[criterion_key] = {"weight": 1.0, "recommend_score": 0}
    save_weights()
    print("테스트용 RL 가중치 초기화 완료.")

def load_weights():
    global criterion_rl_data
    if os.path.exists(TEST_WEIGHTS_FILE):
        try:
            with open(TEST_WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                criterion_rl_data = json.load(f)
            
            initial_data_keys = set()
            for category, items in schema_criteria.items():
                for kr_name, _ in items:
                    initial_data_keys.add(f"{category}::{kr_name}")
            
            updated = False
            for key in initial_data_keys:
                if key not in criterion_rl_data:
                    criterion_rl_data[key] = {"weight": 1.0, "recommend_score": 0}
                    updated = True
            
            keys_to_remove = set(criterion_rl_data.keys()) - initial_data_keys
            if keys_to_remove:
                for key in keys_to_remove:
                    del criterion_rl_data[key]
                updated = True

            if updated:
                save_weights()
            print("테스트용 RL 가중치 로드 완료.")

        except json.JSONDecodeError:
            print(f"경고: {TEST_WEIGHTS_FILE} 파일이 손상되었습니다. 가중치를 초기화합니다.")
            criterion_rl_data = {}
            initialize_weights()
    else:
        print(f"{TEST_WEIGHTS_FILE} 파일이 존재하지 않아 가중치를 초기화합니다.")
        initialize_weights()

def save_weights():
    with open(TEST_WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(criterion_rl_data, f, indent=2, ensure_ascii=False)
    print(f"테스트용 RL 가중치가 {TEST_WEIGHTS_FILE} 파일에 저장되었습니다.")

load_weights()