# 전사·예측 서버

FastAPI 기반의 로컬 서버입니다. 전사 작업을 처리하고, 편집 상태를 로컬 언어모델에 전달한 뒤 제안의 대상·형식·문서 상태를 검증합니다.

![편집 상태를 로컬 모델에 전달하고 검증한 제안을 화면에 표시하는 구조](../docs/images/architecture.png)

## 실행

Python 3.13 이상과 uv가 필요합니다. `server/`에서 실행합니다.

```sh
uv sync --locked
uv run --locked python run.py
```

데스크톱 앱은 서버를 자동으로 실행하므로, 앱을 사용할 때에는 서버를 따로 켤 필요가 없습니다.

## 전사 설정

### ElevenLabs Scribe

앱을 실행할 터미널에 `ELEVENLABS_PERSONAL_API_KEY`를 설정한 뒤, 가져오기 화면에서 **ElevenLabs Scribe v2 (cloud)**를 선택합니다. 한국어·영어·자동 언어 감지를 지원합니다.

```sh
export ELEVENLABS_PERSONAL_API_KEY="your-api-key"
cd ../app
npm start
```

선택한 녹음이 ElevenLabs로 전송되며 API 사용 요금이 발생할 수 있습니다. 키는 저장소나 프로젝트 파일에 넣지 않습니다.

### 로컬 Vosk

`server/`에서 추가 의존성을 설치하고 앱의 모델 관리 화면에서 Vosk 모델을 내려받습니다.

```sh
uv sync --locked --extra local-asr
```

화자 분리에는 `HF_TOKEN`과 `pyannote/speaker-diarization-community-1` 모델 접근 권한도 필요합니다.

## 로컬 코파일럿

편집 예측은 별도로 실행한 `llama-server`에 연결합니다. 모델 가중치, 채팅 템플릿, 모델 등록 정보와 학습 토큰 검증 자료를 먼저 준비해야 합니다. 보고서의 실행 구성은 일곱 편집 동작을 학습한 Gemma 4 26B-A4B의 Q4_K_M 모델입니다.

저장소 루트에서 해당 모델과 일치하는 설정을 만듭니다. 출력에는 아직 없는 새 경로를 사용합니다.

```sh
server/.venv/bin/python server/scripts/prepare_copilot_config.py \
  --contract-version passive-nep-actions-v1 \
  --registry /path/to/model-registry.json \
  --release /path/to/frozen-training-release \
  --candidate your-model-id \
  --output /tmp/transcript-copilot-local --port 18080
```

생성된 설정에는 모델·템플릿·런타임의 식별 정보와 한국어·영어 토큰 검증 자료가 들어갑니다. 설정 생성만으로 모델을 내려받거나 실행하지는 않습니다.

모델을 실행할 터미널:

```sh
sh /tmp/transcript-copilot-local/start-model.sh
```

앱을 실행할 터미널:

```sh
cd app
AUDAPOLIS_NEP_CONFIG=/tmp/transcript-copilot-local/gateway.json npm start
```

편집기에서 **Local copilot**을 켜면 제안을 받을 수 있습니다. `AUDAPOLIS_NEP_CONFIG`가 없으면 코파일럿만 비활성화됩니다. 모델 서버는 루프백 주소에 연결하며, 호환되지 않는 설정이나 오래된 문서 상태의 제안은 적용하지 않습니다.

기존 로컬 실험 파일을 보유한 경우 `server/.venv/bin/python server/scripts/run_local_copilot_cached.py check`로 준비 상태를 확인하고, 마지막 인자를 `start`로 바꾸어 모델과 앱을 함께 실행할 수 있습니다. 이 실행기는 해당 실험의 파일 구성을 그대로 요구합니다.

## 검사

`server/`에서 실행합니다. 테스트는 저장된 예시와 모의 응답을 사용합니다.

```sh
uv run --locked python -m unittest discover -s tests
uv run --locked ruff check app tests scripts/prepare_copilot_config.py scripts/run_local_copilot*.py
uv run --locked ruff format --check app tests scripts/prepare_copilot_config.py scripts/run_local_copilot*.py
```
