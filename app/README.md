# 데스크톱 편집기

Electron, React, Redux 기반의 전사 편집 화면입니다. 전사문과 음성 구간을 연결하고, 로컬 모델의 제안 미리보기, 수락과 거절, 되돌리기를 처리합니다.

프로젝트 소개와 사용 화면은 [루트 README](../README.md)를 참고하세요.

## 실행

먼저 [서버 의존성](../server/README.md#실행)을 설치합니다. 아래 명령은 `app/`에서 실행합니다.

```sh
npm ci
npm start
```

개발 모드에서는 화면 변경을 자동 반영하고 Python 서버를 함께 시작합니다. 모델과 검증된 배포 설정을 별도로 준비한 경우 모델도 함께 시작하며, 문서 열기, 커서 이동과 편집 후 제안을 자동으로 표시합니다. Tab 수락, Esc 거절과 undo를 사용할 수 있습니다. 모델이 없으면 편집기는 열리지만 제안은 사용할 수 없습니다. 모델 준비와 외부 런타임 연결은 [서버 설정](../server/README.md#로컬-코파일럿)을 따릅니다.

## 코드 구성

| 경로 | 역할 |
| --- | --- |
| `main_process/` | 창 관리, Python 서버 실행, 데스크톱 메뉴 |
| `src/pages/Editor/` | 전사문과 쉼 편집, 제안 미리보기, 세션 기록 화면 |
| `src/core/` | 문서와 원본 음성 연결, 재생, 편집 검증 |
| `src/state/` | 편집 상태, 실행 취소, 예측 요청과 사용자 반응 |
| `src/server_api/` | Python 서버와의 통신 |

## 검사

```sh
npm run check
npm run test:fast -- --runInBand
node --test scripts/copilot_ui_helpers.test.cjs
MODE=test npx jest --runInBand '.*\.pup\.spec.ts'
```

마지막 명령은 실제 Electron 창을 사용하는 화면 검사입니다. Xvfb가 설치된 Linux에서는 `MODE=test npm run test:puppeteer`로 가상 화면에서 검사할 수 있습니다.

코드 서식은 `npm run fmt`로 정리합니다. 화면 검사에는 서버 의존성도 필요합니다.
