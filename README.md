# Alpamayo 2 Super CoC 데이터셋 생성 파이프라인

NVIDIA Alpamayo 2 Super (34B) teacher 모델로 자율주행 클립에서 Chain-of-Causation 추론 데이터셋을 만드는 코드입니다. 목적은 이 데이터로 Jetson Thor에 올릴 수 있는 소형 student 모델을 증류하는 것입니다.

teacher는 69GB라 차량에 못 싣고 B200에서도 추론에 1.4초가 걸립니다. GPU 서버를 쓸 수 있을 때 teacher 출력을 최대한 뽑아두고, 나중에 작은 장비에서 student를 학습시키는 구조입니다.

## 무엇을 만드나

클립 하나에서 2초 간격으로 시점(t0)을 뽑고, 각 시점마다 teacher에게 6개 샘플을 생성시킵니다. 저장하는 것은 다음과 같습니다.

| 테이블 | 행 단위 | 내용 |
| --- | --- | --- |
| `frames/` | (clip, t0) | 7카메라 x 4프레임 = JPEG 28장, 타임스탬프 |
| `samples/` | (clip, t0, sample) | CoC 텍스트, 예측 궤적, GT 궤적, 과거 궤적, 토큰 분포, ADE |
| `ranges/` | clip | t0가 가능한 시간 구간 (재개 최적화용) |

용량은 t0당 약 1.15MB이고 그중 98%가 이미지입니다.

## 왜 토큰 분포까지 저장하나

teacher가 샘플링한 텍스트만 저장하면 sequence level SFT 데이터가 됩니다. 각 토큰 위치의 상위 20개 확률분포를 함께 저장하면 token level KL 증류가 가능해집니다.

실제로 저장된 값을 보면 teacher가 첫 토큰에서 "Adapt"(42%)와 "Nudge"(37%)를 거의 반반으로 놓고 고민하는 것이 보입니다. 샘플링된 텍스트만 남기면 이 37%는 영영 사라집니다.

저장 비용은 t0당 24KB로 프레임 대비 2% 수준이고, 34B와 Qwen3-VL student의 텍스트 토크나이저가 완전히 동일(151,669 토큰 전부 일치)해서 이 분포를 그대로 KL 타깃으로 쓸 수 있습니다.

나중에 추가할 수 없는 항목이라 처음부터 담았습니다. 프레임과 달리 재계산이 아니라 teacher 재실행이 필요합니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `tools/generate_coc_34b.py` | 메인 생성기 |
| `tools/run.sh` | 시작, 중단, 상태 확인 |
| `tools/build_clip_queue.py` | 작업 큐 생성 (vla_golden 우선) |
| `tools/validate_dataset.py` | 무결성 검사 (생성 중에도 실행 가능) |
| `tools/compact_dataset.py` | 중복 제거 및 shard 정리 |
| `tools/bench_cameras.py` | 카메라 수별 추론 지연 측정 |
| `tools/train_student_smoke.py` | student 학습 검증 |
| `tools/generate_coc_10b_legacy.py` | Alpamayo 1.5 (10B) 버전, 참고용 |
| `pyfix/sitecustomize.py` | huggingface.co 접속 장애 우회 |

## 사용법

환경 변수를 먼저 설정합니다. `env.sh.example`을 복사해 HF 토큰을 채우고 `env.sh`로 저장하세요.

```bash
source /path/to/alpamayo/env.sh

./tools/run.sh queue     # 클립 큐 생성 (최초 1회)
./tools/run.sh start     # 생성 시작
./tools/run.sh status    # 진행 상황
./tools/run.sh stop      # 중단, GPU 반납
```

`stop`은 언제 눌러도 안전합니다. 각 워커가 메모리에 있던 데이터를 저장하고 종료하며, `start`하면 남은 지점부터 이어갑니다. 강제 종료된 경우에도 마지막 자동 저장(5분 주기) 이후 분량만 잃습니다.

GPU를 일부만 쓰려면 환경 변수로 조절합니다.

```bash
NUM_GPUS=3 ./tools/run.sh start          # GPU 3장만
WORKERS_PER_GPU=1 ./tools/run.sh start   # GPU당 워커 1개
```

## 설계 결정과 근거

전부 실측으로 정한 값입니다. 추측으로 정한 것은 없습니다.

### 카메라 7대를 전부 저장한다

trajectory 태스크는 카메라 (0,1,2,3,5,6)을 쓰고 vqa는 (0,1,2,3,4,5)를 씁니다. 합집합이 7대입니다. 추론에는 태스크가 요구하는 부분집합만 들어가지만 저장은 전부 합니다.

학습할 때 카메라를 빼는 것은 언제든 가능하지만 없는 것을 나중에 추가할 수는 없습니다. 실제로 이 프로젝트에서 처음에 4카메라로 만든 데이터를 통째로 버린 적이 있습니다. 10B는 4대를 쓰는데 34B는 6대를 써서 호환되지 않았습니다.

7대와 6대의 저장 비용 차이는 3GB 정도입니다.

### JPEG를 먼저 만들고 그것을 디코드해서 추론한다

순서가 중요합니다. 원본으로 추론하고 JPEG를 저장하면 저장된 이미지와 저장된 CoC가 서로 다른 입력에서 나온 것이 됩니다.

통제 실험 결과입니다.

| 비교 | CoC 일치 | ADE 최대차 |
| --- | --- | --- |
| 원본 vs 원본 (같은 seed 재실행) | 6/6 | 0.0000 |
| 원본 vs 리사이즈만 | 6/6 | 0.06 |
| 리사이즈 vs JPEG q92 | 5/6 | 1.14 |

JPEG 압축만으로 6개 중 1개의 CoC가 뒤집힙니다. 인코딩을 먼저 하면 저장 데이터만으로 비트 단위 재현이 됩니다. 재검증에서 CoC 6/6 일치, ADE 차이 0.0000을 확인했습니다.

### t0 간격은 2초

logprob 조건을 고정하고 A/B 한 결과입니다.

| 간격 | t0/h | clips/h |
| --- | --- | --- |
| 1초 | 1,278 | 78 |
| 2초 | 1,371 | 152 |

두 지표 모두 2초가 우세합니다. 처음에는 1초가 나아 보였는데 그 비교에 logprob 유무가 섞여 있었습니다.

### num_traj_samples는 6

k=12는 GPU 메모리 95GB를 써서 GPU당 워커 2개(183GB)를 초과합니다. 텍스트 다양성은 토큰 분포가 대신 담당하므로 6으로 충분합니다.

### top-k는 20

k를 늘려야 하는지 검토하고 기각했습니다. 38,430 토큰으로 측정한 결과입니다.

| k | 누적 확률 |
| --- | --- |
| 20 | 0.8195 (실측) |
| 50 | 0.8214 (추정) |
| 100 | 0.8215 (추정) |

확률이 1위와 2위에 몰린 뒤 곧바로 평평한 꼬리로 흩어집니다. 21위 이하 15만 개가 나눠 갖는 18%는 토큰당 1e-6 수준이라 k를 늘려도 담기지 않습니다. 재생성 비용 9시간에 이득 0.2%p이므로 하지 않았습니다.

토큰별로 보면 79%는 top-20으로 99% 이상 담깁니다. 평균을 끌어내리는 20%는 teacher가 실제로 갈피를 못 잡는 분기점입니다.

## 성능

실측값입니다. 워커 8개, B200 4장 기준입니다.

| 항목 | 값 |
| --- | --- |
| 처리량 | 약 1,500 t0/h, 168 clips/h |
| 크기 | 1.147 MB/t0 |
| 프롬프트 | 4,580 토큰 (6카메라 x 4프레임) |
| 추론 | 2.8초/t0 (6샘플) |

병목은 GPU가 아닙니다. 구간별 계측 결과입니다.

| 구간 | 비중 |
| --- | --- |
| tokenize (이미지 전처리 포함) | 19% |
| decode (영상) | 22% |
| open (네트워크 스트리밍) | 20% |
| infer (GPU) | 15% |
| encode (JPEG) | 6% |

처음에는 JPEG 인코딩이 병목이라 보고 스레드 병렬화까지 했는데 실제로는 6%였습니다. 계측 결과 tokenize가 39%로 최대 병목이었고, 원인은 `helper.prepare_model_inputs`가 t0마다 `AutoProcessor.from_pretrained`를 호출하는 것이었습니다. 프로세서를 한 번만 만들도록 고쳐서 39%에서 19%로 줄였고 GPU 사용률이 18%에서 62%로 올랐습니다.

## 카메라 수별 추론 지연

Thor 배포 설계용으로 측정한 값입니다. B200, num_traj_samples=1 기준입니다.

| 카메라 | 이미지 | 토큰 | 지연 | 속도 향상 |
| --- | --- | --- | --- | --- |
| 1 | 4 | 838 | 0.591s | 2.38x |
| 2 | 8 | 1,585 | 0.913s | 1.54x |
| 3 | 12 | 2,333 | 1.033s | 1.36x |
| 4 | 16 | 3,082 | 1.211s | 1.16x |
| 6 | 24 | 4,580 | 1.408s | 1.00x |

토큰 수는 카메라에 정확히 선형입니다. 이미지 1장당 187토큰입니다.

지연은 선형이 아닙니다. 토큰이 5.5배 늘어날 때 지연은 2.4배만 늡니다. 고정 비용(diffusion expert, CoC 디코딩)이 지배해서 이미지 4장짜리 최소 구성에서도 0.59초가 듭니다.

함의는 카메라 축소만으로 실시간(10Hz)에 도달할 수 없다는 것입니다. 6대에서 3대로 줄여도 1.36배인데 14배가 필요합니다. 모델 축소가 필수이고 카메라 축소는 보조 수단입니다.

## student 학습 검증

teacher가 사라진 뒤에는 데이터를 다시 만들 수 없으므로 포맷이 학습에 물리는지 미리 확인했습니다.

Qwen3-VL-2B 백본으로 Alpamayo2Super 구조를 만들어 실제 데이터로 학습시킨 결과입니다.

```
사전학습 로드: 완전복사 624, 부분복사 2(vocab 확장), 건너뜀 0
step  1: loss 38.33
step  5: loss 26.59
step 10: loss 21.13
step 15: loss 19.47
```

사전학습 가중치가 하나도 버려지지 않고(건너뜀 0) 들어갔고, loss가 단조 감소합니다.

학습 경로에서 막혔던 지점 두 가지를 기록해둡니다.

첫째, teacher config를 로드해서 백본 이름만 바꾸면 안 됩니다. `vlm_config`가 34B로 굳어 있고 `traj_ids`도 34B 토크나이저 기준으로 계산돼 있습니다. `vlm_name_or_path`만 주고 처음부터 생성하면 config가 토크나이저 확장과 traj_ids를 백본에 맞춰 다시 계산합니다.

둘째, `helper.prepare_model_inputs`는 `generation_mode=True`로 하드코딩돼 있어 학습에 쓸 수 없습니다. 추론 모드는 생성할 대상을 시퀀스에서 빼기 때문에 future 궤적 자리표시자가 0개가 되고 `fuse_traj_tokens`가 128개를 넣지 못해 실패합니다. `build_conversation`을 직접 호출해 `generation_mode=False`로 만들어야 합니다.

학습 forward는 diffusion expert를 쓰지 않습니다. 궤적이 이산 토큰으로 융합되어 텍스트와 함께 next token loss로 학습됩니다. expert는 추론 시 그 토큰을 연속 궤적으로 정제하는 역할입니다.

## 운영 중 알아야 할 것

### 네트워크 우회가 필요합니다

이 노드에서는 `huggingface.co`가 자주 죽습니다. CloudFront anycast라 IP가 여러 개인데 그중 다수가 SYN에 응답하지 않습니다. 파이썬 기본 연결 타임아웃이 무한이라 죽은 IP를 뽑으면 오류 없이 영원히 멈춥니다.

`pyfix/sitecustomize.py`가 `PYTHONPATH`를 통해 자동 적용됩니다. 원리는 CloudFront 엣지가 SNI로 배포를 고르므로, 살아있는 다른 HF 엣지(`cdn-lfs.hf.co` 등)에 `huggingface.co`라는 SNI로 붙는 것입니다. IP를 하드코딩하지 않아 엣지가 바뀌어도 동작합니다.

증상 확인은 `ss -tnp | grep <PID>`로 `SYN-SENT`가 보이는지 확인하면 됩니다.

### 검증과 압축

```bash
# 검증 (생성 중에도 가능, nice로 우선순위 낮춤)
nice -n 15 python tools/validate_dataset.py

# 압축 (반드시 stop 후)
./tools/run.sh stop
python tools/compact_dataset.py           # 미리보기
python tools/compact_dataset.py --apply   # 적용
./tools/run.sh start
```

압축은 새 파일을 먼저 쓰고 검증한 뒤에 원본을 지웁니다. 중간에 실패해도 원본이 남습니다.

별도 디렉토리에서 실험을 돌린 뒤 본 출력에 병합하면 중복이 생길 수 있습니다. 같은 출력 디렉토리를 쓰면 재개 로직이 막아줍니다.

### 학습 시 train과 val은 clip_id 단위로 나눌 것

t0 간격이 2초인데 예측 지평이 6.4초라 인접 t0끼리 미래 궤적의 70%가 겹칩니다. t0로 쪼개면 val에 train과 거의 같은 장면이 들어가 성능이 부풀려집니다.

## 데이터 읽기

```python
import pyarrow.parquet as pq, glob, numpy as np, io
from PIL import Image

D = "path/to/coc_34b_v1"
s = pq.read_table(sorted(glob.glob(f"{D}/samples/*.parquet"))).to_pandas()
f = pq.read_table(sorted(glob.glob(f"{D}/frames/*.parquet"))).to_pandas()

r = s.iloc[0]
traj = np.array(r.pred_xyz).reshape(64, 3)
dist = np.array(r.topk_logprobs).reshape(r.num_gen_tokens, r.topk_k)

fr = f[(f.clip_id == r.clip_id) & (f.t0_us == r.t0_us)].iloc[0]
imgs = [np.array(Image.open(io.BytesIO(b))) for b in fr.jpegs]
```

카메라 순서는 항상 0부터 6까지 고정입니다. cross_left, front_wide, cross_right, rear_left, rear_tele, rear_right, front_tele 순이고 각 카메라마다 프레임 4장이 이어집니다.

`pass_filter`는 `ade <= 1.0m` 여부를 표시하는 참고용 플래그입니다. ADE 원본이 저장돼 있어 임계값을 나중에 바꿔도 재생성이 필요 없습니다.

## 관련 저장소

이 코드는 다음 NVIDIA 저장소에 의존합니다.

1. NVlabs/alpamayo2 : 34B 추론 코드
2. NVlabs/alpamayo-recipes : SFT와 RL post training 레시피 (Alpamayo 1.5용)
3. nvidia/Alpamayo2-Super : 모델 가중치 (gated)
4. nvidia/PhysicalAI-Autonomous-Vehicles : 원본 주행 데이터 (gated)
