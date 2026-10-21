# QNRS — Quantum Nanorobot Runtime

**Русский** · [English](docs/README.en.md)

> Шаг 2 из плана разработки ПО квантовых нанороботов: **слой API, который
> работает с реальными квантовыми инструментами** — единый интерфейс поверх
> Qiskit, Silq, PennyLane, Labber, QCoDeS и ROS 2.

Это не набросок на 200 строк, а запускаемый пакет: 10 виртуальных устройств,
единый REST/WebSocket-шлюз, ROS 2-мост, 86 тестов и end-to-end демо.

```bash
pip install -e ".[all]"
qnrs demo          # полный проход по всем адаптерам
qnrs serve         # HTTP + WebSocket шлюз на :8080
```

---

## 1. Что решает этот слой

Разнородные квантовые приборы говорят на разных языках: Qiskit — логические
схемы, Silq — высокоуровневый язык, PennyLane — дифференцируемые схемы,
Labber/QCoDeS — импульсы и калибровочные параметры, ROS 2 — сообщения робота.
QNRS вводит один контракт между ними:

```
приложение / ROS 2 / HTTP-клиент
        │
        ▼
Unified Quantum Abstraction Layer   ← device model, discovery, negotiation,
        │                              state tracking, pulse IR, timing budget
        ├── QiskitAdapter        (gate_sequence, measurement_feedback)
        ├── SilqAdapter          (front-end → circuit IR)
        ├── PennyLaneAdapter     (differentiable + variational ML)
        ├── LabberAdapter        (pulse_sequence, analog_control) ← JSON-RPC мост
        ├── QCoDeSAdapter        (pulse_sequence, analog_control) ← параметры/свипы
        └── NumpySimulatorAdapter(эталон, работает всегда, без SDK)
```

Ключевая идея: **адаптер — это одновременно и `ProtocolNegotiator`
(обнаружение + проверка живости), и исполнитель (схема/импульс)**. Шлюз
маршрутизирует задание по `device_id`, а не по имени вендора.

## 2. Стек → адаптер → протокол

| Компонент | Роль в QNRS | Что реально используется | Протокол |
|---|---|---|---|
| **Qiskit** | исполнение логических схем | `qiskit.primitives.StatevectorSampler`, `qiskit_aer.AerSimulator`, IBM Runtime (если есть токен) | `gate_sequence`, `measurement_feedback` |
| **Silq** | высокоуровневый front-end | собственный парсер подмножества Silq → IR | `gate_sequence` |
| **PennyLane** | дифференцируемые схемы и вариационная оптимизация | `default.qubit`, `AdamOptimizer` | `gate_sequence` |
| **Labber** | импульсное управление приборами | JSON-RPC 2.0 мост (+ встроенный mock-сервер) | `pulse_sequence`, `analog_control` |
| **QCoDeS** | параметры, калибровка, свипы | `Station`, `DummyInstrument` (fallback — внутренний store) | `pulse_sequence`, `analog_control` |
| **ROS 2** | интеграция с роботом | ament-пакет `qnrs_ros2`: msg/srv + rclpy-узел | pub/sub + сервисы |

## 3. Быстрый старт

```bash
# 1. Установка (ядро — только numpy; extras ставят вендорные SDK)
pip install -e ".[all]"     # qiskit, pennylane, qcodes, fastapi, uvicorn
# на старом pip (<23) добавьте --no-build-isolation:
#   pip install -e ".[all]" --no-build-isolation

# 2. Демо: обнаружение, план, схема Белла, Silq, импульс, вариационная подготовка
qnrs demo

# 3. CLI
qnrs discover                       # список устройств
qnrs negotiate numpy:simulator --required gate_sequence
qnrs plan --bell                    # ранжирование устройств под схему
qnrs circuit --bell --statevector
qnrs silq --file examples/nanorobot_scan.silq
qnrs pulse
qnrs optimize --layers 3 --steps 80

# 4. HTTP/WebSocket шлюз
qnrs serve --host 0.0.0.0 --port 8080
# Swagger: http://localhost:8080/docs
```

Пример работы с шлюзом:

```bash
curl -s localhost:8080/v1/devices | jq '.[].device_id'

curl -s -X POST localhost:8080/v1/jobs/circuit \
  -H 'content-type: application/json' \
  -d '{"program":{"num_qubits":2,
        "operations":[{"name":"H","qubits":[0]},{"name":"CNOT","qubits":[0,1]}],
        "shots":4096}}' | jq '.result.counts'
# => { "00": 2054, "11": 2042 }

curl -s -X POST localhost:8080/v1/jobs/variational \
  -H 'content-type: application/json' \
  -d '{"target_state":{"real":[0.7071,0,0,0.7071]},"layers":2,"steps":60}' \
  | jq '.variational.final_fidelity'
# => 0.999
```

## 4. Структура

```
quantum-nanorobot-stack/
├── src/qnrs/
│   ├── core/            device model, discovery, negotiation, state, errors, pulse IR, timing
│   ├── adapters/        qiskit · silq · pennylane · labber · qcodes · numpy-эталон
│   ├── protocols/       labber_rpc (JSON-RPC 2.0 клиент + mock-сервер)
│   ├── sim/             numpy state-vector симулятор
│   ├── api/             FastAPI шлюз + pydantic-модели
│   ├── ros2/            transport-agnostic мост (тестируется без ROS)
│   ├── runtime.py       фасад: discovery → планирование → маршрутизация → job store
│   └── cli.py           qnrs ...
├── ros2_ws/src/qnrs_ros2/   ament-пакет: msg/, srv/, rclpy-узел
├── tests/               86 тестов
├── examples/            Silq-программа, JSON-схема, импульсная последовательность
├── scripts/             upload_to_github.py — публикация через GitHub-коннектор
├── .github/workflows/ci.yml  CI: Python 3.10–3.12 + core-only (numpy) прогон
├── docs/ARCHITECTURE.md      слои, потоки данных, модель времени
├── docs/PROTOCOLS.md         спецификации: discovery, Labber JSON-RPC, circuit IR, pulse IR, ROS 2
├── docs/ROADMAP.md           привязка к 24-месячному плану
├── docs/README.en.md         английская версия README
├── CHANGELOG.md         история версий
├── LICENSE              MIT
└── Makefile             install / test / demo / serve / examples
```

## 5. Что реализовано (и как проверено)

| Возможность | Проверка |
|---|---|
| Модель устройства + валидация калибровки (в т.ч. физическое `T2 ≤ 2·T1`) | `tests/test_discovery.py` |
| Подписанные (HMAC-SHA256) сообщения обнаружения + защита от replay | `tests/test_discovery.py` |
| Согласование протокола с прозрачным скорингом | `tests/test_discovery.py` |
| Точная эволюция состояния (`exp(-iHt)` через eigendecomposition) | `tests/test_state.py` |
| Каналы шума Крауса (amplitude + phase damping), проверка CPTP | `tests/test_state.py` |
| Чекпоинты состояния и восстановление после декогеренции | `tests/test_state.py` |
| Модель логических ошибок surface-кода `p_L ≈ A·(p/p_th)^((d+1)/2)` | `tests/test_state.py` |
| Импульсный IR: overlap-детекция, DRAG, sample-accurate waveform | `tests/test_pulses.py` |
| Qiskit: схема → counts, сверка с эталоном | `tests/test_adapters.py` |
| Silq-подмножество: парсер, `pi/4`, `measure`, `forget` | `tests/test_adapters.py` |
| PennyLane: вариационная подготовка состояния (fidelity > 0.99) | `tests/test_adapters.py` |
| Labber: JSON-RPC мост, mock-сервер, pulse.execute, mixer.tune | `tests/test_adapters.py` |
| QCoDeS: параметры, снапшоты, импульсные свипы | `tests/test_adapters.py` |
| Планировщик: ранжирование по фиделити × окну когерентности | `tests/test_runtime.py` |
| REST/WebSocket шлюз (13 маршрутов) | `tests/test_api.py` |
| ROS 2-мост (msg/srv-контракты) | `tests/test_ros_bridge.py` |

```
$ python -m pytest -q
86 passed
```

## 6. Честные ограничения

Это важно для планирования — здесь нет выдаваемой желаемого за действительное.

1. **Silq-адаптер — это front-end подмножества**, а не официальный компилятор
   Silq из ETH Zurich. Поддержаны `H/X/Y/Z/S/T`, `RX/RY/RZ`, `CNOT/CX/CZ/SWAP`,
   `measure`, `forget`, константы `pi`. Полный Silq (`qfree`, типы, `if`/`for`
   над квантовыми данными) не реализован.
2. **Labber — проприетарный продукт.** Адаптер говорит на задокументированном
   JSON-RPC мосте ([PROTOCOLS.md](docs/PROTOCOLS.md)); на стороне Labber нужен
   небольшой скрипт-мост. По умолчанию поднимается встроенный mock-сервер,
   чтобы весь контур был исполняем без станции.
3. **Qiskit Pulse удалён в Qiskit 1.x.** Поэтому Qiskit-адаптер больше не
   заявляет `pulse_sequence`: он отдаёт расписание (`status="delegated"`),
   которое исполняет Labber/QCoDeS.
4. **«Коррекция ошибок» — это модель, а не реальный surface-код.** Мы считаем
   оценку логической ошибки и точку восстановления по чекпоинту; декодер
   (MWPM) и синдромы не реализованы. В исходном ответе Duck.ai фиделити
   умножалась на выдуманный коэффициент 1.15 — здесь этого нет.
5. **ROS 2-пакет не собирался в этом окружении** (нет `rclpy`/`colcon`).
   Логика моста вынесена в чистый Python и покрыта тестами; узел `rclpy`
   запускается в обычном ROS 2 workspace (`colcon build`).
6. **Числа производительности — оценки**, не бенчмарки железа.

## 7. Дальше

См. [ROADMAP.md](docs/ROADMAP.md): привязка к фазам 1–4 (симуляция → HAL → интеграция
→ автономность) и конкретные следующие шаги: реальный Labber-мост, декодер
surface-кода, планировщик роя, железные бэкенды.

## 8. Публикация в GitHub

Проект кладётся в репозиторий одной командой через GitHub-коннектор
(токены остаются на стороне сервиса, в песочницу не попадают):

```bash
# посмотреть, что уедет
python scripts/upload_to_github.py --repo <owner>/<repo> --dry-run

# залить
python scripts/upload_to_github.py --repo <owner>/<repo> --branch main \
    --message "feat: QNRS 0.2.0 — слой API для квантовых инструментов"
```

Скрипт бьёт файлы на батчи по 25 и делает несколько обозримых коммитов через
`github.files.commit`. Требования: репозиторий уже создан, а у GitHub-коннектора
есть доступ к нему (GitHub App installation с выбранными репозиториями).
Проверка доступа:

```bash
connector-cli provider invoke github.repository.get --input '{"repo":"<owner>/<repo>"}'
```

Если доступа к репозиториям нет, остаётся локальный путь: скачать
`qnrs-0.2.0.bundle` и выполнить `scripts/push_from_local.sh <bundle>` на своей
машине — это `git clone` из бандла + `git push` в целевой репозиторий.

---

Лицензия: MIT (см. [LICENSE](LICENSE)).
