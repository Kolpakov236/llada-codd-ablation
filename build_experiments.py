"""Сборщик экспериментов 2-5: из exp_core_template.py делает exp{N}_core.py и блокноты-пульты exp{N}_<имя>.ipynb.
Ничего не запускает: только собирает файлы и статически проверяет (ast.parse, соответствие встроенного ядра, запрет магий и длинных тире).
Запуск:  python3 -I build_experiments.py [папка_проекта]
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = sys.argv[1] if len(sys.argv) > 1 else HERE

EXPS = {
    2: {"EXP_NAME": "exp2_hmm_order_only", "WORDS_FROM": "base", "ORDER_FROM": "pc", "THEME_MODE": "joint",
        "TITLE": "Эксперимент 2: HMM меняет только ПОРЯДОК раскрытия",
        "WHAT": "слова-кандидаты берутся из логитов нейросети ДО HMM (`logits_base`), уверенность (значит, и порядок раскрытия) считается по логитам ПОСЛЕ HMM (`logits_pc`)."},
    3: {"EXP_NAME": "exp3_hmm_words_only", "WORDS_FROM": "pc", "ORDER_FROM": "base", "THEME_MODE": "joint",
        "TITLE": "Эксперимент 3: HMM меняет только СЛОВА",
        "WHAT": "слова-кандидаты берутся из логитов ПОСЛЕ HMM (`logits_pc`), уверенность и порядок раскрытия считаются по логитам ДО HMM (`logits_base`)."},
    4: {"EXP_NAME": "exp4_hmm_independent", "WORDS_FROM": "pc", "ORDER_FROM": "pc", "THEME_MODE": "independent",
        "TITLE": "Эксперимент 4: HMM есть, но тема в каждой клетке выбирается НЕЗАВИСИМО",
        "WHAT": "как в полном CoDD (слова и уверенность по логитам после HMM), но скрытая тема (состояние HMM) в каждой клетке блока выбирается независимо из её маргинала p(z_i | открытые слова, оценки нейросети), без связей между клетками. **Исследовательский эксперимент**: реализация независимого выбора проверяется режимом `pctest` (шаг C); без флага `E4_CHECK_PASSED` полный прогон не стартует."},
    5: {"EXP_NAME": "exp5_codd_full", "WORDS_FROM": "pc", "ORDER_FROM": "pc", "THEME_MODE": "joint",
        "TITLE": "Эксперимент 5: CoDD целиком",
        "WHAT": "слова и уверенность по логитам после HMM, одна общая тема на блок на шаг (оригинальный `apply_pc_logits`)."},
}

# ---------------------------------------------------------------------------------------------------------------------
# Ячейки блокнота (шаблон; маркеры @@...@@ подставляются сборщиком). Только стандартная библиотека в самих ячейках.
# ---------------------------------------------------------------------------------------------------------------------
CELLS = []


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src))


md(r'''# @@TITLE@@ - блокнот-пульт

Проект "Токенные зависимости в диффузионной генерации" (абляции метода CoDD, arXiv 2603.00045). Файл: `@@NB_NAME@@`.
Суть: @@WHAT@@

Общая механика (для экспериментов 2-5): за один шаг делается ОДИН forward LLaDA; получаются `logits_base` (до HMM) и `logits_pc` (после `apply_pc_logits`; если HMM на этом шаге не активна, они совпадают). Слова: `x0 = argmax(A)`; уверенность по формулам оригинала (`low_confidence`: softmax(C)[x0]; `margin`: top1 - top2) по логитам C; далее `top-k` как в оригинале. Здесь **A = @@A_NAME@@, C = @@C_NAME@@**, тема: **@@THEME_MODE@@**.

**Схема "пульт + процесс".** Ядро блокнота (DataSphere `mlkernel`) - старое окружение, поэтому здесь **только стандартная библиотека**. Весь тяжёлый код лежит в `@@CORE_NAME@@`, который блокнот записывает на диск и запускает **отдельным процессом** из `venv` (Python 3.10, torch 2.9.1+cu126, transformers 4.57.3, pyjuice 2.4.1, lm_eval из CoDD). Процесс живёт независимо от ядра: после перезапуска ядра ячейка мониторинга снова к нему подключится.

> ## ЗАМЕТНЫЕ ПОМЕТКИ (прочитать до запуска)
> 1. **Заглушка pyjuice ЗАПРЕЩЕНА**: при ошибке импорта pyjuice процесс останавливается.
> 2. `HMM_CKPT = il18/llada-gsm-pc` (по решению автора проекта); веса HMM грузятся ЛОКАЛЬНО из `offline_bundle_hmm/llada-gsm-pc/model.jpc` (`juice.load` + `juice.compile`), не через Hub.
> 3. `GEN_LENGTH = 256` (общая константа; в статье 512). Менять нельзя: хэш общих параметров зашит в ядро и должен совпасть с exp1.
> 4. Блокнот и ядро **не запускались** при подготовке: проверены только синтаксис (`ast.parse`), структура (`nbformat.validate`) и чистые функции на синтетике. Всё непроверенное помечено **НЕ ПРОВЕРЕНО**.
> 5. **Побитовая воспроизводимость на GPU не гарантируется** (недетерминизм ядер Triton/PyJuice). Перед каждой задачей сидируются `random`, `numpy`, `torch`, `cuda` значением из (seed, task_id) - это нужно, потому что случайность выбора темы в PyJuice берётся из Python `random`.
> 6. **Остановка машины**: блокнот не может остановить ВМ. По окончании остановите JupyterLab и ВМ в интерфейсе DataSphere (иначе тарификация GPU продолжится).
''')

code(r'''# ===================== 0. КОНФИГ (первая ячейка с кодом; только стандартная библиотека) =====================
import os
import sys
import json
import time
import signal
import hashlib
import datetime
import subprocess
from pathlib import Path

# ---------- ОБЩИЕ ПАРАМЕТРЫ (одинаковы у всех пяти блокнотов, менять нельзя; хэш сверяется в ядре) ----------
MODEL = "GSAI-ML/LLaDA-8B-Instruct"
HMM_CKPT = "il18/llada-gsm-pc"
DATASET = "gsm8k"
DATASET_SPLIT = "test"
N_TASKS = 100                    # первые 100 задач test-выборки в порядке датасета
GEN_LENGTH = 256
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
NUM_STEPS_LIST = [64, 128]
REMASKING_LIST = ["low_confidence", "margin"]
PC_TEMPERATURE = 0.2
PC_FRAC = 0.5
REVERSE_FRAC = False
SEEDS = [0, 1, 2]

# ---------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ----------
EXP_NAME = "@@EXP_NAME@@"
WORDS_FROM = "@@WORDS_FROM@@"    # откуда берутся слова-кандидаты (A): "base" = логиты до HMM, "pc" = после
ORDER_FROM = "@@ORDER_FROM@@"    # откуда берётся уверенность/порядок (C)
THEME_MODE = "@@THEME_MODE@@"    # "joint" = оригинальный CoDD (одна тема на блок); "independent" = тема в каждой клетке независима (Э4)
E4_IMPL = "auto"                 # только Э4: "auto" (по pctest) | "flows" (маргиналы из потоков) | "pool" (пул независимых цепочек)
E4B_WORD_MARGINAL = False        # только Э4b (ВЫКЛ. по умолчанию): слово из смеси по апостериору темы клетки; трактовка автора НЕ УТОЧНЕНА
E4_N_SAMPLES = 200               # только Э4: число сэмплов тем в статистической проверке (pctest, шаг C)
RUN_SEEDS = [0]                  # какие из SEEDS реально считать (вне хэшей); полный набор [0, 1, 2]
SYSTEM_INSTRUCTION = "You are a helpful assistant."
NUM_FEWSHOT = 0
TASK_NAME = "gsm8k"
MASK_ID = 126336
DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "eager"
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 12345

# ---------- СРЕДА DATASPHERE ----------
PROJECT_DIR_OVERRIDE = None
DATASPHERE_PROJECT_DIR = "/home/jupyter/project"


def detect_project_dir():
    if PROJECT_DIR_OVERRIDE:
        return PROJECT_DIR_OVERRIDE, "PROJECT_DIR_OVERRIDE в конфиге"
    env_value = os.environ.get("PROJECT_DIR")
    if env_value:
        return env_value, "переменная окружения PROJECT_DIR"
    if os.path.isdir(DATASPHERE_PROJECT_DIR):
        return DATASPHERE_PROJECT_DIR, "стандартный путь проекта DataSphere (папка существует)"
    return os.getcwd(), "текущая папка (запасной вариант)"


PROJECT_DIR, PROJECT_DIR_SOURCE = detect_project_dir()
CODD_REPO_DIR = os.path.join(PROJECT_DIR, "CoDD")
BUNDLE_DIR = os.path.join(PROJECT_DIR, "offline_bundle")
MODEL_PATH = os.path.join(BUNDLE_DIR, "weights", "LLaDA-8B-Instruct")
DATA_DIR = os.path.join(BUNDLE_DIR, "data", "gsm8k")
HMM_BUNDLE_DIR = os.path.join(PROJECT_DIR, "offline_bundle_hmm")
HMM_DIR = os.path.join(HMM_BUNDLE_DIR, "llada-gsm-pc")                              # model.jpc
HMM_MANIFEST = os.path.join(HMM_BUNDLE_DIR, "manifest_hmm.json")
VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
WORK_DIR = os.path.join(PROJECT_DIR, "@@RUN_DIR@@")                                # @@CORE_NAME@@, config.json, pid-файл, logs/
OUT_DIR = os.path.join(PROJECT_DIR, "results", EXP_NAME)
EXP1_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp1_llada_default")      # для парного сравнения в сводке
EXP5_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp5_codd_full")          # для сравнения Э4 с Э5
ALLOW_MODEL_CODE_MISMATCH = False  # True: продолжить, даже если modeling_llada.py не совпал по sha256 (configuration_llada.py - только предупреждение)
STRICT_VERSIONS = False
MIN_GPU_TOTAL_GB = 20
MIN_GPU_FREE_GB = 20
RUN_LIMIT = None                 # None = все N_TASKS; число = только первые RUN_LIMIT задач в ПОЛНОМ прогоне
MAX_RUNTIME_MIN = None           # None = без ограничения; число минут: процесс остановится после ближайшей завершённой задачи
REQUIRE_SMOKE = True             # полный прогон требует успешного smoke.json
REQUIRE_PCTEST = True            # smoke и полный прогон требуют успешного pctest.json (проверка HMM и Triton ДО загрузки 8B)
MONITOR_INTERVAL_S = 30
MONITOR_TAIL_LINES = 12

# ---------- ПРОВЕРКИ ПАРАМЕТРОВ ----------
assert MODEL == "GSAI-ML/LLaDA-8B-Instruct" and HMM_CKPT == "il18/llada-gsm-pc", "MODEL/HMM_CKPT - общие параметры"
assert (DATASET, DATASET_SPLIT, N_TASKS) == ("gsm8k", "test", 100), "DATASET/DATASET_SPLIT/N_TASKS - общие параметры"
assert GEN_LENGTH == 256 and BLOCK_LENGTH == 32 and TEMPERATURE == 0.0, "GEN_LENGTH/BLOCK_LENGTH/TEMPERATURE - общие параметры"
assert NUM_STEPS_LIST == [64, 128] and REMASKING_LIST == ["low_confidence", "margin"], "NUM_STEPS_LIST/REMASKING_LIST - общие параметры"
assert (PC_TEMPERATURE, PC_FRAC, REVERSE_FRAC) == (0.2, 0.5, False), "PC_* - общие параметры"
assert SEEDS == [0, 1, 2], "SEEDS - общий параметр"
assert DTYPE == "bfloat16", "DTYPE: для сопоставимости с exp1 только bfloat16"
assert isinstance(RUN_SEEDS, list) and RUN_SEEDS and all(s in SEEDS for s in RUN_SEEDS), "RUN_SEEDS: непустой список из SEEDS"
assert WORDS_FROM in ("base", "pc") and ORDER_FROM in ("base", "pc") and THEME_MODE in ("joint", "independent")
assert E4_IMPL in ("auto", "flows", "pool")
assert RUN_LIMIT is None or (isinstance(RUN_LIMIT, int) and 1 <= RUN_LIMIT <= N_TASKS), "RUN_LIMIT: None или целое от 1 до N_TASKS"
assert MAX_RUNTIME_MIN is None or (isinstance(MAX_RUNTIME_MIN, (int, float)) and MAX_RUNTIME_MIN > 0), "MAX_RUNTIME_MIN: None или число минут > 0"
NUM_BLOCKS = GEN_LENGTH // BLOCK_LENGTH
for _ns in NUM_STEPS_LIST:
    assert _ns % NUM_BLOCKS == 0
print("GEN_LENGTH=%d: блоков %d; токенов за шаг: %s" % (GEN_LENGTH, NUM_BLOCKS, ", ".join("%d шагов -> %g" % (ns, GEN_LENGTH / float(ns)) for ns in NUM_STEPS_LIST)))


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


COMMON_PARAMS = {
    "MODEL": MODEL, "HMM_CKPT": HMM_CKPT, "DATASET": DATASET, "DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS,
    "GEN_LENGTH": GEN_LENGTH, "BLOCK_LENGTH": BLOCK_LENGTH, "TEMPERATURE": TEMPERATURE,
    "NUM_STEPS_LIST": NUM_STEPS_LIST, "REMASKING_LIST": REMASKING_LIST,
    "PC_TEMPERATURE": PC_TEMPERATURE, "PC_FRAC": PC_FRAC, "REVERSE_FRAC": REVERSE_FRAC, "SEEDS": SEEDS,
}
print("sha256 общих параметров:", sha256_json(COMMON_PARAMS))
print("ожидается (как у exp1):  19272a3da75f0b6f992793038ebe987c380dfbac1848739de5b19ee20d2f7db4")
assert sha256_json(COMMON_PARAMS) == "19272a3da75f0b6f992793038ebe987c380dfbac1848739de5b19ee20d2f7db4", "хэш общих параметров не совпал с exp1: остановка"

# Всё, что нужно процессу (читается из config.json). Ключи ОБЯЗАТЕЛЬНЫ: ядро проверяет их наличие.
CORE_CONFIG = {
    "MODEL": MODEL, "HMM_CKPT": HMM_CKPT, "DATASET": DATASET, "DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS,
    "GEN_LENGTH": GEN_LENGTH, "BLOCK_LENGTH": BLOCK_LENGTH, "TEMPERATURE": TEMPERATURE, "NUM_STEPS_LIST": NUM_STEPS_LIST,
    "REMASKING_LIST": REMASKING_LIST, "PC_TEMPERATURE": PC_TEMPERATURE, "PC_FRAC": PC_FRAC, "REVERSE_FRAC": REVERSE_FRAC, "SEEDS": SEEDS,
    "EXP_NAME": EXP_NAME, "WORDS_FROM": WORDS_FROM, "ORDER_FROM": ORDER_FROM, "THEME_MODE": THEME_MODE, "E4_IMPL": E4_IMPL,
    "E4B_WORD_MARGINAL": E4B_WORD_MARGINAL, "SYSTEM_INSTRUCTION": SYSTEM_INSTRUCTION, "NUM_FEWSHOT": NUM_FEWSHOT, "TASK_NAME": TASK_NAME,
    "MASK_ID": MASK_ID, "DTYPE": DTYPE, "ATTN_IMPLEMENTATION": ATTN_IMPLEMENTATION, "N_BOOTSTRAP": N_BOOTSTRAP, "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    "RUN_SEEDS": RUN_SEEDS, "RUN_LIMIT": RUN_LIMIT, "MAX_RUNTIME_MIN": MAX_RUNTIME_MIN, "PROJECT_DIR": PROJECT_DIR, "CODD_REPO_DIR": CODD_REPO_DIR,
    "MODEL_PATH": MODEL_PATH, "DATA_DIR": DATA_DIR, "OUT_DIR": OUT_DIR, "HMM_DIR": HMM_DIR, "HMM_MANIFEST": HMM_MANIFEST,
    "EXP1_RESULTS_DIR": EXP1_RESULTS_DIR, "EXP5_RESULTS_DIR": EXP5_RESULTS_DIR, "ALLOW_MODEL_CODE_MISMATCH": ALLOW_MODEL_CODE_MISMATCH,
    "STRICT_VERSIONS": STRICT_VERSIONS, "MIN_GPU_TOTAL_GB": MIN_GPU_TOTAL_GB, "MIN_GPU_FREE_GB": MIN_GPU_FREE_GB,
    "REQUIRE_SMOKE": REQUIRE_SMOKE, "REQUIRE_PCTEST": REQUIRE_PCTEST, "E4_N_SAMPLES": E4_N_SAMPLES,
}
print("PROJECT_DIR = %s   <- %s" % (PROJECT_DIR, PROJECT_DIR_SOURCE))
print("Интерпретатор процесса:", VENV_PYTHON)
print("Веса 8B:", MODEL_PATH)
print("Веса HMM:", HMM_DIR)
print("Данные:", DATA_DIR)
print("Результаты:", OUT_DIR)
print("Вариант: слова из", WORDS_FROM, "| порядок из", ORDER_FROM, "| тема:", THEME_MODE, "| RUN_SEEDS:", RUN_SEEDS, "| RUN_LIMIT:", RUN_LIMIT, "| MAX_RUNTIME_MIN:", MAX_RUNTIME_MIN)
''')

md(r'''## 1. Проверка окружения venv и файлов HMM (стандартная библиотека + отдельный процесс)

Запускается `venv/bin/python -I` с теми же переменными среды, что и у рабочего процесса: интерпретатор, версии, CUDA, compute capability, память GPU, диск, компилятор и `Python.h` (нужны Triton/PyJuice). Затем (в самом блокноте) сверяются размеры файлов HMM с `manifest_hmm.json`. Подробные проверки (sha256, промпты 3 задач, структура HMM) делает режим `preflight` ниже.
''')

code(r'''# ===================== 1. ПРОВЕРКА ОКРУЖЕНИЯ VENV И ФАЙЛОВ HMM =====================
PROCESS_ENV = ["MPLBACKEND=Agg", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1"]

assert os.path.isfile(VENV_PYTHON), "Нет интерпретатора venv: %s (проверьте PROJECT_DIR и каталог venv)" % VENV_PYTHON
for _label, _path in (("CoDD", CODD_REPO_DIR), ("веса 8B", MODEL_PATH), ("данные GSM8K", DATA_DIR), ("папка HMM", HMM_DIR)):
    print("%-14s %-8s %s" % (_label, "есть" if os.path.isdir(_path) else "НЕТ!", _path))
for _split in ("test", "train"):
    _f = os.path.join(DATA_DIR, "main", "%s-00000-of-00001.parquet" % _split)
    print("%-14s %-8s %s" % ("parquet " + _split, "есть" if os.path.isfile(_f) else "НЕТ!", _f))

# ---- размеры файлов HMM по manifest_hmm.json (пути в manifest отсчитываются от PROJECT_DIR) ----
if os.path.isfile(HMM_MANIFEST):
    _man = json.loads(Path(HMM_MANIFEST).read_text(encoding="utf-8"))
    _bad = []
    for _e in _man["files"]:
        _p = os.path.join(PROJECT_DIR, _e["path"])
        _size = os.path.getsize(_p) if os.path.isfile(_p) else None
        _ok = _size == _e["size"]
        print("HMM %-48s %12s байт  %s" % (_e["path"], _size, "OK" if _ok else "РАСХОЖДЕНИЕ (ожидалось %d)" % _e["size"]))
        if not _ok:
            _bad.append(_e["path"])
    assert not _bad, "Файлы HMM не совпали по размеру: %s (загрузите заново по README_HMM.md)" % _bad
else:
    print("ВНИМАНИЕ: нет", HMM_MANIFEST, "- загрузите offline_bundle_hmm по README_HMM.md")

ENV_CHECK_SNIPPET = r"""
import sys, os, shutil, sysconfig, importlib.metadata as md
print("sys.executable:", sys.executable)
print("python:", sys.version.split()[0])
def ver(*names):
    for n in names:
        try:
            return md.version(n)
        except Exception:
            pass
    return "нет"
for label, names in (("torch", ("torch",)), ("transformers", ("transformers",)), ("pyjuice", ("pyjuice",)),
                     ("lm_eval", ("lm_eval", "lm-eval")), ("huggingface_hub", ("huggingface-hub",)), ("numpy", ("numpy",)),
                     ("accelerate", ("accelerate",)), ("datasets", ("datasets",)), ("triton", ("triton",))):
    print("  %-16s %s" % (label, ver(*names)))
try:
    import torch
    print("CUDA доступна:", torch.cuda.is_available(), "| сборка CUDA в torch:", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print("GPU:", p.name, "| compute capability %d.%d" % (p.major, p.minor))
        print("видеопамять: всего %.2f ГиБ, свободно %.2f ГиБ" % (total / 2.0 ** 30, free / 2.0 ** 30))
        print("bf16 аппаратно поддерживается:", p.major >= 8)
except Exception as e:
    print("torch НЕ импортируется:", type(e).__name__, str(e)[:300])
try:
    import pyjuice
    print("import pyjuice: OK")
except Exception as e:
    print("import pyjuice: ОШИБКА", type(e).__name__, str(e)[:300])
try:
    import lm_eval
    print("import lm_eval: OK ->", os.path.dirname(lm_eval.__file__))
except Exception as e:
    print("import lm_eval: ОШИБКА", type(e).__name__, str(e)[:300])
cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
inc = sysconfig.get_paths().get("include")
print("компилятор C (нужен Triton):", cc or "НЕ НАЙДЕН")
print("Python.h:", "есть" if inc and os.path.isfile(os.path.join(inc, "Python.h")) else "НЕТ (" + str(inc) + ")")
du = shutil.disk_usage(sys.argv[1])
print("диск проекта: свободно %.1f ГиБ из %.1f ГиБ" % (du.free / 2.0 ** 30, du.total / 2.0 ** 30))
"""

_res = subprocess.run(["env"] + PROCESS_ENV + [VENV_PYTHON, "-I", "-c", ENV_CHECK_SNIPPET, PROJECT_DIR],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=600)
print("-" * 80)
print(_res.stdout)
if _res.stderr.strip():
    print("stderr (хвост):")
    print(_res.stderr[-1500:])
print("код выхода:", _res.returncode)
assert _res.returncode == 0, "Проверка окружения завершилась с ошибкой (см. вывод выше)"
''')

md(r'''## 2. Запись `@@CORE_NAME@@` и `config.json`

Ячейка записывает ядро расчёта и конфиг в `@@RUN_DIR@@/` и проверяет синтаксис интерпретатором venv. Её нужно перезапускать после любой правки конфига выше (запуск процесса тоже перезаписывает `config.json`).
''')

code(r'''# ===================== 2. ЗАПИСЬ @@CORE_NAME@@ И config.json =====================
CORE_SOURCE = r"""@@CORE@@"""
CORE_NAME = "@@CORE_NAME@@"


def write_run_files():
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(WORK_DIR, "logs").mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    core_path = Path(WORK_DIR, CORE_NAME)
    core_path.write_text(CORE_SOURCE, encoding="utf-8")
    cfg_path = Path(WORK_DIR, "config.json")
    cfg_path.write_text(json.dumps(CORE_CONFIG, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return core_path, cfg_path


_core_path, _cfg_path = write_run_files()
print("записан:", _core_path, "(%d строк, sha256 %s)" % (CORE_SOURCE.count("\n"), hashlib.sha256(CORE_SOURCE.encode("utf-8")).hexdigest()[:16]))
print("записан:", _cfg_path)
_chk = subprocess.run([VENV_PYTHON, "-I", "-c", "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('синтаксис ядра (Python venv): OK')", str(_core_path)],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print(_chk.stdout.strip() or _chk.stderr.strip())
assert _chk.returncode == 0, "ядро не проходит синтаксическую проверку в venv"
''')

md(r'''## 3. Управление процессом и мониторинг

`start_core(mode)`: каждый запуск **удаляет прежний pid-файл, пишет НОВЫЙ pid и НОВЫЙ лог** `@@RUN_DIR@@/logs/@@SHORT@@_<режим>_<время>.log`, передаёт окружение (`MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`) и запускает `venv/bin/python -I -u @@CORE_NAME@@ --mode <режим>` через `Popen(start_new_session=True)`. `monitor()` читает именно этот pid и этот лог (цикл с `sleep`, ядро остаётся занятым); после перезапуска ядра он находит процесс по pid-файлу. `stop_core()` - SIGTERM группе процессов.
''')

code(r'''# ===================== 3. УПРАВЛЕНИЕ ПРОЦЕССОМ И МОНИТОРИНГ =====================
PID_FILE = Path(WORK_DIR, "@@SHORT@@.pid")
STATE_FILE = Path(WORK_DIR, "@@SHORT@@_state.json")
LOG_DIR = Path(WORK_DIR, "logs")
CORE_PROC = {"popen": None}
_MON_HIST = []
MODES = ("preflight", "pctest", "smoke", "full", "summary")


def _utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pid_alive(pid):
    # Жив, если существует, не зомби и в командной строке есть имя ядра (защита от переиспользования pid).
    popen = CORE_PROC.get("popen")
    if popen is not None and popen.pid == pid:
        return popen.poll() is None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        cmdline = Path("/proc/%d/cmdline" % pid).read_bytes()
        stat = Path("/proc/%d/stat" % pid).read_text()
    except OSError:
        return False
    state = stat.rsplit(")", 1)[-1].split()[0] if ")" in stat else "?"
    return CORE_NAME.encode("utf-8") in cmdline and state != "Z"


def current_process():
    state = read_state()
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except ValueError:
            pid = None
    if pid is not None and pid_alive(pid):
        return pid, state
    return None, state


def start_core(mode):
    assert mode in MODES, mode
    pid, _state = current_process()
    if pid is not None:
        raise RuntimeError("Процесс уже запущен (pid %d). Дождитесь его окончания или вызовите stop_core()." % pid)
    write_run_files()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for _old in (PID_FILE, STATE_FILE):
        try:
            _old.unlink()
        except OSError:
            pass
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / ("@@SHORT@@_%s_%s.log" % (mode, stamp))
    cmd = ["env"] + PROCESS_ENV + [VENV_PYTHON, "-I", "-u", CORE_NAME, "--mode", mode]
    logf = open(str(log_path), "ab")
    try:
        proc = subprocess.Popen(cmd, cwd=WORK_DIR, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        logf.close()
    CORE_PROC["popen"] = proc
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    STATE_FILE.write_text(json.dumps({"pid": proc.pid, "mode": mode, "started_utc": _utc_now(), "log": str(log_path), "cmd": cmd}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("запущен процесс pid=%d, режим %s" % (proc.pid, mode))
    print("команда:", " ".join(cmd))
    print("лог:", log_path)
    return proc.pid


def stop_core(timeout_s=240):
    pid, state = current_process()
    if pid is None:
        print("Живого процесса нет.")
        return False
    print("Останавливаю pid=%d (SIGTERM группе процессов). Режим full дописывает кэш и steplog и останавливается после текущей задачи." % pid)
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as e:
        print("killpg:", e)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not pid_alive(pid):
            print("Процесс остановлен за %.0f c." % (time.time() - t0))
            return True
        time.sleep(3)
    print("Не остановился за %d c: SIGKILL (кэш согласован: незавершённая задача будет пропущена при возобновлении)." % timeout_s)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError as e:
        print("killpg:", e)
    time.sleep(3)
    return not pid_alive(pid)


def tail_file(path, n):
    try:
        data = Path(path).read_bytes()[-20000:]
    except OSError:
        return []
    text = data.decode("utf-8", "replace").replace("\r", "\n")
    return [l for l in text.splitlines() if l.strip()][-n:]


def _read_rows(path):
    rows = []
    try:
        data = Path(path).read_bytes()
    except OSError:
        return rows
    cut = data.rfind(b"\n") + 1
    for raw in data[:cut].split(b"\n"):
        if raw.strip():
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except ValueError:
                pass
    return rows


def progress_snapshot():
    target = RUN_LIMIT if RUN_LIMIT is not None else N_TASKS
    snap = []
    for sd in RUN_SEEDS:
        for rm in REMASKING_LIST:
            for st in NUM_STEPS_LIST:
                rows = _read_rows(Path(OUT_DIR, "%s_steps%d_seed%d.jsonl" % (rm, st, sd)))
                secs = [r["sec"] for r in rows if isinstance(r.get("sec"), (int, float))]
                snap.append({"seed": sd, "remasking": rm, "steps": st, "done": len(rows), "target": target,
                             "correct": sum(int(r.get("correct", 0)) for r in rows),
                             "mean_sec": (sum(secs) / len(secs)) if secs else None})
    return snap


def _fmt_td(seconds):
    return str(datetime.timedelta(seconds=int(max(0, seconds))))


def render_status(extra_header=""):
    pid, state = current_process()
    lines = []
    if extra_header:
        lines.append(extra_header)
    if state:
        lines.append("режим: %s | pid: %s | запущен: %s | %s" % (state.get("mode"), state.get("pid"), state.get("started_utc"), "ЖИВ" if pid is not None else "не работает"))
    else:
        lines.append("процесс ещё не запускался")
    try:
        st = json.loads(Path(OUT_DIR, "status.json").read_text(encoding="utf-8"))
        lines.append("статус процесса: %s %s (обновлён %s)" % (st.get("state"), st.get("message", ""), st.get("updated_utc")))
    except (OSError, ValueError):
        pass
    snap = progress_snapshot()
    total_done = sum(s["done"] for s in snap)
    total_target = sum(s["target"] for s in snap)
    if total_done or (state and state.get("mode") == "full"):
        lines.append("-" * 78)
        lines.append("%-5s %-16s %6s %12s %10s %12s" % ("seed", "remasking", "steps", "готово", "верно", "сек/задачу"))
        for s in snap:
            acc = ("%.0f%%" % (100.0 * s["correct"] / s["done"])) if s["done"] else "-"
            lines.append("%-5d %-16s %6d %12s %10s %12s" % (s["seed"], s["remasking"], s["steps"], "%d/%d" % (s["done"], s["target"]), acc, ("%.1f" % s["mean_sec"]) if s["mean_sec"] else "-"))
        now = time.time()
        _MON_HIST.append((now, total_done))
        while len(_MON_HIST) > 2 and now - _MON_HIST[0][0] > 900:
            _MON_HIST.pop(0)
        speed = None
        if len(_MON_HIST) >= 2 and _MON_HIST[-1][0] > _MON_HIST[0][0]:
            speed = (_MON_HIST[-1][1] - _MON_HIST[0][1]) / (_MON_HIST[-1][0] - _MON_HIST[0][0]) * 60.0
        per_step = [s["mean_sec"] / s["steps"] for s in snap if s["mean_sec"]]
        per_step_avg = (sum(per_step) / len(per_step)) if per_step else None
        remain = 0.0
        ok = True
        for s in snap:
            left = s["target"] - s["done"]
            sec = s["mean_sec"] if s["mean_sec"] else (per_step_avg * s["steps"] if per_step_avg else None)
            if left > 0:
                if sec is None:
                    ok = False
                else:
                    remain += left * sec
        lines.append("-" * 78)
        lines.append("всего %d/%d задач (%.0f%%)%s" % (total_done, total_target, 100.0 * total_done / max(1, total_target),
                                                       (" | скорость за последние минуты: %.2f задач/мин" % speed) if speed is not None else ""))
        if ok and total_done < total_target:
            lines.append("прогноз до конца по среднему времени задачи: ~%s (оценка)" % _fmt_td(remain))
        elif total_done >= total_target:
            lines.append("все задачи готовы")
    log_path = (state or {}).get("log")
    if not log_path:
        logs = sorted(LOG_DIR.glob("@@SHORT@@_*.log"))
        log_path = str(logs[-1]) if logs else None
    if log_path:
        lines.append("-" * 78)
        lines.append("хвост лога (%s):" % log_path)
        lines.extend(tail_file(log_path, MONITOR_TAIL_LINES))
    return "\n".join(lines), pid


def status():
    text, _pid = render_status()
    print(text)


def monitor(interval=None, max_minutes=None):
    # Цикл с sleep: ядро остаётся занятым (ВМ не простаивает). Завершается, когда процесс закончил. Interrupt kernel - выйти, процесс продолжит работу.
    interval = MONITOR_INTERVAL_S if interval is None else interval
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None
    t_start = time.time()
    try:
        while True:
            text, pid = render_status("[%s UTC] мониторинг, обновление каждые %d c (Interrupt kernel - выйти; процесс продолжит работу)" % (_utc_now(), interval))
            if clear_output is not None:
                clear_output(wait=True)
            print(text)
            if pid is None:
                print("\nПроцесс не работает. Мониторинг завершён.")
                break
            if max_minutes is not None and (time.time() - t_start) > max_minutes * 60:
                print("\nДостигнут лимит времени мониторинга; процесс продолжает работу.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nМониторинг остановлен вручную. Процесс продолжает работать (status() / monitor() подключатся снова).")


_pid, _state = current_process()
print("процесс сейчас:", ("pid %d (режим %s)" % (_pid, (_state or {}).get("mode"))) if _pid else "не запущен")
''')

md(r'''## 4. Preflight (`--mode preflight`, без GPU и без загрузки модели)

Быстро (минуты) находит то, что иначе вскрылось бы после загрузки 8B: версии и окружение; хэши общих параметров; sha256 исходников модели (расхождение `configuration_llada.py` - ПРЕДУПРЕЖДЕНИЕ с обоими хэшами и первыми 60 строками diff в лог и `env.json`; расхождение `modeling_llada.py` - остановка); импорт pyjuice (заглушка запрещена); наличие, размер и sha256 `model.jpc`; безопасное чтение структуры HMM (pickle без исполнения кода); чтение parquet и построение промптов 100 задач тем же кодом `lm_eval`, сверка хэша списка id; компилятор и `Python.h`; свободное место. Результат: `preflight.json`.
''')

code(r'''# ===================== 4. PREFLIGHT =====================
start_core("preflight")
monitor()
''')

md(r'''## 5. Проверка HMM на GPU до загрузки 8B (`--mode pctest`)

Грузит ТОЛЬКО HMM (`juice.load` + `juice.compile(...).to(cuda)`), вызывает оригинальный `apply_pc_logits` на случайных логитах реальной формы блока (время первого вызова с компиляцией Triton и повторных, пик памяти), оценивает, помещается ли HMM рядом с 8B с запасом не менее 2 ГиБ (пик exp1 16.2 ГиБ + пик HMM). Без компилятора останавливается с понятным сообщением. @@PCTEST_E4@@Результат: `pctest.json`.
''')

code(r'''# ===================== 5. PCTEST =====================
start_core("pctest")
monitor()
''')

md(r'''## 6. Дымовой тест (`--mode smoke`)

3 задачи, 8 шагов, 64 токена, ОБА правила `remasking`, HMM активна (на 4-м шаге каждого блока). Проверяется: (а) копия генерации при выключенной HMM побитово совпадает с оригиналом `llada_diffusion_generate`; (б) копия при A = C = pc и общей теме совпадает с оригинальным CoDD при том же seed (оригинал запускается дважды: если он сам невоспроизводим, расхождение допускается только ПОСЛЕ первой активации HMM и фиксируется); (в) логиты конечны, журнал шагов согласован; (г) пик видеопамяти при реальном `GEN_LENGTH` с активной HMM, запас не менее 2 ГиБ; (д) секунды на задачу и прогноз для 4 комбинаций x 100 задач x число seeds (отдельно для `RUN_SEEDS`). `smoke.json` пишется только при успехе. Загрузка модели занимает несколько минут.
''')

code(r'''# ===================== 6. SMOKE =====================
start_core("smoke")
monitor()
''')

md(r'''## 7. Полный прогон (`--mode full`)

4 комбинации (`remasking` x `num_steps`) по `N_TASKS` задач для каждого seed из `RUN_SEEDS`. Требуются успешные `pctest.json` и `smoke.json` (при том же конфиге)@@FULL_E4@@. Прогон **идемпотентен**: после обрыва сессии просто запустите ячейку снова (после перезапуска ядра сначала ячейки 0-3); готовые задачи пропускаются. Перед каждой задачей процесс сидирует `random`, `numpy`, `torch`, `cuda` значением из (seed, task_id). По завершении процесс сам считает сводку (`summary.json`).
''')

code(r'''# ===================== 7. ПОЛНЫЙ ПРОГОН =====================
start_core("full")
monitor()
''')

md(r'''### Подключиться к уже идущему процессу / состояние / остановить

- `monitor()` - подключиться к процессу после перезапуска ядра;
- `status()` - разовый снимок;
- `stop_core()` - остановить (для `full` корректно: после текущей задачи). Затем можно продолжить, запустив ячейку полного прогона снова.
''')

code(r'''# ===================== 7a. ПОДКЛЮЧЕНИЕ К ПРОЦЕССУ (после перезапуска ядра) =====================
_pid, _state = current_process()
if _pid:
    print("найден живой процесс pid=%d (режим %s) по pid-файлу %s" % (_pid, (_state or {}).get("mode"), PID_FILE))
    monitor()
else:
    status()
''')

code(r'''# ===================== 7b. ОСТАНОВКА ПРОЦЕССА (раскомментируйте вызов) =====================
# stop_core()
status()
''')

md(r'''## 8. Сводка (`--mode summary`)

Таблицы по каждому seed и усреднённо по seeds (доля верных под `flexible-extract`, 95% percentile bootstrap по задачам), среднее число активных шагов HMM, средние `delta_words` и `delta_order` (диагностика на шагах с активной HMM). **Парное сравнение с exp1 по `task_id`**: разность точностей с парным bootstrap и точный (двусторонний биномиальный) тест Макнемара@@SUMMARY_E4@@. Модель не загружается. Доверительный интервал отражает выбор 100 задач; случайность темы HMM даёт разброс между seeds.
''')

code(r'''# ===================== 8. СВОДКА (процесс из venv) =====================
_pid, _state = current_process()
assert _pid is None, "Рабочий процесс ещё идёт (pid %d): дождитесь окончания или stop_core()" % _pid
write_run_files()
_res = subprocess.run(["env"] + PROCESS_ENV + [VENV_PYTHON, "-I", "-u", CORE_NAME, "--mode", "summary"], cwd=WORK_DIR,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
print(_res.stdout[-8000:])
assert _res.returncode == 0, "сводка завершилась с ошибкой (см. вывод выше)"
''')

code(r'''# ===================== 9. ФАКТЫ ДЛЯ ОТЧЁТА И НАПОМИНАНИЕ =====================
def _load(name):
    try:
        return json.loads(Path(OUT_DIR, name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_env, _tasks, _smoke, _pct = _load("env.json"), _load("tasks.json"), _load("smoke.json"), _load("pctest.json")
print("Вариант:", EXP_NAME, "| слова из", WORDS_FROM, "| порядок из", ORDER_FROM, "| тема:", THEME_MODE)
print("pyjuice:", (_env or {}).get("pyjuice_mode"), "| HMM загрузка:", (_env or {}).get("hmm_load"))
print("GEN_LENGTH:", GEN_LENGTH, "| DTYPE:", DTYPE, "| GPU:", ((_env or {}).get("gpu") or {}).get("name"))
print("Сверка кода модели (sha256):", dict((k, v.get("verdict")) for k, v in ((_env or {}).get("model_code_check") or {}).items()))
print("sha256 списка id задач:", (_tasks or {}).get("ids_sha256"))
print("sha256 общих параметров:", sha256_json(COMMON_PARAMS))
print("pctest:", ("ok=%s %s" % (_pct.get("ok"), _pct.get("finished_utc"))) if _pct else "не проводился")
print("дымовой тест:", ("пройден " + str(_smoke.get("finished_utc"))) if _smoke else "не проводился")
print("Результаты:", OUT_DIR)
if os.path.isdir(OUT_DIR):
    for _name in sorted(os.listdir(OUT_DIR)):
        print("   ", _name)
print()
print("!" * 100)
print("НАПОМИНАНИЕ: этот блокнот НЕ может остановить машину. После окончания работы вручную остановите JupyterLab и ВМ")
print("в интерфейсе Yandex DataSphere, иначе продолжится тарификация GPU. Точные пункты меню НЕ ПРОВЕРЕНЫ.")
print("!" * 100)
''')

md(r'''## Как запускать

1. Загрузите на сервер `offline_bundle_hmm` (см. `README_HMM.md`) и убедитесь, что есть `CoDD/`, `venv/`, `offline_bundle/` (как для exp1).
2. Выполните ячейки 0-3 подряд (конфиг, окружение и размеры HMM, запись ядра, функции управления). Конфиг правьте только в ячейке 0.
3. Ячейка 4 - `preflight` (минуты, без GPU): должна закончиться "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ".
4. Ячейка 5 - `pctest` (GPU, до загрузки 8B): проверяет HMM, Triton и память@@HOW_E4@@.
5. Ячейка 6 - `smoke` (загрузка 8B несколько минут): пишет `smoke.json` и прогноз времени. Для оценки времени можно задать `RUN_LIMIT = 5`.
6. Ячейка 7 - `full`: по умолчанию `RUN_SEEDS = [0]`; для всех seeds поставьте `[0, 1, 2]` в ячейке 0 и повторите ячейки 2-3.
7. Сорвалась сессия или перезапустилось ядро: выполните ячейки 0-3, затем 7a (`monitor()`); процесс продолжает работать сам.
8. Остановить: `stop_core()` (ячейка 7b). Продолжить: ячейка 7 снова, готовые задачи пропускаются.
9. Ячейка 8 - сводка и парное сравнение с exp1 (`results/exp1_llada_default` должен лежать на сервере).
10. По окончании остановите JupyterLab и ВМ в интерфейсе DataSphere: блокнот сделать это не может.
''')


# ---------------------------------------------------------------------------------------------------------------------
def build_one(num, spec, core_template):
    exp_name = spec["EXP_NAME"]
    nb_name = exp_name + ".ipynb"
    core_name = "exp%d_core.py" % num
    experiment = {k: spec[k] for k in ("EXP_NAME", "WORDS_FROM", "ORDER_FROM", "THEME_MODE")}
    marker = "EXPERIMENT = {}  # @@EXPERIMENT@@"
    assert core_template.count(marker) == 1
    core = core_template.replace(marker, "EXPERIMENT = " + json.dumps(experiment, ensure_ascii=False) + "  # подставлено сборщиком build_experiments.py")
    problems = []
    if '"""' in core:
        problems.append('в ядре есть """')
    if not core.endswith("\n") or core.rstrip("\n").endswith("\\"):
        problems.append("ядро заканчивается слэшем или без перевода строки")
    ast.parse(core, filename=core_name)
    if problems:
        raise SystemExit("; ".join(problems))
    with open(os.path.join(PROJECT, core_name), "w", encoding="utf-8") as f:
        f.write(core)

    is_e4 = spec["THEME_MODE"] == "independent"
    subs = {
        "@@TITLE@@": spec["TITLE"], "@@WHAT@@": spec["WHAT"], "@@NB_NAME@@": nb_name, "@@CORE_NAME@@": core_name, "@@RUN_DIR@@": "exp%d_run" % num,
        "@@EXP_NAME@@": exp_name, "@@WORDS_FROM@@": spec["WORDS_FROM"], "@@ORDER_FROM@@": spec["ORDER_FROM"], "@@THEME_MODE@@": spec["THEME_MODE"],
        "@@A_NAME@@": "logits_base (до HMM)" if spec["WORDS_FROM"] == "base" else "logits_pc (после HMM)",
        "@@C_NAME@@": "logits_base (до HMM)" if spec["ORDER_FROM"] == "base" else "logits_pc (после HMM)",
        "@@SHORT@@": "exp%d" % num,
        "@@PCTEST_E4@@": ("**Для Э4 дополнительно - шаг C**: 200 сэмплов тем в режимах `joint` и `independent` для 4 реальных состояний блока (токены эталонных решений GSM8K, ~44% масок), частоты тем по позициям против маргиналов и взаимная информация между позициями с перестановочным нулём; при успехе пишется флаг `E4_CHECK_PASSED.json`, без него полный прогон не стартует. Мягкие оценки нейросети в этом тесте СИНТЕТИЧЕСКИЕ (модель 8B ещё не загружена). ") if is_e4 else "",
        "@@FULL_E4@@": " и флаг `E4_CHECK_PASSED.json` (проверка независимого выбора темы)" if is_e4 else "",
        "@@SUMMARY_E4@@": "; для этого эксперимента дополнительно сравнение с Э5 (`results/exp5_codd_full`)" if is_e4 else "",
        "@@HOW_E4@@": " и корректность независимого выбора темы (шаг C; без него Э4 не запустится)" if is_e4 else "",
    }
    cells = []
    for idx, (kind, src) in enumerate(CELLS):
        src = src.replace("@@CORE@@", core)
        for k in sorted(subs, key=len, reverse=True):
            src = src.replace(k, subs[k])
        assert "@@" not in src.replace(core, ""), "неподставленный маркер в ячейке %d" % idx
        src = src.strip("\n") + "\n"
        lines = src.splitlines(True)
        lines[-1] = lines[-1].rstrip("\n")
        cell_id = "c%03d" % idx
        if kind == "markdown":
            for ch in ("—", "–"):
                if ch in src:
                    problems.append("ячейка %d: длинное тире" % idx)
            cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines})
            continue
        cells.append({"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {}, "outputs": [], "source": lines})
        try:
            tree = ast.parse(src, filename="cell_%d" % idx, feature_version=(3, 8))
        except SyntaxError as e:
            problems.append("ячейка %d: SyntaxError: %s" % (idx, e))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CORE_SOURCE" for t in node.targets):
                if ast.literal_eval(node.value) != core:
                    problems.append("ячейка %d: встроенный CORE_SOURCE не совпал с %s" % (idx, core_name))
                else:
                    print("  CORE_SOURCE в ячейке %d побайтно равен %s (%d строк)" % (idx, core_name, core.count("\n")))
        if "%pip" in src or src.lstrip().startswith("!"):
            problems.append("ячейка %d: магия IPython" % idx)
    if problems:
        print("ПРОБЛЕМЫ:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(PROJECT, nb_name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print("OK Э%d: %s (%d ячеек, из них кода %d) + %s" % (num, nb_name, len(cells), n_code, core_name))


def main():
    with open(os.path.join(HERE, "exp_core_template.py"), encoding="utf-8") as f:
        template = f.read()
    for num, spec in sorted(EXPS.items()):
        build_one(num, spec, template)


if __name__ == "__main__":
    main()
