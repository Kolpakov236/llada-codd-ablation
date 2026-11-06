# exp1_core.py - тяжёлая часть эксперимента 1 (обычная LLaDA-8B-Instruct, HMM не используется).
#
# Запускается ОТДЕЛЬНЫМ процессом из окружения venv (не из ядра блокнота):
#   env MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#       /home/jupyter/project/venv/bin/python -I exp1_core.py --mode smoke|full|summary
# Параметры читаются из config.json рядом со скриптом (его пишет блокнот-пульт). Интернета нет.
#
# Режимы:
#   smoke   проверка окружения, загрузка модели, сверка кода, задачи, дымовой тест (копия == оригинал), пик памяти, прогноз времени
#   full    полный прогон (требует успешного smoke), кэш после каждой задачи, возобновление, RUN_LIMIT, MAX_RUNTIME_MIN
#   summary таблица точности с bootstrap-интервалами (модель не загружается)
#
# Совместимость: Python 3.10 (без вложенных одинаковых кавычек в f-строках, без match).
# НЕ ПРОВЕРЕНО: скрипт ни разу не запускался (проверен только синтаксис через ast.parse).

import argparse
import copy
import datetime
import glob
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(msg=""):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print("[" + stamp + "] " + str(msg), flush=True)


def dist_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))


# ===================== 0. КОНФИГ (из config.json) И ПРОВЕРКА ПАРАМЕТРОВ =====================
def _parse_args():
    ap = argparse.ArgumentParser(description="exp1_core: тяжёлая часть эксперимента 1 (обычная LLaDA, HMM не используется)")
    ap.add_argument("--mode", required=True, choices=["smoke", "full", "summary"])
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    return ap.parse_args()


ARGS = _parse_args()
MODE = ARGS.mode
START_UTC = utc_now()
with open(ARGS.config, encoding="utf-8") as _f:
    CFG = json.load(_f)

_REQUIRED_KEYS = [
    "MODEL", "HMM_CKPT", "DATASET", "DATASET_SPLIT", "N_TASKS", "GEN_LENGTH", "BLOCK_LENGTH", "TEMPERATURE", "NUM_STEPS_LIST",
    "REMASKING_LIST", "PC_TEMPERATURE", "PC_FRAC", "REVERSE_FRAC", "SEEDS", "EXP_NAME", "USE_PC", "SEED_USED", "SYSTEM_INSTRUCTION",
    "NUM_FEWSHOT", "TASK_NAME", "MASK_ID", "DTYPE", "ATTN_IMPLEMENTATION", "N_BOOTSTRAP", "BOOTSTRAP_SEED", "PROJECT_DIR",
    "CODD_REPO_DIR", "MODEL_PATH", "DATA_DIR", "OUT_DIR", "ALLOW_PYJUICE_STUB", "ALLOW_MODEL_CODE_MISMATCH", "STRICT_VERSIONS",
    "MIN_GPU_TOTAL_GB", "MIN_GPU_FREE_GB", "RUN_LIMIT", "MAX_RUNTIME_MIN", "REQUIRE_SMOKE",
]
_missing_keys = [k for k in _REQUIRED_KEYS if k not in CFG]
if _missing_keys:
    raise SystemExit("В config.json нет ключей: " + str(_missing_keys))

MODEL = CFG["MODEL"]
HMM_CKPT = CFG["HMM_CKPT"]
DATASET = CFG["DATASET"]
DATASET_SPLIT = CFG["DATASET_SPLIT"]
N_TASKS = CFG["N_TASKS"]
GEN_LENGTH = CFG["GEN_LENGTH"]
BLOCK_LENGTH = CFG["BLOCK_LENGTH"]
TEMPERATURE = CFG["TEMPERATURE"]
NUM_STEPS_LIST = CFG["NUM_STEPS_LIST"]
REMASKING_LIST = CFG["REMASKING_LIST"]
PC_TEMPERATURE = CFG["PC_TEMPERATURE"]
PC_FRAC = CFG["PC_FRAC"]
REVERSE_FRAC = CFG["REVERSE_FRAC"]
SEEDS = CFG["SEEDS"]
EXP_NAME = CFG["EXP_NAME"]
USE_PC = CFG["USE_PC"]
SEED_USED = CFG["SEED_USED"]
SYSTEM_INSTRUCTION = CFG["SYSTEM_INSTRUCTION"]
NUM_FEWSHOT = CFG["NUM_FEWSHOT"]
TASK_NAME = CFG["TASK_NAME"]
MASK_ID = CFG["MASK_ID"]
DTYPE = CFG["DTYPE"]
ATTN_IMPLEMENTATION = CFG["ATTN_IMPLEMENTATION"]
N_BOOTSTRAP = CFG["N_BOOTSTRAP"]
BOOTSTRAP_SEED = CFG["BOOTSTRAP_SEED"]
PROJECT_DIR = CFG["PROJECT_DIR"]
CODD_REPO_DIR = CFG["CODD_REPO_DIR"]
MODEL_PATH = CFG["MODEL_PATH"]
DATA_DIR = CFG["DATA_DIR"]
OUT_DIR = CFG["OUT_DIR"]
ALLOW_PYJUICE_STUB = CFG["ALLOW_PYJUICE_STUB"]
ALLOW_MODEL_CODE_MISMATCH = CFG["ALLOW_MODEL_CODE_MISMATCH"]
STRICT_VERSIONS = CFG["STRICT_VERSIONS"]
MIN_GPU_TOTAL_GB = CFG["MIN_GPU_TOTAL_GB"]
MIN_GPU_FREE_GB = CFG["MIN_GPU_FREE_GB"]
RUN_LIMIT = CFG["RUN_LIMIT"]
MAX_RUNTIME_MIN = CFG["MAX_RUNTIME_MIN"]
REQUIRE_SMOKE = CFG["REQUIRE_SMOKE"]

# интернета нет: принудительно офлайн; кэши HF и datasets - внутри проекта
os.environ["MPLBACKEND"] = "Agg"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
HF_CACHE_DIR = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")

# ---------- проверки параметров (те же, что были в ячейке конфига) ----------
assert MODEL == "GSAI-ML/LLaDA-8B-Instruct", "MODEL - общий параметр, менять нельзя"
assert HMM_CKPT == "il18/llada-gsm-pc", "HMM_CKPT - общий параметр (в exp1 не используется)"
assert (DATASET, DATASET_SPLIT, N_TASKS) == ("gsm8k", "test", 100), "DATASET/DATASET_SPLIT/N_TASKS - общие параметры"
assert GEN_LENGTH in (256, 512), "GEN_LENGTH должен быть 256 или 512 (выбор автора проекта)"
assert BLOCK_LENGTH == 32 and TEMPERATURE == 0.0, "BLOCK_LENGTH/TEMPERATURE - общие параметры"
assert NUM_STEPS_LIST == [64, 128] and REMASKING_LIST == ["low_confidence", "margin"], "NUM_STEPS_LIST/REMASKING_LIST - общие параметры"
assert (PC_TEMPERATURE, PC_FRAC, REVERSE_FRAC) == (0.2, 0.5, False), "PC_* - общие параметры"
assert SEEDS == [0, 1, 2], "SEEDS - общий параметр (в exp1 игнорируется)"
assert USE_PC is False, "exp1 - это эксперимент БЕЗ HMM"
assert DTYPE in ("bfloat16", "float16"), "DTYPE должен быть 'bfloat16' или 'float16', получено " + repr(DTYPE)
assert RUN_LIMIT is None or (isinstance(RUN_LIMIT, int) and 1 <= RUN_LIMIT <= N_TASKS), "RUN_LIMIT: None или целое от 1 до N_TASKS"
assert MAX_RUNTIME_MIN is None or (isinstance(MAX_RUNTIME_MIN, (int, float)) and MAX_RUNTIME_MIN > 0), "MAX_RUNTIME_MIN: None или число минут > 0"
for _gl in sorted({256, 512, GEN_LENGTH}):
    assert _gl % BLOCK_LENGTH == 0, "GEN_LENGTH=" + str(_gl) + " не делится на BLOCK_LENGTH"
    for _ns in NUM_STEPS_LIST:
        assert _ns % (_gl // BLOCK_LENGTH) == 0, "num_steps=" + str(_ns) + " не делится на число блоков при GEN_LENGTH=" + str(_gl)
NUM_BLOCKS = GEN_LENGTH // BLOCK_LENGTH

COMMON_PARAMS = {
    "MODEL": MODEL, "HMM_CKPT": HMM_CKPT, "DATASET": DATASET, "DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS,
    "GEN_LENGTH": GEN_LENGTH, "BLOCK_LENGTH": BLOCK_LENGTH, "TEMPERATURE": TEMPERATURE,
    "NUM_STEPS_LIST": NUM_STEPS_LIST, "REMASKING_LIST": REMASKING_LIST,
    "PC_TEMPERATURE": PC_TEMPERATURE, "PC_FRAC": PC_FRAC, "REVERSE_FRAC": REVERSE_FRAC, "SEEDS": SEEDS,
}
EXP_PARAMS = dict(COMMON_PARAMS)
EXP_PARAMS.update({
    "EXP_NAME": EXP_NAME, "USE_PC": USE_PC, "SEED_USED": SEED_USED, "SYSTEM_INSTRUCTION": SYSTEM_INSTRUCTION,
    "NUM_FEWSHOT": NUM_FEWSHOT, "TASK_NAME": TASK_NAME, "MASK_ID": MASK_ID,
    "DTYPE": DTYPE, "ATTN_IMPLEMENTATION": ATTN_IMPLEMENTATION,
})
COMMON_SHA256 = sha256_json(COMMON_PARAMS)
EXP_SHA256 = sha256_json(EXP_PARAMS)

ENV_PATH = os.path.join(OUT_DIR, "env.json")
CONFIG_PATH = os.path.join(OUT_DIR, "config.json")
TASKS_PATH = os.path.join(OUT_DIR, "tasks.json")
SMOKE_PATH = os.path.join(OUT_DIR, "smoke.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")
STATUS_PATH = os.path.join(OUT_DIR, "status.json")

try:
    import numpy as np
except Exception as _e:
    raise SystemExit("numpy не импортируется в этом окружении: " + str(_e))

# глобальное состояние, заполняется по ходу работы
torch = None
F = None
TORCH_DTYPE = None
ENV_INFO = {}
PYJUICE_MODE = None
model = None
tokenizer = None
PLAIN = None
lm = None
task = None
INSTANCES = []
TASK_IDS = []
PROMPT_IDS = []
UNTIL = []
IDS_SHA256 = None
QUESTIONS_SHA256 = None
MODEL_CODE_CHECK = {}
STOP_REQUESTED = {"flag": False}


def set_status(state, message=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    write_json_atomic(STATUS_PATH, {"mode": MODE, "state": state, "message": message, "pid": os.getpid(), "updated_utc": utc_now()})
    log("[EXP1] STATUS: " + state + " " + message)


def _on_stop_signal(signum, frame):
    if STOP_REQUESTED["flag"]:
        log("повторный сигнал остановки: немедленный выход (кэш к этому моменту согласован)")
        os._exit(143)
    STOP_REQUESTED["flag"] = True
    log("получен сигнал остановки: заканчиваю текущую задачу, дописываю кэш и steplog, затем остановлюсь")


# ===================== 1. ПРОВЕРКА ОКРУЖЕНИЯ И GPU =====================
EXPECTED_VERSIONS = {"torch": "2.9.1", "transformers": "4.57.3", "pyjuice": "2.4.1", "huggingface-hub": "0.36.0", "numpy": "2.2.6"}
REQUIRED_IMPORTS = ["accelerate", "datasets", "evaluate", "jinja2", "jsonlines", "pytablewriter", "rouge_score", "sacrebleu", "sklearn",
                    "sqlitedict", "zstandard", "dill", "word2number", "more_itertools", "pyarrow", "safetensors", "tokenizers"]
OPTIONAL_IMPORTS = ["peft"]     # не нужен для расчёта: отсутствие только предупреждение
REQUIRED_REPO_FILES = [
    "eval/llada/codd_llada.py", "eval/llada/llada_generate.py", "eval/llada/modeling_llada.py",
    "eval/llada/configuration_llada.py", "codd/model_class.py", "codd/__init__.py",
    "eval/eval.py", "eval/harness.py", "lm-evaluation-harness/lm_eval/__init__.py",
    "lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k.yaml", "lm-evaluation-harness/pyproject.toml",
]


def check_environment():
    global torch, F, TORCH_DTYPE, ENV_INFO
    log("Python: " + sys.version.split()[0] + " | " + sys.executable + " | " + platform.platform())
    assert sys.version_info >= (3, 10), "Нужен Python >= 3.10"
    try:
        import torch as _torch
        import torch.nn.functional as _F
    except Exception as e:
        raise RuntimeError("torch не импортируется в venv: " + str(e)) from e
    torch = _torch
    F = _F
    TORCH_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
    log("torch: " + torch.__version__ + " | CUDA (сборка torch): " + str(torch.version.cuda) + " | DTYPE: " + DTYPE)
    assert torch.cuda.is_available(), "GPU не найден (torch.cuda.is_available() == False): нужен запуск на GPU"
    gpu_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_index)
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    log("GPU: " + props.name + " | compute capability " + str(props.major) + "." + str(props.minor))
    log("Видеопамять фактически: всего " + format(total / gib, ".2f") + " ГиБ (" + str(total) + " байт), свободно " + format(free / gib, ".2f") + " ГиБ (" + str(free) + " байт)")
    assert total / gib >= MIN_GPU_TOTAL_GB, "Памяти GPU " + format(total / gib, ".2f") + " ГиБ < MIN_GPU_TOTAL_GB=" + str(MIN_GPU_TOTAL_GB)
    assert free / gib >= MIN_GPU_FREE_GB, "Свободно " + format(free / gib, ".2f") + " ГиБ < MIN_GPU_FREE_GB=" + str(MIN_GPU_FREE_GB) + ": освободите GPU (другие процессы)"
    if props.major < 8:
        log("!" * 90)
        log("ПРЕДУПРЕЖДЕНИЕ: compute capability " + str(props.major) + "." + str(props.minor) + " < 8.0: bfloat16 аппаратно не поддерживается; DTYPE=" + DTYPE + " автоматически НЕ меняется")
        log("!" * 90)
    disk = shutil.disk_usage(PROJECT_DIR)
    log("Диск (" + PROJECT_DIR + "): свободно " + format(disk.free / gib, ".1f") + " ГиБ из " + format(disk.total / gib, ".1f"))
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = os.path.join(OUT_DIR, ".write_probe")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe)

    missing_files = [p for p in REQUIRED_REPO_FILES if not os.path.isfile(os.path.join(CODD_REPO_DIR, p))]
    assert not missing_files, "В CODD_REPO_DIR=" + CODD_REPO_DIR + " нет файлов: " + str(missing_files)
    weights_dir = Path(MODEL_PATH)
    assert weights_dir.is_dir(), "Нет папки весов: " + MODEL_PATH
    names = set(os.listdir(MODEL_PATH))
    for need in ("config.json", "tokenizer.json", "modeling_llada.py", "configuration_llada.py", "model.safetensors.index.json"):
        assert need in names, "В " + MODEL_PATH + " нет файла " + need + " (склеены ли шарды и скопированы ли файлы?)"
    shards = sorted(n for n in names if n.endswith(".safetensors"))
    assert len(shards) == 6, "Ожидалось 6 шардов *.safetensors (склеенных), найдено " + str(len(shards)) + ": " + str(shards)
    assert not any(".part" in n for n in names), "В папке весов остались не склеенные части *.partNNN"
    weights_bytes = sum(os.path.getsize(os.path.join(MODEL_PATH, n)) for n in shards)
    log("Веса: " + str(len(shards)) + " шардов, " + format(weights_bytes / 1e9, ".2f") + " ГБ")
    assert weights_bytes > 15_000_000_000, "суммарный размер шардов " + str(weights_bytes) + " байт подозрительно мал (ожидается около 16.0 ГБ): склейка неполная?"

    missing_imports = [m for m in REQUIRED_IMPORTS if importlib.util.find_spec(m) is None]
    assert not missing_imports, "В venv нет пакетов: " + str(missing_imports) + " (интернета нет, pip в venv нет: доложите в wheelhouse)"
    missing_optional = [m for m in OPTIONAL_IMPORTS if importlib.util.find_spec(m) is None]
    if missing_optional:
        log("ПРЕДУПРЕЖДЕНИЕ: нет необязательных пакетов " + str(missing_optional) + " (на расчёт не влияет)")

    log("Версии (ожидается по сборке бандла):")
    mismatches = []
    for name, want in EXPECTED_VERSIONS.items():
        have = dist_version(name)
        ok = have is not None and have.split("+")[0] == want
        if not ok:
            mismatches.append([name, want, have])
        log("   " + name.ljust(18) + (want or "").ljust(10) + str(have).ljust(16) + ("OK" if ok else "РАСХОЖДЕНИЕ"))
    if mismatches:
        msg = "Расхождения версий: " + str(mismatches)
        if STRICT_VERSIONS:
            raise RuntimeError(msg)
        log("ПРЕДУПРЕЖДЕНИЕ (не ошибка): " + msg)
    log("НЕ ПРОВЕРЕНО: совместимость этих версий с кодом модели (modeling_llada.py), CoDD и lm_eval; проверена только сборка бандла.")
    ENV_INFO = {
        "started_utc": START_UTC, "mode": MODE, "python": sys.version.split()[0], "python_executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "gpu_name": props.name, "gpu_total_gb": round(total / gib, 2), "gpu_total_bytes": total,
        "gpu_capability": str(props.major) + "." + str(props.minor), "dtype": DTYPE,
        "hf_reachable": "offline (HF_HUB_OFFLINE=1)", "pypi_reachable": "offline",
        "project_dir": PROJECT_DIR, "model_path": MODEL_PATH, "local_files_only": True, "hf_cache_dir": HF_CACHE_DIR,
        "transformers": dist_version("transformers"), "huggingface_hub": dist_version("huggingface-hub"), "numpy": dist_version("numpy"),
        "version_mismatches_vs_expected": mismatches, "weights_gib": round(weights_bytes / gib, 2),
    }
    return gpu_index, total


# ===================== 2. PYJUICE И КОД РЕПОЗИТОРИЯ =====================
def prepare_pyjuice():
    global PYJUICE_MODE
    try:
        import pyjuice  # noqa: F401
        PYJUICE_MODE = "real"
        log("pyjuice импортирован, версия " + str(dist_version("pyjuice")) + " (ожидается 2.4.1)")
    except Exception as e:
        for k in [k for k in sys.modules if k == "pyjuice" or k.startswith("pyjuice.")]:
            del sys.modules[k]
        log("pyjuice НЕ импортируется: " + type(e).__name__ + ": " + str(e))
        if not ALLOW_PYJUICE_STUB:
            raise RuntimeError("pyjuice недоступен. Для exp1 (HMM не используется) можно поставить ALLOW_PYJUICE_STUB=true в config.json, но ТОЛЬКО в exp1: в блокнотах 2-5 заглушка запрещена.") from e
        import types
        stub = types.ModuleType("pyjuice")

        def _forbidden(*args, **kwargs):
            raise RuntimeError("pyjuice - ЗАГЛУШКА (exp1). Путь HMM здесь вызываться не должен.")

        stub.load = _forbidden
        stub.compile = _forbidden
        stub.save = _forbidden
        stub.queries = types.SimpleNamespace(sample=_forbidden)
        stub.__is_stub__ = True
        sys.modules["pyjuice"] = stub
        PYJUICE_MODE = "stub"
        log("=" * 90)
        log("ВНИМАНИЕ: ИСПОЛЬЗУЕТСЯ ЗАГЛУШКА pyjuice. Допустимо ТОЛЬКО в exp1 (use_pc=False). В блокнотах 2-5 заглушка ЗАПРЕЩЕНА.")
        log("=" * 90)
    ENV_INFO["pyjuice"] = dist_version("pyjuice") if PYJUICE_MODE == "real" else "stub"
    ENV_INFO["pyjuice_mode"] = PYJUICE_MODE


def git_info(repo):
    def run(args):
        return subprocess.run(["git", "-C", repo] + args, capture_output=True, text=True, timeout=20)
    try:
        head = run(["rev-parse", "HEAD"])
        if head.returncode != 0:
            return {"commit": "unknown (нет git-метаданных)", "dirty": None}
        status = run(["status", "--porcelain"])
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except Exception as e:
        return {"commit": "unknown (" + type(e).__name__ + ")", "dirty": None}


def import_repo_code():
    global CoddLlada, LLaDAModelLM, ORIG_GEN_MODULE, llada_diffusion_generate, add_gumbel_noise, get_num_transfer_tokens, _is_codd_model
    global lm_eval, TaskManager, get_task_dict, HFLM, handle_stop_sequences, postprocess_generated_text
    for p in (os.path.join(CODD_REPO_DIR, "lm-evaluation-harness"), CODD_REPO_DIR):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)    # репозиторий перекрывает любые другие версии (в -I режиме cwd не в sys.path)
    importlib.invalidate_caches()
    from eval.llada.codd_llada import CoddLlada
    from eval.llada.modeling_llada import LLaDAModelLM
    from eval.llada import llada_generate as ORIG_GEN_MODULE
    from eval.llada.llada_generate import llada_diffusion_generate, add_gumbel_noise, get_num_transfer_tokens, _is_codd_model
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text
    repo_real = os.path.realpath(CODD_REPO_DIR) + os.sep
    for name, mod in (("eval.llada.llada_generate", ORIG_GEN_MODULE), ("lm_eval", lm_eval)):
        f = os.path.realpath(mod.__file__)
        assert f.startswith(repo_real), name + " импортирован НЕ из репозитория: " + f + " (ожидалась папка " + repo_real + ")"
    log("Код репозитория импортирован из: " + CODD_REPO_DIR)
    pyproject = open(os.path.join(CODD_REPO_DIR, "lm-evaluation-harness", "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    git = git_info(CODD_REPO_DIR)
    ENV_INFO.update({
        "lm_eval_version_pyproject": m.group(1) if m else None, "codd_commit": git["commit"], "codd_dirty": git["dirty"],
        "llada_generate_py_sha256": sha256_file(os.path.join(CODD_REPO_DIR, "eval", "llada", "llada_generate.py")),
        "common_params_sha256": COMMON_SHA256, "exp_params_sha256": EXP_SHA256,
    })


def write_env_and_config():
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            old_env = json.load(f)
        prev = old_env.pop("previous_sessions", [])
        ENV_INFO["previous_sessions"] = prev + [old_env]     # история прежних сессий (smoke/full/возобновление)
    write_json_atomic(ENV_PATH, ENV_INFO)
    log("env.json записан: " + ENV_PATH)
    config_doc = {"common_params": COMMON_PARAMS, "common_sha256": COMMON_SHA256, "exp_params": EXP_PARAMS, "exp_sha256": EXP_SHA256}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            old_cfg = json.load(f)
        if old_cfg.get("exp_sha256") != EXP_SHA256:
            keys = set(old_cfg.get("exp_params", {})) | set(EXP_PARAMS)
            diff = {}
            for k in keys:
                a, b = old_cfg.get("exp_params", {}).get(k), EXP_PARAMS.get(k)
                if a != b:
                    diff[k] = [a, b]
            raise RuntimeError("В " + OUT_DIR + " уже лежат результаты с ДРУГИМ конфигом (параметр: [было, стало]): " + str(diff) + ". Верните прежние значения или укажите другой OUT_DIR.")
        log("config.json совпадает с текущим конфигом - возобновление допустимо.")
    else:
        write_json_atomic(CONFIG_PATH, config_doc)
        log("config.json записан: " + CONFIG_PATH)
    log("codd_commit: " + str(ENV_INFO.get("codd_commit")) + " | dirty: " + str(ENV_INFO.get("codd_dirty")))


# ===================== 3. ЗАГРУЗКА МОДЕЛИ И СВЕРКА КОДА =====================
class PlainView:
    # Та же модель как "обычная": вызов базового forward без CoDD-аргументов. Атрибута pc_model нет специально:
    # _is_codd_model(PlainView(...)) == False, поэтому оригинальная функция идёт по базовой ветке (model(x).logits).
    def __init__(self, m):
        self._m = m

    @property
    def device(self):
        return self._m.device

    def __call__(self, x):
        return LLaDAModelLM.forward(self._m, x)


def load_model():
    global model, tokenizer, PLAIN
    t0 = time.time()
    log("Загрузка модели с диска (local_files_only=True) из " + MODEL_PATH)
    model = CoddLlada.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        local_files_only=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="cuda",
    )
    ENV_INFO["model_load_seconds"] = round(time.time() - t0, 1)
    log("Модель загружена за " + format(time.time() - t0, ".0f") + " c")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = model.tokenizer
    assert tokenizer is not None, "model.tokenizer is None: токенизатор не загрузился"
    assert model.pc_model is None, "pc_model должен быть None: в exp1 HMM не используется и не должен загружаться"
    assert model.codd_config.mask_id == MASK_ID, "mask_id модели " + str(model.codd_config.mask_id) + " != " + str(MASK_ID)
    emb = model.get_input_embeddings().weight
    assert emb.shape[0] > MASK_ID, "Размер словаря " + str(emb.shape[0]) + " не содержит mask_id=" + str(MASK_ID)
    assert next(model.parameters()).dtype == TORCH_DTYPE, "ожидается " + DTYPE + ", а параметры модели в " + str(next(model.parameters()).dtype)
    assert next(model.parameters()).device.type == "cuda", "модель должна быть на GPU"
    assert not model.training, "модель должна быть в eval-режиме"
    log("dtype модели: " + str(next(model.parameters()).dtype) + " | mask_id " + str(MASK_ID) + " -> токен " + str(tokenizer.convert_ids_to_tokens(MASK_ID)) + " | eos_token_id " + str(tokenizer.eos_token_id))
    log("Память GPU занята: " + format(torch.cuda.memory_allocated() / 1024 ** 3, ".1f") + " ГиБ")
    PLAIN = PlainView(model)
    assert _is_codd_model(model) is True and _is_codd_model(PLAIN) is False


def check_model_code():
    # вместо загрузки второй модели: sha256 исходников из репозитория против файлов рядом с весами
    global MODEL_CODE_CHECK
    def sha_norm(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    for fname in ("modeling_llada.py", "configuration_llada.py"):
        local = os.path.join(CODD_REPO_DIR, "eval", "llada", fname)
        snap = os.path.join(MODEL_PATH, fname)
        if not os.path.isfile(snap):
            MODEL_CODE_CHECK[fname] = {"verdict": "НЕ ПРОВЕРЕНО", "reason": "нет файла " + fname + " рядом с весами"}
            log(fname + ": НЕ ПРОВЕРЕНО - нет файла рядом с весами")
            continue
        a, b = sha256_file(local), sha256_file(snap)
        MODEL_CODE_CHECK[fname] = {"verdict": "MATCH" if a == b else "MISMATCH", "repo_sha256": a, "snapshot_sha256": b,
                                   "snapshot_source": "локальная папка весов", "equal_after_newline_normalization": sha_norm(local) == sha_norm(snap)}
        log(fname + ": " + MODEL_CODE_CHECK[fname]["verdict"] + " repo=" + a[:16] + "... snapshot=" + b[:16] + "...")
    ENV_INFO["model_code_check"] = MODEL_CODE_CHECK
    write_json_atomic(ENV_PATH, ENV_INFO)
    bad = [k for k, v in MODEL_CODE_CHECK.items() if v["verdict"] == "MISMATCH"]
    if bad and not ALLOW_MODEL_CODE_MISMATCH:
        raise RuntimeError("Исходники модели из репозитория CoDD отличаются от файлов рядом с весами: " + str(bad) + ". Остановка до решения автора (ALLOW_MODEL_CODE_MISMATCH=true, чтобы продолжить осознанно).")


# ===================== 4. ЗАДАЧИ GSM8K (ОФЛАЙН-parquet ЧЕРЕЗ КОД lm_eval) =====================
def find_parquet(split):
    patterns = [os.path.join("main", split + "-*.parquet"), split + "-*.parquet", os.path.join("main", split + ".parquet"), split + ".parquet"]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(DATA_DIR, pat)))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError("В " + DATA_DIR + " несколько файлов для сплита " + split + ": " + str(hits))
    return None


def load_task():
    # Как в lm_eval: dict-конфиг с task=gsm8k сливается с gsm8k.yaml ({**base_task_config, **name_or_config}), а ConfigurableTask.download
    # вызывает datasets.load_dataset(path=DATASET_PATH, name=DATASET_NAME, **dataset_kwargs). Подменяем только источник данных:
    # path="parquet", name=None, data_files={train, test}. doc_to_text, фильтры, метрика, build_all_requests - код lm_eval без изменений.
    train, test = find_parquet("train"), find_parquet("test")
    assert test is not None, "В " + DATA_DIR + " не найден test-parquet GSM8K"
    if train is None:
        log("ВНИМАНИЕ: train-parquet не найден, вместо него подставлен test (на 0-shot это не влияет).")
        train = test
    log("Данные: train=" + train + " | test=" + test)
    override = {
        "task": TASK_NAME,
        "dataset_path": "parquet",
        "dataset_name": None,
        "dataset_kwargs": {"data_files": {"train": train, "test": test}},
    }
    task_dict = get_task_dict([override], task_manager=TaskManager())
    assert len(task_dict) == 1, "ожидалась одна задача " + TASK_NAME + ", получено: " + str(list(task_dict))
    return next(iter(task_dict.values()))


def gold_final(answer_text):
    m = re.search(r"####\s*([^\n]+)", answer_text)
    return m.group(1).strip() if m else None


def setup_tasks():
    global task, lm, INSTANCES, TASK_IDS, PROMPT_IDS, UNTIL, IDS_SHA256, QUESTIONS_SHA256, MODEL_MAX_LEN
    task = load_task()
    task.set_config(key="num_fewshot", value=NUM_FEWSHOT)
    assert task.get_config("num_fewshot") == NUM_FEWSHOT == 0
    assert task.OUTPUT_TYPE == "generate_until", "ожидался generate_until, а не " + str(task.OUTPUT_TYPE)
    gk = task.get_config("generation_kwargs")
    assert gk.get("do_sample") is False and float(gk.get("temperature", 0.0)) == TEMPERATURE == 0.0, "generation_kwargs задачи не жадные: " + str(gk)
    n_docs = len(task.eval_docs)
    log("Документов в " + DATASET_SPLIT + "-выборке: " + str(n_docs) + ("" if n_docs == 1319 else " (ВНИМАНИЕ: у стандартного GSM8K test 1319 задач)"))
    assert n_docs >= N_TASKS, "в выборке " + str(n_docs) + " < N_TASKS=" + str(N_TASKS)
    assert {f.name for f in task._filters} == {"strict-match", "flexible-extract"}, "неожиданные фильтры задачи: " + str([f.name for f in task._filters])

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    log("HFLM backend: " + str(lm.backend) + " | max_length (HFLM): " + str(lm.max_length) + " | tokenizer_name: " + str(lm.tokenizer_name))
    task.build_all_requests(
        limit=N_TASKS, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
        system_instruction=SYSTEM_INSTRUCTION, apply_chat_template=True, fewshot_as_multiturn=False,
        chat_template=lm.apply_chat_template, tokenizer_name=lm.tokenizer_name,
    )
    INSTANCES = list(task.instances)
    assert len(INSTANCES) == N_TASKS, "получено " + str(len(INSTANCES)) + " запросов вместо " + str(N_TASKS)
    for i, inst in enumerate(INSTANCES):
        assert inst.doc_id == i, "порядок задач нарушен: позиция " + str(i) + ", doc_id " + str(inst.doc_id)
        assert inst.request_type == "generate_until" and inst.repeats == 1
    TASK_IDS = [inst.doc_id for inst in INSTANCES]
    assert TASK_IDS == list(range(N_TASKS))

    gen_kwargs = copy.deepcopy(INSTANCES[0].args[1])
    UNTIL = handle_stop_sequences(gen_kwargs.pop("until", None), eos=lm.tok_decode(lm.eot_token_id, skip_special_tokens=False))
    log("until (стоп-последовательности из задачи + EOS): " + str(UNTIL))
    assert all(inst.args[1] == INSTANCES[0].args[1] for inst in INSTANCES), "generation_kwargs различаются между задачами"

    PROMPT_IDS = []
    for inst in INSTANCES:
        ids, _ = lm.tok_batch_encode([inst.args[0]], left_truncate_len=None, truncation=False)
        assert ids.shape[0] == 1
        PROMPT_IDS.append(ids[0].tolist())
        assert MASK_ID not in PROMPT_IDS[-1], "в промпте встретился mask_id"
    limits = [lm.max_length]
    if hasattr(model.config, "max_sequence_length") and model.config.max_sequence_length:
        limits.append(int(model.config.max_sequence_length))
    MODEL_MAX_LEN = min(limits)
    max_prompt = max(len(p) for p in PROMPT_IDS)
    for gl in sorted({256, 512, GEN_LENGTH}):
        assert max_prompt + gl <= MODEL_MAX_LEN, "Самый длинный промпт (" + str(max_prompt) + " ток.) + GEN_LENGTH=" + str(gl) + " > лимита длины модели " + str(MODEL_MAX_LEN)
    log("Промпты: длина " + str(min(len(p) for p in PROMPT_IDS)) + ".." + str(max_prompt) + " токенов; лимит модели " + str(MODEL_MAX_LEN) + "; GEN_LENGTH=" + str(GEN_LENGTH) + " помещается (проверено и для 256, и для 512).")
    log("-" * 80)
    log("ПРОМПТ задачи 0 (как видит модель):")
    print(INSTANCES[0].args[0], flush=True)
    log("-" * 80)

    IDS_SHA256 = hashlib.sha256(json.dumps(TASK_IDS, separators=(",", ":")).encode("utf-8")).hexdigest()
    QUESTIONS_SHA256 = hashlib.sha256(json.dumps([inst.doc["question"] for inst in INSTANCES], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    tasks_doc = {
        "dataset": DATASET, "split": DATASET_SPLIT, "n_tasks": N_TASKS, "data_source": "локальные parquet через lm_eval (dataset_path=parquet): " + DATA_DIR,
        "system_instruction": SYSTEM_INSTRUCTION, "num_fewshot": NUM_FEWSHOT, "chat_template_applied": True,
        "ids_sha256": IDS_SHA256,
        "ids_sha256_definition": 'sha256(json.dumps(ids, separators=(",",":")).encode("utf-8")), ids = [0..N_TASKS-1]',
        "questions_sha256": QUESTIONS_SHA256, "ids": TASK_IDS,
        "tasks": [
            {"task_id": inst.doc_id, "question": inst.doc["question"], "gold_final": gold_final(inst.doc["answer"]),
             "prompt_sha256": hashlib.sha256(inst.args[0].encode("utf-8")).hexdigest()}
            for inst in INSTANCES
        ],
    }
    assert all(t["gold_final"] is not None for t in tasks_doc["tasks"]), "у некоторых задач нет '#### ответ' в эталоне"
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, encoding="utf-8") as f:
            old = json.load(f)
        assert old["ids_sha256"] == IDS_SHA256, "список id отличается от сохранённого в tasks.json - возобновление небезопасно"
        assert old["questions_sha256"] == QUESTIONS_SHA256, "тексты вопросов отличаются от сохранённых в tasks.json (другая версия датасета?)"
        old_prompts = {t["task_id"]: t["prompt_sha256"] for t in old["tasks"]}
        new_prompts = {t["task_id"]: t["prompt_sha256"] for t in tasks_doc["tasks"]}
        assert old_prompts == new_prompts, "промпты отличаются от сохранённых в tasks.json (другой шаблон чата/токенизатор?) - возобновление небезопасно"
        log("tasks.json совпадает с текущими задачами и промптами - возобновление допустимо.")
    else:
        write_json_atomic(TASKS_PATH, tasks_doc)
        log("tasks.json записан: " + TASKS_PATH)
    log("=" * 80)
    log("sha256 списка id задач: " + IDS_SHA256)
    log("(справочно) sha256 текстов вопросов: " + QUESTIONS_SHA256)
    log("=" * 80)


# ===================== 5. КОПИЯ llada_diffusion_generate С ЖУРНАЛОМ ШАГОВ =====================
def llada_generate_logged(model, prompt, num_steps=128, gen_length=128, block_length=128,
                          temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
                          use_pc=False, pc_temperature=None, pc_frac=None, reverse_frac=None):
    # Копия llada_diffusion_generate (CoDD/eval/llada/llada_generate.py, коммит d9760f4) + журнал шагов.
    # Отличия от оригинала (на вычисления не влияют): assert use_pc is False; нет tqdm; steplog; topk сохраняет значения.
    assert use_pc is False, "exp1: HMM не используется (use_pc=False)"
    assert prompt.dim() == 2 and prompt.shape[0] == 1, "ожидается промпт формы (1, L)"
    with torch.no_grad():
        is_codd = _is_codd_model(model)

        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert num_steps % num_blocks == 0
        steps = num_steps // num_blocks

        steplog = []
        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, start_idx:end_idx] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == mask_id)
                n_masked_block_before = int(mask_index[0, start_idx:end_idx].sum().item())

                # ---------- forward pass ----------
                if is_codd:
                    # CoDD.forward() handles CFG + PC modification internally
                    pc_block_range = (start_idx, end_idx) if use_pc else None
                    output = model(
                        x,
                        pc_block_range=pc_block_range,
                        use_pc=use_pc,
                        cfg_scale=cfg_scale,
                        prompt_index=prompt_index if cfg_scale > 0. else None,
                        pc_temperature=pc_temperature,
                        pc_frac=pc_frac,
                        reverse_frac=reverse_frac,
                    )
                    logits = output.logits
                else:
                    # Plain base model path (backward compatible)
                    if cfg_scale > 0.:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x).logits

                # ---------- sampling ----------
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                elif remasking == 'margin':
                    probs = torch.softmax(logits_with_noise, dim=-1)
                    top_probs, _ = torch.topk(probs, k=2, dim=-1)
                    x0_p = top_probs[..., 0] - top_probs[..., 1]
                elif remasking == 'entropy':
                    probs = torch.softmax(logits_with_noise, dim=-1)
                    epsilon = 1e-10
                    log_probs = torch.log(probs + epsilon)
                    x0_p = torch.sum(probs * log_probs, dim=-1)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, end_idx:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        conf_vals, select_index = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_index] = x0[j, select_index]
                        pos_abs = select_index.tolist()
                        steplog.append({
                            "block": num_block,
                            "step_in_block": i,
                            "global_step": num_block * steps + i,
                            "n_masked_block_before": n_masked_block_before,
                            "positions_abs": pos_abs,
                            "positions_rel": [p_ - prompt.shape[1] for p_ in pos_abs],
                            "token_ids": x0[j, select_index].tolist(),
                            "confidence": [float(c) for c in conf_vals.float().tolist()],
                        })

    return x, steplog


# ===================== 6. ДЕКОДИРОВАНИЕ, ПРОВЕРКА ОТВЕТА (КОД lm_eval), КЭШ =====================
FILTER_NAMES = ("strict-match", "flexible-extract")
REQUIRED_ROW_KEYS = ("task_id", "seed", "correct", "prediction", "num_steps", "remasking", "pc_active_steps", "delta_words", "delta_order")


def score_response(inst, text):
    # Ровно то, что делает evaluator.evaluate: resps -> фильтры задачи -> task.process_results(doc, [filtered_resps]).
    inst.resps = [text]
    inst.filtered_resps = {}
    for filter_ensemble in task._filters:
        filter_ensemble.apply([inst])
    scores, preds = {}, {}
    for name in FILTER_NAMES:
        assert name in inst.filtered_resps, "нет результата фильтра " + name + ": " + str(list(inst.filtered_resps))
        pred = inst.filtered_resps[name]
        assert isinstance(pred, str), "ожидалась строка после фильтра " + name + ", получено " + type(pred).__name__ + ": " + repr(pred)
        metrics = task.process_results(inst.doc, [pred])
        assert "exact_match" in metrics, "process_results не вернул exact_match: " + str(metrics)
        value = float(metrics["exact_match"])
        assert value in (0.0, 1.0), "exact_match должен быть 0 или 1, получено " + str(value)
        scores[name] = value
        preds[name] = pred
    return scores, preds


def decode_generation(gen_token_ids):
    text_raw = lm.tok_decode(gen_token_ids)
    return postprocess_generated_text(generation=text_raw, stop=UNTIL, think_end_token=None)


def run_generation(inst, remasking, num_steps, gen_length=None, block_length=None):
    gen_length = GEN_LENGTH if gen_length is None else gen_length
    block_length = BLOCK_LENGTH if block_length is None else block_length
    prompt = torch.tensor([PROMPT_IDS[inst.doc_id]], dtype=torch.long, device=model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    x, steplog = llada_generate_logged(
        model, prompt, num_steps=num_steps, gen_length=gen_length, block_length=block_length,
        temperature=TEMPERATURE, cfg_scale=0.0, remasking=remasking, mask_id=MASK_ID, use_pc=False)
    torch.cuda.synchronize()
    return x, steplog, prompt.shape[1], time.time() - t0


def process_task(inst, remasking, num_steps):
    task_id = inst.doc_id
    x, steplog, prompt_len, sec = run_generation(inst, remasking, num_steps)
    gen_ids = x[0, prompt_len:].tolist()
    assert len(gen_ids) == GEN_LENGTH
    assert MASK_ID not in gen_ids, "задача " + str(task_id) + ": в ответе остались маски (mask_id=" + str(MASK_ID) + ") - несостыковка, остановка"
    assert len(steplog) == num_steps, "задача " + str(task_id) + ": в журнале " + str(len(steplog)) + " шагов вместо " + str(num_steps)
    text = decode_generation(gen_ids)
    scores, preds = score_response(inst, text)
    row = {
        "task_id": task_id, "seed": SEED_USED, "correct": int(scores["flexible-extract"] == 1.0), "prediction": preds["flexible-extract"],
        "num_steps": num_steps, "remasking": remasking, "pc_active_steps": 0, "delta_words": None, "delta_order": None,
        # --- дополнительные поля (не входят в обязательный формат) ---
        "correct_strict": int(scores["strict-match"] == 1.0), "prediction_strict": preds["strict-match"],
        "gold": gold_final(inst.doc["answer"]), "generation": text, "gen_length": GEN_LENGTH, "prompt_tokens": prompt_len, "sec": round(sec, 3),
    }
    step_rows = []
    for rec in steplog:
        d = {"task_id": task_id, "seed": SEED_USED, "remasking": remasking, "num_steps": num_steps}
        d.update(rec)
        d.update({"delta_words": None, "delta_order": None, "pc_active_steps": 0})
        step_rows.append(d)
    return row, step_rows


def result_path(remasking, num_steps):
    return os.path.join(OUT_DIR, remasking + "_steps" + str(num_steps) + "_seed" + str(SEED_USED) + ".jsonl")


def steplog_path(remasking, num_steps):
    return os.path.join(OUT_DIR, "steplog_" + remasking + "_steps" + str(num_steps) + "_seed" + str(SEED_USED) + ".jsonl")


def read_jsonl_tolerant(path):
    # Читает только полные строки (оканчивающиеся на \n). Оборванный хвост (обрыв во время записи) отбрасывается и физически
    # обрезается; повреждённая ПОЛНАЯ строка - ошибка (автоматически не чиним).
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    if tail:
        if tail.strip():
            log("ВНИМАНИЕ: " + path + ": оборванная последняя строка (" + str(len(tail)) + " байт) отброшена")
        with open(path, "r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
    rows = []
    for line_no, raw in enumerate(data[:cut].split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except Exception as e:
            raise RuntimeError(path + ": строка " + str(line_no) + " повреждена (" + type(e).__name__ + ": " + str(e) + "). Исправьте или удалите файл вручную.") from e
    return rows


def append_jsonl(path, records):
    payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    with open(path, "a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def repair_steplog(path, done_ids):
    rows = read_jsonl_tolerant(path)
    keep = [r for r in rows if r["task_id"] in done_ids]
    if len(keep) != len(rows):
        dropped = sorted({r["task_id"] for r in rows if r["task_id"] not in done_ids})
        log("steplog " + os.path.basename(path) + ": удалены строки незавершённых задач " + str(dropped))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return keep


def validate_result_row(row, remasking, num_steps):
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    assert not missing, "в строке результата нет полей " + str(missing) + ": " + str(row.get("task_id"))
    assert row["remasking"] == remasking and row["num_steps"] == num_steps, "строка не из этой комбинации: " + str(row["remasking"]) + "/" + str(row["num_steps"])
    assert row["seed"] == SEED_USED and row["pc_active_steps"] == 0 and row["delta_words"] is None and row["delta_order"] is None
    assert row["correct"] in (0, 1) and isinstance(row["prediction"], str) and isinstance(row["task_id"], int)
    assert row.get("gen_length", GEN_LENGTH) == GEN_LENGTH, "в файле результаты с gen_length=" + str(row.get("gen_length")) + ", а GEN_LENGTH=" + str(GEN_LENGTH)


def run_combo(remasking, num_steps, instances_to_run, deadline=None):
    # Возвращает "done", "stopped" (deadline = MAX_RUNTIME_MIN) или "signal" (получен SIGTERM/SIGINT). Остановка - только ПОСЛЕ
    # полностью записанной задачи (steplog, затем результат, fsync), поэтому кэш всегда согласован.
    rpath, spath = result_path(remasking, num_steps), steplog_path(remasking, num_steps)
    rows = read_jsonl_tolerant(rpath)
    for r in rows:
        validate_result_row(r, remasking, num_steps)
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids)), rpath + ": повторяющиеся task_id"
    done = set(ids)
    repair_steplog(spath, done)
    todo = [inst for inst in instances_to_run if inst.doc_id not in done]
    log("[" + remasking + ", steps=" + str(num_steps) + "] уже готово " + str(len(done)) + ", осталось " + str(len(todo)))
    for k, inst in enumerate(todo, 1):
        row, step_rows = process_task(inst, remasking, num_steps)
        append_jsonl(spath, step_rows)    # 1) журнал шагов
        append_jsonl(rpath, [row])        # 2) строка результата = маркер завершения задачи
        torch.cuda.empty_cache()          # между задачами
        log("[" + remasking + "/" + str(num_steps) + "] задача " + str(inst.doc_id) + " (" + str(k) + "/" + str(len(todo)) + ") correct=" + str(row["correct"]) + " pred=" + repr(row["prediction"]) + " gold=" + repr(row["gold"]) + " " + format(row["sec"], ".1f") + " c")
        if STOP_REQUESTED["flag"]:
            log("[" + remasking + ", steps=" + str(num_steps) + "] остановка по сигналу после задачи " + str(inst.doc_id))
            return "signal"
        if deadline is not None and time.time() >= deadline:
            log("[" + remasking + ", steps=" + str(num_steps) + "] остановка по MAX_RUNTIME_MIN после задачи " + str(inst.doc_id))
            return "stopped"
    return "done"


def validate_combo(remasking, num_steps, expected_ids):
    rows = read_jsonl_tolerant(result_path(remasking, num_steps))
    for r in rows:
        validate_result_row(r, remasking, num_steps)
    ids = [r["task_id"] for r in rows]
    assert sorted(ids) == sorted(expected_ids), remasking + "/" + str(num_steps) + ": ожидались task_id " + str(len(expected_ids)) + " шт., в файле " + str(len(ids)) + " (повторы или пропуски)"
    steps_rows = read_jsonl_tolerant(steplog_path(remasking, num_steps))
    per_task = {}
    for s in steps_rows:
        per_task.setdefault(s["task_id"], []).append(s)
    assert set(per_task) == set(ids), remasking + "/" + str(num_steps) + ": steplog и результаты содержат разные task_id"
    for tid, recs in per_task.items():
        assert len(recs) == num_steps, remasking + "/" + str(num_steps) + ", задача " + str(tid) + ": в steplog " + str(len(recs)) + " шагов вместо " + str(num_steps)
        assert [r["global_step"] for r in recs] == list(range(num_steps)), remasking + "/" + str(num_steps) + ", задача " + str(tid) + ": нарушен порядок шагов"
    return rows


# ===================== 7. ДЫМОВОЙ ТЕСТ =====================
def assert_same(name_a, a, name_b, b, prompt_len, where, hint=""):
    if a.shape == b.shape and bool((a == b).all()):
        return
    if a.shape == b.shape:
        diff_pos = (a != b).nonzero()[:, 1].tolist()
        first = diff_pos[0]
        detail = ("позиций отличается: " + str(len(diff_pos)) + "; первая: " + str(first) + " (позиция " + str(first - prompt_len) + " в ответе); токены "
                  + str(a[0, first].item()) + " (" + name_a + ") и " + str(b[0, first].item()) + " (" + name_b + ")")
    else:
        detail = "формы различаются: " + str(tuple(a.shape)) + " и " + str(tuple(b.shape))
    raise AssertionError("РАСХОЖДЕНИЕ [" + where + "]: " + name_a + " != " + name_b + ". " + detail + ". " + hint + "Остановка: полный прогон запрещён, пока причина не найдена.")


def check_steplog(steplog, x, prompt_len, gen_length, block_length, num_steps, where):
    num_blocks = gen_length // block_length
    steps = num_steps // num_blocks
    assert len(steplog) == num_steps, "[" + where + "] в журнале " + str(len(steplog)) + " шагов вместо " + str(num_steps)
    seen = set()
    revealed_in_block = {}
    for k, rec in enumerate(steplog):
        b, i = divmod(k, steps)
        assert (rec["block"], rec["step_in_block"], rec["global_step"]) == (b, i, k), "[" + where + "] нарушен порядок шагов на " + str(k)
        start, end = prompt_len + b * block_length, prompt_len + (b + 1) * block_length
        expected_n = block_length // steps + (1 if i < block_length % steps else 0)
        pos = rec["positions_abs"]
        assert len(pos) == expected_n, "[" + where + "] шаг " + str(k) + ": раскрыто " + str(len(pos)) + " позиций вместо " + str(expected_n)
        assert all(start <= p < end for p in pos), "[" + where + "] шаг " + str(k) + ": позиции вне текущего блока [" + str(start) + ", " + str(end) + "): " + str(pos)
        assert not (set(pos) & seen), "[" + where + "] шаг " + str(k) + ": позиция раскрыта повторно"
        seen |= set(pos)
        assert rec["positions_rel"] == [p - prompt_len for p in pos], "[" + where + "] шаг " + str(k) + ": positions_rel не согласованы"
        assert rec["token_ids"] == [x[0, p].item() for p in pos], "[" + where + "] шаг " + str(k) + ": токены журнала не совпали с результатом"
        done_before = revealed_in_block.get(b, 0)
        assert rec["n_masked_block_before"] == block_length - done_before, "[" + where + "] шаг " + str(k) + ": неверное число масок в блоке"
        revealed_in_block[b] = done_before + len(pos)
        conf = rec["confidence"]
        assert all(math.isfinite(c) and -1e-2 <= c <= 1 + 1e-2 for c in conf), "[" + where + "] шаг " + str(k) + ": уверенность вне [0, 1] или не конечна: " + str(conf)
        assert all(conf[t] >= conf[t + 1] for t in range(len(conf) - 1)), "[" + where + "] шаг " + str(k) + ": уверенность не убывает внутри шага"
    assert seen == set(range(prompt_len, prompt_len + gen_length)), "[" + where + "] раскрыты не все позиции ответа"


def run_smoke(gpu_index):
    import io
    import contextlib
    smoke_n, smoke_steps, smoke_gen, smoke_block = 3, 8, 64, 32
    assert smoke_gen % smoke_block == 0 and smoke_steps % (smoke_gen // smoke_block) == 0
    smoke_insts = INSTANCES[:smoke_n]

    log("A. Проверка пути оценки на синтетических ответах")
    for inst in smoke_insts:
        gold_text = inst.doc["answer"]
        s_ok, p_ok = score_response(inst, gold_text)
        assert s_ok["flexible-extract"] == 1.0 and s_ok["strict-match"] == 1.0, "эталонный ответ задачи " + str(inst.doc_id) + " оценён как неверный: " + str(p_ok) + " - путь оценки сломан"
        gold_num = gold_final(gold_text).replace(",", "")
        try:
            wrong_num = str(int(float(gold_num)) + 1)
        except ValueError:
            log("   задача " + str(inst.doc_id) + ": эталон не число, проверка искажённого ответа пропущена")
            continue
        wrong_text = gold_text.rsplit("####", 1)[0] + "#### " + wrong_num
        s_bad, p_bad = score_response(inst, wrong_text)
        assert s_bad["flexible-extract"] == 0.0 and s_bad["strict-match"] == 0.0, "искажённый ответ задачи " + str(inst.doc_id) + " оценён как верный: " + str(p_bad) + " - путь оценки сломан"
        log("   задача " + str(inst.doc_id) + ": эталон -> верно (" + repr(p_ok["flexible-extract"]) + "); искажённый -> неверно (" + repr(p_bad["flexible-extract"]) + ")")

    log("B. Проверка промпта")
    for inst in smoke_insts:
        ctx_text = inst.args[0]
        assert SYSTEM_INSTRUCTION in ctx_text, "в промпте задачи " + str(inst.doc_id) + " нет system-инструкции"
        assert inst.doc["question"] in ctx_text, "в промпте задачи " + str(inst.doc_id) + " нет текста вопроса"
        manual = lm.apply_chat_template([{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": task.doc_to_text(inst.doc)}], add_generation_prompt=True)
        log("   задача " + str(inst.doc_id) + ": совпадает с ручной сборкой system+user через chat template: " + str(manual == ctx_text))

    def quiet_call(fn, *a, **kw):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*a, **kw)

    log("C0. Логиты конечны (защита от NaN/inf; особенно важно при DTYPE=float16)")
    for inst in smoke_insts:
        p_len = len(PROMPT_IDS[inst.doc_id])
        probe = torch.full((1, p_len + smoke_gen), MASK_ID, dtype=torch.long, device=model.device)
        probe[0, :p_len] = torch.tensor(PROMPT_IDS[inst.doc_id], dtype=torch.long, device=model.device)
        with torch.no_grad():
            lg = model(probe).logits
        assert bool(torch.isfinite(lg).all()), "В логитах задачи " + str(inst.doc_id) + " есть NaN/inf при DTYPE=" + DTYPE + ". Остановка."
        log("   задача " + str(inst.doc_id) + ": логиты конечны, форма " + str(tuple(lg.shape)) + ", dtype " + str(lg.dtype))
        del probe, lg
        torch.cuda.empty_cache()

    log("C. Токены: копия против оригинала (GPU)")
    samples = {}
    for remasking in REMASKING_LIST:
        for inst in smoke_insts:
            prompt = torch.tensor([PROMPT_IDS[inst.doc_id]], dtype=torch.long, device=model.device)
            P = prompt.shape[1]
            common = dict(num_steps=smoke_steps, gen_length=smoke_gen, block_length=smoke_block, temperature=TEMPERATURE, cfg_scale=0., remasking=remasking, mask_id=MASK_ID)
            where = "remasking=" + remasking + ", задача " + str(inst.doc_id)
            x_copy, log_copy = llada_generate_logged(model, prompt, use_pc=False, **common)
            x_copy_2, _ = llada_generate_logged(model, prompt, use_pc=False, **common)
            x_copy_plain, _ = llada_generate_logged(PLAIN, prompt, use_pc=False, **common)
            with torch.no_grad():
                x_orig_plain = quiet_call(llada_diffusion_generate, PLAIN, prompt, **common)
                x_orig_codd = quiet_call(llada_diffusion_generate, model, prompt, use_pc=False, **common)
            assert_same("копия (прогон 2)", x_copy_2, "копия (прогон 1)", x_copy, P, where, hint="Копия расходится сама с собой: недетерминизм GPU/ядер, а не ошибка логики. ")
            assert_same("копия на PlainView", x_copy_plain, "копия на CoddLlada", x_copy, P, where)
            assert_same("копия на PlainView", x_copy_plain, "оригинал на PlainView", x_orig_plain, P, where)
            assert_same("копия на CoddLlada", x_copy, "оригинал на CoddLlada(use_pc=False)", x_orig_codd, P, where)
            assert bool((x_copy[:, :P] == prompt).all()), "[" + where + "] промпт в результате изменился"
            assert not bool((x_copy[:, P:] == MASK_ID).any()), "[" + where + "] в ответе остались маски"
            check_steplog(log_copy, x_copy, P, smoke_gen, smoke_block, smoke_steps, where)
            samples[(remasking, inst.doc_id)] = tokenizer.decode(x_copy[0, P:].tolist(), skip_special_tokens=True)
            log("   OK [" + where + "]: 5 прогонов совпали по токенам, журнал согласован")
            torch.cuda.empty_cache()
    log("Пример ответа дымового теста (задача 0, " + REMASKING_LIST[0] + ", 8 шагов, 64 токена; текст может быть бессмысленным - шагов мало):")
    log(repr(samples[(REMASKING_LIST[0], 0)][:300]))
    log("ДЫМОВОЙ ТЕСТ (корректность): ПРОЙДЕН")

    # --- пиковая видеопамять при РЕАЛЬНОМ GEN_LENGTH на самой длинной по токенам задаче ---
    gib = 1024 ** 3
    longest = max(INSTANCES, key=lambda i: len(PROMPT_IDS[i.doc_id]))
    total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
    peak_report = {}
    for rm in REMASKING_LIST:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_generation(longest, rm, NUM_BLOCKS)
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()
        peak_report[rm] = {"peak_allocated_gib": peak_alloc / gib, "peak_reserved_gib": peak_res / gib, "headroom_gib": (total_mem - peak_alloc) / gib}
        log("Пик памяти [" + rm + "], самая длинная задача (" + str(len(PROMPT_IDS[longest.doc_id])) + " ток. промпта + " + str(GEN_LENGTH) + " ответа): allocated " + format(peak_alloc / gib, ".2f")
            + " ГиБ, reserved " + format(peak_res / gib, ".2f") + " ГиБ из " + format(total_mem / gib, ".2f") + " ГиБ; запас (total - allocated) " + format((total_mem - peak_alloc) / gib, ".2f") + " ГиБ")
        assert total_mem - peak_alloc >= 2 * gib, "Запас видеопамяти " + format((total_mem - peak_alloc) / gib, ".2f") + " ГиБ < 2 ГиБ при remasking=" + rm + ", GEN_LENGTH=" + str(GEN_LENGTH) + ", DTYPE=" + DTYPE
        if total_mem - peak_res < 2 * gib:
            log("   ВНИМАНИЕ: с учётом кэша аллокатора (reserved) запас " + format((total_mem - peak_res) / gib, ".2f") + " ГиБ < 2 ГиБ; empty_cache между задачами включён.")
        torch.cuda.empty_cache()

    # --- секунды на задачу и прогноз ---
    timing_n = 2
    timing_remasking = REMASKING_LIST[0]
    timing_steps = NUM_STEPS_LIST[0]
    secs = []
    for inst in INSTANCES[:timing_n]:
        _x, _log, _plen, sec = run_generation(inst, timing_remasking, timing_steps)
        secs.append(sec)
        torch.cuda.empty_cache()
    sec_per_task = sum(secs) / len(secs)
    sec_per_forward = sec_per_task / timing_steps
    log("Замер: " + str(timing_n) + " задачи, remasking=" + timing_remasking + ", num_steps=" + str(timing_steps) + ", GEN_LENGTH=" + str(GEN_LENGTH))
    log("Секунд на задачу: " + format(sec_per_task, ".2f") + " (" + str(timing_steps) + " шагов), на один проход модели: " + format(sec_per_forward, ".3f") + " c")
    n_eff = RUN_LIMIT if RUN_LIMIT is not None else N_TASKS
    per_combo = {}
    log("Прогноз времени (пропорционально числу проходов модели; оценка, НЕ гарантия; задач на комбинацию: " + str(n_eff) + "):")
    for rm in REMASKING_LIST:
        for st in NUM_STEPS_LIST:
            per_combo[(rm, st)] = sec_per_forward * st * n_eff
            log("   " + rm.ljust(16) + " steps=" + str(st).ljust(4) + " ~" + format(sec_per_forward * st, "6.1f") + " c/задачу, ~" + str(datetime.timedelta(seconds=int(per_combo[(rm, st)]))) + " на комбинацию")
    total_forecast = sum(per_combo.values())
    log("ИТОГО на " + str(len(per_combo)) + " комбинации x " + str(n_eff) + " задач: ~" + str(datetime.timedelta(seconds=int(total_forecast))) + " (~" + format(total_forecast / 3600, ".1f") + " ч)")
    if MAX_RUNTIME_MIN is not None:
        log("MAX_RUNTIME_MIN=" + str(MAX_RUNTIME_MIN) + ": за один запуск будет сделано примерно " + format(MAX_RUNTIME_MIN * 60 / total_forecast * 100, ".0f") + "% полного прогона (оценка).")
    write_json_atomic(SMOKE_PATH, {
        "ok": True, "finished_utc": utc_now(), "exp_sha256": EXP_SHA256, "gen_length": GEN_LENGTH,
        "sec_per_task_first_steps": sec_per_task, "sec_per_forward": sec_per_forward,
        "forecast_seconds_total": total_forecast, "forecast_tasks_per_combo": n_eff,
        "pyjuice_mode": PYJUICE_MODE, "dtype": DTYPE, "gpu_name": ENV_INFO.get("gpu_name"), "gpu_capability": ENV_INFO.get("gpu_capability"),
        "peak_memory": peak_report, "total_gpu_memory_gib": total_mem / gib,
    })
    log("smoke.json записан: полный прогон разрешён.")


# ===================== 8. ПОЛНЫЙ ПРОГОН =====================
def check_smoke_done():
    if not REQUIRE_SMOKE:
        log("REQUIRE_SMOKE=false: проверка smoke.json пропущена")
        return
    assert os.path.exists(SMOKE_PATH), "Нет " + SMOKE_PATH + ": сначала запустите режим smoke и дождитесь успешного завершения"
    with open(SMOKE_PATH, encoding="utf-8") as f:
        sm = json.load(f)
    assert sm.get("ok") is True, "smoke.json: ok != true"
    assert sm.get("exp_sha256") == EXP_SHA256, "smoke.json получен при другом конфиге (exp_sha256 не совпал): повторите smoke"
    assert sm.get("dtype") == DTYPE and sm.get("gen_length") == GEN_LENGTH, "smoke.json получен при другом dtype/GEN_LENGTH: повторите smoke"
    log("smoke.json: успешный дымовой тест от " + str(sm.get("finished_utc")) + " найден")


def run_full():
    run_instances = INSTANCES if RUN_LIMIT is None else INSTANCES[:RUN_LIMIT]
    if RUN_LIMIT is not None:
        log("=" * 90)
        log("ВНИМАНИЕ: RUN_LIMIT=" + str(RUN_LIMIT) + ". Будут обработаны только первые " + str(RUN_LIMIT) + " задач из " + str(N_TASKS) + ". ПРОГОН НЕПОЛНЫЙ.")
        log("=" * 90)
    log("GEN_LENGTH=" + str(GEN_LENGTH) + ", BLOCK_LENGTH=" + str(BLOCK_LENGTH) + ", задач: " + str(len(run_instances)) + ", комбинаций: " + str(len(REMASKING_LIST) * len(NUM_STEPS_LIST)))
    t_all = time.time()
    deadline = None if MAX_RUNTIME_MIN is None else t_all + 60 * MAX_RUNTIME_MIN
    if deadline is not None:
        log("MAX_RUNTIME_MIN=" + str(MAX_RUNTIME_MIN) + ": после завершённой задачи, на которой истечёт лимит, прогон корректно остановится.")
    stopped = None
    for remasking in REMASKING_LIST:
        for num_steps in NUM_STEPS_LIST:
            if STOP_REQUESTED["flag"]:
                stopped = "signal"
                break
            if deadline is not None and time.time() >= deadline:
                stopped = "stopped"
                break
            status = run_combo(remasking, num_steps, run_instances, deadline=deadline)
            if status != "done":
                stopped = status
                break
        if stopped:
            break
    log("Цикл прогона завершён за " + str(datetime.timedelta(seconds=int(time.time() - t_all))) + " (в этом запуске), остановка: " + str(stopped))

    for remasking in REMASKING_LIST:
        for num_steps in NUM_STEPS_LIST:
            present = sorted(r["task_id"] for r in read_jsonl_tolerant(result_path(remasking, num_steps)))
            if stopped:
                validate_combo(remasking, num_steps, present)      # прогон неполный: проверяем только согласованность готового
            elif RUN_LIMIT is None:
                validate_combo(remasking, num_steps, TASK_IDS)
            else:
                wanted = [inst.doc_id for inst in run_instances]
                assert set(wanted) <= set(present), remasking + "/" + str(num_steps) + ": нет результатов для задач " + str(sorted(set(wanted) - set(present)))
                validate_combo(remasking, num_steps, present)
            log("   проверено: " + os.path.basename(result_path(remasking, num_steps)) + ": " + str(len(present)) + " задач, steplog согласован")
    if stopped:
        return "STOPPED", "остановлено (" + stopped + "); чтобы продолжить - запустите режим full ещё раз, готовые задачи будут пропущены"
    if RUN_LIMIT is None:
        return "DONE", "все " + str(len(REMASKING_LIST) * len(NUM_STEPS_LIST)) + " комбинации по " + str(N_TASKS) + " задач"
    return "DONE", "НЕПОЛНЫЙ прогон (RUN_LIMIT=" + str(RUN_LIMIT) + ")"


# ===================== 9. СВОДКА =====================
def bootstrap_ci(values, n_boot=None, seed=None, alpha=0.05):
    # Percentile bootstrap по задачам: ресэмплируем задачи с возвратом, считаем долю верных.
    n_boot = N_BOOTSTRAP if n_boot is None else n_boot
    seed = BOOTSTRAP_SEED if seed is None else seed
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and v.size > 0, "пустой набор значений"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def run_summary():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids_sha = None
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, encoding="utf-8") as f:
            ids_sha = json.load(f).get("ids_sha256")
    rows_out = []
    for remasking in REMASKING_LIST:
        for num_steps in NUM_STEPS_LIST:
            rows = sorted(read_jsonl_tolerant(result_path(remasking, num_steps)), key=lambda r: r["task_id"])
            if not rows:
                log("нет результатов для " + remasking + "/" + str(num_steps))
                continue
            acc, lo, hi = bootstrap_ci([r["correct"] for r in rows])
            acc_s, lo_s, hi_s = bootstrap_ci([r["correct_strict"] for r in rows])
            rows_out.append({"remasking": remasking, "num_steps": num_steps, "n": len(rows), "accuracy": acc, "ci_low": lo, "ci_high": hi,
                             "accuracy_strict": acc_s, "ci_low_strict": lo_s, "ci_high_strict": hi_s})
    log("exp1 (обычная LLaDA, HMM не используется) | GSM8K test, первые " + str(N_TASKS) + " задач | GEN_LENGTH=" + str(GEN_LENGTH) + ", BLOCK_LENGTH=" + str(BLOCK_LENGTH) + ", greedy | bootstrap: " + str(N_BOOTSTRAP) + " повторов, 95%")
    log("sha256 списка id задач: " + str(ids_sha))
    header = "remasking".ljust(16) + "steps".rjust(6) + "n".rjust(5) + "   " + "accuracy (flexible-extract)".ljust(30) + "accuracy (strict-match)"
    print(header, flush=True)
    print("-" * (len(header) + 6), flush=True)
    for s in rows_out:
        flag = "" if s["n"] == N_TASKS else "  <- НЕПОЛНО"
        a = format(100 * s["accuracy"], "5.1f") + "% [" + format(100 * s["ci_low"], "5.1f") + "; " + format(100 * s["ci_high"], "5.1f") + "]"
        b = format(100 * s["accuracy_strict"], "5.1f") + "% [" + format(100 * s["ci_low_strict"], "5.1f") + "; " + format(100 * s["ci_high_strict"], "5.1f") + "]"
        print(s["remasking"].ljust(16) + str(s["num_steps"]).rjust(6) + str(s["n"]).rjust(5) + "   " + a.ljust(30) + b + flag, flush=True)
    print("Доверительный интервал отражает только выбор задач (разброс по задачам), а не случайность генерации: случайности нет.", flush=True)
    complete = (len(rows_out) == len(REMASKING_LIST) * len(NUM_STEPS_LIST)) and all(s["n"] == N_TASKS for s in rows_out)
    write_json_atomic(SUMMARY_PATH, {
        "experiment": EXP_NAME, "gen_length": GEN_LENGTH, "block_length": BLOCK_LENGTH, "n_tasks": N_TASKS, "ids_sha256": ids_sha,
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "complete": complete, "rows": rows_out, "updated_utc": utc_now(),
    })
    log("summary.json записан: " + SUMMARY_PATH + " | полный: " + str(complete))
    return complete


# ===================== MAIN =====================
def main():
    if MODE == "summary":
        set_status("running", "summary")
        run_summary()
        set_status("done", "summary")
        return 0
    set_status("running", "запуск режима " + MODE)
    if MODE == "full":
        signal.signal(signal.SIGTERM, _on_stop_signal)
        signal.signal(signal.SIGINT, _on_stop_signal)
    gpu_index, _total = check_environment()
    prepare_pyjuice()
    import_repo_code()
    write_env_and_config()
    if MODE == "full":
        check_smoke_done()
    load_model()
    check_model_code()
    setup_tasks()
    if MODE == "smoke":
        run_smoke(gpu_index)
        set_status("done", "smoke пройден")
        return 0
    state, message = run_full()
    run_summary()
    set_status(state.lower(), message)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        try:
            set_status("failed", type(exc).__name__ + ": " + str(exc)[:500])
        except Exception:
            pass
        sys.exit(1)
