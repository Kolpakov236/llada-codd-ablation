# f_core_template.py - ШАБЛОН ядра полных прогонов f1-f5 (GSM8K test 1319 задач, 512 токенов; f1 - без HMM, f2-f5 - HMM поверх замороженной LLaDA-8B-Instruct).
# Получен из exp_core_template.py (Э2-Э5, 100 задач, 256 токенов), который не менялся.
# Сборщик build_experiments.py подставляет параметры варианта в строку EXPERIMENT и пишет exp2_core.py ... exp5_core.py
# (каждый файл самодостаточен). Ядро запускается ОТДЕЛЬНЫМ процессом из окружения venv:
#   env MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#       /home/jupyter/project/venv/bin/python -I -u expN_core.py --mode preflight|pctest|smoke|full|summary
# Параметры читаются из config.json рядом со скриптом (его пишет блокнот). Интернета нет.
#
# Варианты (A - логиты, из которых берутся слова-кандидаты; C - логиты, из которых берётся уверенность и порядок):
#   Э2: A = logits_base, C = logits_pc   Э3: A = logits_pc, C = logits_base   Э5: A = C = logits_pc (тема одна на блок)
#   Э4: A = C = logits_pc, но тема каждой клетки выбирается НЕЗАВИСИМО из её маргинала (без связей между клетками)
# logits_base - логиты LLaDA до HMM, logits_pc - после apply_pc_logits (или её аналога для Э4). Один сэмпл темы на шаг для обоих наборов.
#
# Совместимость: Python 3.10 (без вложенных одинаковых кавычек в f-строках, без match).
# НЕ ПРОВЕРЕНО: ядро ни разу не запускалось (проверены только синтаксис и чистые функции на синтетических данных).

import argparse
import copy
import datetime
import difflib
import functools
import glob
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import traceback
import types
from pathlib import Path

import numpy as np

EXPERIMENT = {'EXP_NAME': 'f5_codd_full512', 'WORDS_FROM': 'pc', 'ORDER_FROM': 'pc', 'THEME_MODE': 'joint', 'USE_PC': True}  # подставлено сборщиком build_full512.py

EXPECTED_COMMON_SHA256 = "404c3db6d6f737fb72158943ee43e3c7effcb7871e81a8dfba6492b8152361c0"   # N_TASKS=200 случайных из 1319 (seed выбора 3), GEN_LENGTH=512, steps [128], margin, SEEDS [3]
EXPECTED_IDS_SHA256 = "42c43df159e0927be57f182036ed0d7cee10430d4d50fa22d1fbf43da00ab2e9"   # отсортированные id 200 выбранных задач (seed выбора 3)
EXP1_PEAK_GIB = 16.2          # пик видеопамяти exp1 при GEN_LENGTH=256 (проверено на сервере); при 512 пик больше - это лишь оценка в pctest, реальный пик измерит smoke
GIB = 1024 ** 3


# ===================== 0. УТИЛИТЫ =====================
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


def derive_seed(seed, task_id):
    # Детерминированный сид из (seed, task_id): им сидируются random, numpy, torch и cuda перед каждой задачей.
    # Побитовая воспроизводимость на GPU НЕ гарантируется (недетерминизм ядер).
    return (int(seed) * 1000003 + int(task_id) * 7919 + 12345) % (2 ** 31)


# ===================== 1. ЧИСТЫЕ ФУНКЦИИ (проверяются без GPU) =====================
def select_task_ids(test_size, n, sample_seed):
    # Детерминированный "случайный" выбор n задач из test_size, не зависящий от версий Python/numpy:
    # id сортируются по sha256("seed:id"), берутся первые n, результат сортируется по возрастанию id.
    keyed = sorted(range(test_size), key=lambda i: hashlib.sha256(("%d:%d" % (sample_seed, i)).encode("utf-8")).hexdigest())
    return sorted(keyed[:n])


def transfer_schedule(block_length, steps):
    # как get_num_transfer_tokens из оригинала: base токенов за шаг, первые remainder шагов - на один больше
    base, rem = divmod(block_length, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def count_active_steps(num_steps, gen_length, block_length, pc_frac):
    # Сколько шагов генерации HMM активна: условие mask_ratio < pc_frac считается по состоянию блока на начало шага.
    num_blocks = gen_length // block_length
    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks
    m = block_length
    per_block = 0
    for k in transfer_schedule(block_length, steps):
        if (m / float(block_length)) < pc_frac:
            per_block += 1
        m -= k
    return per_block * num_blocks


def jaccard_delta_order(set_base, set_pc):
    # delta_order = 1 - |S_base ∩ S_pc| / |S_base ∪ S_pc| (пустые множества: 0.0)
    a, b = set(set_base), set(set_pc)
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / float(len(union))


def delta_words_value(base_ids, pc_ids, masked_flags):
    # Доля ЗАКРЫТЫХ клеток блока, где argmax(logits_base) != argmax(logits_pc). Возвращает (доля, число закрытых, число отличий).
    n = 0
    diff = 0
    for b, p, m in zip(base_ids, pc_ids, masked_flags):
        if m:
            n += 1
            if int(b) != int(p):
                diff += 1
    return (diff / float(n) if n else 0.0), n, diff


def mcnemar_exact(b, c):
    # Точный двусторонний тест Макнемара: b, c - числа несогласных пар (A верно/B неверно и наоборот).
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    p = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * p)


def bootstrap_ci(values, n_boot=10000, seed=12345, alpha=0.05):
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and v.size > 0, "пустой набор значений"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot=10000, seed=12345, alpha=0.05):
    # Парный bootstrap по задачам: разность mean(a) - mean(b) с интервалом.
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    assert d.ndim == 1 and d.size > 0, "пустой набор значений"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def bin_by_top_states(samples, top_states):
    out = np.full(np.asarray(samples).shape, len(top_states), dtype=np.int64)
    for b, s in enumerate(top_states):
        out[np.asarray(samples) == s] = b
    return out


def mutual_info_plugin(x, y, nbins):
    n = len(x)
    joint = np.zeros((nbins, nbins), dtype=np.float64)
    np.add.at(joint, (np.asarray(x), np.asarray(y)), 1.0)
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    prod = px @ py
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / prod[nz])).sum())


def mi_permutation_z(x, y, nbins, n_perm, rng):
    # plug-in MI и z-оценка относительно перестановочного нуля (нулевая гипотеза: независимость). Смещение MI при малом числе сэмплов
    # одинаково для данных и перестановок, поэтому z осмысленнее самого MI.
    mi = mutual_info_plugin(x, y, nbins)
    nulls = np.empty(n_perm, dtype=np.float64)
    y = np.asarray(y)
    for i in range(n_perm):
        nulls[i] = mutual_info_plugin(x, rng.permutation(y), nbins)
    sd = float(nulls.std())
    z = (mi - float(nulls.mean())) / sd if sd > 0 else 0.0
    return mi, float(nulls.mean()), sd, float(z)


def read_jsonl(path, repair=False):
    # repair=False: ТОЛЬКО чтение (оборванный хвост игнорируется, файл не меняется) - для чужих результатов (exp1, exp5).
    # repair=True: оборванный хвост физически обрезается (для своих файлов). Повреждённая ПОЛНАЯ строка - всегда ошибка.
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    if tail and repair:
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


def read_jsonl_tolerant(path):
    return read_jsonl(path, repair=True)


def append_jsonl(path, records):
    payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    with open(path, "a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


# ---------- безопасное чтение структуры .jpc (pickle) без исполнения кода ----------
class _Stub:
    # Инертная заглушка для ЛЮБОГО неизвестного класса/функции из pickle: ничего не исполняет, только запоминает аргументы.
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        self._state = state


_BUILTIN_OK = set(("builtins", n) for n in ("set", "frozenset", "tuple", "list", "dict", "int", "float", "str", "bytes", "bytearray", "complex", "slice", "range"))
_BUILTIN_OK.add(("collections", "OrderedDict"))
_NUMPY_NAMES = set([("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
                    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")])


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _BUILTIN_OK:
            return getattr(importlib.import_module(module), name)
        if (module, name) in _NUMPY_NAMES:
            if name == "ndarray":
                return np.ndarray
            if name == "dtype":
                return np.dtype
            mod = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
            return getattr(mod, name)
        return type(name, (_Stub,), {"_origin": module + "." + name})


def inspect_jpc(path):
    with open(path, "rb") as f:
        nodes_list = SafeUnpickler(f).load()
    assert isinstance(nodes_list, list), "ожидался список словарей, получено " + type(nodes_list).__name__
    s = {"num_node_sets": len(nodes_list), "by_type": {}, "total_param_count": 0, "total_nodes_input": 0, "total_nodes_sum": 0, "total_nodes_prod": 0,
         "param_dtypes": {}, "input_scope_vars": 0, "dist_classes": {}}
    scope_vars = set()
    for ns in nodes_list:
        t = ns["type"]
        nodes = (ns.get("num_node_blocks") or 0) * (ns.get("block_size") or 1)
        s["by_type"][t] = s["by_type"].get(t, 0) + 1
        if t == "Input":
            s["total_nodes_input"] += nodes
            scope_vars.update(int(v) for v in np.asarray(ns["scope"]).ravel().tolist())
            d = SafeUnpickler(io.BytesIO(ns["dist"])).load()
            origin = getattr(d, "_origin", type(d).__name__)
            s["dist_classes"][origin] = s["dist_classes"].get(origin, 0) + 1
        elif t == "Sum":
            s["total_nodes_sum"] += nodes
        else:
            s["total_nodes_prod"] += nodes
        if "params" in ns:
            arr = ns["params"]
            s["total_param_count"] += int(arr.size)
            s["param_dtypes"][str(arr.dtype)] = s["param_dtypes"].get(str(arr.dtype), 0) + int(arr.size)
    s["input_scope_vars"] = len(scope_vars)
    s["input_nodes_per_var_if_uniform"] = (s["total_nodes_input"] / float(len(scope_vars))) if scope_vars else None
    s["params_fp32_gib_lower_bound"] = s["total_param_count"] * 4 / float(GIB)
    return s


# ===================== 2. КОНФИГ =====================
MODE = None
CFG = None
MODEL = None
HMM_CKPT = None
DATASET = None
DATASET_SPLIT = None
N_TASKS = None
TEST_SIZE = None
TASK_SAMPLE_SEED = None
GEN_LENGTH = None
BLOCK_LENGTH = None
TEMPERATURE = None
NUM_STEPS_LIST = None
REMASKING_LIST = None
PC_TEMPERATURE = None
PC_FRAC = None
REVERSE_FRAC = None
SEEDS = None
EXP_NAME = None
WORDS_FROM = None
ORDER_FROM = None
THEME_MODE = None
USE_PC = None
E4_IMPL = None
E4B_WORD_MARGINAL = None
SYSTEM_INSTRUCTION = None
NUM_FEWSHOT = None
TASK_NAME = None
MASK_ID = None
DTYPE = None
ATTN_IMPLEMENTATION = None
N_BOOTSTRAP = None
BOOTSTRAP_SEED = None
RUN_SEEDS = None
RUN_LIMIT = None
MAX_RUNTIME_MIN = None
PROJECT_DIR = None
CODD_REPO_DIR = None
MODEL_PATH = None
DATA_DIR = None
OUT_DIR = None
HMM_DIR = None
HMM_MANIFEST = None
EXP1_RESULTS_DIR = None
EXP5_RESULTS_DIR = None
ALLOW_MODEL_CODE_MISMATCH = None
STRICT_VERSIONS = None
MIN_GPU_TOTAL_GB = None
MIN_GPU_FREE_GB = None
REQUIRE_SMOKE = None
REQUIRE_PCTEST = None
E4_N_SAMPLES = None
HMM_FILE = None
START_UTC = None
COMMON_PARAMS = None
EXP_PARAMS = None
COMMON_SHA256 = None
EXP_SHA256 = None
HMM_SHA256 = None
ENV_PATH = CONFIG_PATH = TASKS_PATH = SMOKE_PATH = SUMMARY_PATH = STATUS_PATH = PREFLIGHT_PATH = PCTEST_PATH = E4_FLAG_PATH = None

_REQUIRED_KEYS = [
    "MODEL", "HMM_CKPT", "DATASET", "DATASET_SPLIT", "N_TASKS", "TEST_SIZE", "TASK_SAMPLE_SEED", "GEN_LENGTH", "BLOCK_LENGTH", "TEMPERATURE", "NUM_STEPS_LIST", "REMASKING_LIST",
    "PC_TEMPERATURE", "PC_FRAC", "REVERSE_FRAC", "SEEDS", "EXP_NAME", "WORDS_FROM", "ORDER_FROM", "THEME_MODE", "USE_PC", "E4_IMPL", "E4B_WORD_MARGINAL",
    "SYSTEM_INSTRUCTION", "NUM_FEWSHOT", "TASK_NAME", "MASK_ID", "DTYPE", "ATTN_IMPLEMENTATION", "N_BOOTSTRAP", "BOOTSTRAP_SEED", "RUN_SEEDS",
    "RUN_LIMIT", "MAX_RUNTIME_MIN", "PROJECT_DIR", "CODD_REPO_DIR", "MODEL_PATH", "DATA_DIR", "OUT_DIR", "HMM_DIR", "HMM_MANIFEST",
    "EXP1_RESULTS_DIR", "EXP5_RESULTS_DIR", "ALLOW_MODEL_CODE_MISMATCH", "STRICT_VERSIONS", "MIN_GPU_TOTAL_GB", "MIN_GPU_FREE_GB",
    "REQUIRE_SMOKE", "REQUIRE_PCTEST", "E4_N_SAMPLES",
]


def sanitize_for_summary_only():
    return None


def init_config(cfg, mode):
    # Присваивает глобальные константы из config.json, проверяет параметры и хэши.
    global MODE, CFG, HMM_FILE, START_UTC, COMMON_PARAMS, EXP_PARAMS, COMMON_SHA256, EXP_SHA256, HMM_SHA256
    global ENV_PATH, CONFIG_PATH, TASKS_PATH, SMOKE_PATH, SUMMARY_PATH, STATUS_PATH, PREFLIGHT_PATH, PCTEST_PATH, E4_FLAG_PATH
    MODE = mode
    CFG = cfg
    START_UTC = utc_now()
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit("В config.json нет ключей: " + str(missing))
    for k in _REQUIRED_KEYS:
        globals()[k] = cfg[k]
    # config.json должен относиться именно к этому эксперименту (защита от запуска чужого ядра)
    for k in ("EXP_NAME", "WORDS_FROM", "ORDER_FROM", "THEME_MODE", "USE_PC"):
        assert cfg[k] == EXPERIMENT[k], "config.json (" + k + "=" + str(cfg[k]) + ") не соответствует этому ядру (" + str(EXPERIMENT[k]) + "): запущено не то ядро или не тот конфиг"
    assert WORDS_FROM in ("base", "pc") and ORDER_FROM in ("base", "pc") and THEME_MODE in ("joint", "independent")
    assert E4_IMPL in ("auto", "flows", "pool")
    assert isinstance(USE_PC, bool), "USE_PC: True/False"
    assert USE_PC or (WORDS_FROM == "base" and ORDER_FROM == "base" and THEME_MODE == "joint"), "при USE_PC=False допустимо только base/base/joint"
    # ---------- общие параметры (идентичны exp1; менять нельзя) ----------
    assert MODEL == "GSAI-ML/LLaDA-8B-Instruct", "MODEL - общий параметр"
    assert HMM_CKPT == "il18/llada-gsm-pc", "HMM_CKPT - общий параметр"
    assert (DATASET, DATASET_SPLIT, N_TASKS) == ("gsm8k", "test", 200), "DATASET/DATASET_SPLIT/N_TASKS - общие параметры (200 случайных задач test)"
    assert (TEST_SIZE, TASK_SAMPLE_SEED) == (1319, 3), "TEST_SIZE/TASK_SAMPLE_SEED - общие параметры"
    assert GEN_LENGTH == 512 and BLOCK_LENGTH == 32 and TEMPERATURE == 0.0, "GEN_LENGTH/BLOCK_LENGTH/TEMPERATURE - общие параметры"
    assert NUM_STEPS_LIST == [128] and REMASKING_LIST == ["margin"], "NUM_STEPS_LIST/REMASKING_LIST - общие параметры"
    assert (PC_TEMPERATURE, PC_FRAC, REVERSE_FRAC) == (0.2, 0.5, False), "PC_* - общие параметры"
    assert SEEDS == [3], "SEEDS - общий параметр"
    assert DTYPE == "bfloat16", "DTYPE - bfloat16 (L4 поддерживает bf16 аппаратно; dtype общий для всех экспериментов)"
    assert isinstance(RUN_SEEDS, list) and RUN_SEEDS and all(s in SEEDS for s in RUN_SEEDS), "RUN_SEEDS: непустой список из SEEDS"
    assert RUN_LIMIT is None or (isinstance(RUN_LIMIT, int) and 1 <= RUN_LIMIT <= N_TASKS), "RUN_LIMIT: None или целое от 1 до N_TASKS"
    assert MAX_RUNTIME_MIN is None or (isinstance(MAX_RUNTIME_MIN, (int, float)) and MAX_RUNTIME_MIN > 0), "MAX_RUNTIME_MIN: None или число минут > 0"
    for ns in NUM_STEPS_LIST:
        assert ns % (GEN_LENGTH // BLOCK_LENGTH) == 0
    COMMON_PARAMS = {
        "MODEL": MODEL, "HMM_CKPT": HMM_CKPT, "DATASET": DATASET, "DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS,
        "TEST_SIZE": TEST_SIZE, "TASK_SAMPLE_SEED": TASK_SAMPLE_SEED, "GEN_LENGTH": GEN_LENGTH, "BLOCK_LENGTH": BLOCK_LENGTH, "TEMPERATURE": TEMPERATURE,
        "NUM_STEPS_LIST": NUM_STEPS_LIST, "REMASKING_LIST": REMASKING_LIST,
        "PC_TEMPERATURE": PC_TEMPERATURE, "PC_FRAC": PC_FRAC, "REVERSE_FRAC": REVERSE_FRAC, "SEEDS": SEEDS,
    }
    COMMON_SHA256 = sha256_json(COMMON_PARAMS)
    assert COMMON_SHA256 == EXPECTED_COMMON_SHA256, "Хэш общих параметров " + COMMON_SHA256 + " != ожидаемому " + EXPECTED_COMMON_SHA256 + ": общие параметры изменены. Остановка."
    # ---------- пути ----------
    HMM_FILE = os.path.join(HMM_DIR, "model.jpc")
    ENV_PATH = os.path.join(OUT_DIR, "env.json")
    CONFIG_PATH = os.path.join(OUT_DIR, "config.json")
    TASKS_PATH = os.path.join(OUT_DIR, "tasks.json")
    SMOKE_PATH = os.path.join(OUT_DIR, "smoke.json")
    SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")
    STATUS_PATH = os.path.join(OUT_DIR, "status.json")
    PREFLIGHT_PATH = os.path.join(OUT_DIR, "preflight.json")
    PCTEST_PATH = os.path.join(OUT_DIR, "pctest.json")
    E4_FLAG_PATH = os.path.join(OUT_DIR, "E4_CHECK_PASSED.json")
    # ---------- офлайн-среда ----------
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    hf_cache = os.path.join(PROJECT_DIR, ".hf_cache")
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")
    if mode != "summary" and USE_PC:
        HMM_SHA256 = hmm_sha_from_manifest()
    EXP_PARAMS = dict(COMMON_PARAMS)
    EXP_PARAMS.update({
        "EXP_NAME": EXP_NAME, "WORDS_FROM": WORDS_FROM, "ORDER_FROM": ORDER_FROM, "THEME_MODE": THEME_MODE, "E4_IMPL": E4_IMPL,
        "E4B_WORD_MARGINAL": E4B_WORD_MARGINAL, "USE_PC": USE_PC, "SYSTEM_INSTRUCTION": SYSTEM_INSTRUCTION, "NUM_FEWSHOT": NUM_FEWSHOT,
        "TASK_NAME": TASK_NAME, "MASK_ID": MASK_ID, "DTYPE": DTYPE, "ATTN_IMPLEMENTATION": ATTN_IMPLEMENTATION, "HMM_SHA256": HMM_SHA256,
    })
    EXP_SHA256 = sha256_json(EXP_PARAMS)


def hmm_sha_from_manifest():
    # sha256 файла HMM берётся из manifest_hmm.json (его составил download_hmm.py на Mac); сам файл сверяется в preflight.
    if not os.path.isfile(HMM_MANIFEST):
        return None
    with open(HMM_MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    for e in man.get("files", []):
        if e["path"].endswith("/model.jpc") or e["path"] == "model.jpc":
            return e["sha256"]
    return None


# ===================== 3. СОСТОЯНИЕ ПРОЦЕССА =====================
torch = None
F = None
TORCH_DTYPE = None
ENV_INFO = {}
PYJUICE_MODE = None
juice = None
CoddLlada = None
LLaDAModelLM = None
ORIG_GEN_MODULE = None
llada_diffusion_generate = None
add_gumbel_noise = None
get_num_transfer_tokens = None
_is_codd_model = None
apply_pc_logits = None
lm_eval = None
TaskManager = None
get_task_dict = None
HFLM = None
handle_stop_sequences = None
postprocess_generated_text = None
model = None
tokenizer = None
PLAIN = None
lm = None
task = None
INSTANCES = []
TASK_IDS = []
PROMPT_IDS = {}
UNTIL = []
IDS_SHA256 = None
QUESTIONS_SHA256 = None
MODEL_MAX_LEN = None
MODEL_CODE_CHECK = {}
STOP_REQUESTED = {"flag": False}
VAR_INDEX = {"pc": None, "block_length": None}
E4_IMPL_RESOLVED = {"value": None}


def set_status(state, message=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    write_json_atomic(STATUS_PATH, {"experiment": EXP_NAME, "mode": MODE, "state": state, "message": message, "pid": os.getpid(), "updated_utc": utc_now()})
    log("[EXP] STATUS: " + state + " " + message)


def _on_stop_signal(signum, frame):
    if STOP_REQUESTED["flag"]:
        log("повторный сигнал остановки: немедленный выход (кэш к этому моменту согласован)")
        os._exit(143)
    STOP_REQUESTED["flag"] = True
    log("получен сигнал остановки: заканчиваю текущую задачу, дописываю кэш и steplog, затем остановлюсь")


def seed_everything(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ===================== 4. ПРОВЕРКА ОКРУЖЕНИЯ =====================
EXPECTED_VERSIONS = {"torch": "2.9.1", "transformers": "4.57.3", "pyjuice": "2.4.1", "huggingface-hub": "0.36.0", "numpy": "2.2.6"}
REQUIRED_IMPORTS = ["accelerate", "datasets", "evaluate", "jinja2", "jsonlines", "pytablewriter", "rouge_score", "sacrebleu", "sklearn",
                    "sqlitedict", "zstandard", "dill", "word2number", "more_itertools", "pyarrow", "safetensors", "tokenizers", "triton", "numba", "networkx"]
OPTIONAL_IMPORTS = ["peft"]
REQUIRED_REPO_FILES = [
    "eval/llada/codd_llada.py", "eval/llada/llada_generate.py", "eval/llada/modeling_llada.py", "eval/llada/configuration_llada.py",
    "codd/model_class.py", "codd/__init__.py", "lm-evaluation-harness/lm_eval/__init__.py",
    "lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k.yaml", "lm-evaluation-harness/pyproject.toml",
]


def check_compiler():
    # Triton (PyJuice) собирает маленькие C-заглушки: нужны компилятор и Python.h
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    inc = sysconfig.get_paths().get("include")
    py_h = bool(inc and os.path.isfile(os.path.join(inc, "Python.h")))
    return {"compiler": cc, "python_include": inc, "python_h_exists": py_h, "ok": bool(cc and py_h)}


def check_environment(need_gpu):
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
    TORCH_DTYPE = torch.bfloat16
    log("torch: " + torch.__version__ + " | CUDA (сборка torch): " + str(torch.version.cuda) + " | DTYPE: " + DTYPE)
    gpu = {"available": bool(torch.cuda.is_available())}
    gpu_index, total = None, None
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_index)
        free, total = torch.cuda.mem_get_info()
        gpu.update({"name": props.name, "capability": str(props.major) + "." + str(props.minor), "total_bytes": total, "free_bytes": free})
        log("GPU: " + props.name + " | compute capability " + gpu["capability"] + " | всего " + format(total / GIB, ".2f") + " ГиБ, свободно " + format(free / GIB, ".2f") + " ГиБ")
        if need_gpu:
            assert total / GIB >= MIN_GPU_TOTAL_GB, "Памяти GPU " + format(total / GIB, ".2f") + " ГиБ < MIN_GPU_TOTAL_GB=" + str(MIN_GPU_TOTAL_GB)
            assert free / GIB >= MIN_GPU_FREE_GB, "Свободно " + format(free / GIB, ".2f") + " ГиБ < MIN_GPU_FREE_GB=" + str(MIN_GPU_FREE_GB) + ": освободите GPU"
            assert props.major >= 8, "compute capability " + gpu["capability"] + " < 8.0: bf16 не поддерживается аппаратно"
    else:
        log("GPU не найден" + ("" if not need_gpu else " - ОШИБКА для этого режима"))
        assert not need_gpu, "GPU не найден (torch.cuda.is_available() == False): этот режим требует GPU"
    disk = shutil.disk_usage(PROJECT_DIR)
    log("Диск (" + PROJECT_DIR + "): свободно " + format(disk.free / GIB, ".1f") + " ГиБ")
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = os.path.join(OUT_DIR, ".write_probe")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe)

    missing_files = [p for p in REQUIRED_REPO_FILES if not os.path.isfile(os.path.join(CODD_REPO_DIR, p))]
    assert not missing_files, "В CODD_REPO_DIR=" + CODD_REPO_DIR + " нет файлов: " + str(missing_files)
    names = set(os.listdir(MODEL_PATH)) if os.path.isdir(MODEL_PATH) else set()
    for need in ("config.json", "tokenizer.json", "modeling_llada.py", "configuration_llada.py", "model.safetensors.index.json"):
        assert need in names, "В " + MODEL_PATH + " нет файла " + need
    shards = sorted(n for n in names if n.endswith(".safetensors"))
    assert len(shards) == 6, "Ожидалось 6 шардов *.safetensors (склеенных), найдено " + str(len(shards))
    assert not any(".part" in n for n in names), "В папке весов остались не склеенные части *.partNNN"
    weights_bytes = sum(os.path.getsize(os.path.join(MODEL_PATH, n)) for n in shards)
    assert weights_bytes > 15_000_000_000, "суммарный размер шардов подозрительно мал: " + str(weights_bytes)
    missing_imports = [m for m in REQUIRED_IMPORTS if importlib.util.find_spec(m) is None]
    assert not missing_imports, "В venv нет пакетов: " + str(missing_imports) + " (интернета нет, pip не использовать)"
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
        log("   " + name.ljust(18) + want.ljust(10) + str(have).ljust(16) + ("OK" if ok else "РАСХОЖДЕНИЕ"))
    if mismatches:
        msg = "Расхождения версий: " + str(mismatches)
        if STRICT_VERSIONS:
            raise RuntimeError(msg)
        log("ПРЕДУПРЕЖДЕНИЕ (не ошибка): " + msg)
    ENV_INFO = {
        "started_utc": START_UTC, "experiment": EXP_NAME, "mode": MODE, "python": sys.version.split()[0], "python_executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "gpu": gpu, "dtype": DTYPE,
        "hf_reachable": "offline (HF_HUB_OFFLINE=1)", "project_dir": PROJECT_DIR, "model_path": MODEL_PATH, "hmm_file": HMM_FILE,
        "transformers": dist_version("transformers"), "huggingface_hub": dist_version("huggingface-hub"), "numpy": dist_version("numpy"),
        "triton": dist_version("triton"), "version_mismatches_vs_expected": mismatches, "weights_gib": round(weights_bytes / GIB, 2),
        "compiler_check": check_compiler(),
    }
    return gpu_index, total


def prepare_pyjuice():
    # В экспериментах с HMM заглушка pyjuice ЗАПРЕЩЕНА: при ошибке импорта - остановка.
    global PYJUICE_MODE, juice
    try:
        import pyjuice as _juice
    except Exception as e:
        raise RuntimeError("pyjuice не импортируется: " + type(e).__name__ + ": " + str(e) + ". Заглушка pyjuice в экспериментах с HMM ЗАПРЕЩЕНА. "
                           "Проверьте gcc и Python.h (шаг preflight) и совместимость triton.") from e
    juice = _juice
    PYJUICE_MODE = "real"
    log("pyjuice импортирован, версия " + str(dist_version("pyjuice")) + " (ожидается 2.4.1)")
    ENV_INFO["pyjuice"] = dist_version("pyjuice")
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
    global apply_pc_logits, lm_eval, TaskManager, get_task_dict, HFLM, handle_stop_sequences, postprocess_generated_text
    for p in (os.path.join(CODD_REPO_DIR, "lm-evaluation-harness"), CODD_REPO_DIR):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    importlib.invalidate_caches()
    from eval.llada.codd_llada import CoddLlada
    from eval.llada.modeling_llada import LLaDAModelLM
    from eval.llada import llada_generate as ORIG_GEN_MODULE
    from eval.llada.llada_generate import llada_diffusion_generate, add_gumbel_noise, get_num_transfer_tokens, _is_codd_model
    from codd.model_class import apply_pc_logits
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text
    repo_real = os.path.realpath(CODD_REPO_DIR) + os.sep
    for name, mod in (("eval.llada.llada_generate", ORIG_GEN_MODULE), ("lm_eval", lm_eval)):
        f = os.path.realpath(mod.__file__)
        assert f.startswith(repo_real), name + " импортирован НЕ из репозитория: " + f
    log("Код репозитория импортирован из: " + CODD_REPO_DIR)
    pyproject = open(os.path.join(CODD_REPO_DIR, "lm-evaluation-harness", "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    git = git_info(CODD_REPO_DIR)
    ENV_INFO.update({
        "lm_eval_version_pyproject": m.group(1) if m else None, "codd_commit": git["commit"], "codd_dirty": git["dirty"],
        "llada_generate_py_sha256": sha256_file(os.path.join(CODD_REPO_DIR, "eval", "llada", "llada_generate.py")),
        "model_class_py_sha256": sha256_file(os.path.join(CODD_REPO_DIR, "codd", "model_class.py")),
        "common_params_sha256": COMMON_SHA256, "exp_params_sha256": EXP_SHA256,
    })


def write_env_and_config(final=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            old_env = json.load(f)
        prev = old_env.pop("previous_sessions", [])
        ENV_INFO["previous_sessions"] = (prev + [old_env])[-20:]
    write_json_atomic(ENV_PATH, ENV_INFO)
    config_doc = {"common_params": COMMON_PARAMS, "common_sha256": COMMON_SHA256, "exp_params": EXP_PARAMS, "exp_sha256": EXP_SHA256}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            old_cfg = json.load(f)
        if old_cfg.get("exp_sha256") != EXP_SHA256:
            diff = {}
            for k in set(old_cfg.get("exp_params", {})) | set(EXP_PARAMS):
                a, b = old_cfg.get("exp_params", {}).get(k), EXP_PARAMS.get(k)
                if a != b:
                    diff[k] = [a, b]
            raise RuntimeError("В " + OUT_DIR + " уже лежат результаты с ДРУГИМ конфигом (параметр: [было, стало]): " + str(diff) + ". Верните прежние значения или укажите другой OUT_DIR.")
        log("config.json совпадает с текущим конфигом - возобновление допустимо.")
    else:
        write_json_atomic(CONFIG_PATH, config_doc)
        log("config.json записан: " + CONFIG_PATH)


def check_model_code():
    # modeling_llada.py - расхождение = ОСТАНОВКА; configuration_llada.py - ПРЕДУПРЕЖДЕНИЕ (оба хэша и первые 60 строк unified diff в лог и env.json)
    global MODEL_CODE_CHECK
    fatal_names = {"modeling_llada.py": True, "configuration_llada.py": False}
    for fname, fatal in fatal_names.items():
        local = os.path.join(CODD_REPO_DIR, "eval", "llada", fname)
        snap = os.path.join(MODEL_PATH, fname)
        if not os.path.isfile(snap):
            MODEL_CODE_CHECK[fname] = {"verdict": "НЕ ПРОВЕРЕНО", "reason": "нет файла " + fname + " рядом с весами"}
            log(fname + ": НЕ ПРОВЕРЕНО - нет файла рядом с весами")
            continue
        a, b = sha256_file(local), sha256_file(snap)
        rec = {"verdict": "MATCH" if a == b else "MISMATCH", "repo_sha256": a, "snapshot_sha256": b, "fatal_if_mismatch": fatal}
        if a != b:
            with open(local, encoding="utf-8", errors="replace") as f1, open(snap, encoding="utf-8", errors="replace") as f2:
                diff = list(difflib.unified_diff(f1.read().splitlines(), f2.read().splitlines(), fromfile="repo/" + fname, tofile="weights/" + fname, lineterm=""))
            rec["diff_first_60_lines"] = diff[:60]
            rec["diff_total_lines"] = len(diff)
        MODEL_CODE_CHECK[fname] = rec
        log(fname + ": " + rec["verdict"] + " repo=" + a[:16] + "... weights=" + b[:16] + "...")
        if a != b:
            if fatal:
                log("ОШИБКА: расхождение " + fname + " останавливает работу (код модели).")
            else:
                log("ПРЕДУПРЕЖДЕНИЕ: " + fname + " в репозитории CoDD и рядом с весами различается (строк diff: " + str(rec["diff_total_lines"]) + "); первые 60 строк:")
                for line in rec["diff_first_60_lines"]:
                    print("    " + line, flush=True)
    ENV_INFO["model_code_check"] = MODEL_CODE_CHECK
    write_json_atomic(ENV_PATH, ENV_INFO)
    bad = [k for k, v in MODEL_CODE_CHECK.items() if v["verdict"] == "MISMATCH" and v.get("fatal_if_mismatch")]
    if bad and not ALLOW_MODEL_CODE_MISMATCH:
        raise RuntimeError("Исходники модели из репозитория CoDD отличаются от файлов рядом с весами: " + str(bad) + ". Остановка до решения автора (ALLOW_MODEL_CODE_MISMATCH=true, чтобы продолжить осознанно).")


# ===================== 5. МОДЕЛЬ И HMM =====================
class PlainView:
    # Та же модель как "обычная": базовый forward без CoDD-аргументов (атрибута pc_model нет специально).
    def __init__(self, m):
        self._m = m

    @property
    def device(self):
        return self._m.device

    def __call__(self, x):
        return LLaDAModelLM.forward(self._m, x)


def load_hmm(device):
    # PC грузится ЛОКАЛЬНО: juice.load(локальный .jpc) + juice.compile(...).to(device) (так делает CoddLlada.from_pretrained, но без Hub).
    assert os.path.isfile(HMM_FILE), "Нет файла HMM: " + HMM_FILE + " (загрузите offline_bundle_hmm по README_HMM.md)"
    t0 = time.time()
    raw = juice.load(HMM_FILE)
    t1 = time.time()
    pc = juice.compile(raw)
    t2 = time.time()
    pc = pc.to(device)
    torch.cuda.synchronize()
    t3 = time.time()
    info = {"load_s": round(t1 - t0, 2), "compile_s": round(t2 - t1, 2), "to_device_s": round(t3 - t2, 2),
            "gpu_allocated_gib_after": round(torch.cuda.memory_allocated() / GIB, 3), "num_vars": int(getattr(pc, "num_vars", -1)),
            "num_nodes": int(getattr(pc, "num_nodes", -1)), "num_elements": int(getattr(pc, "num_elements", -1))}
    log("HMM загружена: " + json.dumps(info, ensure_ascii=False))
    return pc, info


def load_model(with_hmm=True):
    global model, tokenizer, PLAIN
    t0 = time.time()
    log("Загрузка модели с диска (local_files_only=True) из " + MODEL_PATH)
    model = CoddLlada.from_pretrained(MODEL_PATH, torch_dtype=TORCH_DTYPE, local_files_only=True, attn_implementation=ATTN_IMPLEMENTATION, device_map="cuda")
    ENV_INFO["model_load_seconds"] = round(time.time() - t0, 1)
    log("Модель загружена за " + format(time.time() - t0, ".0f") + " c")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = model.tokenizer
    assert tokenizer is not None and model.pc_model is None
    assert model.codd_config.mask_id == MASK_ID
    assert next(model.parameters()).dtype == TORCH_DTYPE and next(model.parameters()).device.type == "cuda" and not model.training
    log("Память GPU занята моделью: " + format(torch.cuda.memory_allocated() / GIB, ".1f") + " ГиБ")
    PLAIN = PlainView(model)
    assert _is_codd_model(model) is True and _is_codd_model(PLAIN) is False
    if with_hmm:
        pc, info = load_hmm(model.device)
        model.pc_model = pc
        model._pc_compiled = False
        ENV_INFO["hmm_load"] = info
        log("Память GPU: модель + HMM = " + format(torch.cuda.memory_allocated() / GIB, ".2f") + " ГиБ")


# ===================== 6. ЗАДАЧИ GSM8K (ОФЛАЙН-parquet ЧЕРЕЗ КОД lm_eval) =====================
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
    # dict-конфиг с task=gsm8k сливается с gsm8k.yaml; подменяется только источник данных (path="parquet", локальные data_files).
    train, test = find_parquet("train"), find_parquet("test")
    assert test is not None, "В " + DATA_DIR + " не найден test-parquet GSM8K"
    if train is None:
        log("ВНИМАНИЕ: train-parquet не найден, вместо него подставлен test (на 0-shot это не влияет).")
        train = test
    override = {"task": TASK_NAME, "dataset_path": "parquet", "dataset_name": None, "dataset_kwargs": {"data_files": {"train": train, "test": test}}}
    task_dict = get_task_dict([override], task_manager=TaskManager())
    assert len(task_dict) == 1, "ожидалась одна задача " + TASK_NAME
    return next(iter(task_dict.values()))


def gold_final(answer_text):
    m = re.search(r"####\s*([^\n]+)", answer_text)
    return m.group(1).strip() if m else None


def configure_task(t):
    t.set_config(key="num_fewshot", value=NUM_FEWSHOT)
    assert t.get_config("num_fewshot") == NUM_FEWSHOT == 0
    assert t.OUTPUT_TYPE == "generate_until"
    gk = t.get_config("generation_kwargs")
    assert gk.get("do_sample") is False and float(gk.get("temperature", 0.0)) == TEMPERATURE == 0.0, "generation_kwargs задачи не жадные: " + str(gk)
    assert len(t.eval_docs) == TEST_SIZE, "в test-выборке " + str(len(t.eval_docs)) + " задач, ожидалось " + str(TEST_SIZE)
    assert set(f.name for f in t._filters) == set(["strict-match", "flexible-extract"])


def build_prompts_cpu():
    # Быстрый отказ: строим контексты ТЕХ ЖЕ 200 выбранных задач ТЕМ ЖЕ кодом lm_eval, но без модели: HFLM.apply_chat_template / tok_batch_encode
    # вызываются как есть на минимальной обёртке (tokenizer + атрибуты, которые эти методы читают).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    shim = types.SimpleNamespace(tokenizer=tok, chat_template_args={}, backend="causal", add_bos_token=None)
    chat_template = functools.partial(HFLM.apply_chat_template, shim)
    t = load_task()
    configure_task(t)
    t.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
                         system_instruction=SYSTEM_INSTRUCTION, apply_chat_template=True, fewshot_as_multiturn=False,
                         chat_template=chat_template, tokenizer_name=tok.name_or_path.replace("/", "__"))
    all_insts = sorted(t.instances, key=lambda i: i.doc_id)
    assert [i.doc_id for i in all_insts] == list(range(TEST_SIZE))
    insts = [all_insts[i] for i in select_task_ids(TEST_SIZE, N_TASKS, TASK_SAMPLE_SEED)]
    assert len(insts) == N_TASKS
    ids = [i.doc_id for i in insts]
    prompt_lens = []
    for inst in insts:
        enc, _ = HFLM.tok_batch_encode(shim, [inst.args[0]], left_truncate_len=None, truncation=False)
        prompt_lens.append(int(enc.shape[1]))
    return ids, prompt_lens, insts


def setup_tasks():
    global task, lm, INSTANCES, TASK_IDS, PROMPT_IDS, UNTIL, IDS_SHA256, QUESTIONS_SHA256, MODEL_MAX_LEN
    task = load_task()
    configure_task(task)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    log("HFLM backend: " + str(lm.backend) + " | max_length (HFLM): " + str(lm.max_length))
    task.build_all_requests(
        limit=None, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
        system_instruction=SYSTEM_INSTRUCTION, apply_chat_template=True, fewshot_as_multiturn=False,
        chat_template=lm.apply_chat_template, tokenizer_name=lm.tokenizer_name,
    )
    all_instances = sorted(task.instances, key=lambda inst: inst.doc_id)
    assert [inst.doc_id for inst in all_instances] == list(range(TEST_SIZE)), "запросы построены не для всех задач test"
    selected_ids = select_task_ids(TEST_SIZE, N_TASKS, TASK_SAMPLE_SEED)
    INSTANCES = [all_instances[i] for i in selected_ids]
    assert len(INSTANCES) == N_TASKS
    for inst in INSTANCES:
        assert inst.request_type == "generate_until" and inst.repeats == 1
    TASK_IDS = [inst.doc_id for inst in INSTANCES]
    assert TASK_IDS == selected_ids
    gen_kwargs = copy.deepcopy(INSTANCES[0].args[1])
    UNTIL = handle_stop_sequences(gen_kwargs.pop("until", None), eos=lm.tok_decode(lm.eot_token_id, skip_special_tokens=False))
    log("until: " + str(UNTIL))
    assert all(inst.args[1] == INSTANCES[0].args[1] for inst in INSTANCES)
    PROMPT_IDS = {}      # {task_id: токены промпта}
    for inst in INSTANCES:
        ids, _ = lm.tok_batch_encode([inst.args[0]], left_truncate_len=None, truncation=False)
        PROMPT_IDS[inst.doc_id] = ids[0].tolist()
        assert MASK_ID not in PROMPT_IDS[inst.doc_id]
    limits = [lm.max_length]
    if hasattr(model.config, "max_sequence_length") and model.config.max_sequence_length:
        limits.append(int(model.config.max_sequence_length))
    MODEL_MAX_LEN = min(limits)
    max_prompt = max(len(p) for p in PROMPT_IDS.values())
    assert max_prompt + GEN_LENGTH <= MODEL_MAX_LEN, "Самый длинный промпт (" + str(max_prompt) + ") + GEN_LENGTH > лимита длины модели " + str(MODEL_MAX_LEN)
    log("Промпты: длина " + str(min(len(p) for p in PROMPT_IDS.values())) + ".." + str(max_prompt) + " токенов; лимит модели " + str(MODEL_MAX_LEN))
    IDS_SHA256 = hashlib.sha256(json.dumps(TASK_IDS, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert IDS_SHA256 == EXPECTED_IDS_SHA256, "Хэш списка task_id " + IDS_SHA256 + " != ожидаемому " + EXPECTED_IDS_SHA256 + ". Остановка."
    QUESTIONS_SHA256 = hashlib.sha256(json.dumps([inst.doc["question"] for inst in INSTANCES], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    tasks_doc = {
        "dataset": DATASET, "split": DATASET_SPLIT, "n_tasks": N_TASKS, "test_size": TEST_SIZE, "task_sample_seed": TASK_SAMPLE_SEED, "selection_rule": "id сортируются по sha256(seed:id), берутся первые n, итог по возрастанию id", "data_source": "локальные parquet через lm_eval (dataset_path=parquet): " + DATA_DIR,
        "system_instruction": SYSTEM_INSTRUCTION, "num_fewshot": NUM_FEWSHOT, "chat_template_applied": True, "ids_sha256": IDS_SHA256,
        "ids_sha256_definition": 'sha256(json.dumps(ids, separators=(",",":")).encode("utf-8")), ids = выбранные id (select_task_ids), по возрастанию',
        "questions_sha256": QUESTIONS_SHA256, "ids": TASK_IDS,
        "tasks": [{"task_id": inst.doc_id, "question": inst.doc["question"], "gold_final": gold_final(inst.doc["answer"]),
                   "prompt_sha256": hashlib.sha256(inst.args[0].encode("utf-8")).hexdigest()} for inst in INSTANCES],
    }
    assert all(t["gold_final"] is not None for t in tasks_doc["tasks"])
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, encoding="utf-8") as f:
            old = json.load(f)
        assert old["ids_sha256"] == IDS_SHA256 and old["questions_sha256"] == QUESTIONS_SHA256, "tasks.json от другой выборки: возобновление небезопасно"
        assert {t["task_id"]: t["prompt_sha256"] for t in old["tasks"]} == {t["task_id"]: t["prompt_sha256"] for t in tasks_doc["tasks"]}, "промпты отличаются от сохранённых"
    else:
        write_json_atomic(TASKS_PATH, tasks_doc)
        log("tasks.json записан")
    log("sha256 списка id задач: " + IDS_SHA256 + " (совпал с ожидаемым)")


# ===================== 7. ПРИМЕНЕНИЕ HMM: joint (оригинал), independent (Э4) =====================
def apply_joint(m, x, logits_base, start_idx, end_idx):
    # ОРИГИНАЛЬНЫЙ apply_pc_logits на клоне базовых логитов; правило record_cudagraph - как в CoddLlada.forward.
    cfg = m.codd_config
    record = not m._pc_compiled
    out = apply_pc_logits(pc_model=m.pc_model, input_ids=x, logits=logits_base.clone(), start_idx=start_idx, end_idx=end_idx,
                          mask_id=cfg.mask_id, vocab_size=cfg.vocab_size, pc_temperature=PC_TEMPERATURE, pc_frac=PC_FRAC,
                          reverse_frac=REVERSE_FRAC, record_cudagraph=record)
    if record:
        m._pc_compiled = True
    return out


def num_layers_of(group):
    # у группы слоёв PyJuice число слоёв - атрибут num_layers (len() группы не определён)
    n = getattr(group, "num_layers", None)
    return int(n) if n is not None else len(group)


def get_var_index(pc, block_length):
    # Для каждой позиции блока: локальные индексы входных узлов (скрытые состояния) во входном слое. Один раз на HMM.
    if VAR_INDEX["pc"] is pc and VAR_INDEX["block_length"] == block_length:
        return VAR_INDEX["idx"]
    assert num_layers_of(pc.input_layer_group) == 1, "ожидался один входной слой, найдено " + str(num_layers_of(pc.input_layer_group))
    layer = pc.input_layer_group[0]
    sid, eid = layer._output_ind_range
    vids = layer.vids[:, 0]
    var_nodes = []
    for v in range(block_length):
        nodes = (vids == v).nonzero(as_tuple=True)[0]
        assert nodes.numel() > 0, "у позиции " + str(v) + " нет входных узлов"
        var_nodes.append(nodes)
    idx = {"layer": layer, "sid": int(sid), "eid": int(eid), "var_nodes": var_nodes, "states_per_var": [int(n.numel()) for n in var_nodes]}
    VAR_INDEX["pc"] = pc
    VAR_INDEX["block_length"] = block_length
    VAR_INDEX["idx"] = idx
    return idx


def _prepare_pc_inputs(x, logits_base, start_idx, end_idx):
    # Тот же препроцессинг, что в apply_pc_logits (деление на температуру, log_softmax, маска известных слов)
    block_length = end_idx - start_idx
    curr_x = x[:, start_idx:end_idx]
    curr_logits = logits_base[:, start_idx:end_idx, :]
    if PC_TEMPERATURE > 0 and PC_TEMPERATURE < 1:
        curr_logits = curr_logits / PC_TEMPERATURE
    pc_block_x = curr_x.contiguous().view(-1, block_length)
    pc_logits = curr_logits.contiguous().view(-1, block_length, curr_logits.size(-1)).float()
    ext = F.log_softmax(pc_logits, dim=-1)
    pc_value_mask = ~(pc_block_x == MASK_ID)
    return pc_block_x, ext, pc_value_mask


def pc_forward(pc, pc_block_x, ext, pc_value_mask, record):
    if record:
        pc(pc_block_x, external_categorical_logps=ext, extern_product_categorical_mode="unnormalized_ll",
           external_categorical_value_mask=pc_value_mask, record_cudagraph=True)
    pc(pc_block_x, external_categorical_logps=ext, extern_product_categorical_mode="unnormalized_ll", external_categorical_value_mask=pc_value_mask)


def theme_posteriors(pc, pc_block_x):
    # Апостериорные вероятности состояний по позициям: backward ТОЛЬКО по внутренним слоям, без параметров (compute_param_flows=False).
    dummy = torch.zeros(pc_block_x.size(0), pc.num_vars, dtype=torch.long, device=pc_block_x.device)
    pc.backward(inputs=dummy, compute_param_flows=False, _inner_layers_only=True)
    return pc.node_flows


def apply_independent(m, x, logits_base, start_idx, end_idx):
    # Э4, вариант "потоки": тема каждой позиции выбирается НЕЗАВИСИМО из маргинала p(z_i | открытые слова, оценки нейросети).
    pc = m.pc_model
    V = m.codd_config.vocab_size
    block_length = end_idx - start_idx
    idx = get_var_index(pc, block_length)
    layer, sid = idx["layer"], idx["sid"]
    record = not m._pc_compiled
    logits = logits_base.clone()
    pc_block_x, ext, pc_value_mask = _prepare_pc_inputs(x, logits_base, start_idx, end_idx)
    pc_forward(pc, pc_block_x, ext, pc_value_mask, record)
    if record:
        m._pc_compiled = True
    flows = theme_posteriors(pc, pc_block_x)
    chosen, probs_list = [], []
    for v in range(block_length):
        nodes = idx["var_nodes"][v]
        f = flows[sid + nodes, 0].float().clamp_min(0.0)
        tot = f.sum()
        assert float(tot) > 0.0, "нулевая сумма потоков у позиции " + str(v)
        p = f / tot
        probs_list.append(p)
        chosen.append(nodes[torch.multinomial(p, 1)])
    if E4B_WORD_MARGINAL:
        # Э4b (выкл. по умолчанию; МОЯ трактовка): вместо одной темы - смесь слов по апостериору состояний
        mix = torch.empty_like(ext)
        for v in range(block_length):
            nodes = idx["var_nodes"][v]
            ps = layer.s_pids[nodes].tolist()
            acc = torch.zeros(V, device=ext.device, dtype=ext.dtype)
            for kk, psid in enumerate(ps):
                acc += probs_list[v][kk] * torch.softmax(ext[0, v, :] + layer.params[psid:psid + V].log(), dim=-1)
            mix[0, v, :] = torch.log(acc.clamp_min(1e-30))
        sampled = mix
    else:
        chosen_t = torch.cat(chosen)
        psids = layer.s_pids[chosen_t].tolist()
        sampled = ext.clone()
        for v in range(block_length):
            sampled[0, v, :] += layer.params[psids[v]:psids[v] + V].log()
    logits[:, start_idx:end_idx, :] = sampled
    return logits


def apply_pool(m, x, logits_base, start_idx, end_idx):
    # Э4, вариант "пул независимых цепочек": блок дублируется L раз по батчу, один вызов sample, для позиции p берётся тема цепочки p.
    pc = m.pc_model
    V = m.codd_config.vocab_size
    block_length = end_idx - start_idx
    idx = get_var_index(pc, block_length)
    layer, sid = idx["layer"], idx["sid"]
    logits = logits_base.clone()
    pc_block_x, ext, pc_value_mask = _prepare_pc_inputs(x, logits_base, start_idx, end_idx)
    L = block_length
    bx = pc_block_x.expand(L, -1).contiguous()
    ex = ext.expand(L, -1, -1).contiguous()
    vm = pc_value_mask.expand(L, -1).contiguous()
    pc_forward(pc, bx, ex, vm, False)
    node_samples = juice.queries.sample(pc, conditional=True, external_categorical_logps=ex, _sample_input_ns=False)
    chosen = []
    for p in range(L):
        col = node_samples[:, p]
        nodes = col[col >= 0] - sid
        sel = nodes[layer.vids[nodes, 0] == p]
        assert sel.numel() == 1, "в цепочке " + str(p) + " для позиции p найдено узлов: " + str(int(sel.numel()))
        chosen.append(sel[0:1])
    chosen_t = torch.cat(chosen)
    psids = layer.s_pids[chosen_t].tolist()
    sampled = ext.clone()
    for v in range(block_length):
        sampled[0, v, :] += layer.params[psids[v]:psids[v] + V].log()
    logits[:, start_idx:end_idx, :] = sampled
    return logits


def resolve_e4_impl():
    if THEME_MODE != "independent":
        return None
    if E4_IMPL != "auto":
        return E4_IMPL
    assert os.path.exists(PCTEST_PATH), "E4_IMPL=auto требует pctest.json (режим pctest)"
    with open(PCTEST_PATH, encoding="utf-8") as f:
        pt = json.load(f)
    v = (pt.get("e4") or {}).get("impl_resolved")
    assert v in ("flows", "pool"), "в pctest.json нет выбранной реализации Э4 (e4.impl_resolved)"
    return v


def apply_theme(m, x, logits_base, start_idx, end_idx, theme_mode, impl):
    if theme_mode == "joint":
        return apply_joint(m, x, logits_base, start_idx, end_idx)
    if impl == "flows":
        return apply_independent(m, x, logits_base, start_idx, end_idx)
    if impl == "pool":
        return apply_pool(m, x, logits_base, start_idx, end_idx)
    raise ValueError("неизвестная реализация независимого выбора темы: " + str(impl))


# ===================== 8. ГЕНЕРАЦИЯ: копия llada_diffusion_generate с A/C и диагностикой =====================
def confidence_from(logits, x0, remasking, end_idx):
    # Те же формулы, что в оригинале llada_diffusion_generate, на полных тензорах (чтобы top-k совпадал с оригиналом).
    logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "margin":
        probs = torch.softmax(logits_with_noise, dim=-1)
        top_probs, _ = torch.topk(probs, k=2, dim=-1)
        x0_p = top_probs[..., 0] - top_probs[..., 1]
    else:
        raise NotImplementedError(remasking)
    x0_p[:, end_idx:] = -np.inf
    return x0_p


def generate_variant(m, prompt, num_steps, gen_length, block_length, remasking, words_from, order_from, theme_mode, use_hmm, impl=None,
                     check_finite=False):
    # Возвращает (x, steplog). use_hmm=False: HMM не вызывается (logits_pc = logits_base) - эквивалент оригинала с use_pc=False.
    assert prompt.dim() == 2 and prompt.shape[0] == 1
    assert remasking in ("low_confidence", "margin")
    with torch.no_grad():
        x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long).to(m.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert num_steps % num_blocks == 0
        steps = num_steps // num_blocks
        steplog = []
        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length
            block_mask_index = (x[:, start_idx:end_idx] == MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == MASK_ID)
                n_masked_block_before = int(mask_index[0, start_idx:end_idx].sum().item())
                mask_ratio = (x[:, start_idx:end_idx] == MASK_ID).float().mean()
                logits_base = LLaDAModelLM.forward(m, x).logits
                active = bool(use_hmm and (mask_ratio < PC_FRAC))
                if active:
                    logits_pc = apply_theme(m, x, logits_base, start_idx, end_idx, theme_mode, impl)
                    if check_finite:
                        assert bool(torch.isfinite(logits_pc[:, start_idx:end_idx]).all()), "NaN/inf в логитах после HMM (блок " + str(num_block) + ", шаг " + str(i) + ")"
                else:
                    logits_pc = logits_base
                logits_a = logits_pc if words_from == "pc" else logits_base
                logits_c = logits_pc if order_from == "pc" else logits_base
                logits_with_noise_a = add_gumbel_noise(logits_a, temperature=TEMPERATURE)
                x0 = torch.argmax(logits_with_noise_a, dim=-1)
                x0_p = confidence_from(logits_c, x0, remasking, end_idx)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                num_tokens = num_transfer_tokens[0, i].item()
                rec = {"block": num_block, "step_in_block": i, "global_step": num_block * steps + i, "n_masked_block_before": n_masked_block_before,
                       "pc_active": active, "delta_words": 0.0, "delta_order": 0.0, "n_masked_block": n_masked_block_before, "n_word_diff": 0, "k": int(num_tokens)}
                if num_tokens > 0:
                    conf_vals, select_index = torch.topk(confidence[0], k=num_tokens)
                    x[0, select_index] = x0[0, select_index]
                    pos_abs = select_index.tolist()
                    rec.update({"positions_abs": pos_abs, "positions_rel": [p_ - prompt.shape[1] for p_ in pos_abs],
                                "token_ids": x0[0, select_index].tolist(), "confidence": [float(c) for c in conf_vals.float().tolist()]})
                    if active:
                        # диагностика: слова (base vs pc) и порядок (base vs pc при одних и тех же кандидатах x0)
                        bm = mask_index[0, start_idx:end_idx].tolist()
                        base_ids = torch.argmax(logits_base[0, start_idx:end_idx], dim=-1).tolist()
                        pc_ids = torch.argmax(logits_pc[0, start_idx:end_idx], dim=-1).tolist()
                        dw, n_m, n_d = delta_words_value(base_ids, pc_ids, bm)
                        sets = {}
                        for name, lg in (("base", logits_base), ("pc", logits_pc)):
                            if (name == "pc") == (order_from == "pc"):
                                conf_n = confidence
                            else:
                                conf_n = torch.where(mask_index, confidence_from(lg, x0, remasking, end_idx), -np.inf)
                            _v, idx_n = torch.topk(conf_n[0], k=num_tokens)
                            sets[name] = idx_n.tolist()
                        rec.update({"delta_words": dw, "n_masked_block": n_m, "n_word_diff": n_d, "delta_order": jaccard_delta_order(sets["base"], sets["pc"])})
                else:
                    rec.update({"positions_abs": [], "positions_rel": [], "token_ids": [], "confidence": []})
                steplog.append(rec)
    return x, steplog


FILTER_NAMES = ("strict-match", "flexible-extract")
REQUIRED_ROW_KEYS = ("task_id", "seed", "correct", "prediction", "num_steps", "remasking", "pc_active_steps", "delta_words", "delta_order")


def score_response(inst, text):
    inst.resps = [text]
    inst.filtered_resps = {}
    for filter_ensemble in task._filters:
        filter_ensemble.apply([inst])
    scores, preds = {}, {}
    for name in FILTER_NAMES:
        assert name in inst.filtered_resps
        pred = inst.filtered_resps[name]
        assert isinstance(pred, str)
        metrics = task.process_results(inst.doc, [pred])
        value = float(metrics["exact_match"])
        assert value in (0.0, 1.0)
        scores[name] = value
        preds[name] = pred
    return scores, preds


def decode_generation(gen_token_ids):
    return postprocess_generated_text(generation=lm.tok_decode(gen_token_ids), stop=UNTIL, think_end_token=None)


def run_generation(inst, remasking, num_steps, gen_length=None, block_length=None, use_hmm=None, theme_mode=None, words_from=None, order_from=None, impl=None, check_finite=False):
    gen_length = GEN_LENGTH if gen_length is None else gen_length
    block_length = BLOCK_LENGTH if block_length is None else block_length
    prompt = torch.tensor([PROMPT_IDS[inst.doc_id]], dtype=torch.long, device=model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    x, steplog = generate_variant(model, prompt, num_steps, gen_length, block_length, remasking,
                                  WORDS_FROM if words_from is None else words_from, ORDER_FROM if order_from is None else order_from,
                                  THEME_MODE if theme_mode is None else theme_mode, (USE_PC if use_hmm is None else use_hmm), impl=impl, check_finite=check_finite)
    torch.cuda.synchronize()
    return x, steplog, prompt.shape[1], time.time() - t0


def eos_token_ids():
    # EOS: <|endoftext|> (eos_token_id токенайзера) и <|eot_id|> (eot_token_id HFLM)
    ids = set()
    for v in (getattr(tokenizer, "eos_token_id", None), getattr(lm, "eot_token_id", None)):
        if v is not None:
            ids.add(int(v))
    return ids


def mean_or_none(vals):
    return (sum(vals) / float(len(vals))) if vals else None


def process_task(inst, remasking, num_steps, seed, impl):
    task_id = inst.doc_id
    seed_everything(derive_seed(seed, task_id))
    x, steplog, prompt_len, sec = run_generation(inst, remasking, num_steps, impl=impl)
    gen_ids = x[0, prompt_len:].tolist()
    assert len(gen_ids) == GEN_LENGTH
    assert MASK_ID not in gen_ids, "задача " + str(task_id) + ": в ответе остались маски - несостыковка, остановка"
    assert len(steplog) == num_steps
    text = decode_generation(gen_ids)
    eos_ids = eos_token_ids()
    eos_pos = next((i for i, t_ in enumerate(gen_ids) if t_ in eos_ids), None)    # None: за 512 токенов EOS не встретился = ответ неоконченный
    scores, preds = score_response(inst, text)
    active = [r for r in steplog if r["pc_active"]]
    row = {
        "task_id": task_id, "seed": seed, "correct": int(scores["flexible-extract"] == 1.0), "prediction": preds["flexible-extract"],
        "num_steps": num_steps, "remasking": remasking, "pc_active_steps": len(active),
        "delta_words": mean_or_none([r["delta_words"] for r in active]), "delta_order": mean_or_none([r["delta_order"] for r in active]),
        # --- дополнительные поля ---
        "correct_strict": int(scores["strict-match"] == 1.0), "prediction_strict": preds["strict-match"], "gold": gold_final(inst.doc["answer"]),
        "generation": text, "finished": eos_pos is not None, "eos_pos": eos_pos, "correct_if_finished": int(scores["flexible-extract"] == 1.0 and eos_pos is not None), "gen_length": GEN_LENGTH, "prompt_tokens": prompt_len, "sec": round(sec, 3),
        "variant": {"use_pc": USE_PC, "words_from": WORDS_FROM, "order_from": ORDER_FROM, "theme_mode": THEME_MODE, "impl": impl},
    }
    step_rows = []
    for rec in steplog:
        d = {"task_id": task_id, "seed": seed, "remasking": remasking, "num_steps": num_steps}
        d.update(rec)
        d["pc_active_steps"] = len(active)
        step_rows.append(d)
    return row, step_rows


# ===================== 9. КЭШ И ВОЗОБНОВЛЕНИЕ =====================
def result_path(remasking, num_steps, seed):
    return os.path.join(OUT_DIR, remasking + "_steps" + str(num_steps) + "_seed" + str(seed) + ".jsonl")


def steplog_path(remasking, num_steps, seed):
    return os.path.join(OUT_DIR, "steplog_" + remasking + "_steps" + str(num_steps) + "_seed" + str(seed) + ".jsonl")


def repair_steplog(path, done_ids):
    rows = read_jsonl_tolerant(path)
    keep = [r for r in rows if r["task_id"] in done_ids]
    if len(keep) != len(rows):
        log("steplog " + os.path.basename(path) + ": удалены строки незавершённых задач " + str(sorted({r["task_id"] for r in rows if r["task_id"] not in done_ids})))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return keep


def validate_result_row(row, remasking, num_steps, seed):
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    assert not missing, "в строке результата нет полей " + str(missing)
    assert row["remasking"] == remasking and row["num_steps"] == num_steps and row["seed"] == seed
    assert row["correct"] in (0, 1) and isinstance(row["prediction"], str) and isinstance(row["task_id"], int)
    assert isinstance(row["pc_active_steps"], int) and 0 <= row["pc_active_steps"] <= num_steps
    assert row["delta_words"] is None or 0.0 <= row["delta_words"] <= 1.0
    assert row["delta_order"] is None or 0.0 <= row["delta_order"] <= 1.0
    assert row.get("gen_length", GEN_LENGTH) == GEN_LENGTH


def run_combo(remasking, num_steps, seed, instances_to_run, impl, deadline=None):
    # "done" | "stopped" (MAX_RUNTIME_MIN) | "signal" (SIGTERM/SIGINT). Остановка только ПОСЛЕ полностью записанной задачи.
    rpath, spath = result_path(remasking, num_steps, seed), steplog_path(remasking, num_steps, seed)
    rows = read_jsonl_tolerant(rpath)
    for r in rows:
        validate_result_row(r, remasking, num_steps, seed)
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids)), rpath + ": повторяющиеся task_id"
    done = set(ids)
    repair_steplog(spath, done)
    todo = [inst for inst in instances_to_run if inst.doc_id not in done]
    tag = "[" + remasking + "/" + str(num_steps) + "/seed" + str(seed) + "]"
    log(tag + " уже готово " + str(len(done)) + ", осталось " + str(len(todo)))
    for k, inst in enumerate(todo, 1):
        row, step_rows = process_task(inst, remasking, num_steps, seed, impl)
        append_jsonl(spath, step_rows)
        append_jsonl(rpath, [row])
        torch.cuda.empty_cache()
        log(tag + " задача " + str(inst.doc_id) + " (" + str(k) + "/" + str(len(todo)) + ") correct=" + str(row["correct"]) + " pred=" + repr(row["prediction"]) + " gold="
            + repr(row["gold"]) + " активных шагов HMM=" + str(row["pc_active_steps"]) + " " + format(row["sec"], ".1f") + " c")
        if STOP_REQUESTED["flag"]:
            log(tag + " остановка по сигналу после задачи " + str(inst.doc_id))
            return "signal"
        if deadline is not None and time.time() >= deadline:
            log(tag + " остановка по MAX_RUNTIME_MIN после задачи " + str(inst.doc_id))
            return "stopped"
    return "done"


def validate_combo(remasking, num_steps, seed, expected_ids):
    rows = read_jsonl_tolerant(result_path(remasking, num_steps, seed))
    for r in rows:
        validate_result_row(r, remasking, num_steps, seed)
    ids = [r["task_id"] for r in rows]
    assert sorted(ids) == sorted(expected_ids), remasking + "/" + str(num_steps) + "/seed" + str(seed) + ": ожидались task_id " + str(len(expected_ids)) + " шт., в файле " + str(len(ids))
    per_task = {}
    for s in read_jsonl_tolerant(steplog_path(remasking, num_steps, seed)):
        per_task.setdefault(s["task_id"], []).append(s)
    assert set(per_task) == set(ids), "steplog и результаты содержат разные task_id"
    for tid, recs in per_task.items():
        assert len(recs) == num_steps and [r["global_step"] for r in recs] == list(range(num_steps)), "задача " + str(tid) + ": steplog неполный или нарушен порядок"
    return rows


# ===================== 10. PREFLIGHT (CPU, без модели) =====================
def run_preflight():
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append({"check": name, "ok": True, "detail": detail})
            log("OK   " + name + (": " + str(detail)[:300] if detail else ""))
        except BaseException as e:
            results.append({"check": name, "ok": False, "detail": type(e).__name__ + ": " + str(e)[:500]})
            log("FAIL " + name + ": " + type(e).__name__ + ": " + str(e)[:500])

    log("PREFLIGHT: все проверки без GPU и без загрузки модели")
    check("параметры и хэши (общие = ожидаемому)", lambda: {"common_sha256": COMMON_SHA256, "exp_sha256": EXP_SHA256})
    check("окружение, версии, веса модели, импорты", lambda: (check_environment(False), "версии и файлы проверены")[1])
    if USE_PC:
        check("компилятор и Python.h (нужны Triton)", lambda: (lambda c: c if c["ok"] else (_ for _ in ()).throw(RuntimeError("нет компилятора или Python.h: " + json.dumps(c, ensure_ascii=False))))(check_compiler()))
    check("pyjuice импортируется (заглушка запрещена)", lambda: (prepare_pyjuice(), "ok")[1])
    check("код репозитория CoDD и lm_eval импортируются", lambda: (import_repo_code(), "ok")[1])
    check("sha256 исходников модели (warning/stop)", lambda: (check_model_code(), {k: v["verdict"] for k, v in MODEL_CODE_CHECK.items()})[1])

    def hmm_file_check():
        assert os.path.isfile(HMM_FILE), "нет файла " + HMM_FILE
        assert HMM_SHA256, "нет manifest_hmm.json или в нём нет model.jpc: " + HMM_MANIFEST
        with open(HMM_MANIFEST, encoding="utf-8") as f:
            man = json.load(f)
        entry = [e for e in man["files"] if e["path"].endswith("/model.jpc")][0]
        size = os.path.getsize(HMM_FILE)
        assert size == entry["size"], "размер " + str(size) + " != manifest " + str(entry["size"])
        sha = sha256_file(HMM_FILE)
        assert sha == entry["sha256"], "sha256 файла HMM не совпал с manifest"
        return {"size": size, "sha256": sha}
    if USE_PC:
        check("файл HMM: размер и sha256 по manifest", hmm_file_check)

    def hmm_structure():
        s = inspect_jpc(HMM_FILE)
        ENV_INFO["hmm_structure"] = s
        return {k: s[k] for k in ("by_type", "total_param_count", "total_nodes_input", "input_scope_vars", "input_nodes_per_var_if_uniform", "params_fp32_gib_lower_bound")}
    if USE_PC:
        check("структура HMM (чтение pickle без исполнения кода)", hmm_structure)

    def prompts_check():
        ids, lens, insts = build_prompts_cpu()
        h = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert h == EXPECTED_IDS_SHA256, "хэш task_id " + h + " != ожидаемому"
        assert max(lens) + GEN_LENGTH <= 4096, "самый длинный промпт " + str(max(lens)) + " + GEN_LENGTH > 4096"
        for inst in insts[:3]:
            assert SYSTEM_INSTRUCTION in inst.args[0] and inst.doc["question"] in inst.args[0]
        return {"ids_sha256": h, "prompt_tokens_min_max": [min(lens), max(lens)], "first_prompt_head": insts[0].args[0][:200]}
    check("GSM8K из parquet, промпты 200 выбранных задач (код lm_eval), хэш id", prompts_check)
    check("места на диске для результатов", lambda: {"free_gib": round(shutil.disk_usage(PROJECT_DIR).free / GIB, 1)})
    ok = all(r["ok"] for r in results)
    write_json_atomic(PREFLIGHT_PATH, {"experiment": EXP_NAME, "ok": ok, "finished_utc": utc_now(), "exp_sha256": EXP_SHA256, "checks": results})
    log("PREFLIGHT: " + ("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if ok else "ЕСТЬ ОШИБКИ: " + str([r["check"] for r in results if not r["ok"]])))
    if not ok:
        raise SystemExit("preflight не пройден (см. выше и preflight.json)")


# ===================== 11. PCTEST (GPU, ДО загрузки 8B) =====================
def real_blocks_from_gold(n_states):
    # блоки по 32 токена из эталонных решений реальных задач GSM8K (токены - токенайзером с диска), ~44% позиций скрыты
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    table = pq.read_table(find_parquet("test"))
    answers = table.column("answer").to_pylist()
    rng = np.random.default_rng(2024)
    states = []
    for t in range(n_states):
        ids = tok.encode(answers[t], add_special_tokens=False)
        ids = (ids + [tok.eos_token_id] * BLOCK_LENGTH)[:BLOCK_LENGTH]
        masked = np.zeros(BLOCK_LENGTH, dtype=bool)
        masked[rng.choice(BLOCK_LENGTH, size=14, replace=False)] = True       # 14/32 = 0.4375 < PC_FRAC: HMM активна
        states.append({"task": t, "ids": ids, "masked": masked})
    return states


def synthetic_soft_logits(state, V, rng, peak=3.0):
    # Мягкие оценки "нейросети" для ТЕСТА: шум + пик на эталонном слове. НЕ логиты 8B (в pctest модель не загружается).
    lg = torch.from_numpy(rng.standard_normal((BLOCK_LENGTH, V)).astype(np.float32))
    for i, tid in enumerate(state["ids"]):
        lg[i, tid] += peak
    return lg


def e4_statistical_check(pc, V, device):
    idx = get_var_index(pc, BLOCK_LENGTH)
    layer, sid = idx["layer"], idx["sid"]
    rng = np.random.default_rng(777)
    states = real_blocks_from_gold(4)
    n = int(E4_N_SAMPLES)
    report = {"n_samples": n, "states": [], "states_per_var": sorted(set(idx["states_per_var"])), "criteria": {
        "flows_sum_tol": 1e-3, "marginal_tolerance_sigma": 6.0, "independent_z_ok": 3.5, "independent_z_max": 5.0, "joint_z_min": 4.0, "joint_marginal_abs_floor": 0.08}}
    ok_flows = ok_marg = ok_indep = True
    joint_dependent_pairs = 0
    for st in states:
        x_ids = torch.tensor([[(MASK_ID if st["masked"][i] else st["ids"][i]) for i in range(BLOCK_LENGTH)]], dtype=torch.long, device=device)
        lg = synthetic_soft_logits(st, V, rng).to(device)
        ext = F.log_softmax(lg.unsqueeze(0) / PC_TEMPERATURE, dim=-1)
        vm = ~(x_ids == MASK_ID)
        pc_forward(pc, x_ids, ext, vm, False)
        flows = theme_posteriors(pc, x_ids)
        pis = []
        for v in range(BLOCK_LENGTH):
            f = flows[sid + idx["var_nodes"][v], 0].float().clamp_min(0.0)
            tot = float(f.sum())
            if abs(tot - 1.0) > report["criteria"]["flows_sum_tol"]:
                ok_flows = False
            pis.append((f / max(tot, 1e-30)).cpu().numpy())
        # --- joint: n сэмплов тем оригинальным sample (батчами по 50) ---
        node_to_state = []
        for v in range(BLOCK_LENGTH):
            lookup = {int(nd): j for j, nd in enumerate(idx["var_nodes"][v].tolist())}
            node_to_state.append(lookup)
        joint = np.zeros((n, BLOCK_LENGTH), dtype=np.int64)
        done_n = 0
        while done_n < n:
            b = min(50, n - done_n)
            bx = x_ids.expand(b, -1).contiguous()
            ex = ext.expand(b, -1, -1).contiguous()
            vmb = vm.expand(b, -1).contiguous()
            pc_forward(pc, bx, ex, vmb, False)
            ns = juice.queries.sample(pc, conditional=True, external_categorical_logps=ex, _sample_input_ns=False)
            for j in range(b):
                col = ns[:, j]
                nodes = (col[col >= 0] - sid)
                vids = layer.vids[nodes, 0].tolist()
                for nd, v in zip(nodes.tolist(), vids):
                    if v < BLOCK_LENGTH:
                        joint[done_n + j, v] = node_to_state[v][nd]
            done_n += b
        # --- independent: n независимых выборов из потоков ---
        indep = np.zeros((n, BLOCK_LENGTH), dtype=np.int64)
        for v in range(BLOCK_LENGTH):
            indep[:, v] = np.random.default_rng(1000 + v).choice(len(pis[v]), size=n, p=pis[v])
        # (i) частоты против потоков
        worst = 0.0
        worst_by = {"joint": 0.0, "indep": 0.0}
        worst_pos = {}

        for v in range(BLOCK_LENGTH):
            for name, smp in (("joint", joint), ("indep", indep)):
                freq = np.bincount(smp[:, v], minlength=len(pis[v])) / float(n)
                top = np.argsort(-pis[v])[:5]
                for s_ in top:
                    pi = pis[v][s_]
                    tol = report["criteria"]["marginal_tolerance_sigma"] * math.sqrt(max(pi * (1 - pi), 1e-12) / n) + 1.0 / n
                    if name == "joint":
                        tol = max(tol, report["criteria"]["joint_marginal_abs_floor"])   # ослабленный допуск маргиналов joint
                    dev = abs(freq[s_] - pi)
                    worst = max(worst, dev / tol)
                    if dev / tol > worst_by[name]:
                        worst_by[name] = dev / tol
                        worst_pos[name] = [v, int(s_), round(float(pi), 4), round(float(freq[s_]), 4), bool(st["masked"][v])]
                    if dev > tol:
                        ok_marg = False
        # (ii) зависимость между темами соседних закрытых позиций
        mrng = np.random.default_rng(5)
        masked_pos = [i for i in range(BLOCK_LENGTH) if st["masked"][i]]
        pairs = [(a, b) for a in masked_pos for b in masked_pos if b == a + 1] or [(masked_pos[0], masked_pos[1])]
        pair_rows = []
        for (a, b) in pairs:
            top_a = [int(s_) for s_ in np.argsort(-pis[a])[:4]]
            top_b = [int(s_) for s_ in np.argsort(-pis[b])[:4]]
            row = {"pair": [a, b]}
            for name, smp in (("joint", joint), ("indep", indep)):
                xa = bin_by_top_states(smp[:, a], top_a)
                xb = bin_by_top_states(smp[:, b], top_b)
                mi, null_mean, null_sd, z = mi_permutation_z(xa, xb, 5, 200, mrng)
                row[name] = {"mi": round(mi, 5), "null_mean": round(null_mean, 5), "z": round(z, 2)}
            pair_rows.append(row)
            if abs(row["indep"]["z"]) > report["criteria"]["independent_z_max"]:
                ok_indep = False
            if row["joint"]["z"] > report["criteria"]["joint_z_min"]:
                joint_dependent_pairs += 1
        frac_bad = sum(1 for r in pair_rows if abs(r["indep"]["z"]) > report["criteria"]["independent_z_ok"]) / float(len(pair_rows))
        if frac_bad > 0.30:
            ok_indep = False
        report["states"].append({"task": st["task"], "masked_positions": masked_pos, "worst_marginal_dev_over_tol": round(worst, 3), "worst_by": worst_by, "worst_pos": worst_pos, "pairs": pair_rows,
                                 "indep_frac_pairs_z_gt_3.5": round(frac_bad, 3)})
    joint_confirms_dependence = joint_dependent_pairs >= 1
    report.update({"flows_sum_to_one": ok_flows, "marginals_match": ok_marg, "independent_ok": ok_indep, "joint_pairs_with_dependence": joint_dependent_pairs,
                   "joint_shows_dependence": joint_confirms_dependence})
    report["passed"] = bool(ok_flows and ok_marg and ok_indep and joint_confirms_dependence)
    if not joint_confirms_dependence:
        report["verdict_note"] = "НЕ ПОДТВЕРЖДЕНО: в режиме joint зависимость между темами не обнаружена ни в одном состоянии - тест не различает joint и independent"
    return report


def run_pctest():
    if not USE_PC:
        raise SystemExit("pctest не нужен: USE_PC=False (HMM не используется)")
    gpu_index, total = check_environment(True)
    comp = check_compiler()
    assert comp["ok"], "нет компилятора или Python.h для Triton: " + json.dumps(comp, ensure_ascii=False) + " (установите gcc / python3-dev; интернета нет - попросите администратора образа)"
    prepare_pyjuice()
    import_repo_code()
    write_env_and_config()
    V = 126464
    res = {"experiment": EXP_NAME, "started_utc": START_UTC, "compiler": comp}
    try:
        torch.cuda.reset_peak_memory_stats()
        pc, info = load_hmm(torch.device("cuda", torch.cuda.current_device()))
        res["hmm_load"] = info
        res["hmm_resident_gib"] = round(torch.cuda.memory_allocated() / GIB, 3)
        states = real_blocks_from_gold(1)
        st = states[0]
        L = 160
        start_idx = L - BLOCK_LENGTH
        x = torch.tensor([[st["ids"][i % BLOCK_LENGTH] for i in range(L)]], dtype=torch.long, device=f"cuda:{torch.cuda.current_device()}")
        for i in range(BLOCK_LENGTH):
            if st["masked"][i]:
                x[0, start_idx + i] = MASK_ID
        g = torch.Generator(device="cpu").manual_seed(1)
        logits_base = (torch.randn((1, L, V), generator=g) * 3.0).to(device=f"cuda:{torch.cuda.current_device()}", dtype=torch.bfloat16)
        cfg_vocab = V
        times = []
        for call in range(6):
            lg = logits_base.clone()
            torch.cuda.synchronize()
            t0 = time.time()
            apply_pc_logits(pc_model=pc, input_ids=x, logits=lg, start_idx=start_idx, end_idx=L, mask_id=MASK_ID, vocab_size=cfg_vocab,
                            pc_temperature=PC_TEMPERATURE, pc_frac=PC_FRAC, reverse_frac=REVERSE_FRAC, record_cudagraph=(call == 0))
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            assert bool(torch.isfinite(lg[:, start_idx:L]).all()), "NaN/inf после apply_pc_logits"
        res["apply_pc_logits_first_call_s"] = round(times[0], 3)
        res["apply_pc_logits_repeat_s"] = [round(t_, 4) for t_ in times[1:]]
        res["gpu_peak_gib_hmm_only"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
        res["gpu_total_gib"] = round(total / GIB, 2)
        projected = EXP1_PEAK_GIB + res["gpu_peak_gib_hmm_only"]
        res["projected_peak_gib_with_8b"] = round(projected, 2)
        res["projected_headroom_gib"] = round(total / GIB - projected, 2)
        log("pctest: первый вызов " + str(res["apply_pc_logits_first_call_s"]) + " c, повторные " + str(res["apply_pc_logits_repeat_s"]) + " c; HMM занимает " + str(res["hmm_resident_gib"])
            + " ГиБ, пик " + str(res["gpu_peak_gib_hmm_only"]) + " ГиБ; прогноз пика с 8B " + str(res["projected_peak_gib_with_8b"]) + " ГиБ, запас " + str(res["projected_headroom_gib"]) + " ГиБ")
        res["fits_with_8b"] = res["projected_headroom_gib"] >= 2.0
        if not res["fits_with_8b"]:
            res["ok"] = False
            res["message"] = "HMM + 8B не помещаются с запасом >= 2 ГиБ: полный прогон невозможен на этой карте"
        else:
            res["ok"] = True
        if THEME_MODE == "independent" and res["ok"]:
            log("Э4: Шаг C - статистическая проверка независимого выбора темы (синтетические мягкие оценки, реальные блоки токенов)")
            t0 = time.time()
            impl_order = ["flows", "pool"] if E4_IMPL == "auto" else [E4_IMPL]
            e4 = {"tried": []}
            for impl in impl_order:
                if impl == "flows":
                    try:
                        rep = e4_statistical_check(pc, V, torch.device("cuda", torch.cuda.current_device()))
                    except BaseException as e:
                        rep = {"passed": False, "error": type(e).__name__ + ": " + str(e)[:400], "traceback": traceback.format_exc()}
                        print(rep["traceback"], flush=True)
                    e4["tried"].append({"impl": "flows", "report": rep})
                    if rep.get("passed"):
                        e4["impl_resolved"] = "flows"
                        e4["report"] = rep
                        break
                else:
                    e4["tried"].append({"impl": "pool", "report": {"passed": False, "note": "статистическая проверка пула цепочек не автоматизирована: требует 200 вызовов sample; реализация apply_pool доступна, но не подтверждена"}})
            e4["seconds"] = round(time.time() - t0, 1)
            e4["passed"] = "impl_resolved" in e4
            res["e4"] = e4
            if e4["passed"]:
                write_json_atomic(E4_FLAG_PATH, {"experiment": EXP_NAME, "E4_CHECK_PASSED": True, "impl": e4["impl_resolved"], "exp_sha256": EXP_SHA256, "finished_utc": utc_now()})
                log("E4_CHECK_PASSED записан: реализация '" + e4["impl_resolved"] + "'")
            else:
                if os.path.exists(E4_FLAG_PATH):
                    os.remove(E4_FLAG_PATH)
                res["ok"] = False
                res["message"] = "Э4: проверка независимого выбора темы НЕ пройдена - полный прогон Э4 не будет запущен (см. e4.tried в pctest.json)"
                log("Э4: проверка НЕ пройдена")
    except BaseException as e:
        res["ok"] = False
        res["error"] = type(e).__name__ + ": " + str(e)[:800]
        res["traceback"] = traceback.format_exc()
        print(res["traceback"], flush=True)
        log("pctest: ОШИБКА: " + res["error"])
        if "gcc" in str(e).lower() or "compil" in str(e).lower() or "Python.h" in str(e):
            res["hint"] = "похоже на ошибку компиляции Triton: проверьте gcc, Python.h (шаг preflight)"
    res["exp_sha256"] = EXP_SHA256
    res["finished_utc"] = utc_now()
    write_json_atomic(PCTEST_PATH, res)
    if not res.get("ok"):
        raise SystemExit("pctest не пройден: " + str(res.get("message") or res.get("error")))
    log("pctest пройден")


# ===================== 12. ДЫМОВОЙ ТЕСТ =====================
def assert_same(name_a, a, name_b, b, prompt_len, where, hint=""):
    if a.shape == b.shape and bool((a == b).all()):
        return
    if a.shape == b.shape:
        diff_pos = (a != b).nonzero()[:, 1].tolist()
        first = diff_pos[0]
        detail = "позиций отличается: " + str(len(diff_pos)) + "; первая: " + str(first - prompt_len) + " в ответе; токены " + str(a[0, first].item()) + " (" + name_a + ") и " + str(b[0, first].item()) + " (" + name_b + ")"
    else:
        detail = "формы различаются"
    raise AssertionError("РАСХОЖДЕНИЕ [" + where + "]: " + name_a + " != " + name_b + ". " + detail + ". " + hint + "Остановка.")


def check_steplog(steplog, x, prompt_len, gen_length, block_length, num_steps, where):
    num_blocks = gen_length // block_length
    steps = num_steps // num_blocks
    assert len(steplog) == num_steps, "[" + where + "] в журнале " + str(len(steplog)) + " шагов вместо " + str(num_steps)
    seen = set()
    revealed_in_block = {}
    for k, rec in enumerate(steplog):
        b, i = divmod(k, steps)
        assert (rec["block"], rec["step_in_block"], rec["global_step"]) == (b, i, k)
        start, end = prompt_len + b * block_length, prompt_len + (b + 1) * block_length
        expected_n = block_length // steps + (1 if i < block_length % steps else 0)
        pos = rec["positions_abs"]
        assert len(pos) == expected_n and all(start <= p < end for p in pos) and not (set(pos) & seen), "[" + where + "] шаг " + str(k) + ": позиции некорректны"
        seen |= set(pos)
        assert rec["token_ids"] == [x[0, p].item() for p in pos]
        done_before = revealed_in_block.get(b, 0)
        assert rec["n_masked_block_before"] == block_length - done_before
        revealed_in_block[b] = done_before + len(pos)
        conf = rec["confidence"]
        assert all(math.isfinite(c) and -1e-2 <= c <= 1 + 1e-2 for c in conf) and all(conf[t] >= conf[t + 1] for t in range(len(conf) - 1)), "[" + where + "] шаг " + str(k) + ": уверенность некорректна"
        assert 0.0 <= rec["delta_words"] <= 1.0 and 0.0 <= rec["delta_order"] <= 1.0
        assert rec["pc_active"] or (rec["delta_words"] == 0.0 and rec["delta_order"] == 0.0), "[" + where + "] неактивный шаг с ненулевой дельтой"
    assert seen == set(range(prompt_len, prompt_len + gen_length))


def first_diff_gen_pos(a, b, prompt_len):
    d = (a != b).nonzero()
    return None if d.numel() == 0 else int(d[0, 1].item()) - prompt_len


def run_smoke(gpu_index):
    import contextlib
    smoke_n, smoke_steps, smoke_gen, smoke_block = 3, 8, 64, 32
    smoke_insts = INSTANCES[:smoke_n]
    impl = resolve_e4_impl()
    log("Вариант: слова из " + WORDS_FROM + ", уверенность/порядок из " + ORDER_FROM + ", тема: " + THEME_MODE + ("/" + str(impl) if impl else ""))

    def quiet_call(fn, *a, **kw):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*a, **kw)

    log("A. Путь оценки на синтетических ответах")
    for inst in smoke_insts:
        gold_text = inst.doc["answer"]
        s_ok, p_ok = score_response(inst, gold_text)
        assert s_ok["flexible-extract"] == 1.0 and s_ok["strict-match"] == 1.0, "эталонный ответ оценён как неверный: путь оценки сломан"
        gold_num = gold_final(gold_text).replace(",", "")
        try:
            wrong_num = str(int(float(gold_num)) + 1)
        except ValueError:
            continue
        s_bad, _p = score_response(inst, gold_text.rsplit("####", 1)[0] + "#### " + wrong_num)
        assert s_bad["flexible-extract"] == 0.0 and s_bad["strict-match"] == 0.0, "искажённый ответ оценён как верный"
    log("   OK")

    log("B. Логиты базовой модели конечны")
    for inst in smoke_insts:
        p_len = len(PROMPT_IDS[inst.doc_id])
        probe = torch.full((1, p_len + smoke_gen), MASK_ID, dtype=torch.long, device=model.device)
        probe[0, :p_len] = torch.tensor(PROMPT_IDS[inst.doc_id], dtype=torch.long, device=model.device)
        with torch.no_grad():
            lg = LLaDAModelLM.forward(model, probe).logits
        assert bool(torch.isfinite(lg).all()), "NaN/inf в логитах базовой модели"
        del probe, lg
        torch.cuda.empty_cache()
    log("   OK")

    equivalence = {"hmm_off_vs_original": [], "pc_joint_vs_original": []}
    log("C. Копия против оригинала (HMM выключена; затем HMM активна, тема joint, A=C=pc, одинаковый seed)")
    for remasking in REMASKING_LIST:
        for inst in smoke_insts:
            prompt = torch.tensor([PROMPT_IDS[inst.doc_id]], dtype=torch.long, device=model.device)
            P = prompt.shape[1]
            common = dict(num_steps=smoke_steps, gen_length=smoke_gen, block_length=smoke_block, temperature=TEMPERATURE, cfg_scale=0., remasking=remasking, mask_id=MASK_ID)
            where = "remasking=" + remasking + ", задача " + str(inst.doc_id)
            # (a) HMM выключена: копия == оригинал (база и путь CoddLlada use_pc=False)
            x_off, log_off = generate_variant(model, prompt, smoke_steps, smoke_gen, smoke_block, remasking, "base", "base", "joint", False, check_finite=True)
            with torch.no_grad():
                x_o_plain = quiet_call(llada_diffusion_generate, PLAIN, prompt, **common)
                x_o_codd = quiet_call(llada_diffusion_generate, model, prompt, use_pc=False, **common)
            assert_same("копия (HMM выкл.)", x_off, "оригинал на PlainView", x_o_plain, P, where)
            assert_same("копия (HMM выкл.)", x_off, "оригинал на CoddLlada(use_pc=False)", x_o_codd, P, where)
            check_steplog(log_off, x_off, P, smoke_gen, smoke_block, smoke_steps, where + " [HMM выкл.]")
            assert all(not r["pc_active"] for r in log_off)
            equivalence["hmm_off_vs_original"].append(where)
            if not USE_PC:
                torch.cuda.empty_cache()
                continue      # f1: HMM не используется, пункт (b) не применим
            # (b) HMM активна, joint, A=C=pc: копия против оригинального CoDD при одинаковом seed
            s = derive_seed(RUN_SEEDS[0], inst.doc_id)
            seed_everything(s)
            x_pc, log_pc = generate_variant(model, prompt, smoke_steps, smoke_gen, smoke_block, remasking, "pc", "pc", "joint", True, check_finite=True)
            check_steplog(log_pc, x_pc, P, smoke_gen, smoke_block, smoke_steps, where + " [HMM, joint]")
            n_active = sum(1 for r in log_pc if r["pc_active"])
            assert n_active >= 1, "[" + where + "] HMM ни разу не была активна за 8 шагов: smoke не проверяет HMM"
            seed_everything(s)
            with torch.no_grad():
                x_o1 = quiet_call(llada_diffusion_generate, model, prompt, use_pc=True, pc_temperature=PC_TEMPERATURE, pc_frac=PC_FRAC, reverse_frac=REVERSE_FRAC, **common)
            seed_everything(s)
            with torch.no_grad():
                x_o2 = quiet_call(llada_diffusion_generate, model, prompt, use_pc=True, pc_temperature=PC_TEMPERATURE, pc_frac=PC_FRAC, reverse_frac=REVERSE_FRAC, **common)
            orig_deterministic = bool((x_o1 == x_o2).all())
            same = bool((x_pc == x_o1).all())
            rec = {"where": where, "pc_active_steps": n_active, "original_self_consistent": orig_deterministic, "copy_equals_original": same}
            if not same:
                p_gen = first_diff_gen_pos(x_pc, x_o1, P)
                step_of_p = [r["global_step"] for r in log_pc if (P + p_gen) in r["positions_abs"]][0]
                first_active = min(r["global_step"] for r in log_pc if r["pc_active"])
                rec.update({"first_diff_gen_pos": p_gen, "revealed_at_step": step_of_p, "first_active_step": first_active})
                assert step_of_p >= first_active, "[" + where + "] расхождение копии с оригиналом ДО первой активации HMM (шаг " + str(step_of_p) + " < " + str(first_active) + "): ошибка в копии"
                assert not orig_deterministic, "[" + where + "] копия расходится с оригиналом при активной HMM, хотя оригинал воспроизводим при том же seed: RNG/операции копии не совпадают с оригиналом CoDD"
                log("   ВНИМАНИЕ [" + where + "]: расхождение после активации HMM, оригинал сам невоспроизводим при том же seed (недетерминизм GPU/Triton) - ограничение зафиксировано")
            else:
                log("   OK [" + where + "]: копия == оригинал CoDD при одном seed (оригинал воспроизводим: " + str(orig_deterministic) + "), активных шагов HMM: " + str(n_active))
            equivalence["pc_joint_vs_original"].append(rec)
            torch.cuda.empty_cache()

    log("D. Собственный вариант эксперимента (слова из " + WORDS_FROM + ", порядок из " + ORDER_FROM + ", тема " + THEME_MODE + "): конечность и журнал")
    for remasking in REMASKING_LIST:
        for inst in smoke_insts:
            prompt = torch.tensor([PROMPT_IDS[inst.doc_id]], dtype=torch.long, device=model.device)
            seed_everything(derive_seed(RUN_SEEDS[0], inst.doc_id))
            x_v, log_v = generate_variant(model, prompt, smoke_steps, smoke_gen, smoke_block, remasking, WORDS_FROM, ORDER_FROM, THEME_MODE, USE_PC, impl=impl, check_finite=True)
            check_steplog(log_v, x_v, prompt.shape[1], smoke_gen, smoke_block, smoke_steps, "вариант, remasking=" + remasking + ", задача " + str(inst.doc_id))
            assert (not USE_PC) or any(r["pc_active"] for r in log_v)
            torch.cuda.empty_cache()
    log("   OK")

    # --- пик видеопамяти при РЕАЛЬНОМ GEN_LENGTH с активной HMM, на самой длинной задаче ---
    longest = max(INSTANCES, key=lambda i: len(PROMPT_IDS[i.doc_id]))
    total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
    peak_report = {}
    steps_for_peak = NUM_BLOCKS_GLOBAL() * 4
    for rm in REMASKING_LIST:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_everything(derive_seed(RUN_SEEDS[0], longest.doc_id))
        _x, _log, _plen, _sec = run_generation(longest, rm, steps_for_peak, impl=impl, check_finite=True)
        assert (not USE_PC) or sum(1 for r in _log if r["pc_active"]) >= 1, "HMM не была активна в замере пика"
        peak_alloc, peak_res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        peak_report[rm] = {"peak_allocated_gib": peak_alloc / GIB, "peak_reserved_gib": peak_res / GIB, "headroom_gib": (total_mem - peak_alloc) / GIB}
        log("Пик памяти [" + rm + "], самая длинная задача, HMM активна: allocated " + format(peak_alloc / GIB, ".2f") + " ГиБ, reserved " + format(peak_res / GIB, ".2f")
            + " ГиБ из " + format(total_mem / GIB, ".2f") + " ГиБ; запас " + format((total_mem - peak_alloc) / GIB, ".2f") + " ГиБ")
        assert total_mem - peak_alloc >= 2 * GIB, "Запас видеопамяти " + format((total_mem - peak_alloc) / GIB, ".2f") + " ГиБ < 2 ГиБ с HMM (remasking=" + rm + "). Полный прогон невозможен."
        torch.cuda.empty_cache()

    # --- время: базовый шаг без HMM, HMM-шаг, прогноз ---
    timing_steps = NUM_STEPS_LIST[0]
    inst0, inst1 = INSTANCES[0], INSTANCES[1]
    seed_everything(derive_seed(RUN_SEEDS[0], inst0.doc_id))
    _x, _l, _p, t_off = run_generation(inst0, REMASKING_LIST[0], timing_steps, use_hmm=False)
    torch.cuda.empty_cache()
    t_on = []
    n_act = []
    for inst in (inst0, inst1):
        seed_everything(derive_seed(RUN_SEEDS[0], inst.doc_id))
        _x, _l, _p, t_ = run_generation(inst, REMASKING_LIST[0], timing_steps, impl=impl)
        t_on.append(t_)
        n_act.append(sum(1 for r in _l if r["pc_active"]))
        torch.cuda.empty_cache()
    sec_per_step_base = t_off / timing_steps
    mean_on = sum(t_on) / len(t_on)
    mean_active = sum(n_act) / float(len(n_act))
    sec_per_active_step_extra = max(0.0, (mean_on - t_off)) / max(1.0, mean_active)
    if not USE_PC:
        sec_per_active_step_extra = 0.0
    log("Замер (steps=" + str(timing_steps) + ", " + REMASKING_LIST[0] + "): без HMM " + format(t_off, ".1f") + " c/задачу (" + format(sec_per_step_base, ".3f") + " c/шаг); с HMM " + format(mean_on, ".1f")
        + " c/задачу при " + format(mean_active, ".1f") + " активных шагах => доплата за активный шаг HMM ~" + format(sec_per_active_step_extra, ".2f") + " c")
    forecast = {}

    def forecast_seconds(seeds_n, n_tasks):
        total = 0.0
        for rm in REMASKING_LIST:
            for st in NUM_STEPS_LIST:
                na = count_active_steps(st, GEN_LENGTH, BLOCK_LENGTH, PC_FRAC)
                total += n_tasks * seeds_n * (sec_per_step_base * st + sec_per_active_step_extra * na)
        return total

    n_eff = RUN_LIMIT if RUN_LIMIT is not None else N_TASKS
    forecast["all_seeds_seconds"] = forecast_seconds(len(SEEDS), n_eff)
    forecast["run_seeds_seconds"] = forecast_seconds(len(RUN_SEEDS), n_eff)
    for st in NUM_STEPS_LIST:
        log("   steps=" + str(st) + ": активных шагов HMM за задачу " + str(count_active_steps(st, GEN_LENGTH, BLOCK_LENGTH, PC_FRAC)) + " из " + str(st))
    log("ПРОГНОЗ (оценка, не гарантия; 4 комбинации x " + str(n_eff) + " задач): на RUN_SEEDS=" + str(RUN_SEEDS) + " ~" + str(datetime.timedelta(seconds=int(forecast["run_seeds_seconds"])))
        + " (~" + format(forecast["run_seeds_seconds"] / 3600, ".1f") + " ч); на все SEEDS=" + str(SEEDS) + " ~" + str(datetime.timedelta(seconds=int(forecast["all_seeds_seconds"]))) + " (~" + format(forecast["all_seeds_seconds"] / 3600, ".1f") + " ч)")
    write_json_atomic(SMOKE_PATH, {
        "ok": True, "experiment": EXP_NAME, "finished_utc": utc_now(), "exp_sha256": EXP_SHA256, "gen_length": GEN_LENGTH, "dtype": DTYPE,
        "e4_impl": impl, "equivalence": equivalence, "peak_memory": peak_report, "total_gpu_memory_gib": total_mem / GIB,
        "sec_per_step_base": sec_per_step_base, "sec_per_active_step_extra": sec_per_active_step_extra, "mean_active_steps_timing_run": mean_active,
        "forecast": forecast, "run_seeds": RUN_SEEDS, "n_tasks_forecast": n_eff,
    })
    log("smoke.json записан: полный прогон разрешён.")


def NUM_BLOCKS_GLOBAL():
    return GEN_LENGTH // BLOCK_LENGTH


# ===================== 13. ПОЛНЫЙ ПРОГОН =====================
def load_json_if(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_gates_for_full():
    if REQUIRE_SMOKE:
        sm = load_json_if(SMOKE_PATH)
        assert sm and sm.get("ok") is True, "Нет успешного smoke.json: сначала режим smoke"
        assert sm.get("exp_sha256") == EXP_SHA256, "smoke.json получен при другом конфиге: повторите smoke"
        log("smoke.json: успешный дымовой тест от " + str(sm.get("finished_utc")))
    if REQUIRE_PCTEST and USE_PC:
        pt = load_json_if(PCTEST_PATH)
        assert pt and pt.get("ok") is True and pt.get("exp_sha256") == EXP_SHA256, "Нет успешного pctest.json для этого конфига: сначала режим pctest"
    if THEME_MODE == "independent":
        flag = load_json_if(E4_FLAG_PATH)
        assert flag and flag.get("E4_CHECK_PASSED") is True and flag.get("exp_sha256") == EXP_SHA256, "Э4: нет флага E4_CHECK_PASSED (проверка независимого выбора темы не пройдена или не проводилась): полный прогон запрещён"


def run_full():
    impl = resolve_e4_impl()
    run_instances = INSTANCES if RUN_LIMIT is None else INSTANCES[:RUN_LIMIT]
    if RUN_LIMIT is not None:
        log("=" * 80)
        log("ВНИМАНИЕ: RUN_LIMIT=" + str(RUN_LIMIT) + ": ПРОГОН НЕПОЛНЫЙ.")
        log("=" * 80)
    log("Эксперимент " + EXP_NAME + ": seeds=" + str(RUN_SEEDS) + ", задач " + str(len(run_instances)) + ", комбинаций " + str(len(REMASKING_LIST) * len(NUM_STEPS_LIST)) + ", реализация темы: " + str(impl))
    t_all = time.time()
    deadline = None if MAX_RUNTIME_MIN is None else t_all + 60 * MAX_RUNTIME_MIN
    stopped = None
    for seed in RUN_SEEDS:
        for remasking in REMASKING_LIST:
            for num_steps in NUM_STEPS_LIST:
                if STOP_REQUESTED["flag"]:
                    stopped = "signal"
                    break
                if deadline is not None and time.time() >= deadline:
                    stopped = "stopped"
                    break
                status = run_combo(remasking, num_steps, seed, run_instances, impl, deadline=deadline)
                if status != "done":
                    stopped = status
                    break
            if stopped:
                break
        if stopped:
            break
    log("Цикл прогона завершён за " + str(datetime.timedelta(seconds=int(time.time() - t_all))) + ", остановка: " + str(stopped))
    for seed in RUN_SEEDS:
        for remasking in REMASKING_LIST:
            for num_steps in NUM_STEPS_LIST:
                rows = read_jsonl_tolerant(result_path(remasking, num_steps, seed))
                present = sorted(r["task_id"] for r in rows)
                if stopped or RUN_LIMIT is not None:
                    validate_combo(remasking, num_steps, seed, present)
                else:
                    validate_combo(remasking, num_steps, seed, TASK_IDS)
    if stopped:
        return "STOPPED", "остановлено (" + stopped + "); запустите режим full ещё раз, готовые задачи будут пропущены"
    return ("DONE", "все комбинации для seeds " + str(RUN_SEEDS)) if RUN_LIMIT is None else ("DONE", "НЕПОЛНЫЙ прогон (RUN_LIMIT=" + str(RUN_LIMIT) + ")")


# ===================== 14. СВОДКА =====================
def row_value(r, kind):
    # kind "asis": flexible-extract как есть; "fin": неоконченные ответы (нет EOS в 512 токенах) считаются неверными
    if kind == "fin":
        return int(int(r["correct"]) == 1 and bool(r.get("finished")))
    return int(r["correct"])


def per_task_values(path, kind):
    # {task_id: 0/1} из файла результатов (только чтение)
    return dict((r["task_id"], row_value(r, kind)) for r in read_jsonl(path, repair=False))


def paired_vs(other_dir, label, remasking, num_steps, my_by_seed, kind):
    # Парное сравнение с другим экспериментом по task_id: для каждого seed (если есть и там) и усреднённо по seeds.
    out = {"label": label, "kind": kind, "by_seed": {}, "seed_averaged": None}
    other_all = {}
    for seed in my_by_seed:
        other = per_task_values(os.path.join(other_dir, remasking + "_steps" + str(num_steps) + "_seed" + str(seed) + ".jsonl"), kind)
        if not other:
            continue
        other_all[seed] = other
        mine = my_by_seed[seed]
        common = sorted(set(mine) & set(other))
        if not common:
            continue
        a = [mine[t] for t in common]
        b = [other[t] for t in common]
        n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        d, lo, hi = paired_bootstrap_diff(a, b, N_BOOTSTRAP, BOOTSTRAP_SEED)
        out["by_seed"][str(seed)] = {"n_common": len(common), "diff_mean": d, "diff_ci": [lo, hi], "mine_correct_other_wrong": n10, "mine_wrong_other_correct": n01,
                                     "mcnemar_exact_p": mcnemar_exact(n10, n01)}
    if len(other_all) >= 1:
        seeds_both = sorted(other_all)
        tasks = sorted(set.intersection(*[set(my_by_seed[s_]) & set(other_all[s_]) for s_ in seeds_both]))
        if tasks:
            a = [sum(my_by_seed[s_][t] for s_ in seeds_both) / float(len(seeds_both)) for t in tasks]
            b = [sum(other_all[s_][t] for s_ in seeds_both) / float(len(seeds_both)) for t in tasks]
            d, lo, hi = paired_bootstrap_diff(a, b, N_BOOTSTRAP, BOOTSTRAP_SEED)
            out["seed_averaged"] = {"seeds": seeds_both, "n_common": len(tasks), "diff_mean": d, "diff_ci": [lo, hi]}
    return out


def run_summary():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids_sha = (load_json_if(TASKS_PATH) or {}).get("ids_sha256")
    table = []
    compare = []
    same_as_f1 = os.path.realpath(EXP1_RESULTS_DIR) == os.path.realpath(OUT_DIR)
    for remasking in REMASKING_LIST:
        for num_steps in NUM_STEPS_LIST:
            by_seed = {"asis": {}, "fin": {}}
            for seed in SEEDS:
                rows = sorted(read_jsonl(result_path(remasking, num_steps, seed), repair=False), key=lambda r: r["task_id"])
                if not rows:
                    continue
                for kind in ("asis", "fin"):
                    by_seed[kind][seed] = dict((r["task_id"], row_value(r, kind)) for r in rows)
                acc, lo, hi = bootstrap_ci([row_value(r, "asis") for r in rows], N_BOOTSTRAP, BOOTSTRAP_SEED)
                acc_f, lo_f, hi_f = bootstrap_ci([row_value(r, "fin") for r in rows], N_BOOTSTRAP, BOOTSTRAP_SEED)
                acc_s, lo_s, hi_s = bootstrap_ci([r["correct_strict"] for r in rows], N_BOOTSTRAP, BOOTSTRAP_SEED)
                unf, lo_u, hi_u = bootstrap_ci([0 if r.get("finished") else 1 for r in rows], N_BOOTSTRAP, BOOTSTRAP_SEED)
                act = [r["pc_active_steps"] for r in rows]
                dws = [r["delta_words"] for r in rows if r["delta_words"] is not None]
                dos = [r["delta_order"] for r in rows if r["delta_order"] is not None]
                table.append({"remasking": remasking, "num_steps": num_steps, "seed": seed, "n": len(rows), "accuracy": acc, "ci_low": lo, "ci_high": hi,
                              "accuracy_unfinished_wrong": acc_f, "ci_low_unfinished_wrong": lo_f, "ci_high_unfinished_wrong": hi_f,
                              "unfinished_share": unf, "unfinished_ci": [lo_u, hi_u], "n_unfinished": sum(1 for r in rows if not r.get("finished")),
                              "accuracy_strict": acc_s, "ci_low_strict": lo_s, "ci_high_strict": hi_s, "mean_pc_active_steps": sum(act) / float(len(act)),
                              "mean_delta_words": mean_or_none(dws), "mean_delta_order": mean_or_none(dos)})
            if by_seed["asis"]:
                for kind in ("asis", "fin"):
                    entry = {"remasking": remasking, "num_steps": num_steps, "kind": kind}
                    if not same_as_f1:
                        entry["vs_f1"] = paired_vs(EXP1_RESULTS_DIR, "f1_base_full512", remasking, num_steps, by_seed[kind], kind)
                    if THEME_MODE == "independent":
                        entry["vs_f5"] = paired_vs(EXP5_RESULTS_DIR, "f5_codd_full512", remasking, num_steps, by_seed[kind], kind)
                    compare.append(entry)
    log(EXP_NAME + " | GSM8K test, " + str(N_TASKS) + " задач | GEN_LENGTH=" + str(GEN_LENGTH) + " | bootstrap " + str(N_BOOTSTRAP) + " повторов, 95% | sha256 id: " + str(ids_sha))
    log("Неоконченный ответ: в 512 сгенерированных токенах нет EOS (<|endoftext|> или <|eot_id|>). 'как есть' = flexible-extract; 'нео. неверны' = неоконченные считаются неверными.")
    header = "remasking".ljust(15) + "steps".rjust(6) + "seed".rjust(5) + "n".rjust(6) + "  " + "acc как есть [95% CI]".ljust(25) + "acc нео.неверны".rjust(17) + "неоконч.".rjust(10) + "HMM-шагов".rjust(10) + "d_words".rjust(9) + "d_order".rjust(9)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in table:
        a = format(100 * r["accuracy"], "5.1f") + "% [" + format(100 * r["ci_low"], "5.1f") + "; " + format(100 * r["ci_high"], "5.1f") + "]"
        print(r["remasking"].ljust(15) + str(r["num_steps"]).rjust(6) + str(r["seed"]).rjust(5) + str(r["n"]).rjust(6) + "  " + a.ljust(25)
              + (format(100 * r["accuracy_unfinished_wrong"], "5.1f") + "%").rjust(17) + (format(100 * r["unfinished_share"], "4.1f") + "%").rjust(10)
              + format(r["mean_pc_active_steps"], ".1f").rjust(10)
              + (format(r["mean_delta_words"], ".4f") if r.get("mean_delta_words") is not None else "-").rjust(9)
              + (format(r["mean_delta_order"], ".4f") if r.get("mean_delta_order") is not None else "-").rjust(9), flush=True)
    for c in compare:
        for key in ("vs_f1", "vs_f5"):
            if key not in c:
                continue
            blk = c[key]
            tag = "как есть" if c["kind"] == "asis" else "нео. неверны"
            if not blk["by_seed"] and not blk["seed_averaged"]:
                print("  [" + c["remasking"] + "/" + str(c["num_steps"]) + ", " + tag + "] парное сравнение с " + blk["label"] + ": файлов для сравнения нет", flush=True)
                continue
            for seed, v in blk["by_seed"].items():
                print("  [" + c["remasking"] + "/" + str(c["num_steps"]) + " seed" + seed + ", " + tag + "] разность с " + blk["label"] + ": " + format(100 * v["diff_mean"], "+.1f") + " п.п. ["
                      + format(100 * v["diff_ci"][0], "+.1f") + "; " + format(100 * v["diff_ci"][1], "+.1f") + "], несогласных пар " + str(v["mine_correct_other_wrong"]) + "/"
                      + str(v["mine_wrong_other_correct"]) + ", Макнемар p=" + format(v["mcnemar_exact_p"], ".4f"), flush=True)
    complete = bool(table) and all(r["n"] == N_TASKS for r in table)
    write_json_atomic(SUMMARY_PATH, {"experiment": EXP_NAME, "gen_length": GEN_LENGTH, "n_tasks": N_TASKS, "ids_sha256": ids_sha, "n_bootstrap": N_BOOTSTRAP,
                                     "bootstrap_seed": BOOTSTRAP_SEED, "complete": complete, "table": table, "paired_comparisons": compare, "updated_utc": utc_now(),
                                     "unfinished_definition": "в сгенерированных GEN_LENGTH токенах нет EOS (eos_token_id токенайзера или eot_token_id HFLM)",
                                     "note": "Доверительный интервал отражает выбор задач; один seed, разброса по seeds нет"})
    log("summary.json записан: " + SUMMARY_PATH)
    return complete


# ===================== MAIN =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Ядро эксперимента " + str(EXPERIMENT.get("EXP_NAME")))
    ap.add_argument("--mode", required=True, choices=["preflight", "pctest", "smoke", "full", "summary"])
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    init_config(cfg, args.mode)
    if MODE == "summary":
        set_status("running", "summary")
        run_summary()
        set_status("done", "summary")
        return 0
    set_status("running", "запуск режима " + MODE)
    if MODE == "preflight":
        run_preflight()
        set_status("done", "preflight пройден")
        return 0
    if MODE == "pctest":
        run_pctest()
        set_status("done", "pctest пройден")
        return 0
    if MODE == "full":
        signal.signal(signal.SIGTERM, _on_stop_signal)
        signal.signal(signal.SIGINT, _on_stop_signal)
    gpu_index, _total = check_environment(True)
    prepare_pyjuice()
    import_repo_code()
    write_env_and_config()
    if MODE == "smoke" and REQUIRE_PCTEST and USE_PC:
        pt = load_json_if(PCTEST_PATH)
        assert pt and pt.get("ok") is True and pt.get("exp_sha256") == EXP_SHA256, "Нет успешного pctest.json для этого конфига: сначала режим pctest"
    if MODE == "full":
        check_gates_for_full()
    load_model(with_hmm=USE_PC)
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
