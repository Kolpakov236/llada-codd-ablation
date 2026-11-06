"""download_hmm.py - скачивает веса HMM il18/llada-gsm-pc и готовит их для загрузки в DataSphere.

Запуск (из venv с huggingface_hub==0.36.0 и numpy):  python -I download_hmm.py
Что делает:
  1. список файлов репозитория и размеры/sha256 из API Hugging Face;
  2. скачивание в offline_bundle_hmm/llada-gsm-pc/ (обычный HTTP, без hf-xet: докачка после обрыва не теряет прогресс);
  3. сверка размеров и sha256 (LFS) с API;
  4. при необходимости (файл > 1.5 ГБ) разрезание на части по 1500 МБ (десятичных);
  5. offline_bundle_hmm/manifest_hmm.json и SHA256SUMS_HMM.txt (пути относительно папки проекта);
  6. БЕЗОПАСНОЕ чтение структуры .jpc (pickle читается распаковщиком-заглушкой: код из файла не исполняется,
     pyjuice не импортируется) -> offline_bundle_hmm/hmm_structure.json.
"""
import datetime
import hashlib
import importlib
import io
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"      # до импорта huggingface_hub

REPO_ID = "il18/llada-gsm-pc"
PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "offline_bundle_hmm"
DEST = OUT_ROOT / "llada-gsm-pc"
SPLIT_THRESHOLD = 1_500_000_000
SPLIT_CHUNK = 1_500_000_000
IO_CHUNK = 64 * 1024 * 1024


def log(msg=""):
    print("[" + datetime.datetime.now().strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def human(n):
    n = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1000 or unit == "ГБ":
            return (format(n, ".0f") if unit == "Б" else format(n, ".2f")) + " " + unit
        n /= 1000.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(IO_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# 1-3. список файлов, скачивание, сверка
# ---------------------------------------------------------------------------------------------
def fetch_listing():
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(REPO_ID)
    info = api.model_info(REPO_ID, files_metadata=True)
    meta = {}
    for s in info.siblings:
        meta[s.rfilename] = {"size": s.size, "lfs_sha256": (s.lfs.sha256 if getattr(s, "lfs", None) else None)}
    log("Репозиторий " + REPO_ID + ": gated=" + str(info.gated) + ", private=" + str(info.private) + ", обновлён " + str(info.last_modified))
    log("list_repo_files: " + str(files))
    for name in sorted(meta):
        log("   " + name.ljust(24) + str(meta[name]["size"]).rjust(14) + " Б   lfs.sha256=" + str(meta[name]["lfs_sha256"]))
    return files, meta


def download(files):
    from huggingface_hub import snapshot_download
    DEST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(repo_id=REPO_ID, local_dir=str(DEST), max_workers=4)
    shutil.rmtree(DEST / ".cache", ignore_errors=True)
    log("Скачивание завершено за " + format((time.time() - t0) / 60, ".1f") + " мин")


def verify(meta):
    problems = []
    log("Сверка с API Hugging Face:")
    for name in sorted(meta):
        p = DEST / name
        parts = sorted(DEST.glob(name + ".part*"))
        if p.exists():
            size = p.stat().st_size
            sha = sha256_file(p)
        elif parts:
            size = sum(x.stat().st_size for x in parts)
            h = hashlib.sha256()
            for x in parts:
                with open(x, "rb") as f:
                    for chunk in iter(lambda: f.read(IO_CHUNK), b""):
                        h.update(chunk)
            sha = h.hexdigest()
        else:
            problems.append(name + ": файла нет")
            continue
        ok_size = (size == meta[name]["size"])
        lfs = meta[name]["lfs_sha256"]
        ok_sha = (lfs is None) or (sha == lfs)
        log("   " + name.ljust(24) + str(size).rjust(14) + " Б  размер " + ("OK" if ok_size else "ОШИБКА") + "  sha256 " + sha[:16] + "... "
            + ("(совпал с LFS)" if (lfs and ok_sha) else ("(НЕ совпал с LFS!)" if lfs else "(не LFS-файл: сверен только размер)")))
        if not ok_size or not ok_sha:
            problems.append(name + ": размер или sha256 не совпали")
    if problems:
        raise SystemExit("Проблемы: " + str(problems))
    log("Все файлы совпали с API.")


# ---------------------------------------------------------------------------------------------
# 4. разрезание
# ---------------------------------------------------------------------------------------------
def maybe_split():
    records = {}
    for p in sorted(DEST.rglob("*")):
        if not p.is_file() or ".part" in p.suffix or p.stat().st_size <= SPLIT_THRESHOLD:
            continue
        rel = p.relative_to(PROJECT).as_posix()
        log("Режу " + rel + " (" + human(p.stat().st_size) + ") на части по " + human(SPLIT_CHUNK))
        whole = hashlib.sha256()
        parts = []
        with open(p, "rb") as src:
            idx = 0
            while True:
                part_path = p.parent / (p.name + ".part" + format(idx, "03d"))
                ph = hashlib.sha256()
                written = 0
                with open(part_path, "wb") as dst:
                    while written < SPLIT_CHUNK:
                        buf = src.read(min(IO_CHUNK, SPLIT_CHUNK - written))
                        if not buf:
                            break
                        dst.write(buf)
                        whole.update(buf)
                        ph.update(buf)
                        written += len(buf)
                if written == 0:
                    part_path.unlink()
                    break
                parts.append({"path": part_path.relative_to(PROJECT).as_posix(), "size": written, "sha256": ph.hexdigest()})
                idx += 1
        check = hashlib.sha256()
        for part in parts:
            with open(PROJECT / part["path"], "rb") as f:
                for chunk in iter(lambda: f.read(IO_CHUNK), b""):
                    check.update(chunk)
        assert check.hexdigest() == whole.hexdigest(), "склейка частей не совпала с исходником: " + rel
        records[rel] = {"original_path": rel, "original_size": p.stat().st_size, "original_sha256": whole.hexdigest(), "parts": parts,
                        "join_command": "cat " + p.name + ".part* > " + p.name}
        p.unlink()
        log("   разрезано на " + str(len(parts)) + " частей, склейка == исходник")
    return records


# ---------------------------------------------------------------------------------------------
# 5. manifest и SHA256SUMS
# ---------------------------------------------------------------------------------------------
def write_manifest(meta, split_records):
    files = []
    for p in sorted(OUT_ROOT.rglob("*")):
        if not p.is_file() or p.name in ("manifest_hmm.json", "SHA256SUMS_HMM.txt") or p.name.startswith("."):
            continue
        rel = p.relative_to(PROJECT).as_posix()
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "tool": "download_hmm.py",
        "source": {"repo_id": REPO_ID, "files_in_repo": sorted(meta)},
        "split": {"threshold_bytes": SPLIT_THRESHOLD, "chunk_bytes": SPLIT_CHUNK, "note": "1 МБ = 10^6 байт; склейка: cat <имя>.part* > <имя>"},
        "files": files,
        "split_files": list(split_records.values()),
        "totals": {"files": len(files), "bytes": sum(f["size"] for f in files)},
    }
    (OUT_ROOT / "manifest_hmm.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_ROOT / "SHA256SUMS_HMM.txt").write_text("".join(f["sha256"] + "  " + f["path"] + "\n" for f in files), encoding="utf-8")
    log("manifest_hmm.json и SHA256SUMS_HMM.txt записаны: " + str(len(files)) + " файлов, " + human(manifest["totals"]["bytes"]))
    return manifest


# ---------------------------------------------------------------------------------------------
# 6. безопасное чтение структуры .jpc
# ---------------------------------------------------------------------------------------------
class _Stub:
    # Инертная заглушка для ЛЮБОГО неизвестного класса/функции из pickle: ничего не исполняет, только запоминает аргументы.
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        self._state = state


_BUILTIN_OK = {("builtins", n) for n in ("set", "frozenset", "tuple", "list", "dict", "int", "float", "str", "bytes", "bytearray", "complex", "slice", "range")}
_BUILTIN_OK.add(("collections", "OrderedDict"))
_NUMPY_NAMES = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _BUILTIN_OK:
            return getattr(importlib.import_module(module), name)
        if (module, name) in _NUMPY_NAMES:
            import numpy as np
            if name == "ndarray":
                return np.ndarray
            if name == "dtype":
                return np.dtype
            mod = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
            return getattr(mod, name)
        stub = type(name, (_Stub,), {"_origin": module + "." + name})
        return stub


def safe_loads(data):
    return SafeUnpickler(io.BytesIO(data)).load()


def inspect_jpc(path):
    import numpy as np
    with open(path, "rb") as f:
        nodes_list = SafeUnpickler(f).load()
    assert isinstance(nodes_list, list), "ожидался список словарей, получено " + type(nodes_list).__name__
    summary = {"file": str(path), "num_node_sets": len(nodes_list), "by_type": {}, "groups": {}, "total_param_count": 0, "total_param_bytes_as_stored": 0,
               "param_dtypes": {}, "input": {}, "sum_edges_blocks": 0, "total_nodes_input": 0, "total_nodes_sum": 0, "total_nodes_prod": 0}
    all_scope_vars = set()
    dist_origins = {}
    for ns in nodes_list:
        t = ns["type"]
        nb, bs = ns.get("num_node_blocks"), ns.get("block_size")
        nodes = (nb or 0) * (bs or 1)
        summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        key = t + " blocks=" + str(nb) + " bs=" + str(bs)
        if t == "Input":
            summary["total_nodes_input"] += nodes
            scope = list(ns["scope"])
            all_scope_vars.update(scope)
            key += " scope_len=" + str(len(scope))
            d = safe_loads(ns["dist"])
            dist_origins[getattr(d, "_origin", type(d).__name__)] = dist_origins.get(getattr(d, "_origin", type(d).__name__), 0) + 1
            args = getattr(d, "_args", None)
            if args:
                summary["input"].setdefault("dist_args_examples", [])
                if len(summary["input"]["dist_args_examples"]) < 3:
                    summary["input"]["dist_args_examples"].append([repr(a) for a in args])
        elif t == "Sum":
            summary["total_nodes_sum"] += nodes
            e = ns.get("edge_ids")
            if e is not None:
                summary["sum_edges_blocks"] += int(e.shape[1]) if getattr(e, "ndim", 0) == 2 else int(len(e))
        else:
            summary["total_nodes_prod"] += nodes
        if "params" in ns:
            arr = ns["params"]
            summary["total_param_count"] += int(arr.size)
            summary["total_param_bytes_as_stored"] += int(arr.size * arr.dtype.itemsize)
            dt = str(arr.dtype)
            summary["param_dtypes"][dt] = summary["param_dtypes"].get(dt, 0) + int(arr.size)
        summary["groups"][key] = summary["groups"].get(key, 0) + 1
    summary["input"]["scope_vars"] = len(all_scope_vars)
    summary["input"]["scope_var_ids_min_max"] = [min(all_scope_vars), max(all_scope_vars)] if all_scope_vars else None
    summary["input"]["nodes_per_var_if_uniform"] = (summary["total_nodes_input"] / len(all_scope_vars)) if all_scope_vars else None
    summary["input"]["dist_classes"] = dist_origins
    summary["estimate_gpu_params_fp32_bytes"] = summary["total_param_count"] * 4
    summary["note"] = ("Число параметров и узлов прочитано из файла. Оценка памяти GPU ограничена снизу весами (fp32); индексные тензоры слоёв "
                       "и буферы node_mars/element_mars/flows считает PyJuice при compile - их размер НЕ ПРОВЕРЕН без GPU (его измерит pctest).")
    return summary


def main():
    log("Папка проекта: " + str(PROJECT) + " | свободно на диске: " + format(shutil.disk_usage(PROJECT).free / 1024 ** 3, ".1f") + " ГиБ")
    files, meta = fetch_listing()
    download(files)
    verify(meta)
    records = maybe_split()
    write_manifest(meta, records)
    jpc = DEST / "model.jpc"
    if jpc.exists():
        log("Безопасное чтение структуры " + jpc.name + " (без исполнения кода из файла)")
        s = inspect_jpc(jpc)
        (OUT_ROOT / "hmm_structure.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        log(json.dumps({k: s[k] for k in ("num_node_sets", "by_type", "total_param_count", "param_dtypes", "total_nodes_input", "total_nodes_sum",
                                          "total_nodes_prod", "sum_edges_blocks", "estimate_gpu_params_fp32_bytes", "input")}, ensure_ascii=False, indent=1))
        # hmm_structure.json - служебный файл рядом с manifest; обновим manifest, чтобы он в нём тоже был
        write_manifest(meta, records)
    else:
        log("model.jpc лежит частями: структура не читалась (склейте и запустите ещё раз)")
    log("ГОТОВО")


if __name__ == "__main__":
    main()
