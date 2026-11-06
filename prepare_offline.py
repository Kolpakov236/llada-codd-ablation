#!/usr/bin/env python3
"""prepare_offline.py - собирает папку offline_bundle/ для Yandex DataSphere БЕЗ интернета на целевой машине.

Целевая среда: Linux x86_64, Python 3.10, драйвер NVIDIA 535 (CUDA 12.2), GPU L4.
Запускать из venv с huggingface_hub==0.36.0 и pyarrow, ключ -I рекомендуется:

    python -I prepare_offline.py <шаг> --out offline_bundle

Шаги (каждый запускается отдельно и идемпотентен):
    preflight   проверка диска и версий, создание структуры папок
    weights     скачать GSAI-ML/LLaDA-8B-Instruct целиком в weights/LLaDA-8B-Instruct
    dataset     скачать main/test и main/train датасета openai/gsm8k в data/gsm8k
    resolve     разрешить зависимости под Linux/cp310 (pip --dry-run --report) и записать requirements*.txt
    wheelhouse  скачать колёса в wheelhouse/ и проверить, что всё разрешается без сети
    sdists      безопасно проверить и собрать колёса для чисто питоновских пакетов без колёс (--pkg name==ver)
    split       разрезать файлы > 2 ГБ на части по 1500 МБ
    manifest    manifest.json, SHA256SUMS*.txt, README_OFFLINE.md
    verify      полная перепроверка sha256 по manifest.json и склейки частей

Размеры: 1 ГБ = 10^9 байт, 1 МБ = 10^6 байт (десятичные), чтобы части были < 1.5 ГБ в любом толковании.
Ничего из скачанного не исполняется, кроме сборки чисто питоновских sdist (шаг sdists, только по явной команде).
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_DIRNAME = "LLaDA-8B-Instruct"
DATASET_ID = "openai/gsm8k"
DATASET_FILES = ["main/test-00000-of-00001.parquet", "main/train-00000-of-00001.parquet"]
DATASET_ROWS = {"main/test-00000-of-00001.parquet": 1319, "main/train-00000-of-00001.parquet": 7473}

PYPI_INDEX = "https://pypi.org/simple"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TARGET_PYTHON = "3.10"

# pip --platform не раскрывает совместимость сам: перечисляем все теги, под которыми бывают колёса Linux x86_64.
# torch 2.9.1 и triton 3.5.1 помечены manylinux_2_28/2_27, поэтому одного manylinux2014 недостаточно.
PLATFORMS = [
    "manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_26_x86_64", "manylinux_2_24_x86_64",
    "manylinux_2_17_x86_64", "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64",
]

SPLIT_THRESHOLD = 2_000_000_000
SPLIT_CHUNK = 1_500_000_000
IO_CHUNK = 64 * 1024 * 1024
EXCLUDE_NAMES_FROM_REQUIREMENTS = {"typing"}   # устаревший backport typing 3.7.4.3 ломает Python 3.10


def log(msg=""):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def human(n):
    n = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if n < 1000 or unit == "ТБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.2f} {unit}"
        n /= 1000


def sha256_file(path, progress_label=None):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    t0 = time.time()
    last = t0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(IO_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_label and size > 500_000_000 and time.time() - last > 15:
                last = time.time()
                log(f"   sha256 {progress_label}: {done / size * 100:.0f}% ({human(done)} из {human(size)})")
    return h.hexdigest()


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def bundle_dir(args):
    return Path(args.out).resolve()


PIP_CACHE = {"dir": None}   # временный кэш pip (в scratchpad), чтобы резолвер и download не качали большие колёса дважды


def pip_cmd():
    return [sys.executable, "-m", "pip"]


def cache_args():
    return ["--cache-dir", PIP_CACHE["dir"]] if PIP_CACHE["dir"] else ["--no-cache-dir"]


def platform_args():
    a = []
    for p in PLATFORMS:
        a += ["--platform", p]
    a += ["--python-version", TARGET_PYTHON, "--implementation", "cp", "--abi", "cp310", "--only-binary=:all:"]
    return a


def index_args(with_torch=True):
    a = ["--index-url", PYPI_INDEX]
    if with_torch:
        a += ["--extra-index-url", TORCH_INDEX]
    return a


def run(cmd, **kw):
    log("$ " + " ".join(shlex.quote(str(c)) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def require_hf():
    try:
        import huggingface_hub  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Нужен huggingface_hub (pip install huggingface_hub==0.36.0): {e}")


# ----------------------------------------------------------------------------------------------
def cmd_preflight(args):
    out = bundle_dir(args)
    log(f"Python скрипта: {sys.version.split()[0]} ({sys.executable})")
    log(f"pip: {subprocess.run(pip_cmd() + ['--version'], capture_output=True, text=True).stdout.strip()}")
    require_hf()
    import huggingface_hub
    log(f"huggingface_hub {huggingface_hub.__version__}")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("weights", "data/gsm8k", "wheelhouse"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    log(f"Папка бандла: {out}")
    log(f"Свободно на диске: {free / 1024 ** 3:.1f} ГиБ (нужно не меньше {args.min_free_gib})")
    if free / 1024 ** 3 < args.min_free_gib:
        raise SystemExit("Недостаточно свободного места. Остановка.")
    log("preflight OK")


def hf_expected(repo_id, repo_type):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id, files_metadata=True) if repo_type == "model" else HfApi().dataset_info(repo_id, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}


def cmd_weights(args):
    require_hf()
    from huggingface_hub import snapshot_download
    out = bundle_dir(args)
    dest = out / "weights" / MODEL_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    expected = hf_expected(MODEL_ID, "model")
    log(f"{MODEL_ID}: файлов {len(expected)}, всего {human(sum(expected.values()))}")
    # шарды, уже разрезанные на части (шаг split), повторно не качаем
    already_split = [n for n in expected if (dest / (n + ".part000")).exists()]
    if already_split:
        log(f"Уже разрезаны (пропускаю загрузку): {already_split}")
    log("Скачивание (докачка продолжается с места обрыва)...")
    t0 = time.time()
    snapshot_download(repo_id=MODEL_ID, local_dir=str(dest), max_workers=4, ignore_patterns=already_split or None)
    log(f"Загрузка завершена за {(time.time() - t0) / 60:.1f} мин")
    shutil.rmtree(dest / ".cache", ignore_errors=True)   # служебные метаданные загрузчика, в бандл не нужны

    log("Проверка по размерам из API Hugging Face:")
    problems = []
    rows = []
    for name in sorted(expected):
        p = dest / name
        if p.exists():
            got = p.stat().st_size
        else:
            parts = sorted(dest.glob(name + ".part*"))
            got = sum(x.stat().st_size for x in parts) if parts else None
        ok = got == expected[name]
        rows.append((name, expected[name], got, ok))
        if not ok:
            problems.append(name)
    w = max(len(r[0]) for r in rows)
    for name, exp, got, ok in rows:
        log(f"   {name:<{w}}  {exp:>14,}  {human(exp):>10}  {'OK' if ok else 'ОШИБКА (получено ' + str(got) + ')'}")
    log(f"Итого: {human(sum(r[1] for r in rows))} ({sum(r[1] for r in rows):,} байт)")
    for must in ("modeling_llada.py", "configuration_llada.py", "tokenizer.json", "config.json"):
        if must not in expected:
            problems.append("нет в репозитории: " + must)
    if problems:
        raise SystemExit(f"Проблемы: {problems}")
    log("weights OK")


def cmd_dataset(args):
    require_hf()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    out = bundle_dir(args)
    dest = out / "data" / "gsm8k"
    dest.mkdir(parents=True, exist_ok=True)
    expected = hf_expected(DATASET_ID, "dataset")
    log(f"Датасет {DATASET_ID}: в репозитории {len(expected)} файлов; качаю только {DATASET_FILES}")
    for fname in DATASET_FILES:
        assert fname in expected, f"в {DATASET_ID} нет {fname}; есть: {sorted(expected)}"
        hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=fname, local_dir=str(dest))
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    for fname in DATASET_FILES:
        p = dest / fname
        assert p.stat().st_size == expected[fname], f"{fname}: размер {p.stat().st_size} != {expected[fname]}"
        t = pq.read_table(p)
        assert {"question", "answer"} <= set(t.column_names), f"{fname}: столбцы {t.column_names}"
        assert t.num_rows == DATASET_ROWS[fname], f"{fname}: строк {t.num_rows}, ожидалось {DATASET_ROWS[fname]}"
        log(f"   {fname}: {p.stat().st_size:,} Б, строк {t.num_rows}, столбцы {t.column_names}  OK")
    log("dataset OK")


# ----------------------------------------------------------------------------------------------
def parse_env_yml_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s+-\s+([A-Za-z0-9_.\-]+)==([^\s]+)\s*$", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def cmd_resolve(args):
    out = bundle_dir(args)
    req_in, req_out = out / args.req_in, out / args.req_out
    assert req_in.exists(), f"нет {req_in}"
    tmp = Path(tempfile.mkdtemp(prefix="resolve_"))
    report = tmp / "report.json"
    cmd = pip_cmd() + ["install", "--dry-run", "--ignore-installed", "--quiet", "--target", str(tmp / "t"),
                       "--report", str(report)] + cache_args() + platform_args() + index_args()
    wh = out / "wheelhouse"
    if any(wh.glob("*.whl")):
        cmd += ["--find-links", str(wh)]
    if args.constraint:
        cmd += ["-c", str(out / args.constraint)]
    cmd += ["-r", str(req_in)]
    r = run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("РЕЗОЛВЕР ЗАВЕРШИЛСЯ ОШИБКОЙ:")
        print((r.stdout or "")[-3000:])
        print((r.stderr or "")[-6000:])
        raise SystemExit(1)
    data = json.loads(report.read_text(encoding="utf-8"))
    items = []
    for it in data["install"]:
        meta = it["metadata"]
        name, ver = meta["name"], meta["version"]
        items.append((norm(name), name, ver, it["download_info"]["url"]))
    items.sort()
    excluded = [i for i in items if i[0] in EXCLUDE_NAMES_FROM_REQUIREMENTS]
    items = [i for i in items if i[0] not in EXCLUDE_NAMES_FROM_REQUIREMENTS]
    lines = [f"# Сгенерировано prepare_offline.py resolve {utc_now()}",
             f"# Платформа: Linux x86_64 (manylinux_2_28 и старше), Python {TARGET_PYTHON}, cp310; torch: {TORCH_INDEX}",
             "# Установка: pip install --no-index --find-links wheelhouse -r requirements.txt"]
    lines += [f"{i[1]}=={i[2]}" for i in items]
    req_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Записан {req_out.name}: {len(items)} пакетов (исключено: {[i[1] for i in excluded]})")

    sizes = {}
    # отклонения от environment.yml
    env_pins = parse_env_yml_pins(args.env_yml) if args.env_yml else {}
    dev = []
    for n, name, ver, url in items:
        if n in env_pins and env_pins[n] != ver:
            dev.append((name, env_pins[n], ver))
    if dev:
        log("Отклонения от environment.yml (имя: было -> стало):")
        for name, a, b in dev:
            log(f"   {name}: {a} -> {b}")
    (out / (args.req_out + ".deviations.json")).write_text(json.dumps(
        {"created_utc": utc_now(), "deviations_from_environment_yml": [{"name": a, "env": b, "resolved": c} for a, b, c in dev],
         "sources": {i[1]: i[3] for i in items}}, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    log("resolve OK")


def wheel_key(filename):
    parts = filename[:-4].split("-")
    return norm(parts[0]), parts[1]


def cmd_wheelhouse(args):
    out = bundle_dir(args)
    wh = out / "wheelhouse"
    wh.mkdir(parents=True, exist_ok=True)
    req = out / args.req
    # Медленный и рвущийся канал. pip download сохраняет файлы только в конце всего запуска, и сбой одного пакета обнуляет
    # остальные, поэтому качаем ПО ОДНОМУ пакету: готовые колёса остаются в wheelhouse, сбойные повторяются.
    t0 = time.time()
    lines = [l.strip() for l in req.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    failed = []
    for i, spec in enumerate(lines, 1):
        name, ver = spec.split("==")
        have = {wheel_key(f.name) for f in wh.glob("*.whl")}
        if (norm(name), ver) in have or (norm(name), ver.replace("+", ".")) in have:
            log(f"[{i}/{len(lines)}] {spec}: уже есть, пропуск")
            continue
        ok = False
        for attempt in range(1, args.attempts + 1):
            log(f"[{i}/{len(lines)}] {spec}: попытка {attempt}/{args.attempts}")
            cmd = pip_cmd() + ["download", "--no-deps", "--retries", "10", "--timeout", "60", "--resume-retries", "30",
                               "-d", str(wh), spec] + cache_args() + platform_args() + index_args()
            if any(wh.glob("*.whl")):
                cmd += ["--find-links", str(wh)]
            if run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode == 0:
                ok = True
                break
            log("   ошибка загрузки, повтор через 20 с")
            time.sleep(20)
        if not ok:
            failed.append(spec)
    log(f"Загрузка колёс завершена за {(time.time() - t0) / 60:.1f} мин; не скачано: {failed}")
    if failed:
        raise SystemExit(f"Не удалось скачать: {failed}. Запустите шаг ещё раз (готовые пропускаются).")
    if args.pyjuice:
        r = run(pip_cmd() + ["download", "--no-deps", "--no-cache-dir", "-d", str(wh), "pyjuice==2.4.1"] + platform_args() + index_args(False))
        if r.returncode != 0:
            raise SystemExit("не удалось скачать pyjuice==2.4.1")
    have = {}
    for f in wh.glob("*.whl"):
        have[wheel_key(f.name)] = f.name
    missing = []
    for line in req.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, ver = line.split("==")
        ver_n = ver.replace("+", "+")
        if (norm(name), ver_n) not in have and (norm(name), ver_n.replace("+", ".")) not in have:
            missing.append(line)
    if missing:
        raise SystemExit(f"В wheelhouse нет колёс для: {missing}")
    total = sum(f.stat().st_size for f in wh.iterdir() if f.is_file())
    big = sorted(((f.stat().st_size, f.name) for f in wh.glob("*.whl")), reverse=True)[:8]
    log(f"wheelhouse: {len(have)} колёс, {human(total)}; крупнейшие:")
    for s, n in big:
        log(f"   {human(s):>10}  {n}")
    # проверка: всё разрешается ЛОКАЛЬНО, без сети
    tmp = Path(tempfile.mkdtemp(prefix="offline_check_"))
    chk = pip_cmd() + ["install", "--dry-run", "--ignore-installed", "--no-index", "--quiet", "--no-cache-dir", "--no-deps", "--target", str(tmp / "t"),
                       "--find-links", str(wh), "-r", str(req)] + platform_args()
    r = run(chk, capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit("Проверка установки без сети НЕ прошла")
    log("Проверка `pip install --no-index --find-links wheelhouse -r ...` (dry-run, платформа цели): OK")


# ----------------------------------------------------------------------------------------------
SUSPICIOUS_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".pyx", ".pxd", ".pxi", ".f", ".f90", ".rs", ".cu", ".cuh", ".go", ".swift"}
SUSPICIOUS_FILES = {"cmakelists.txt", "cargo.toml", "meson.build"}
SUSPICIOUS_TEXT = ["ext_modules", "Extension(", "cythonize", "maturin", "setuptools-rust", "setuptools_rust", "scikit-build", "meson",
                   "pybind11", "cffi_modules", "numpy.distutils", "cmake"]
PURE_BACKENDS = ("setuptools", "hatchling", "flit_core", "flit", "poetry.core", "pdm.backend")


def inspect_sdist(path):
    """Возвращает (чисто_питоновский: bool, причины: list). Код пакета НЕ исполняется."""
    reasons = []
    texts = {}
    names = []
    if str(path).endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                names.append(m.name)
                base = os.path.basename(m.name).lower()
                if m.isfile() and base in ("setup.py", "setup.cfg", "pyproject.toml") and m.size < 2_000_000:
                    texts[m.name] = tf.extractfile(m).read().decode("utf-8", "replace")
    else:
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                names.append(n)
                base = os.path.basename(n).lower()
                if base in ("setup.py", "setup.cfg", "pyproject.toml"):
                    texts[n] = zf.read(n).decode("utf-8", "replace")
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if ext in SUSPICIOUS_EXT:
            reasons.append(f"исходник на не-Python: {n}")
        if os.path.basename(n).lower() in SUSPICIOUS_FILES:
            reasons.append(f"файл сборки: {n}")
    for n, t in texts.items():
        for key in SUSPICIOUS_TEXT:
            if key in t:
                reasons.append(f"в {os.path.basename(n)} есть '{key}'")
        m = re.search(r'build-backend\s*=\s*"([^"]+)"', t)
        if m and not m.group(1).startswith(PURE_BACKENDS):
            reasons.append(f"нестандартный build-backend: {m.group(1)}")
    if not texts:
        reasons.append("нет setup.py/setup.cfg/pyproject.toml: нечего проверять")
    return (not reasons), reasons


def pypi_sdist(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    for u in data["urls"]:
        if u["packagetype"] == "sdist":
            return u["url"], u["filename"], u["digests"]["sha256"]
    return None


def cmd_sdists(args):
    out = bundle_dir(args)
    wh = out / "wheelhouse"
    wh.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sdist_"))
    report = []
    for spec in args.pkg:
        name, ver = spec.split("==")
        log(f"--- {spec}")
        found = pypi_sdist(name, ver)
        if not found:
            log("   sdist на PyPI не найден")
            report.append({"pkg": spec, "result": "нет sdist"})
            continue
        url, fname, digest = found
        dest = work / fname
        urllib.request.urlretrieve(url, dest)
        got = sha256_file(dest)
        if got != digest:
            raise SystemExit(f"{fname}: sha256 не совпал с PyPI ({got} != {digest})")
        pure, reasons = inspect_sdist(dest)
        if not pure:
            log(f"   НЕ чисто питоновский (или не удалось убедиться): {reasons[:6]}; колесо НЕ собираю")
            report.append({"pkg": spec, "result": "не собирал", "reasons": reasons})
            continue
        log("   исходники проверены без исполнения: компилируемых расширений нет; собираю pip wheel --no-deps (исполняется build-backend пакета)")
        bdir = work / ("wheel_" + norm(name))
        bdir.mkdir()
        r = run(pip_cmd() + ["wheel", "--no-deps", "--no-cache-dir", "-w", str(bdir), str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-3000:])
            report.append({"pkg": spec, "result": "сборка не удалась"})
            continue
        whl = list(bdir.glob("*.whl"))
        if len(whl) != 1 or not whl[0].name.endswith("-none-any.whl"):
            log(f"   результат не none-any ({[w.name for w in whl]}): отбрасываю")
            report.append({"pkg": spec, "result": "колесо не none-any, отброшено", "files": [w.name for w in whl]})
            continue
        shutil.copy2(whl[0], wh / whl[0].name)
        log(f"   OK: {whl[0].name}")
        report.append({"pkg": spec, "result": "собрано", "wheel": whl[0].name, "sdist_sha256": digest})
    shutil.rmtree(work, ignore_errors=True)
    rp = out / "sdists_report.json"
    old = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else []
    rp.write_text(json.dumps(old + report, ensure_ascii=False, indent=1), encoding="utf-8")
    log("sdists: " + json.dumps(report, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------
TARGET_ENV = {
    "implementation_name": "cpython", "implementation_version": "3.10.12", "os_name": "posix", "platform_machine": "x86_64",
    "platform_python_implementation": "CPython", "platform_release": "5.15.0", "platform_system": "Linux",
    "platform_version": "#1 SMP", "python_full_version": "3.10.12", "python_version": "3.10", "sys_platform": "linux",
}


def wheel_metadata(path):
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if not names:
            raise RuntimeError(f"{path.name}: нет METADATA")
        return zf.read(names[0]).decode("utf-8", "replace")


def cmd_closure(args):
    """Пересчёт замыкания зависимостей по METADATA колёс с маркерами ЦЕЛИ (Linux x86_64, Python 3.10),
    потому что pip на Mac вычисляет маркеры по хосту (platform_system, python_version)."""
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version
    out = bundle_dir(args)
    wh = out / "wheelhouse"
    pins = {}
    for line in (out / args.req).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            n, v = line.split("==")
            pins[norm(n)] = v
    roots = []
    for line in (out / args.req_in).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            roots.append(Requirement(line))
    for extra_req in args.extra_req:   # дополнительные закреплённые списки (например, requirements_kernel.txt)
        for line in (out / extra_req).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                n, v = line.split("==")
                pins[norm(n)] = v
    for extra_in in args.extra_in:
        for line in (out / extra_in).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                roots.append(Requirement(line))
    wheels = {}
    for f in wh.glob("*.whl"):
        wheels.setdefault(wheel_key(f.name), f)
    missing, violations, reach = {}, [], set()
    stack = [(norm(r.name), frozenset(r.extras), r) for r in roots]
    seen = set()
    no_wheel = []
    while stack:
        name, extras, req = stack.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        reach.add(name)
        if name not in pins:
            missing.setdefault(name, set()).add(str(req))
            continue
        ver = pins[name]
        if req.specifier and not req.specifier.contains(Version(ver), prereleases=True):
            violations.append(f"{name}=={ver} не удовлетворяет {req.specifier}")
        wf = wheels.get((name, ver)) or wheels.get((name, ver.replace("+", ".")))
        if wf is None:
            no_wheel.append(f"{name}=={ver}")
            continue
        md_text = wheel_metadata(wf)
        from email.parser import Parser
        msg = Parser().parsestr(md_text)
        for rd in msg.get_all("Requires-Dist") or []:
            r2 = Requirement(rd)
            ok = False
            for ex in (extras | {""}):
                env = dict(TARGET_ENV, extra=ex)
                if r2.marker is None or r2.marker.evaluate(env):
                    ok = True
                    if r2.marker is not None and ex == "" and "extra" in str(r2.marker):
                        ok = False   # зависимость относится только к extra, а extra не запрошен
                    break
            if ok:
                stack.append((norm(r2.name), frozenset(r2.extras), r2))
    unreachable = sorted(set(pins) - reach)
    log(f"Замыкание по маркерам цели: достижимо {len(reach & set(pins))} из {len(pins)} закреплённых пакетов")
    if no_wheel:
        log(f"Нет колёс в wheelhouse (скачайте шаг wheelhouse): {no_wheel[:10]}{' ...' if len(no_wheel) > 10 else ''}")
    if missing:
        log("НЕ ХВАТАЕТ в закреплённом списке (нужны на цели, добавьте в requirements.in и повторите resolve):")
        for n, who in sorted(missing.items()):
            log(f"   {n}   (требует: {sorted(who)[:3]})")
    if violations:
        log("Нарушены ограничения версий: " + "; ".join(violations))
    if unreachable:
        log(f"Закреплены, но на цели не нужны: {unreachable}")
    if args.apply_unreachable and unreachable:
        lines = [l for l in (out / args.req).read_text(encoding="utf-8").splitlines()
                 if l.startswith("#") or not l.strip() or norm(l.split("==")[0]) not in set(unreachable)]
        (out / args.req).write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"Из {args.req} удалены ненужные: {unreachable}")
    if missing or violations or no_wheel:
        raise SystemExit("closure: есть проблемы (см. выше)")
    log("closure OK")


def cmd_kernel(args):
    """Колёса ipykernel (опционально): скачивает с зависимостями под цель с constraints=requirements.txt,
    переносит новые колёса в wheelhouse и пишет requirements_kernel.txt (только пакеты, которых нет в requirements.txt)."""
    out = bundle_dir(args)
    wh = out / "wheelhouse"
    main_pins = {}
    for line in (out / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            n, v = line.split("==")
            main_pins[norm(n)] = v
    tmp = Path(tempfile.mkdtemp(prefix="kernel_"))
    cmd = pip_cmd() + ["download", "--retries", "10", "--timeout", "60", "-d", str(tmp), "-c", str(out / "requirements.txt"),
                       "-r", str(out / "requirements_kernel.in")] + cache_args() + platform_args() + index_args(False)
    if run(cmd).returncode != 0:
        raise SystemExit("pip download для ядра завершился с ошибкой")
    new = []
    for f in sorted(tmp.glob("*.whl")):
        n, v = wheel_key(f.name)
        if n in main_pins and main_pins[n] == v:
            continue
        shutil.copy2(f, wh / f.name)
        new.append(f"{n}=={v}")
    shutil.rmtree(tmp, ignore_errors=True)
    (out / "requirements_kernel.txt").write_text(
        "# Сгенерировано prepare_offline.py kernel: пакеты ядра Jupyter сверх requirements.txt\n" + "\n".join(sorted(new)) + "\n", encoding="utf-8")
    log(f"requirements_kernel.txt: {len(new)} новых пакетов: {sorted(new)}")


def cmd_split(args):
    out = bundle_dir(args)
    rec_path = out / "split_records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    cands = [p for p in sorted(out.rglob("*")) if p.is_file() and ".part" not in p.suffix and p.name not in ("manifest.json", "split_records.json")
             and p.stat().st_size > args.threshold]
    if not cands:
        log("Файлов больше порога нет (или уже разрезаны).")
    for p in cands:
        rel = p.relative_to(out).as_posix()
        size = p.stat().st_size
        log(f"Режу {rel}: {human(size)} -> части по {human(args.chunk)}")
        for old in p.parent.glob(p.name + ".part*"):
            old.unlink()
        whole = hashlib.sha256()
        parts = []
        with open(p, "rb") as src:
            idx = 0
            while True:
                part_path = p.parent / f"{p.name}.part{idx:03d}"
                ph = hashlib.sha256()
                written = 0
                with open(part_path, "wb") as dst:
                    while written < args.chunk:
                        buf = src.read(min(IO_CHUNK, args.chunk - written))
                        if not buf:
                            break
                        dst.write(buf)
                        whole.update(buf)
                        ph.update(buf)
                        written += len(buf)
                    dst.flush()
                    os.fsync(dst.fileno())
                if written == 0:
                    part_path.unlink()
                    break
                parts.append({"path": part_path.relative_to(out).as_posix(), "size": written, "sha256": ph.hexdigest()})
                log(f"   {part_path.name}: {human(written)}")
                idx += 1
        orig_sha = whole.hexdigest()
        # проверка: sha256 конкатенации частей == sha256 исходного файла (до удаления исходника)
        check = hashlib.sha256()
        for part in parts:
            with open(out / part["path"], "rb") as f:
                for chunk in iter(lambda: f.read(IO_CHUNK), b""):
                    check.update(chunk)
        assert check.hexdigest() == orig_sha, f"{rel}: склейка частей не совпала с исходником"
        assert sum(x["size"] for x in parts) == size
        log(f"   склейка частей == исходник (sha256 {orig_sha[:16]}...), исходный файл удаляю")
        p.unlink()
        records[rel] = {"original_path": rel, "original_size": size, "original_sha256": orig_sha, "parts": parts,
                        "join_command": f"cat {p.name}.part* > {p.name}"}
        rec_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"split OK (разрезано файлов: {len(cands)}; всего записей: {len(records)})")


# ----------------------------------------------------------------------------------------------
README_TEMPLATE = """# offline_bundle: установка в Yandex DataSphere без интернета

Собрано {created} скриптом prepare_offline.py. Целевая среда: Linux x86_64, Python 3.10, драйвер NVIDIA 535, GPU L4.

## Содержимое
- `weights/LLaDA-8B-Instruct/` - GSAI-ML/LLaDA-8B-Instruct целиком; файлы больше 2 ГБ разрезаны на части `*.partNNN` (по 1500 МБ).
- `data/gsm8k/main/{{test,train}}-00000-of-00001.parquet` - openai/gsm8k (test 1319 строк, train 7473).
- `wheelhouse/` - колёса для Linux x86_64 / cp310 (torch 2.9.1+cu126 и др.). `requirements.txt` - полностью закреплённый список.
- `manifest.json` - путь, размер, sha256 каждого файла; `SHA256SUMS.txt` - то же для `sha256sum -c`.

## 1. Проверка загруженного
```bash
cd offline_bundle
sha256sum -c SHA256SUMS.txt
```

## 2. Склейка частей (все разрезанные файлы)
```bash
cd weights/LLaDA-8B-Instruct
{join_lines}
cd ../..
sha256sum -c SHA256SUMS_JOINED.txt    # проверка склеенных файлов
```
Одиночная команда: `cat model-00001-of-00006.safetensors.part* > model-00001-of-00006.safetensors` (части нумеруются `.part000`, `.part001`, ... и склеиваются в лексикографическом порядке).

## 3. Установка пакетов (рекомендуется отдельное окружение, образ не трогаем)
```bash
python3.10 -m venv ~/venv_exp1 && source ~/venv_exp1/bin/activate
pip install --no-index --find-links wheelhouse -r requirements.txt
pip install --no-index --find-links wheelhouse --no-deps pyjuice==2.4.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
pyjuice ставится отдельно с `--no-deps`: его метаданные требуют пакет `typing` (устаревший backport), который на Python 3.10 ставить нельзя.
Если нужно ядро JupyterLab из этого окружения: `pip install --no-index --find-links wheelhouse -r requirements_kernel.txt && python -m ipykernel install --user --name venv_exp1` (НЕ ПРОВЕРЕНО, что DataSphere позволяет выбрать такое ядро).

## 4. Значения конфига блокнота exp1
- `MODEL_PATH` = путь к `weights/LLaDA-8B-Instruct` (после склейки), `DATA_DIR` = путь к `data/gsm8k`
- `PIP_FIND_LINKS` = путь к `wheelhouse`, `INSTALL_DEPS` можно оставить True (ставятся только недостающие пакеты)
- `CODD_REPO_DIR` = папка клона CoDD (в бандл не входит: возьмите архив/клон отдельно)
- на глубину проверки совместимости версий смотрите отчёт prepare_offline.py (раздел НЕ ПРОВЕРЕНО).
"""


def cmd_manifest(args):
    out = bundle_dir(args)
    rec_path = out / "split_records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    # README нужен до подсчёта хэшей, чтобы попасть в manifest
    join_lines = []
    for rel, rec in sorted(records.items()):
        if rel.startswith("weights/LLaDA-8B-Instruct/"):
            n = Path(rel).name
            join_lines.append(f'cat "{n}".part* > "{n}" && rm "{n}".part*')
    other = [r for r in records if not r.startswith("weights/LLaDA-8B-Instruct/")]
    if other:
        join_lines.append("# ВНИМАНИЕ: разрезаны также файлы вне weights/ (см. manifest.json -> split_files): " + ", ".join(other))
    (out / "README_OFFLINE.md").write_text(README_TEMPLATE.format(created=utc_now(), join_lines="\n".join(join_lines) or "# разрезанных файлов нет"), encoding="utf-8")

    files = []
    part_of = {part["path"]: rel for rel, rec in records.items() for part in rec["parts"]}
    known = {part["path"]: part for rec in records.values() for part in rec["parts"]}
    paths = [p for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "SHA256SUMS.txt", "SHA256SUMS_JOINED.txt")]
    log(f"manifest: файлов {len(paths)}")
    for p in paths:
        rel = p.relative_to(out).as_posix()
        size = p.stat().st_size
        if rel in known and known[rel]["size"] == size:
            sha = known[rel]["sha256"]
        else:
            sha = sha256_file(p, rel)
        e = {"path": rel, "size": size, "sha256": sha}
        if rel in part_of:
            e["part_of"] = part_of[rel]
        files.append(e)
    total = sum(f["size"] for f in files)
    manifest = {
        "created_utc": utc_now(),
        "tool": "prepare_offline.py",
        "target": {"os": "Linux x86_64", "python": TARGET_PYTHON, "driver": "NVIDIA 535 (CUDA 12.2)", "gpu": "L4",
                   "wheel_platforms": PLATFORMS, "torch_index": TORCH_INDEX},
        "split": {"threshold_bytes": SPLIT_THRESHOLD, "chunk_bytes": SPLIT_CHUNK, "naming": "<имя>.part000, .part001, ...",
                  "note": "1 МБ = 10^6 байт; склейка: cat <имя>.part* > <имя>"},
        "files": files,
        "split_files": list(records.values()),
        "totals": {"files": len(files), "bytes": total, "human": human(total)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files if f["path"] != "manifest.json"), encoding="utf-8")
    (out / "SHA256SUMS_JOINED.txt").write_text(
        "".join(f"{rec['original_sha256']}  {rec['original_path']}\n" for rec in records.values()), encoding="utf-8")
    log(f"manifest.json записан: {len(files)} файлов, {human(total)}")


def cmd_verify(args):
    out = bundle_dir(args)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    t0 = time.time()
    for i, f in enumerate(manifest["files"], 1):
        p = out / f["path"]
        if not p.exists():
            bad.append((f["path"], "нет файла"))
            continue
        if p.stat().st_size != f["size"]:
            bad.append((f["path"], "размер"))
            continue
        if sha256_file(p, f["path"]) != f["sha256"]:
            bad.append((f["path"], "sha256"))
    log(f"Проверено файлов: {len(manifest['files'])} за {time.time() - t0:.0f} c; ошибок: {len(bad)}")
    for rec in manifest["split_files"]:
        h = hashlib.sha256()
        for part in rec["parts"]:
            with open(out / part["path"], "rb") as f:
                for chunk in iter(lambda: f.read(IO_CHUNK), b""):
                    h.update(chunk)
        ok = h.hexdigest() == rec["original_sha256"]
        log(f"   склейка {Path(rec['original_path']).name}: {'OK' if ok else 'ОШИБКА'}")
        if not ok:
            bad.append((rec["original_path"], "склейка"))
    if bad:
        raise SystemExit(f"Ошибки: {bad}")
    log(f"verify OK; итого {manifest['totals']['human']} ({manifest['totals']['bytes']:,} байт)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["preflight", "weights", "dataset", "resolve", "wheelhouse", "closure", "kernel", "sdists", "split", "manifest", "verify"])
    ap.add_argument("--extra-req", action="append", default=[], help="closure: дополнительный закреплённый список (внутри бандла)")
    ap.add_argument("--extra-in", action="append", default=[], help="closure: дополнительный входной список (внутри бандла)")
    ap.add_argument("--apply-unreachable", action="store_true", help="closure: удалить из requirements.txt пакеты, не нужные на цели")
    ap.add_argument("--out", default="offline_bundle", help="папка бандла")
    ap.add_argument("--min-free-gib", type=float, default=30.0)
    ap.add_argument("--req-in", default="requirements.in")
    ap.add_argument("--req-out", default="requirements.txt")
    ap.add_argument("--req", default="requirements.txt", help="файл для шага wheelhouse")
    ap.add_argument("--constraint", default=None, help="constraints-файл (внутри бандла) для шага resolve")
    ap.add_argument("--env-yml", default=None, help="CoDD/environment.yml для отчёта об отклонениях")
    ap.add_argument("--pyjuice", action="store_true", help="wheelhouse: дополнительно скачать pyjuice==2.4.1 с --no-deps")
    ap.add_argument("--pkg", action="append", default=[], help="sdists: name==version (можно несколько раз)")
    ap.add_argument("--threshold", type=int, default=SPLIT_THRESHOLD)
    ap.add_argument("--chunk", type=int, default=SPLIT_CHUNK)
    ap.add_argument("--attempts", type=int, default=6, help="wheelhouse: число попыток на один пакет")
    ap.add_argument("--pip-cache", default=None, help="временный каталог кэша pip (удалите после работы); по умолчанию кэш выключен")
    args = ap.parse_args()
    PIP_CACHE["dir"] = args.pip_cache
    {"preflight": cmd_preflight, "weights": cmd_weights, "dataset": cmd_dataset, "resolve": cmd_resolve, "wheelhouse": cmd_wheelhouse, "closure": cmd_closure, "kernel": cmd_kernel,
     "sdists": cmd_sdists, "split": cmd_split, "manifest": cmd_manifest, "verify": cmd_verify}[args.step](args)


if __name__ == "__main__":
    main()
