"""Сборщик полных прогонов f1-f5 (GSM8K test, 1319 задач, 512 токенов, margin, 128 шагов, seed 0).
Из f_core_template.py делает f{N}_core.py, из ячеек build_experiments.py (с заменами) - блокноты f{N}_<имя>.ipynb.
Файлы exp1-exp5 (100 задач) не затрагиваются. Ничего не запускает: только собирает и статически проверяет.
Запуск:  python3 -I build_full512.py
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_experiments as B   # даёт список ячеек CELLS (при импорте ничего не пишет)

PROJECT = HERE

EXPS = {
    1: {"EXP_NAME": "f1_base_full512", "WORDS_FROM": "base", "ORDER_FROM": "base", "THEME_MODE": "joint", "USE_PC": False,
        "TITLE": "f1: обычная LLaDA без HMM, 200 случайных задач GSM8K test, 512 токенов, seed 3",
        "WHAT": "HMM выключена и не загружается (`use_hmm=False`): слова и уверенность считаются по логитам нейросети. Это база для парных сравнений f2-f5."},
    2: {"EXP_NAME": "f2_order_full512", "WORDS_FROM": "base", "ORDER_FROM": "pc", "THEME_MODE": "joint", "USE_PC": True,
        "TITLE": "f2: HMM меняет только ПОРЯДОК раскрытия, 200 случайных задач GSM8K test, 512 токенов, seed 3",
        "WHAT": "слова-кандидаты берутся из логитов нейросети ДО HMM (`logits_base`), уверенность (значит, и порядок раскрытия) считается по логитам ПОСЛЕ HMM (`logits_pc`)."},
    3: {"EXP_NAME": "f3_words_full512", "WORDS_FROM": "pc", "ORDER_FROM": "base", "THEME_MODE": "joint", "USE_PC": True,
        "TITLE": "f3: HMM меняет только СЛОВА, 200 случайных задач GSM8K test, 512 токенов, seed 3",
        "WHAT": "слова-кандидаты берутся из логитов ПОСЛЕ HMM (`logits_pc`), уверенность и порядок раскрытия считаются по логитам ДО HMM (`logits_base`)."},
    4: {"EXP_NAME": "f4_indep_full512", "WORDS_FROM": "pc", "ORDER_FROM": "pc", "THEME_MODE": "independent", "USE_PC": True,
        "TITLE": "f4: HMM есть, но тема в каждой клетке выбирается НЕЗАВИСИМО, 200 случайных задач GSM8K test, 512 токенов, seed 3",
        "WHAT": "как в полном CoDD (слова и уверенность по логитам после HMM), но скрытая тема (состояние HMM) в каждой клетке блока выбирается независимо из её маргинала p(z_i | открытые слова, оценки нейросети), без связей между клетками. **Исследовательский эксперимент**: реализация независимого выбора проверяется режимом `pctest` (шаг C); без флага `E4_CHECK_PASSED` полный прогон не стартует."},
    5: {"EXP_NAME": "f5_codd_full512", "WORDS_FROM": "pc", "ORDER_FROM": "pc", "THEME_MODE": "joint", "USE_PC": True,
        "TITLE": "f5: CoDD целиком, 200 случайных задач GSM8K test, 512 токенов, seed 3",
        "WHAT": "слова и уверенность по логитам после HMM, одна общая тема на блок на шаг (оригинальный `apply_pc_logits`)."},
}

HASH_OLD = "19272a3da75f0b6f992793038ebe987c380dfbac1848739de5b19ee20d2f7db4"
HASH_NEW = "404c3db6d6f737fb72158943ee43e3c7effcb7871e81a8dfba6492b8152361c0"

# (старый текст, новый текст) - применяются ко ВСЕМ ячейкам шаблона build_experiments.py; каждая замена обязана найтись
REPL = [
    ("N_TASKS = 100                    # первые 100 задач test-выборки в порядке датасета",
     "N_TASKS = 200                    # число задач в прогоне: 200 случайных задач из test\nTEST_SIZE = 1319                 # размер test-выборки GSM8K\nTASK_SAMPLE_SEED = 3             # seed выбора задач: id сортируются по sha256(\"seed:id\"), берутся первые N, итог по возрастанию id"),
    ("GEN_LENGTH = 256\nBLOCK_LENGTH = 32", "GEN_LENGTH = 512\nBLOCK_LENGTH = 32"),
    ('NUM_STEPS_LIST = [64, 128]\nREMASKING_LIST = ["low_confidence", "margin"]\n', 'NUM_STEPS_LIST = [128]\nREMASKING_LIST = ["margin"]\n'),
    ("SEEDS = [0, 1, 2]\n\n# ---------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА", "SEEDS = [3]\n\n# ---------- ПАРАМЕТРЫ ЭКСПЕРИМЕНТА"),
    ("RUN_SEEDS = [0]                  # какие из SEEDS реально считать (вне хэшей); полный набор [0, 1, 2]", "RUN_SEEDS = [3]                  # какие из SEEDS реально считать (вне хэшей); SEEDS = [3], других нет"),
    ('assert (DATASET, DATASET_SPLIT, N_TASKS) == ("gsm8k", "test", 100), "DATASET/DATASET_SPLIT/N_TASKS - общие параметры"', 'assert (DATASET, DATASET_SPLIT, N_TASKS) == ("gsm8k", "test", 200), "DATASET/DATASET_SPLIT/N_TASKS - общие параметры"\nassert (TEST_SIZE, TASK_SAMPLE_SEED) == (1319, 3), "TEST_SIZE/TASK_SAMPLE_SEED - общие параметры"'),
    ("assert GEN_LENGTH == 256 and BLOCK_LENGTH == 32", "assert GEN_LENGTH == 512 and BLOCK_LENGTH == 32"),
    ('assert NUM_STEPS_LIST == [64, 128] and REMASKING_LIST == ["low_confidence", "margin"], "NUM_STEPS_LIST/REMASKING_LIST - общие параметры"', 'assert NUM_STEPS_LIST == [128] and REMASKING_LIST == ["margin"], "NUM_STEPS_LIST/REMASKING_LIST - общие параметры"'),
    ('assert SEEDS == [0, 1, 2], "SEEDS - общий параметр"', 'assert SEEDS == [3], "SEEDS - общий параметр"'),
    (HASH_OLD, HASH_NEW),
    ('"DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS,', '"DATASET_SPLIT": DATASET_SPLIT, "N_TASKS": N_TASKS, "TEST_SIZE": TEST_SIZE, "TASK_SAMPLE_SEED": TASK_SAMPLE_SEED,'),
    ("ожидается (как у exp1):", "ожидается (общий для f1-f5):"),
    ("хэш общих параметров не совпал с exp1: остановка", "хэш общих параметров не совпал с ожидаемым для f1-f5: остановка"),
    ('EXP1_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp1_llada_default")      # для парного сравнения в сводке',
     'EXP1_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "f1_base_full512")         # f1: база для парного сравнения в сводке (имя ключа сохранено)'),
    ('EXP5_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp5_codd_full")          # для сравнения Э4 с Э5',
     'EXP5_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "f5_codd_full512")         # для сравнения f4 с f5'),
    ('E4_IMPL = "auto"                 # только Э4:', 'USE_PC = @@USE_PC@@                # False: HMM выключена и не загружается (только f1)\nE4_IMPL = "auto"                 # только Э4:'),
    ('assert E4_IMPL in ("auto", "flows", "pool")\n', 'assert E4_IMPL in ("auto", "flows", "pool")\nassert isinstance(USE_PC, bool) and (USE_PC or (WORDS_FROM == "base" and ORDER_FROM == "base" and THEME_MODE == "joint"))\n'),
    ("REQUIRE_PCTEST = True            # smoke и полный прогон требуют успешного pctest.json (проверка HMM и Triton ДО загрузки 8B)",
     "REQUIRE_PCTEST = USE_PC          # при USE_PC: smoke и полный прогон требуют успешного pctest.json (HMM и Triton проверяются ДО загрузки 8B)"),
    ('"EXP_NAME": EXP_NAME, "WORDS_FROM": WORDS_FROM, "ORDER_FROM": ORDER_FROM, "THEME_MODE": THEME_MODE, "E4_IMPL": E4_IMPL,',
     '"EXP_NAME": EXP_NAME, "WORDS_FROM": WORDS_FROM, "ORDER_FROM": ORDER_FROM, "THEME_MODE": THEME_MODE, "USE_PC": USE_PC, "E4_IMPL": E4_IMPL,'),
    ("if os.path.isfile(HMM_MANIFEST):\n    _man", "if USE_PC and os.path.isfile(HMM_MANIFEST):\n    _man"),
    ('else:\n    print("ВНИМАНИЕ: нет", HMM_MANIFEST,', 'elif USE_PC:\n    print("ВНИМАНИЕ: нет", HMM_MANIFEST,'),
    ("веса HMM грузятся ЛОКАЛЬНО из `offline_bundle_hmm/llada-gsm-pc/model.jpc` (`juice.load` + `juice.compile`), не через Hub.", "@@HMM_LOAD_NOTE@@"),
    ("> 3. `GEN_LENGTH = 256` (общая константа; в статье 512). Менять нельзя: хэш общих параметров зашит в ядро и должен совпасть с exp1.",
     '> 3. Общие параметры f1-f5: `N_TASKS = 200` случайных задач из test (1319; `TASK_SAMPLE_SEED = 3`), `GEN_LENGTH = 512`, `NUM_STEPS_LIST = [128]`, `REMASKING_LIST = ["margin"]`, `SEEDS = [3]`. Хэш зашит в ядро и должен совпадать у всех пяти блокнотов.'),
    ("Общая механика (для экспериментов 2-5):", "Общая механика (для f2-f5; в f1 HMM выключена и не загружается):"),
    ("тема: **@@THEME_MODE@@**.", "тема: **@@THEME_TEXT@@**."),
    ("безопасное чтение структуры HMM (pickle без исполнения кода);", "безопасное чтение структуры HMM (pickle без исполнения кода; только если HMM используется);"),
    ("`smoke.json` пишется только при успехе. Загрузка модели занимает несколько минут.", "`smoke.json` пишется только при успехе. Если HMM не используется (f1), пункт (б) пропускается. Загрузка модели занимает несколько минут."),
    ("Требуются успешные `pctest.json` и `smoke.json` (при том же конфиге)", "Требуются успешный `smoke.json` и, если HMM используется, `pctest.json` (при том же конфиге)"),
    ("4 комбинации (`remasking` x `num_steps`) по `N_TASKS` задач для каждого seed из `RUN_SEEDS`.", "Одна комбинация (`margin`, 128 шагов, 512 токенов) по `N_TASKS` = 200 случайных задач test (`TASK_SAMPLE_SEED = 3`), seed 3."),
    ("**Парное сравнение с exp1 по `task_id`**", "**Парное сравнение с f1 по `task_id`**"),
    ("@@SUMMARY_E4@@. Модель не загружается. Доверительный интервал отражает выбор 100 задач; случайность темы HMM даёт разброс между seeds.",
     "@@SUMMARY_E4@@. Дополнительно: доля неоконченных ответов (за 512 токенов нет EOS: `<|endoftext|>` или `<|eot_id|>`) и точность в двух вариантах - \"как есть\" и \"неоконченные считать неверными\" (парные сравнения считаются для обоих). Модель не загружается. Доверительный интервал отражает выбор 200 задач; seed один, разброса по seeds нет."),
    ("1. Загрузите на сервер `offline_bundle_hmm` (см. `README_HMM.md`) и убедитесь, что есть `CoDD/`, `venv/`, `offline_bundle/` (как для exp1).",
     "1. Для f2-f5 загрузите на сервер `offline_bundle_hmm` (см. `README_HMM.md`; для f1 не нужно) и убедитесь, что есть `CoDD/`, `venv/`, `offline_bundle/`."),
    ("4. Ячейка 5 - `pctest` (GPU, до загрузки 8B): проверяет HMM, Triton и память@@HOW_E4@@.", "4. @@HOW_PCTEST@@"),
    ("6. Ячейка 7 - `full`: по умолчанию `RUN_SEEDS = [0]`; для всех seeds поставьте `[0, 1, 2]` в ячейке 0 и повторите ячейки 2-3.",
     "6. Ячейка 7 - `full` (200 задач, seed 3): запускайте только после проверки отчёта smoke. С `MAX_RUNTIME_MIN` процесс останавливается сам; запустите ячейку снова, готовые задачи пропускаются."),
    ("9. Ячейка 8 - сводка и парное сравнение с exp1 (`results/exp1_llada_default` должен лежать на сервере).", "9. Ячейка 8 - сводка@@VS_TEXT@@."),
]

PCTEST_MD_PREFIX = "## 5. Проверка HMM на GPU"
PCTEST_CODE_PREFIX = "# ===================== 5. PCTEST"
F1_PCTEST_MD = "## 5. pctest не нужен\n\nВ f1 HMM не используется (`USE_PC = False`), поэтому проверка HMM и Triton на GPU пропущена. Нумерация ячеек сохранена: сразу переходите к ячейке 6 (smoke).\n"


def apply_repl(cells):
    texts = ["\n".join(src for _k, src in cells)]
    for old, new in REPL:
        assert old in texts[0], "в ячейках не найдено: " + old[:90]
    out = []
    for kind, src in cells:
        for old, new in REPL:
            src = src.replace(old, new)
        out.append((kind, src))
    return out


def build_one(num, spec, core_template):
    exp_name = spec["EXP_NAME"]
    use_pc = spec["USE_PC"]
    nb_name = exp_name + ".ipynb"
    core_name = "f%d_core.py" % num
    experiment = {k: spec[k] for k in ("EXP_NAME", "WORDS_FROM", "ORDER_FROM", "THEME_MODE", "USE_PC")}
    marker = "EXPERIMENT = {}  # @@EXPERIMENT@@"
    assert core_template.count(marker) == 1
    core = core_template.replace(marker, "EXPERIMENT = " + repr(experiment) + "  # подставлено сборщиком build_full512.py")
    problems = []
    if '"""' in core:
        problems.append('в ядре есть """')
    if not core.endswith("\n") or core.rstrip("\n").endswith("\\"):
        problems.append("ядро заканчивается слэшем или без перевода строки")
    tree_core = ast.parse(core, filename=core_name)
    # ключи конфига ядра
    required = None
    for node in ast.walk(tree_core):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_REQUIRED_KEYS" for t in node.targets):
            required = set(ast.literal_eval(node.value))
    assert required, "не найден _REQUIRED_KEYS"
    if problems:
        raise SystemExit("; ".join(problems))
    with open(os.path.join(PROJECT, core_name), "w", encoding="utf-8") as f:
        f.write(core)

    is_e4 = spec["THEME_MODE"] == "independent"
    a_name = "logits_base (до HMM)" if spec["WORDS_FROM"] == "base" else "logits_pc (после HMM)"
    c_name = "logits_base (до HMM)" if spec["ORDER_FROM"] == "base" else "logits_pc (после HMM)"
    if not use_pc:
        a_name = c_name = "logits_base (HMM выключена)"
    how_e4 = " и корректность независимого выбора темы (шаг C; без него f4 не запустится)" if is_e4 else ""
    subs = {
        "@@TITLE@@": spec["TITLE"], "@@WHAT@@": spec["WHAT"], "@@NB_NAME@@": nb_name, "@@CORE_NAME@@": core_name, "@@RUN_DIR@@": "f%d_run" % num,
        "@@EXP_NAME@@": exp_name, "@@WORDS_FROM@@": spec["WORDS_FROM"], "@@ORDER_FROM@@": spec["ORDER_FROM"], "@@THEME_MODE@@": spec["THEME_MODE"],
        "@@THEME_TEXT@@": spec["THEME_MODE"] if use_pc else "нет (HMM выключена)",
        "@@USE_PC@@": "True" if use_pc else "False",
        "@@A_NAME@@": a_name, "@@C_NAME@@": c_name, "@@SHORT@@": "f%d" % num,
        "@@HMM_LOAD_NOTE@@": ("веса HMM грузятся ЛОКАЛЬНО из `offline_bundle_hmm/llada-gsm-pc/model.jpc` (`juice.load` + `juice.compile`), не через Hub." if use_pc
                              else "в f1 HMM выключена (`USE_PC = False`) и НЕ загружается."),
        "@@PCTEST_E4@@": ("**Для f4 дополнительно - шаг C**: 200 сэмплов тем в режимах `joint` и `independent` для 4 реальных состояний блока (токены эталонных решений GSM8K, ~44% масок), частоты тем по позициям против маргиналов и взаимная информация между позициями с перестановочным нулём; при успехе пишется флаг `E4_CHECK_PASSED.json`, без него полный прогон не стартует. Мягкие оценки нейросети в этом тесте СИНТЕТИЧЕСКИЕ (модель 8B ещё не загружена). ") if is_e4 else "",
        "@@FULL_E4@@": " и флаг `E4_CHECK_PASSED.json` (проверка независимого выбора темы)" if is_e4 else "",
        "@@SUMMARY_E4@@": "; для этого эксперимента дополнительно сравнение с f5 (`results/f5_codd_full512`)" if is_e4 else "",
        "@@HOW_E4@@": how_e4,
        "@@HOW_PCTEST@@": ("Ячейка 5 - `pctest` (GPU, до загрузки 8B): проверяет HMM, Triton и память" + how_e4 + "." if use_pc else "Ячейка 5 отсутствует: HMM не используется (f1)."),
        "@@VS_TEXT@@": (" (f1 - база, сравнение с другими прогонами не требуется)" if num == 1 else " и парное сравнение с f1 (`results/f1_base_full512` должен лежать на сервере)"),
    }
    cells = []
    for idx, (kind, src) in enumerate(apply_repl(B.CELLS)):
        if kind == "markdown" and src.startswith(PCTEST_MD_PREFIX) and not use_pc:
            src = F1_PCTEST_MD
        if kind == "code" and src.startswith(PCTEST_CODE_PREFIX) and not use_pc:
            continue
        src = src.replace("@@CORE@@", core)
        for k in sorted(subs, key=len, reverse=True):
            src = src.replace(k, subs[k])
        assert "@@" not in src.replace(core, ""), "неподставленный маркер в ячейке %d: %s" % (idx, [t for t in src.split() if "@@" in t])
        src = src.strip("\n") + "\n"
        lines = src.splitlines(True)
        lines[-1] = lines[-1].rstrip("\n")
        cell_id = "c%03d" % len(cells)
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
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "CORE_SOURCE" for t in node.targets):
                    if ast.literal_eval(node.value) != core:
                        problems.append("ячейка %d: встроенный CORE_SOURCE не совпал с %s" % (idx, core_name))
                    else:
                        print("  CORE_SOURCE побайтно равен %s (%d строк)" % (core_name, core.count("\n")))
                if any(isinstance(t, ast.Name) and t.id == "CORE_CONFIG" for t in node.targets):
                    keys = set(k.value for k in node.value.keys)
                    if keys != required:
                        problems.append("CORE_CONFIG != _REQUIRED_KEYS: нет %s, лишние %s" % (sorted(required - keys), sorted(keys - required)))
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
    print("OK f%d: %s (%d ячеек, кода %d) + %s" % (num, nb_name, len(cells), n_code, core_name))


def main():
    with open(os.path.join(HERE, "f_core_template.py"), encoding="utf-8") as f:
        template = f.read()
    for num, spec in sorted(EXPS.items()):
        build_one(num, spec, template)


if __name__ == "__main__":
    main()
