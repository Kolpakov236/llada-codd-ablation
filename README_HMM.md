# Веса HMM для экспериментов 2-5 (DataSphere, без интернета)

Что везём: чекпойнт `il18/llada-gsm-pc` (скачан на Mac скриптом `download_hmm.py`, размеры и sha256 сверены с API Hugging Face).

| Файл | Размер, байт | sha256 |
|---|---|---|
| `llada-gsm-pc/model.jpc` | 522 204 554 (522.2 МБ) | `0c8903455fd8cf592e8c8d5489230eb0d343c6bc6adae3208c533d09dc51c25f` |
| `llada-gsm-pc/config.json` | 33 | `2d9e8279c32f8a1444d12320e1058ffb2340297d1732e1670c130cca2f35d5bc` |

Структура `model.jpc` (прочитана безопасным распаковщиком без `import pyjuice`, см. `offline_bundle_hmm/hmm_structure.json`): 32 переменные (позиции блока), по 1024 входных узла на позицию (скрытые состояния), 32 768 входных, 31 745 суммирующих и 32 768 произведений узлов, 130 548 736 параметров float32 (около 0.49 ГиБ). Память на GPU (нижняя оценка по параметрам): ~0.5 ГиБ плюс память под активации и CUDA-граф - **реальный пик измерит режим `pctest`, до него это только оценка**.

## Перенос на DataSphere

Запустите `python -I download_hmm.py` на машине с интернетом. Скрипт скачает веса и создаст манифест. Перенесите папку `offline_bundle_hmm/` в корень проекта на сервере, сохранив структуру и имена файлов. Если переносите папку архивом, распакуйте её перед запуском ноутбука.

## Склейка частей и сверка sha256

Для текущего чекпойнта части не нужны (файл меньше 1.5 ГБ). Если когда-нибудь появятся файлы `*.partNNN` (порог разрезания 1500 МБ, `download_hmm.py`), склейте их в Terminal: `cat model.jpc.part* > model.jpc` (в алфавитном порядке), затем удалите части. В любом случае выполните сверку по `manifest_hmm.json`:

```python
import hashlib, json, os
PROJECT_DIR = "/home/jupyter/project"
man = json.load(open(os.path.join(PROJECT_DIR, "offline_bundle_hmm", "manifest_hmm.json"), encoding="utf-8"))
bad = []
for e in man["files"]:
    p = os.path.join(PROJECT_DIR, e["path"])
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ok = os.path.getsize(p) == e["size"] and h.hexdigest() == e["sha256"]
    print("%-52s %s" % (e["path"], "OK" if ok else "РАСХОЖДЕНИЕ"))
    if not ok:
        bad.append(e["path"])
assert not bad, "не совпали: %s" % bad
```

Эту же сверку (размер и sha256 `model.jpc`) повторяет режим `preflight` каждого эксперимента, так что при ошибке полный прогон не стартует.

## Ожидания и чего НЕ ПРОВЕРЕНО

- Время загрузки (`juice.load`) и компиляции (`juice.compile`) на сервере, реальный пик видеопамяти с HMM и скорость `apply_pc_logits` неизвестны до `pctest` и `smoke`.
- Работа PyJuice 2.4.1 и компиляция Triton на сервере (нужны `gcc` и `Python.h`): `preflight` проверяет их наличие, но не саму компиляцию - её проверяет `pctest`.
- Версия `pyjuice_version` в `config.json` чекпойнта: `2.4.1` (совпадает с версией в venv).
