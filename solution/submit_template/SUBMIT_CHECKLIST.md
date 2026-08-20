# Шаблон надёжного сабмита: защита от нуля (по докам Бабенко + нашей охоте на баги)

Каждый сабмит-пакет перед отправкой сверять с этим списком.

## 1. Файл на выходе появляется ВСЕГДА (watchdog + fallback)

Падение/таймаут = ноль на всех трёх наборах. Правило: main оборачивается так,
что при ЛЮБОЙ ошибке или приближении к лимиту времени пишется csv из дешёвой
модели (или в крайнем случае константа 0.5):

```python
import time, traceback
T0 = time.monotonic()
BUDGET = {1000: 55, 120000: 350, 300000: 700}  # сек по размеру теста, с запасом

def main_safe(items_path, matches_path, output_path):
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    try:
        run_full_model(...)          # основной путь
    except Exception:
        traceback.print_exc()
        run_cheap_fallback(...)      # CatBoost на фичах / rapidfuzz-скор / 0.5
    # контроль: файл существует, строк ровно len(matches), колонки id1,id2,predict
```

Watchdog в цикле инференса: если elapsed > 0.8 * бюджета — остаток пар доскорить
дешёвой моделью, не падать.

## 2. Check-стадия (1000 пар / 60 сек) — фиксированный оверхед

- веса fp16 + safetensors (вдвое меньше читать с холодного диска);
- НИКАКОГО torch.compile;
- импорты torch/transformers — лениво, после чтения matches;
- vocab trimming для XLM-R моделей (250k -> 64k) — вдвое быстрее загрузка;
- прогнать локально холодный старт на 1000 парах и замерить.

## 3. Гомоглифы кириллица/латиница — в нормализацию фичей и текстов

Продавцы пишут 'Ѕаmsung' с кириллическими а/е/о/р/с/х. Без фикса артикулы
и бренды «разные строки». Потокенно (не для чистых слов!):

```python
CYR2LAT = str.maketrans("аеорсухАЕОРСУХ", "aeopcyxAEOPCYX")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
def fix_homoglyphs(tok):
    has_cyr = any('Ѐ' <= c <= 'ӿ' for c in tok)
    has_lat = any(c.isascii() and c.isalpha() for c in tok)
    if not (has_cyr and has_lat):
        return tok
    n_cyr = sum('Ѐ' <= c <= 'ӿ' for c in tok)
    n_lat = sum(c.isascii() and c.isalpha() for c in tok)
    return tok.translate(CYR2LAT if n_lat >= n_cyr else LAT2CYR)
```

## 4. Уже пойманное нашей охотой на баги (не регрессировать)

- category=null у существующего товара: X["category"].fillna("").astype(str)
  перед CatBoost — иначе крэш;
- отсутствующий в items товар не роняет пайплайн (пустые name/attrs);
- предсказание для КАЖДОЙ пары matches в исходном порядке;
- build_text инференса побайтово совпадает с обучением;
- нет NaN/inf в predict.

## 5. Перед отправкой

- локальная эмуляция на items_human+matches: формат, тайминг, воспроизведение
  локальной метрики бит-в-бит;
- два финальных решения в конце соревнования: агрессивное и консервативное.
