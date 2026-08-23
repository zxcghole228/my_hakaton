# Ноутбуки

Здесь хранятся ноутбуки обучения, восстановления моделей, анализа и экспорта.
Ноутбук должен иметь содержательное имя и не должен содержать datasets, веса
или секреты. Итоговые метрики и конфигурации из завершённого запуска нужно
переносить в соответствующую папку `experiments/`.

Текущие направления:

- `e5_small_macro_llm_stageA_v2.ipynb` — обучение V2;
- `e5_small_restore30k_fullval_export.ipynb` — восстановление и экспорт V2;
- `e5_small_macro_v3_fashion_specialist_hybrid.ipynb` — обучение V3 specialist;
- `v3-ozonecup-fix-old-bugs.ipynb` — последующая работа над исправлениями/V4.
- `selective_e5_bge_reranking_v6.ipynb` — воспроизведение V6.2: подбор
  selective BGE reranking поверх E5 Stage-B и экспорт production-конфигурации.
