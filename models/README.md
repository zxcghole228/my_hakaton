# Локальные training artifacts

В этой папке хранятся checkpoints и exports экспериментов. Все вложенные файлы
исключены из Git; отслеживается только этот README.

Runtime-веса готового решения находятся внутри
`solutions/<solution_name>/models/`. Submission-архивы находятся в
`artifacts/submissions/`.

## E5-small Macro V2

```text
models/e5_small_macro_v2/
├── checkpoints/e5_macro_v2_best.zip
├── exports/e5_macro_v2_30k_export.zip
└── hf_export_30k/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

- checkpoint SHA-256:
  `e3ceb948ddb41961dc5db08b55b317019074365836c337d9f3923d25ed1ce29c`;
- export SHA-256:
  `926a31dc7c300e33735c1aa9e0172a2b7a8dd5a3b274250eea4021ea97152837`;
- `model.safetensors` SHA-256:
  `b7263e2c9f39cf73bfd1217c91ef613b9cbc9529fa95a062784215d4c568d92c`.

## E5-small Macro V3

```text
models/e5_small_macro_v3/
├── checkpoints/
│   ├── e5_v3_best_specialist.pt
│   └── e5_v3_resume.pt
└── exports/e5_v3_hybrid_export.zip
```

- best specialist SHA-256:
  `0546814b8950782e7116248ac557ed67ebd04098e5c0f9362aa5d9a8bbca7ada`;
- resume checkpoint SHA-256:
  `0ae0d6a752893faa8bbd4131e03b1d730ffe2fc1a5618887d433aab62fcededf`;
- hybrid export SHA-256:
  `ea84f56b9b0f6bfbabed5d3dfbe75292992234c8beffbd5f9141614c392d99a8`.

## Structured model V4

```text
models/e5_small_macro_v4/
└── structured_catboost.cbm
```

Это исходная CatBoost-модель из experiment export. Submission использует её
скомпилированный вариант `structured_model.so` внутри локальной папки моделей
V4 solution.

## CrossEncoder baseline

Runtime-веса baseline теперь находятся рядом с решением:

```text
solutions/cross_encoder_baseline/models/
└── cross-encoder-ms-marco-MiniLM-L12-v2/
```

Так baseline полностью отделён от training artifacts V2–V4.
