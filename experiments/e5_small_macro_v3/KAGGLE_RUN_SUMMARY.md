# E5 V3 — Fashion Specialist

## Reference V2

- Public LB: **0.4838757641**
- Backbone: `intfloat/multilingual-e5-small`
- checkpoint: step `30000`
- full LLM group holdout Macro PR-AUC: `0.786482`

## V3

- Specialist initialized from V2 weights
- Fashion categories:
  - Обувь
  - Одежда
  - Галантерея и аксессуары
  - Ювелирные изделия
- Training distribution: LLM only
- Balanced candidate pool: up to `200,000` / category
- Mined train: up to `150,000` / category
- hard/random/stable = `50%/30%/20%`
- Variant signals: size/color/article/model/gender/material
- epochs: `2`
- LR: backbone `8e-06`, head `3e-05`
- MAX_LEN: `192`

## Result

- Best specialist step: `4000`
- Best fast hybrid Macro PR-AUC: `0.790609`
- Base full LLM Macro PR-AUC: `0.786482`
- Specialist fashion full Macro PR-AUC: `0.504528`
- **Hybrid V3 full LLM Macro PR-AUC: `0.790182`**
- Delta vs V2: `+0.003700`

## Routing

```json
{
  "Галантерея и аксессуары": "base",
  "Обувь": "specialist",
  "Одежда": "specialist",
  "Ювелирные изделия": "specialist"
}
```

## Export

`/kaggle/working/e5_v3_hybrid_export.zip`
