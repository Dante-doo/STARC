# STARC: Satellite Trails — Treino, Teste e Resultados

[English](#en) • [Português](#pt-br)

---

<a id="en"></a>

## English version

**Quick nav:** [Requirements](#en-req) • [Folders](#en-struct) • [Config](#en-config) • [Train](#en-train) • [Test](#en-test) • [Results](#en-results) • [Outputs](#en-outputs) • [How it works](#en-how) • [Stability & OOM](#en-oom) • [Reproducibility](#en-reprod) • [One-liners](#en-oneliners) • [Português](#pt-br)

---

### <a id="en-req"></a>1) Requirements

* **OS:** Linux (e.g., Ubuntu)
* **Python:** 3.12 (use `venv`)
* **GPU:** 1–2× NVIDIA GTX 1080 Ti (~11 GB)
* **Python packages (pip):**

  * `torch`, `torchvision`
  * `numpy`, `pandas`, `Pillow`, `matplotlib`
  * (optional for external Hough stages: `opencv-python`)

---

### <a id="en-struct"></a>2) Folder structure

```
images/
  full/
    raw/{standart,augment}/...
    hough/...                 # full-res Hough artifacts
  patches/
    raw/{standart,augment}/fullX/fullX_l_c_r[_ang].png
    hough/accumulator/...     # filenames use zero-padded l,c
    combined/{standart,augment}/...
labels/
  train.csv  val.csv  test.csv
models/
  resnet|deepnet/
    <modality>[/aug-<policy>]/ {best.pt,last.pt,best_threshold.json,train_meta.json}
results/
  preds/<arch>/<modality>[/aug-*]/test_preds.csv
  metrics/<arch>/<modality>[/aug-*]/test_metrics.json
  summary/{all_runs.csv,best_by_group.csv,mean_by_group.csv}
  vis/{balacc_*.png,confusion_*.png}
code/
  config.py
  train.py
  test.py
  results.py
```

> **Splits** (`labels/*.csv`) are **grouped by full image** (group split by `fullX`) to prevent patch leakage across train/val/test.

---

### <a id="en-config"></a>3) Key configuration (`code/config.py`)

* **Backbones:** `resnet18` and `deepnet121` (DenseNet121).
* **Modalities:** `raw`, `hough` (1-ch accumulator), `combined` (RGB stack).
* **Defaults:** `TRAIN_ARCH="resnet18"`, `TRAIN_MODALITY="raw"`. All can be overridden via CLI.
* **Batch/Workers:** tune `TRAIN_BATCH_SIZE` and `EVAL_BATCH_SIZE` for your GPU.
* **Performance:** AMP + `channels_last` enabled; TF32 allowed on supported GPUs.
* **Aug policy:** `rebalance` (caps per-base augments and matches train pos_rate to base pos_rate).
* **Threshold sweep (val):** `t ∈ [0.10, 0.90]` (step `0.02`), metric: **balanced accuracy**; best threshold stored to `best_threshold.json`.

---

### <a id="en-train"></a>4) Training

> **DenseNet121 note:** Heavier than ResNet18. For `hough/combined`, use **smaller per-GPU batch** (e.g., 32).

#### 4.1 Single-GPU examples

**ResNet18 / raw / rebalance**

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough / rebalance** (ensure smaller batch in `config.py`)

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch deepnet121 --modality hough --aug rebalance
```

#### 4.2 Two GPUs (DDP) — 1-liner pattern

Stable env flags for this machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train \
  --arch <resnet18|deepnet121> --modality <raw|hough|combined> --aug rebalance
```

Examples:

**ResNet18 / raw**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance
```

**DeepNet121 / combined**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

> Optional allocator flag (only if supported by your stack):
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

### <a id="en-test"></a>5) Testing (writes CSV/JSON per run)

**General form (1 GPU):**

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch <resnet18|deepnet121> --modality <raw|hough|combined> --aug rebalance --ckpt best
```

Examples:

```bash
# ResNet18 / raw
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality raw      --aug rebalance --ckpt best
# DeepNet121 / hough
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough    --aug rebalance --ckpt best
# DeepNet121 / combined
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best
```

If OOM on test, lower `EVAL_BATCH_SIZE` in `config.py`.

---

### <a id="en-results"></a>6) Aggregation & plots (results)

Builds summary CSVs and figures (bar charts and confusion matrices for the group best):

```bash
python -m code.results
# Optional (skip pathological-run filter TNR==0 & recall==1):
# python -m code.results --no-filter
```

---

### <a id="en-outputs"></a>7) Where things are saved

* **Models & thresholds**
  `models/<arch_dir>/<modality>[/aug-<policy>]/{best.pt,last.pt,best_threshold.json,train_meta.json}`
  where `<arch_dir>` ∈ `{resnet, deepnet}`.

* **Predictions & metrics (test)**

  * CSV: `results/preds/<arch_dir>/<modality>[/aug-*]/test_preds.csv`
    (`raw_path`, `mapped_path`, `label`, `prob`, `pred`)
  * JSON: `results/metrics/<arch_dir>/<modality>[/aug-*]/test_metrics.json`
    (includes threshold used, `tp,fp,tn,fn`, metrics)

* **Summaries & plots**
  `results/summary/*.csv`, `results/vis/*.png`

---

### <a id="en-how"></a>8) How it works (short)

* **Dataset & splits:** group split by full image to prevent leakage.
* **Path mapping:** consistent RAW → Hough/Combined mapping; Hough filenames zero-pad `l,c` and preserve rotation subdir; Combined keeps `l,c` (and `ang` for aug).
* **Training:** Focal Loss (default), light TF flips per batch, **rebalance** caps augments and matches train positive rate to base.
* **Validation:** threshold sweep in `[0.10, 0.90]` (step `0.02`), selects **best balanced accuracy**; threshold stored to `best_threshold.json`.
* **Testing:** uses saved threshold (if present); writes CSV + JSON.
* **Results:** aggregates, ranks by balacc/acc, plots bars and confusion matrices.

---

### <a id="en-oom"></a>9) Stability & OOM tips

* DenseNet121 (`hough/combined`) → lower `TRAIN_BATCH_SIZE` (e.g., 32 per GPU) and reduce `EVAL_BATCH_SIZE` if needed.
* For older networking: `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` helps.
* AMP + `channels_last` reduce memory and can speed up.
* Still OOM? Lower batch further; disable `persistent_workers` in extreme cases.

---

### <a id="en-reprod"></a>10) Reproducibility

* Seed (=42) for `random`, `numpy`, and `torch`/`cuda`.
* `cudnn.benchmark=True` (fast; not fully deterministic).
* Logs at `code/train.log`.

---

### <a id="en-oneliners"></a>Useful one-liners

**Train deepnet121 (hough) on 2 GPUs + test + results**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough --aug rebalance --ckpt best && python -m code.results
```

**Train deepnet121 (combined) on 2 GPUs + test + results**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best && python -m code.results
```

[Back to top](#en) • [Português](#pt-br)

---

<a id="pt-br"></a>

## Versão em português

**Navegação rápida:** [Requisitos](#pt-req) • [Pastas](#pt-struct) • [Config](#pt-config) • [Treino](#pt-train) • [Teste](#pt-test) • [Results](#pt-results) • [Saídas](#pt-outputs) • [Como funciona](#pt-how) • [Estabilidade & OOM](#pt-oom) • [Reprodutibilidade](#pt-reprod) • [One-liners](#pt-oneliners) • [English](#en)

---

### <a id="pt-req"></a>1) Requisitos

* **SO:** Linux (ex.: Ubuntu)
* **Python:** 3.12 (use `venv`)
* **GPU:** 1–2× NVIDIA GTX 1080 Ti (~11 GB)
* **Bibliotecas (pip):**

  * `torch`, `torchvision`
  * `numpy`, `pandas`, `Pillow`, `matplotlib`
  * (opcional p/ estágios Hough externos: `opencv-python`)

---

### <a id="pt-struct"></a>2) Estrutura de pastas

```
images/
  full/
    raw/{standart,augment}/...
    hough/...                 # artefatos Hough em resolução cheia
  patches/
    raw/{standart,augment}/fullX/fullX_l_c_r[_ang].png
    hough/accumulator/...     # nomes com l,c zero-padded
    combined/{standart,augment}/...
labels/
  train.csv  val.csv  test.csv
models/
  resnet|deepnet/
    <modality>[/aug-<policy>]/ {best.pt,last.pt,best_threshold.json,train_meta.json}
results/
  preds/<arch>/<modality>[/aug-*]/test_preds.csv
  metrics/<arch>/<modality>[/aug-*]/test_metrics.json
  summary/{all_runs.csv,best_by_group.csv,mean_by_group.csv}
  vis/{balacc_*.png,confusion_*.png}
code/
  config.py
  train.py
  test.py
  results.py
```

> **Splits** (`labels/*.csv`) foram gerados **por imagem-mãe** (split por `fullX`) para evitar vazamento de patches “irmãos” entre treino/val/test.

---

### <a id="pt-config"></a>3) Configuração chave (`code/config.py`)

* **Backbones:** `resnet18` e `deepnet121` (DenseNet121).
* **Modalidades:** `raw`, `hough` (acumulador 1 canal), `combined` (RGB).
* **Padrões:** `TRAIN_ARCH="resnet18"`, `TRAIN_MODALITY="raw"` — pode sobrescrever via CLI.
* **Batch/Workers:** ajuste `TRAIN_BATCH_SIZE` e `EVAL_BATCH_SIZE` conforme sua GPU.
* **Performance:** AMP + `channels_last` ativos; TF32 permitido.
* **Política de aug:** `rebalance` (limita aug por base e casa a taxa de positivos do treino com a base).
* **Sweep de limiar (val):** `t ∈ [0.10, 0.90]` (passo `0.02`), métrica: **balanced accuracy**; salva em `best_threshold.json`.

---

### <a id="pt-train"></a>4) Treino

> **Nota DenseNet121:** mais pesado que ResNet18. Em `hough/combined`, use **batch por GPU menor** (ex.: 32).

#### 4.1 Exemplos 1 GPU

**ResNet18 / raw / rebalance**

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough / rebalance** (garanta batch menor no `config.py`)

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch deepnet121 --modality hough --aug rebalance
```

#### 4.2 Duas GPUs (DDP) — 1 linha

Flags estáveis nesta máquina:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train \
  --arch <resnet18|deepnet121> --modality <raw|hough|combined> --aug rebalance
```

Exemplos:

**ResNet18 / raw**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance
```

**DeepNet121 / combined**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

> Flag opcional do alocador (apenas se suportado):
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

### <a id="pt-test"></a>5) Teste (gera CSV/JSON por execução)

**Forma geral (1 GPU):**

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch <resnet18|deepnet121> --modality <raw|hough|combined> --aug rebalance --ckpt best
```

Exemplos:

```bash
# ResNet18 / raw
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality raw      --aug rebalance --ckpt best
# DeepNet121 / hough
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough    --aug rebalance --ckpt best
# DeepNet121 / combined
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best
```

Se der OOM no teste, reduza `EVAL_BATCH_SIZE` no `config.py`.

---

### <a id="pt-results"></a>6) Agregação & gráficos (results)

Gera CSVs de sumário e figuras (barras e confusões do melhor por grupo):

```bash
python -m code.results
# Opcional (não filtrar execuções patológicas TNR==0 & recall==1):
# python -m code.results --no-filter
```

---

### <a id="pt-outputs"></a>7) Onde salva

* **Modelos & thresholds**
  `models/<arch_dir>/<modality>[/aug-<policy>]/{best.pt,last.pt,best_threshold.json,train_meta.json}`
  `<arch_dir>` ∈ `{resnet, deepnet}`.

* **Predições & métricas (test)**

  * CSV: `results/preds/<arch_dir>/<modality>[/aug-*]/test_preds.csv`
    (`raw_path`, `mapped_path`, `label`, `prob`, `pred`)
  * JSON: `results/metrics/<arch_dir>/<modality>[/aug-*]/test_metrics.json`
    (inclui threshold, `tp,fp,tn,fn`, métricas)

* **Sumários & plots**
  `results/summary/*.csv`, `results/vis/*.png`

---

### <a id="pt-how"></a>8) Como funciona (resumo)

* **Dataset & splits:** split por imagem-mãe, sem vazamento.
* **Mapeamento:** RAW → Hough/Combined consistente; Hough com `l,c` zero-padded e subpasta de rotação; Combined mantém `l,c` (e `ang` em aug).
* **Treino:** Focal Loss (padrão), flips leves por batch, **rebalance** limita aug e casa a taxa de positivos.
* **Validação:** varre `t ∈ [0.10, 0.90]` (passo `0.02`), escolhe **melhor balanced accuracy**; salva `best_threshold.json`.
* **Teste:** usa threshold salvo (se existir); escreve CSV + JSON.
* **Results:** agrega, ordena por balacc/acc, plota barras e matrizes de confusão.

---

### <a id="pt-oom"></a>9) Estabilidade & OOM

* DenseNet121 (`hough/combined`) → reduza `TRAIN_BATCH_SIZE` (ex.: 32 por GPU) e `EVAL_BATCH_SIZE` se necessário.
* Em redes antigas: `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` ajuda.
* AMP + `channels_last` reduzem memória e aceleram.
* Persistindo OOM: reduza mais o batch; desative `persistent_workers` em último caso.

---

### <a id="pt-reprod"></a>10) Reprodutibilidade

* Seed (=42) para `random`, `numpy` e `torch`/`cuda`.
* `cudnn.benchmark=True` (rápido; não 100% determinístico).
* Logs em `code/train.log`.

---

### <a id="pt-oneliners"></a>One-liners úteis

**Treinar deepnet121 (hough) em 2 GPUs + testar + results**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough --aug rebalance --ckpt best && python -m code.results
```

**Treinar deepnet121 (combined) em 2 GPUs + testar + results**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best && python -m code.results
```

[Topo](#pt-br) • [English](#en)
