# STARC: Satellite Trails — Treino, Teste e Resultados

Pipeline para classificar patches (224×224) em **raw / hough / combined** usando **ResNet18** e **DenseNet121** (apelido: *deepnet121*). Inclui treino com **DDP**, varredura de **threshold** no *val*, teste no *test*, e agregação/plots.

## 1) Requisitos

* **SO:** Linux (ex.: Ubuntu)
* **Python:** 3.12 (via `venv`)
* **GPU:** 1–2× NVIDIA GTX 1080 Ti (≈11 GB)
* **Bibliotecas (pip):**

  * `torch` e `torchvision`
  * `numpy`, `pandas`, `Pillow`, `matplotlib`
  * (opcional p/ etapas Hough fora do treino/teste: `opencv-python`)

## 2) Estrutura de pastas

```
images/
  full/
    raw/{standart,augment}/...
    hough/... (artefatos de Hough em resolução cheia)
  patches/
    raw/{standart,augment}/fullX/fullX_l_c_r[_ang].png
    hough/accumulator/... (usa l,c com zero-padding no nome)
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

> **Splits** (`labels/*.csv`) já devem vir estratificados por **imagem-mãe** (group split por fullX) para evitar vazamento entre sets.

## 3) Configuração chave (`code/config.py`)

* **Backbone padrão:** `TRAIN_ARCH="resnet18"` (pode sobrescrever via CLI).
* **Modalidade padrão:** `TRAIN_MODALITY="raw"` (também sobrescrevível).
* **Batch/Workers:** ajuste `TRAIN_BATCH_SIZE` e `EVAL_BATCH_SIZE` conforme GPU.
* **AMP + channels_last:** habilitados no config (ganho de memória/velocidade).
* **Aug policy:** `TRAIN_AUG_POLICY="rebalance"` (limita *aug* por base e casa a taxa de positivos com a base).
* **Threshold sweep (val):** `0.10→0.90` passo `0.02`, métrica: **balanced accuracy**.

## 4) Comandos — Treino

> **Notas**
> • Para **DenseNet121 + hough/combined**, use **batch menor por GPU** (ex.: `TRAIN_BATCH_SIZE=32` no config).
> • Em máquinas antigas/sem IB, os flags `NCCL_*_DISABLE=1` ajudam.
> • Se faltar memória, reduza batch e/ou `EVAL_BATCH_SIZE` (testes).

### 4.1 Treino — 1 GPU (exemplos)

**ResNet18 / raw / rebalance**

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough / rebalance** (garanta `TRAIN_BATCH_SIZE=32`)

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train --arch deepnet121 --modality hough --aug rebalance
```

### 4.2 Treino — 2 GPUs (DDP, 1 linha)

Variáveis “estáveis” para sua máquina:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch <resnet18|deepnet121> --modality <raw|hough|combined> --aug rebalance
```

Exemplos prontos:

**ResNet18 / raw**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality raw --aug rebalance
```

**DeepNet121 / hough** (batch por GPU menor no config)

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance
```

**DeepNet121 / combined** (idem, batch menor)

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

## 5) Comandos — Teste (gera CSV/JSON por run)

**Formato geral (1 GPU):**

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

> **Dica**: se der OOM no teste, reduza `EVAL_BATCH_SIZE` no `config.py`.

## 6) Agregação & gráficos (results)

Gera tabelas (`summary/*.csv`), barras de **balanced accuracy** e **matrizes de confusão** para o melhor de cada grupo.

```bash
python -m code.results
# opcional (não filtrar execuções patológicas TNR==0 & recall==1):
# python -m code.results --no-filter
```

## 7) Onde cada coisa é salva

* **Modelos & thresholds**
  `models/<arch_dir>/<modality>[/aug-<policy>]/{best.pt,last.pt,best_threshold.json,train_meta.json}`
  Onde `<arch_dir>` ∈ `{resnet, deepnet}`.

* **Predições & métricas (test)**

  * CSV: `results/preds/<arch_dir>/<modality>[/aug-*]/test_preds.csv`
    (`raw_path`, `mapped_path`, `label`, `prob`, `pred`)
  * JSON: `results/metrics/<arch_dir>/<modality>[/aug-*]/test_metrics.json`
    (inclui `threshold` usado, `tp,fp,tn,fn`, métricas)

* **Sumários & plots**
  `results/summary/*.csv`, `results/vis/*.png`

## 8) Como funciona (resumo)

* **Dataset & splits**: `labels/train,val,test.csv` vêm de split estratificado por **imagem-mãe**.
* **Mapeamento**: caminhos em `test/train` traduzem **raw → hough/combined** preservando ângulos (aug) e com *zero-padding* no nome (hough).
* **Treino**: Focal Loss (padrão), flips leves no DataSet, **rebalance** limita aug e casa a taxa de positivos com a base.
* **Validação**: varre `t ∈ [0.10,0.90]` (passo `0.02`), escolhe **melhor balanced accuracy** e salva `best_threshold.json`.
* **Teste**: usa o threshold salvo (se houver) e escreve CSV/JSON.
* **Results**: agrega, ordena por balacc/acc, plota barras e matrizes de confusão (melhor de cada grupo).

## 9) Dicas de estabilidade & OOM

* **DenseNet121** em `hough/combined` é bem mais pesado → use `TRAIN_BATCH_SIZE=32` (por GPU) no **treino** e baixe `EVAL_BATCH_SIZE` no **teste** se necessário.
* **Env (2 GPUs)**: `OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1`.
* **AMP** e **channels_last** já ajudam a memória.
* Em caso de OOM persistente: reduza batch, confira que não tem outras cargas usando a GPU, e desative `persistent_workers` se estiver muito perto do limite de RAM.

## 10) Reprodutibilidade

* Seed fixo (=42) para `random`, `numpy` e `torch`/`cuda`.
* `cudnn.benchmark=True` (rápido; não 100% determinístico).
* Logs em `code/train.log`.

---

### One-liners úteis

**Treinar deepnet121 (hough) em 2 GPUs + testar + results:**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough --aug rebalance --ckpt best && python -m code.results
```

**Treinar deepnet121 (combined) em 2 GPUs + testar + results:**

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance && CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best && python -m code.results
```
