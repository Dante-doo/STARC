# README – STARC

> *Por padrão, os **testes** usam `--aug rebalance` apenas para **apontar a pasta do checkpoint**. O **conjunto de teste não recebe nenhum augment**; ele sempre lê `labels/test.csv` com imagens reais. O **treino**, por padrão, usa dados balanceados via `aug rebalance`.*

---

## 🧭 Ordem de execução do pipeline

1) **Slice** → recorta as imagens *full* em patches 224×224.
2) **Hough (full & patches)** → gera artefatos Hough/Coerência/Inverted/Accumulator.
3) **Augment** → rotações (90/180/270) dos *full frames* e patches.
4) **Combined** → gera patches RGB combinando canais (ex.: raw + hough + inv).
5) **Label** → gera `labels/all_labels.csv` com rótulos por patch.
6) **Split** → cria `labels/{train,val,test}.csv` estratificados por *full*.
7) **Train** → treina (ResNet18 / DenseNet121) em `raw | hough | combined`.
8) **Test** → infere no *test split*; salva métricas e predições.
9) **Results** → agrega resultados, gera tabelas e gráficos.

> **Pré-requisitos:** variáveis/caminhos em `code/config.py` devidamente ajustados (pastas de entrada/saída, workers, batch sizes, etc.).

---

## 🧪 Comandos – Preparação de dados

> Execute **nesta ordem**.

### 1) Slice (gera patches RAW & MASK)

```bash
python -m code.slice
```

### 2) Hough (artefatos a partir dos *fulls* e/ou patches)

```bash
python -m code.hough
```

### 3) Augment (rotações dos *fulls* e patches)

```bash
python -m code.augment
```

### 4) Combined (gera patches RGB combinados)

```bash
python -m code.combined
```

### 5) Label (gera rótulos por patch)

```bash
python -m code.label
```

### 6) Split (gera train/val/test estratificados por *full*)

```bash
python -m code.split
```

---

## 🏋️ Treino (DDP 2×GPU)

> **Padrão** de treino: `--aug rebalance` (balanceia positivos; *não* é augment no test). Ajuste `TRAIN_BATCH_SIZE` no `config.py` para *batch por GPU*.

### Treinar **todos os 6 experimentos** (ResNet18 + DenseNet121 × raw/hough/combined)

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality raw      --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality hough    --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality combined --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality raw      --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough    --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

### Treinos por arquitetura

**Só ResNet18 (3 modalidades):**
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality raw      --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality hough    --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18 --modality combined --aug rebalance
```

**Só DenseNet121 (3 modalidades):**
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality raw      --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough    --aug rebalance && \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

### Treinos individuais (seis opções)

```bash
# ResNet18
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality raw      --aug rebalance
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality hough    --aug rebalance
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch resnet18   --modality combined --aug rebalance
# DenseNet121
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality raw      --aug rebalance
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality hough    --aug rebalance
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 -m code.train --arch deepnet121 --modality combined --aug rebalance
```

---

## 🔎 Teste (1×GPU)

> **Importante**: `--aug rebalance` no **teste** *não aplica augment* — apenas seleciona a subpasta onde o checkpoint foi salvo durante o treino.
>
> Estrutura: `models/<arch_dir>/<modality>/[aug-<policy>]/{best,last}.pt`

### Testar **todos os 6 experimentos** (assumindo que foram treinados com `aug rebalance`)

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality raw      --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality hough    --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality combined --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality raw      --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough    --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best
```

### Testes por arquitetura

**Só ResNet18 (3 modalidades):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18 --modality raw      --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18 --modality hough    --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18 --modality combined --aug rebalance --ckpt best
```

**Só DenseNet121 (3 modalidades):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality raw      --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough    --aug rebalance --ckpt best && \
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best
```

### Testes individuais (seis opções)

```bash
# ResNet18
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality raw      --aug rebalance --ckpt best
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality hough    --aug rebalance --ckpt best
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch resnet18   --modality combined --aug rebalance --ckpt best
# DenseNet121
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality raw      --aug rebalance --ckpt best
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality hough    --aug rebalance --ckpt best
CUDA_VISIBLE_DEVICES=0 python -m code.test --arch deepnet121 --modality combined --aug rebalance --ckpt best
```

> **Se treinou com outra política** (ex.: `--aug none`): troque `--aug rebalance` por `--aug none` no comando de teste, pois o script procura o checkpoint na subpasta correspondente.

---

## 📊 Results – Agregação e gráficos

```bash
python -m code.results
```

Saídas principais:
- `results/summary/all_runs.csv`, `best_by_group.csv`, `mean_by_group.csv`.
- `results/vis/balacc_best_by_group.png`, `balacc_mean_by_group.png`.
- Confusions normalizadas por linha: `results/vis/confusion_<arch>_<mod>_<aug>.png` (melhor de cada grupo).

---

## ℹ️ Dicas e troubleshooting

- **OOM na DenseNet121:** reduza `TRAIN_BATCH_SIZE` (é *por GPU*). Para 1080 Ti, valores 16–32 costumam ser seguros em `hough/combined`.
- **TF32 / AMP / channels_last:** controlados via `config.py` (`TRAIN_TF32`, `TRAIN_MIXED_PRECISION`, `TRAIN_CHANNELS_LAST`).
- **DDP**: os comandos já definem variáveis para ambientes sem NVLink/IB: `NCCL_IB_DISABLE=1` e `NCCL_P2P_DISABLE=1`.
- **Seeds:** fixados em `train.py` para Python/NumPy/PyTorch (determinismo desligado no cuDNN para desempenho, mas repetibilidade de alto nível mantida).

---

## 📁 Layout de checkpoints e saídas

```
models/
  resnet/|deepnet121/
    raw|hough|combined/
      aug-rebalance|aug-none|aug-all/
        best.pt, last.pt, best_threshold.json, train_meta.json
results/
  preds/<arch>/<modality>[/aug-...]/test_preds.csv
  metrics/<arch>/<modality>[/aug-...]/test_metrics.json
  summary/*.csv
  vis/*.png
```

---

## 🌐 English TL;DR

- **Test time** uses `--aug rebalance` **only** to point to the checkpoint directory. Test set is **never augmented**; it always reads `labels/test.csv`.
- **Training** defaults to `--aug rebalance` (class balance via curated aug rows).
- Run order: `slice → hough → augment → combined → label → split → train → test → results`.
- Use the bundled one‑liners above to run **all trainings**, **all tests**, per‑arch batches, or individual runs.

Happy training! 🚀

