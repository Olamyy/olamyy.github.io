---
title: "TensorCAS: Content Addressable Storage for ML Checkpoints"
date: 2026-03-24
categories:
    - tensorcas
    - checkpoint storage
    - content-addressable storage
    - machine learning
---

A question I've been thinking about a lot recently is what it would look like to treat model checkpoints as structured object states instead of opaque blobs.
Since a model is a collection of named arrays, if you could identify individual arrays by their content, you would only need to store changes between checkpoints instead of the full model every time.
This is the idea of [content addressable storage](https://en.wikipedia.org/wiki/Content-addressable_storage) and is the basis for git, IPFS, and other systems that need to store large amounts of versioned data efficiently.
<!-- more -->

??? summary "In a hurry?"

    This article explores the design and implementation of a content-addressable storage system for model checkpoints, called **tensorcas**. 
    The core idea is to deduplicate checkpoints at the tensor level so identical tensors can be stored stored once across runs.
    This leads to significant storage savings for scenarios with a lot of tensor reuse, such as warm-start tree models and fine-tuning from a shared pretrained base.
    

    ```python
    from pathlib import Path
    import torch
    import torch.nn as nn
    from tensorcas.store import TensorCasStore
    from tensorcas.adapters.pytorch import PyTorchAdapter
    
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    
    store = TensorCasStore(
        root=Path("./checkpoints"),
        run_id="mlp-run-001",
        adapter=PyTorchAdapter(),
    )
    
    # Save a checkpoint at each epoch
    for epoch in range(1, 21):
        # ... training loop ...
        store.save(model, step=epoch)
    
    # Load any checkpoint
    store.load(step=10, original=model)
    
    # Inspect what was saved
    print(store.stats())
    # {'run_id': 'mlp-run-001', 'checkpoints': 20, 'total_chunks': 60,
    #  'unique_chunks': 8, 'dedup_ratio': 0.1333, 'total_bytes': 245760}
    # dedup_ratio = unique/total chunks — lower means more reuse (0.13 = 87% reused)
    ```

    - ResNet-18 HP sweep, 8 runs × 10 epochs: 3.3 GB → **39.7 MB**
    - Multi-seed sweep, 4 runs × 10 epochs: 1.7 GB → **39.7 MB**
    - Works best with warm-start tree models and transfer learning from a shared pretrained base.
    - Code: [github.com/olamyy/tensorcas](https://github.com/olamyy/tensorcas)

This idea holds for two reasons:

- In a gradient boosted tree, once a tree is added it never changes. Only new trees are appended. 
- In a fine-tuned neural network, the backbone weights are often identical across runs that share the same pretrained base. 

The rest of this article is about validating the idea actually works, documenting what worked, what didn't work and what the numbers look like.


## Do arrays actually stay identical between checkpoints?

The first thing I wanted to check was how often arrays are byte-identical between checkpoints. If this is common, we could introduce a `no-op` save path that skips writing the array entirely. If it's rare, then the whole idea is less compelling.

!!! note
    
    I define the no-op rate as the percentage of arrays that are byte-identical between two checkpoints. For example, if a checkpoint has 100 arrays and 90 of them are identical to the previous checkpoint, the no-op rate is 90%.


#### Tree Models 
For gradient boosted trees the answer is obvious once you think about it. sklearn's [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) with `warm_start=True` adds new trees each step and never modifies existing ones. So if step 10 has 100 trees and step 20 has 200 trees, trees 0 through 99 are byte-identical between those two checkpoints. The no-op rate at steady state is 100% for all existing tree arrays.

#### XGBoost
XGBoost is similar. Individual tree arrays are frozen once added. The exception is that XGBoost serializes the whole model as a single JSON document, and that document includes a round count that increments every step, so the model JSON always changes even when no tree definitions inside it have.

#### Neural Networks
For PyTorch, active training means every weight is updated every step. The no-op rate during a vanilla training run is close to 0%. Transfer learning is a different story. When you fine-tune a pretrained ResNet-18, the convolutional layers inherited from ImageNet pretraining often don't change across runs, especially if you freeze the backbone. Run 4 experiments with the same pretrained base and different hyperparameters, and those backbone tensors are byte-identical across all 4 runs. That cross-run identity is where the big savings come from.

So the premise holds for tree models unconditionally and for neural networks in the specific case of shared pretrained weights. The next question is how to extract arrays in a way that actually captures that identity.

??? note "Notebook output — dedup_analysis.ipynb"

    **sklearn** — GradientBoostingClassifier warm-start

    ```
    Overall: 100.0% of tensor-steps byte-identical

    Tensor                                Identical  Changed  Identical%
    ------------------------------------ ---------- -------- -----------
    tree_000000_features                          9        0      100.0%
    tree_000000_thresholds                        9        0      100.0%
    tree_000000_values                            9        0      100.0%
    tree_000001_features                          9        0      100.0%
    tree_000001_thresholds                        9        0      100.0%
    tree_000001_values                            9        0      100.0%
    tree_000002_features                          9        0      100.0%
    tree_000002_thresholds                        9        0      100.0%
    tree_000002_values                            9        0      100.0%
    tree_000003_features                          9        0      100.0%
    tree_000003_thresholds                        9        0      100.0%
    tree_000003_values                            9        0      100.0%
    tree_000004_features                          9        0      100.0%
    tree_000004_thresholds                        9        0      100.0%
    tree_000004_values                            9        0      100.0%
    tree_000005_features                          9        0      100.0%
    tree_000005_thresholds                        9        0      100.0%
    tree_000005_values                            9        0      100.0%
    tree_000006_features                          9        0      100.0%
    tree_000006_thresholds                        9        0      100.0%
    tree_000006_values                            9        0      100.0%
    tree_000007_features                          9        0      100.0%
    tree_000007_thresholds                        9        0      100.0%
    tree_000007_values                            9        0      100.0%
    tree_000008_features                          9        0      100.0%
    tree_000008_thresholds                        9        0      100.0%
    tree_000008_values                            9        0      100.0%
    tree_000009_features                          9        0      100.0%
    tree_000009_thresholds                        9        0      100.0%
    tree_000009_values                            9        0      100.0%
    ```

    **XGBoost** — warm-start boosting

    ```
    Overall: 90.9% of tensor-steps byte-identical

    Tensor                                Identical  Changed  Identical%
    ------------------------------------ ---------- -------- -----------
    __skeleton__                                  0        9        0.0%
    tree_000000                                   9        0      100.0%
    tree_000001                                   9        0      100.0%
    tree_000002                                   9        0      100.0%
    tree_000003                                   9        0      100.0%
    tree_000004                                   9        0      100.0%
    tree_000005                                   9        0      100.0%
    tree_000006                                   9        0      100.0%
    tree_000007                                   9        0      100.0%
    tree_000008                                   9        0      100.0%
    tree_000009                                   9        0      100.0%
    ```

You can see the full notebook <a href="https://github.com/olamyy/tensorcas/blob/main/notebooks/dedup_analysis.ipynb" target="_blank">here</a>.

---

## The extraction problem

With the idea validated, the next step was figuring out how to extract arrays in a way that preserved identity.

#### sklearn

Assume you have two models A and B. A has 100 trees. B is a warm start continuation of A with 200 trees. Trees 0 through 99 are carried over unchanged and trees 100 through 199 are new. How do you extract the arrays in these models with the identity of these trees preserved?

One option is to concatenate all the tree arrays into one big array per checkpoint. The problem with this is that the concatenated array grows in shape every step, so its hash changes each time, and a different hash means that it's stored as a completely new object for every save, even though the data for the earlier trees is identical.

```
Model A:  [tree_0 | tree_1 | ... | tree_99]                              shape: (412,)  hash: a3f9...
Model B:  [tree_0 | tree_1 | ... | tree_99 | tree_100 | ... | tree_199]  shape: (846,)  hash: 7c2e...
```

The workaround for this is to extract a key per tree. With per-tree keys, the storage layer writes only the 100 new trees and reuses the rest. The no-op rate at this point is 100% for all existing tree arrays, and the storage layer only needs to write the new trees.

```
Model A:  tree_000000_features   shape: (4,)  hash: b1c2...
          tree_000000_thresholds  shape: (4,)  hash: d3e4...
          ...
          tree_000099_values      shape: (4,)  hash: f5a6...

Model B:  tree_000000_features   shape: (4,)  hash: b1c2...  <- same, no write
          tree_000000_thresholds  shape: (4,)  hash: d3e4...  <- same, no write
          ...
          tree_000099_values      shape: (4,)  hash: f5a6...  <- same, no write
          tree_000100_features    shape: (4,)  hash: 9g7h...  <- new, write
          ...
          tree_000199_values      shape: (4,)  hash: 2i3j...  <- new, write
```

#### XGBoost

The question is similar to that of sklearn. Assume you have two models A and B. A is trained for 100 rounds. B is a warm-start continuation of A trained for 200 rounds. The first 100 trees are identical.

XGBoost serializes the whole model as a single JSON document. An approach is to store that document as one array per checkpoint. The problem is that the JSON embeds the round count, so the whole document changes every step even though the individual tree definitions inside it are identical.

```
Model A:  [full model JSON, 100 rounds]  hash: c4d5...
Model B:  [full model JSON, 200 rounds]  hash: e6f7...
```

Here, the workaround was to split the JSON into two parts:

- A `__skeleton__` key for everything except tree definitions, 
- A `uint8` array per tree for the raw JSON bytes of that tree's node structure. 

Since tree arrays are stable once added, the skeleton is the only component that changes every step. The storage layer writes the skeleton every time, but it can reuse all the tree arrays that are identical.

```
Model A:  __skeleton__   hash: c4d5...
          tree_000000    hash: h8i9...
          ...
          tree_000099    hash: j0k1...

Model B:  __skeleton__   hash: l2m3...  <- changed, write
          tree_000000    hash: h8i9...  <- same, no write
          ...
          tree_000099    hash: j0k1...  <- same, no write
          tree_000100    hash: n4o5...  <- new, write
          ...
          tree_000199    hash: p6q7...  <- new, write
```

Is this instance, the no-op rate is closer to 90%.

#### PyTorch

Assume you have a [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) fine-tuned from a shared pretrained across two runs A and B with different hyperparameters. The backbone weights (convolutional layers inherited from ImageNet pretraining) are identical across both runs. Only the head weights will be different.

A naive approach is to save the full ``state_dict`` as a single binary blob per checkpoint. The whole file differs between runs even though most of the weights inside are identical, so the storage layer writes the full 150MB on every save which is exactly what we were trying to avoid.

```
Run A checkpoint:  [full state_dict, 150MB]  hash: r8s9...
Run B checkpoint:  [full state_dict, 150MB]  hash: t0u1...
```

A better approach is to iterate `state_dict()` and store each tensor under its own key while preserving the native dtype. 

!!! note
    
    Upcasting to float64 doubles memory usage and distorts every compression ratio in the benchmarks.


```
Run A:  layer1.weight   hash: v2w3...
        layer1.bias     hash: x4y5...
        ...
        fc.weight       hash: z6a7...

Run B:  layer1.weight   hash: v2w3...  <- same, no write
        layer1.bias     hash: x4y5...  <- same, no write
        ...
        fc.weight       hash: b8c9...  <- changed, write
```

With a working extraction strategy, the next thing to check was whether delta compression could handle the arrays that do change.

---

## Delta compression

Finding identical arrays between checkpoints is straightforward when arrays are frozen. 
For arrays that do change between steps, a question that came up is whether we could store the delta between them instead of the full array. If the delta is smaller than the original, we save space. If it's larger, we can just store the full array instead.

To check this, I calculated the compression ratio τ for each checkpoint pair, where `B - A` is the arithmetic delta between arrays, `zstd(·)` compresses it, and `|B|` is the uncompressed size of the original array:

$$
\tau = \frac{|\text{zstd}(B - A)|}{|B|}
$$

A delta is worth storing when τ < 1. The lower the value, the more compressible the delta.

#### Tree models

For sklearn and XGBoost this question barely applies. Existing tree arrays never change, so there is no delta to compress. The only changing component in XGBoost is the skeleton, which is small enough that storing it in full every step is not a problem.

#### PyTorch

For ResNet-18 on CIFAR-10 during early training, τ was between **1.17 and 1.31**. Gradient updates during early training are high entropy and compress worse than the original. At convergence with a low learning rate, τ drops to around 0.58 to 0.92 at 256KB chunk sizes.

The learning from this is that compression only works at convergence. During early training, deltas are larger than the originals and not worth storing.

You can see the full notebook for these experiments <a href="https://github.com/olamyy/tensorcas/blob/main/notebooks/compression_benchmark.ipynb" target="_blank">here</a>.

---

## The storage layer

The learnings from the previous section established three things. 

- Arrays are the unit of storage, not files. 
- Identity is determined by content hash. 
- Most arrays don't change between checkpoints.

From these constraints, we can design a tensor storage layer where each tensor is split into fixed-size chunks. Each chunk is hashed with [BLAKE3](https://en.wikipedia.org/wiki/BLAKE_(hash_function)) and stored by that hash. If a chunk already exists in the store, it is not written again. Checkpoints that share identical arrays pay nothing for the duplication.

For lookup, a manifest is written for each checkpoint that records the chunks that belong to an array. On load, the manifest is read first, then the chunks are fetched and reassembled.

The data flow for a save:

```
save(model, step)
  extract arrays from model
  for each array:
    check if identical to previous checkpoint  ->  skip if unchanged
    split into 1MB chunks
    hash each chunk (BLAKE3)
    check which chunks already exist in the store
    write only the missing chunks (zstd compressed)
  write a manifest recording which chunks belong to this checkpoint
  register the checkpoint in the index
```

Chunks are stored under `objects/{hash[:2]}/{hash[2:4]}/{hash[4:]}.chunk`, using three levels of directory sharding to avoid inode pressure at scale. The hash is computed on raw uncompressed bytes before compression, so chunk identity is independent of the compression format. Each chunk is then compressed with zstd and written atomically via a tempfile and `os.replace()`, so a reader never sees a partially written chunk.

---

## No-op cost

The no-op check in the save flow is straightforward: hash the full array, compare to the stored hash from the previous checkpoint's manifest. For a sklearn model with 5,000 trees where only 10 are new each step, this means rehashing 4,990 arrays that haven't changed. Hashing is cheap per array but not free, and at that scale it took up a large chunk of save latency.

To improve this, I added a hash cache in memory that maps array names to their last known hash. Before hashing an array, the cache is checked first. If the cached hash matches the manifest hash, the array is skipped with no computation. On a cache miss, shape and dtype are checked first as a fast-fail, then BLAKE3 is computed and the cache is updated.

Before the cache, a 5,000-tree sklearn model took around 1,425ms per save. After:

```
Total trees  |  Median save
-------------|-------------
500          |  63.5ms
1,000        |  70.5ms
5,000        |  70.2ms
```

The hash cache worked in sklearn because nearly everything is frozen. It could not work for XGBoost because the skeleton cannot be frozen. Each save must write the skeleton, which is the model JSON minus the individual tree definitions. 
It embeds the round count, so it changes every step regardless of the hash cache. At 5,000 rounds the skeleton is around 40 KB and must be written on every checkpoint. The save latency in this case grows with round count:

```
Total rounds  |  Median save
--------------|-------------
500           |  50.4ms
5,000         |  208.7ms
```

---

## Garbage collection

A known issue with CAS is that it accumulates chunks over time. This means that deleting a checkpoint run doesn't automatically free the objects it wrote, because those chunks might be shared with other runs. It also meant the storage layer needed a way to find and delete chunks that are no longer referenced by any active checkpoint.

The first attempt to solve this used a `refs` join table: on every save, write one row per new chunk recording which run references it. The garbage collector then queries for chunks with no live references. The logic is straightforward and the schema is simple to reason about.

A problem comes up for large PyTorch models with thousands of chunks, where the `O(new_chunks)` writes per save dominated save latency more than the actual chunk I/O. The increasing number of writes made the `refs` table approach impractical at scale.

To improve this, I used a [mark-and-sweep](https://en.wikipedia.org/wiki/Tracing_garbage_collection) heuristic that runs GC as a separate pass with:

1. **Mark**: scan all manifest files on disk and collect the set of all referenced chunk hashes.
2. **Sweep**: any chunk in the object store not in that set is a deletion candidate.
3. **Grace period**: only delete candidates older than 24 hours, so chunks written by an in-progress save aren't removed before their manifest is committed.

This dropped saved cost back to `O(new_chunks)` of chunk I/O with no join table writes so GC cost scales with the size of the store, not with each individual save.

---

## Benchmark Results

### Storage

Each scenario was run as a sequence of checkpoints saved with `torch.save`, DVC, and tensorcas. The numbers below are total storage for the full sequence.

| Scenario | torch.save | DVC | tensorcas |
|---|---|---|---|
| ResNet-18, 20 checkpoints | 854.5 MB | 790.7 MB | **39.5 MB** (95.0% vs DVC) |
| Multi-seed sweep (4 runs × 10 epochs) | 1.7 GB | 1.7 GB | **39.7 MB** (97.7%) |
| HP sweep (8 runs × 10 epochs) | 3.3 GB | 3.3 GB | **39.7 MB** (98.8%) |
| Mid-training resume (2 snapshots) | 85.5 MB | 85.5 MB | **39.6 MB** (53.7%) |

DVC matches torch.save in every scenario because each checkpoint file is treated as a distinct object even when most of the weights inside are identical. The savings come entirely from array-level deduplication.

The multi-seed and HP sweep numbers reflect the core thesis: four runs from the same pretrained ResNet-18 backbone, DVC stores each independently, tensorcas stores the backbone once and each run adds around 53 KB for the fine-tuned head weights. The HP sweep is the same pattern at larger scale: 8 configs, 10 epochs each, all sharing the same pretrained base.

The mid-training resume case is more modest at 53.7%. Two snapshots of a single run with a frozen backbone. Most of the savings come from the backbone being written once; the head weights and BatchNorm running stats are distinct for each snapshot.

### Latency

Numbers from `pytorch_resnet.ipynb`, ResNet-18 with 122 tensors in the state dict.

| Scenario | Median save |
|---|---|
| Full fine-tune (20 epochs) | 36.0ms |
| Frozen backbone (15 epochs, fc only changes) | 14.0ms |
| LoRA-style (20 epochs, fc + 1 conv changes) | 14.9ms |

Save latency drops significantly once most tensors are frozen. Full fine-tune writes all 122 tensors every step; frozen backbone writes only the FC layer, which is around 18.5 KB.

---

## What didn't work

**Chunk-level deduplication for neural networks.** I expected that even changing weights would share some 1 MB chunks between consecutive epochs. They don't. During active training, chunk-level reuse within an array is essentially zero. All savings come from whole-array identity. Chunk size turns out not to matter for storage in any of these benchmarks.

**Delta compression during active training.** Covered in the delta compression section above: early training deltas are larger than originals. This will matter more at convergence, which is what Phase 2 is for.

**Reference-counting GC.** Covered in the GC section above.

---

## Code

tensorcas is at [github.com/olamyy/tensorcas](https://github.com/olamyy/tensorcas). 

```python
from tensorcas import TensorCasStore

store = TensorCasStore("/path/to/store")

for epoch in range(20):
    train_one_epoch(model, optimizer)
    model.eval()
    store.save(model, run_id="run_a", step=epoch)
    model.train()
```

If you are training tree models with warm-start or running multiple experiments from a shared pretrained backbone, the savings are significant and the API overhead is low enough to checkpoint every step by default.

---

## Things to try next

**Delta encoding.** The delta compression measurements show it only becomes useful at convergence, specifically for fine-tuning scenarios where weights change slowly. For those cases, the delta ratios are low enough that it's worth implementing. The hard part is the policy: when do you store a delta versus a full chunk? Restoration cost has to factor in. A checkpoint that is three deltas deep requires three sequential reads and three reconstructions to load.

**XGBoost skeleton delta encoding.** The skeleton is the one component that changes every step: it embeds the round count and grows with training. At 5,000 rounds it is around 40 KB per save. Delta-encoding the skeleton separately would eliminate that growth, leaving XGBoost saves as flat as sklearn.

**GPT-2 fine-tuning.** The benchmark I haven't run yet. 117M parameters, around 445 MB. My guess is the transfer learning pattern holds: shared pretrained weights stored once, marginal cost per run proportional to what actually changed. The attention weight structure is different from ResNet's conv layers though, and I want the actual numbers before claiming it generalizes.
