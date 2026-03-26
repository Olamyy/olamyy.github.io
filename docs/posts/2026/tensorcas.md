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
This is the idea of <a href="https://en.wikipedia.org/wiki/Content-addressable_storage" target="_blank">content addressable storage</a> and is the basis for git, IPFS, and other systems that need to store large amounts of versioned data efficiently.
<!-- more -->

??? summary "In a hurry?"

    This article explores the design of a content-addressable storage system for model checkpoints, called **tensorcas**. The core idea is that instead of storing full checkpoints each time, we can deduplicate models at the tensor level so identical tensors across checkpoints and runs are stored once.

    The savings are significant for warm-start tree models (only new trees are written each step) and for transfer learning from a shared pretrained base (backbone stored once across all runs). During active full training, savings are modest since every weight changes every epoch.

    The most interesting learning from this is that deduplication only works if you extract tensors correctly. Concatenating all tree arrays into one blob defeats it entirely, since the concatenated array grows in shape every step and gets a new hash. The fix is [one key per tree](#the-extraction-problem).

    A few other things that came up worth knowing: [BatchNorm running stats](#the-batchnorm-problem-pytorch) update even on frozen layers, so `model.eval()` before save is required to get full backbone no-op rates. [GC](#garbage-collection) uses mark-and-sweep over manifest files rather than a per-chunk refs table, which keeps save cost flat. Load is always a single read per tensor regardless of how many checkpoints exist.

    Storage numbers (ResNet-18, shared pretrained base):

    - HP sweep, 8 runs × 10 epochs: 3.3 GB → **39.7 MB**
    - Multi-seed sweep, 4 runs × 10 epochs: 1.7 GB → **39.7 MB**

    ```python
    from pathlib import Path
    from tensorcas.store import TensorCasStore
    from tensorcas.adapters.pytorch import PyTorchAdapter

    store = TensorCasStore(
        root=Path("./checkpoints"),
        run_id="run-001",
        adapter=PyTorchAdapter(),
    )

    for epoch in range(20):
        train_one_epoch(model, optimizer)
        model.eval()
        store.save(model, step=epoch, metrics={"val_loss": val_loss})
        model.train()

    best_epoch = store.best("val_loss", mode="min")
    model = store.load(step=best_epoch, original=model)
    ```

    Code: <a href="https://github.com/olamyy/tensorcas" target="_blank">github.com/olamyy/tensorcas</a>

This works for two reasons:

- In a gradient boosted tree, once a tree is added it never changes; only new trees are appended as training continues.
- In a finetuned neural network, the backbone weights are often identical across runs that share the same pretrained base. 

The rest of this article covers what it took to make that work in practice, where it works well, where it doesn't, and the numbers.


## Do arrays actually stay identical between checkpoints?

The first thing I wanted to check was how often arrays are byte-identical between checkpoints. If this is common, we could introduce a `no-op` save path that skips writing the array entirely. If it's rare, then the whole idea is less compelling.

!!! note
    
    I define the no-op rate as the percentage of arrays that are byte-identical between two checkpoints. For example, if a checkpoint has 100 arrays and 90 of them are identical to the previous checkpoint, the no-op rate is 90%.


#### Tree Models 
For gradient boosted trees the answer is obvious once you think about it. sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html" target="_blank">GradientBoostingClassifier</a> with `warm_start=True` adds new trees each step and never modifies existing ones. So if step 10 has 100 trees and step 20 has 200 trees, trees 0 through 99 are byte-identical between those two checkpoints. The no-op rate at steady state is 100% for all existing tree arrays.

#### XGBoost
XGBoost is similar. Individual tree arrays are frozen once added. The exception is that XGBoost serializes the whole model as a single JSON document, and that document includes a round count that increments every step, so the model JSON always changes even when no tree definitions inside it have.

#### Neural Networks
For PyTorch, active training means every weight is updated every epoch. The no-op rate during a vanilla training run is close to 0%. Transfer learning is a different story. When you fine-tune a pretrained ResNet-18, the convolutional layers inherited from ImageNet pretraining often don't change across runs, especially if you freeze the backbone. Run 4 experiments with the same pretrained base and different hyperparameters, and those backbone tensors are byte-identical across all 4 runs. That cross-run identity is where the big savings come from.

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

With the idea validated, the next step was figuring out how to extract arrays in a way that preserved identity. This is important because we want to be able to compare tensor content hashes. If the same underlying weights produce a different array each time they're extracted, they'll hash differently and be treated as new data on every save, even when nothing has changed.

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

- A `__skeleton__` key for everything except tree definitions
- A `uint8` array per tree for the raw JSON bytes of that tree's node structure

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

In this instance, the no-op rate is closer to 90%.

#### PyTorch

Assume you have a <a href="https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html" target="_blank">ResNet-18</a> finetuned from a shared pretrained across two runs A and B with different hyperparameters. The backbone weights (convolutional layers inherited from ImageNet pretraining) are identical across both runs. Only the head weights will be different.

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

For arrays that are frozen, deduplication handles everything. For arrays that do change between steps, a question that came up is whether I could use the arithmetic delta between consecutive checkpoints instead of the full array as a storage unit. If the delta is more compressible than the original, this would reduce storage for changing arrays as well.

To check this, I calculated the compression ratio τ for each checkpoint pair, where `B - A` is the arithmetic delta between arrays, `zstd(·)` compresses it, and `|B|` is the uncompressed size of the original array:

$$
\tau = \frac{|\text{zstd}(B - A)|}{|B|}
$$

A delta is worth storing when τ < 1, with lower values indicating a more compressible delta.

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

From these constraints, we can design a tensor storage layer where each tensor is split into fixed-size chunks. Each chunk is hashed with <a href="https://en.wikipedia.org/wiki/BLAKE_(hash_function)" target="_blank">BLAKE3</a> and stored by that hash. If a chunk already exists in the store, it is not written again. Checkpoints that share identical arrays pay nothing for the duplication.

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

For lookup, a manifest is written for each checkpoint that records the chunks that belong to an array. On load, the manifest is read first, then the chunks are fetched and reassembled.

The data flow for a load:

```
load(step)
  look up manifest path for (run_id, step) in the registry
  read and parse the manifest
  verify all referenced chunk hashes exist in the object store
  for each tensor in the manifest (in parallel):
    fetch chunks by hash
    decompress each chunk (zstd)
    reassemble chunks into the original array
  pass tensors to the adapter to reconstruct the model
```

Because each checkpoint has its own manifest pointing directly to its chunks, load time is independent of how many checkpoints exist in the store. There is no traversal of prior checkpoints.

---

## No-op cost

The no-op check in the save flow is straightforward: hash the full array, compare to the stored hash from the previous checkpoint's manifest. For a sklearn model with 5,000 trees where only 10 are new each step, this means rehashing 4,990 arrays that haven't changed. Hashing is cheap per array but not free, and at that scale it dominated save latency.

To improve this, I added a hash cache in memory that maps array names to their last known hash. Before hashing an array, the cache is checked first. If the cached hash matches the manifest hash, the array is skipped with no computation. On a cache miss, shape and dtype are checked first as a fast-fail, then BLAKE3 is computed and the cache is updated.

The question is whether the hashing overhead is noticeable during training. At 5,000 trees, the answer is no. Before the cache, a 5,000-tree sklearn model took around 1,425ms per save. After:

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

To improve this, I used a <a href="https://en.wikipedia.org/wiki/Tracing_garbage_collection" target="_blank">mark-and-sweep</a> heuristic that runs GC as a separate pass with:

1. **Mark**: scan all manifest files on disk and collect the set of all referenced chunk hashes.
2. **Sweep**: any chunk in the object store not in that set is a deletion candidate.
3. **Grace period**: only delete candidates older than 24 hours, so chunks written by an in-progress save aren't removed before their manifest is committed.

This dropped save cost back to `O(new_chunks)` of chunk I/O with no join table writes. Save cost is now flat per checkpoint regardless of store size; GC scales with the store, but runs as a separate pass and doesn't affect training.

---

## Benchmark Results

### Storage

All scenarios below are PyTorch/ResNet-18. Each was run as a sequence of checkpoints saved with `torch.save`, DVC, and tensorcas. The numbers are total storage for the full sequence.

| Scenario | torch.save | DVC | tensorcas |
|---|---|---|---|
| ResNet-18, 20 checkpoints | 854.5 MB | 790.7 MB | **39.5 MB** (95.0% vs DVC) |
| Multi-seed sweep (4 runs × 10 epochs) | 1.7 GB | 1.7 GB | **39.7 MB** (97.7%) |
| HP sweep (8 runs × 10 epochs) | 3.3 GB | 3.3 GB | **39.7 MB** (98.8%) |
| Mid-training resume (2 snapshots) | 85.5 MB | 85.5 MB | **39.6 MB** (53.7%) |

DVC matches torch.save in every scenario because each checkpoint file is treated as a distinct object even when most of the weights inside are identical. The savings in tensorcas come from array-level deduplication, which neither tool does.

The multi-seed and HP sweep numbers reflect the core thesis: four runs from the same pretrained ResNet-18 backbone, DVC stores each independently, tensorcas stores the backbone once and each run adds around 53 KB for the finetuned head weights. The HP sweep is the same pattern at larger scale: 8 configs, 10 epochs each, all sharing the same pretrained base.

The mid-training resume case is more modest at 53.7%. Two snapshots of a single run with a frozen backbone. Most of the savings come from the backbone being written once; the head weights and BatchNorm running stats are distinct for each snapshot.

### Latency

Numbers from <a href="https://github.com/olamyy/tensorcas/blob/main/notebooks/pytorch_resnet.ipynb" target="_blank">pytorch_resnet.ipynb</a>, ResNet-18 with 122 tensors in the state dict.

| Scenario | Median save |
|---|---|
| Full fine-tune (20 epochs) | 36.0ms |
| Frozen backbone (15 epochs, fc only changes) | 14.0ms |
| LoRA-style (20 epochs, fc + 1 conv changes) | 14.9ms |

Save latency drops significantly once most tensors are frozen. Full fine-tune writes all 122 tensors every epoch; frozen backbone writes only the FC layer, which is around 18.5 KB.

Delta chaining, where load time would depend on checkpoint depth, is a Phase 2 concern.

---

## What didn't work

**Chunk-level deduplication for neural networks.** I expected that even changing weights would share some 1 MB chunks between consecutive epochs, but during active training, chunk-level reuse within an array is essentially zero. All savings come from whole-array identity, and chunk size turns out not to matter for storage in any of these benchmarks.

**Delta compression during active training.** Covered in the delta compression section above: early training deltas are larger than originals. This will matter more at convergence, which is what Phase 2 is for.

**Reference-counting GC.** The per-chunk `refs` join table was the obvious first approach, but the O(new_chunks) SQLite writes per save made it impractical for large PyTorch models. Mark-and-sweep replaced it entirely.

---

## Code

tensorcas is at <a href="https://github.com/olamyy/tensorcas" target="_blank">github.com/olamyy/tensorcas</a>. 

```python
from pathlib import Path
from tensorcas.store import TensorCasStore
from tensorcas.adapters.pytorch import PyTorchAdapter

store = TensorCasStore(
    root=Path("./checkpoints"),
    run_id="run-001",
    adapter=PyTorchAdapter(),
)

for epoch in range(20):
    train_one_epoch(model, optimizer)
    model.eval()
    store.save(model, step=epoch, metrics={"val_loss": val_loss})
    model.train()

best_epoch = store.best("val_loss", mode="min")
model = store.load(step=best_epoch, original=model)
```

If you are training tree models with warm-start or running multiple experiments from a shared pretrained backbone, the savings are significant and the API overhead is low enough to checkpoint every epoch by default. Here `step` is the epoch number for neural networks and the tree-addition step for sklearn and XGBoost, not a batch step.

For batch-level workflows, the same API applies: pass the global batch index as `step`. During active training the no-op rate will be near zero since every weight changes every batch, so the storage savings will be modest. The value comes from cross-run deduplication: multiple runs sharing the same pretrained base will still store the backbone once regardless of checkpoint frequency.

```python
global_step = 0
for epoch in range(20):
    for batch in dataloader:
        train_one_batch(model, optimizer, batch)
        global_step += 1
        if global_step % save_every == 0:
            model.eval()
            store.save(model, step=global_step)
            model.train()
```

---

## Things to try next

**Delta encoding.** The delta compression measurements show it only becomes useful at convergence, specifically for fine-tuning scenarios where weights change slowly. For those cases, the delta ratios are low enough that it's worth implementing. The hard part is the policy: when do you store a delta versus a full chunk? Restoration cost has to factor in. A checkpoint that is three deltas deep requires three sequential reads and three reconstructions to load.

**XGBoost skeleton delta encoding.** The skeleton is the one component that changes every step: it embeds the round count and grows with training. At 5,000 rounds it is around 40 KB per save and save latency has grown to 208.7ms. Delta-encoding the skeleton separately would eliminate that growth and bring XGBoost save latency down to the same flat range as sklearn (around 70ms regardless of round count).

**Transducers for incremental weight updates.** Delta encoding stores the difference between checkpoints and reconstructs at load time. A complementary idea is to represent weight updates as <a href="https://en.wikipedia.org/wiki/Finite-state_transducer" target="_blank">transducers</a>. This would be useful for cases where multiple finetuned versions share a common base and you want to store the transformation rather than the result. The open question is whether the transducer representation is compact enough to be worth the reconstruction complexity.

**GPT-2 fine-tuning.** The benchmark I haven't run yet. 117M parameters, around 445 MB. My guess is the transfer learning pattern holds: shared pretrained weights stored once, marginal cost per run proportional to what actually changed. The attention weight structure is different from ResNet's conv layers though, and I want the actual numbers before claiming it generalizes.
