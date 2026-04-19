# MAP2 Patch Discovery Pipeline Walkthrough

This document explains the full pipeline we built, from raw file conversion through latent reporting.

The emphasis here is on three things:

1. what each stage is technically doing
2. where your biological and analytical choices enter the framework
3. how "latent patterns" are actually pulled out of the image data


## 1. High-level goal

The goal of this project is not to train a model to say "this is a punctum" in one supervised step.

The goal is to build a discovery pipeline that:

- uses `MAP2` as the only hard spatial prior
- extracts many local image patches from biologically relevant and comparison regions
- represents each patch numerically
- organizes patches into groups in feature space
- identifies recurring structures, ambiguous structures, and unusual structures without hand-labeling puncta

So the project is not a classical puncta classifier.

It is a `latent structure discovery pipeline` for local multi-channel image patches.


## 2. What the user provided vs. what the pipeline learns

It is important to separate `human assumptions` from `machine-learned structure`.

### User-provided assumptions

These are the priors you explicitly chose:

- `MAP2` is the only hard anatomical prior
- `channel 0 = MAP2`
- `channel 1 = FLAG`
- `channel 2 = HA`
- `channel 3 = SHANK2`
- data should start from raw microscopy images, not deconvolved images
- raw `ND2` files should be converted to `OME-TIFF`
- patch size should begin at `48 x 48 px`
- axial context should begin as `11` slices
- the system should learn from local patches, not whole images
- the model should not be given punctum truth labels

These choices shape the framework, but they do not define the latent classes directly.

### What the pipeline learns

The pipeline learns or discovers:

- what combinations of local morphology, channel intensity, and spatial context recur in the data
- which patches are similar to each other
- which patches are different from each other
- which recurring groups are enriched inside or outside the MAP2 ROI
- which recurring groups are enriched in positive or negative conditions

That is the "latent" part: the groupings are not manually defined up front.


## 3. Stage 1: Raw image conversion

### What we started with

Your source images were raw Nikon `ND2` files.

Those are valid microscopy files, but they are awkward for a custom Python pipeline because:

- tooling support is less uniform
- lazy access is more cumbersome
- metadata handling is more brittle across libraries

### What we did

We built a conversion step that:

- scans the raw ND2 folder
- scans the decon ND2 folder
- matches only raw files that have corresponding decon files
- converts only those matching raw files to `OME-TIFF`

### Why conversion matters

This stage does not do machine learning.

It standardizes the input format so the later pipeline can:

- read volumes more reliably
- use lazy-capable TIFF/Zarr-backed access when available
- preserve microscopy metadata more consistently

### Where user input is embedded here

Your choices entered here in three ways:

- which directory contains the raw images
- which directory contains the matching decon references
- the rule that only raw images with corresponding decon files should be converted


## 4. Stage 2: MAP2 mask generation

### Biological role of the MAP2 mask

The MAP2 mask is the only hard prior in this system.

It is not being used as a punctum label.

It is being used as an anatomical ROI that says:

- inside this region is dendrite-associated territory
- just outside this region is nearby comparison territory
- farther away is broader background territory

This is a weak spatial prior, not a truth label.

### What the mask exporter does

We built a standalone mask exporter inside `map2-patch-discovery`.

It reproduces the minimal path from your `v5.2` logic to `connectivity_mask`:

1. load raw MAP2 volume
2. compute a 2D MIP
3. build a binary MIP mask
4. optionally expand it with a halo
5. convert that mask to a soft confidence map
6. broadcast that confidence through Z
7. blend confidence into the raw MAP2 volume
8. run connectivity propagation to generate the final `connectivity_mask`
9. max-project to a 2D analysis mask
10. save `*_analysismask.ome.tif`

### What it does not do

It does not:

- classify puncta
- use ridge score in the exported phase
- use skeletonization as part of the saved analysis mask
- use centerline tracing as the final mask output

### Why this matters

This step translates your prior knowledge into a machine-usable spatial constraint without manually labeling signal.

### Where user input is embedded here

You defined:

- that `MAP2` is channel 0
- that the mask should come from the `v5.2 -> connectivity_mask` logic
- that skeletonization was not needed for phase 1
- that the output must be a 2D mask TIFF usable by the patch pipeline


## 5. Stage 3: Patch extraction

This is the first stage where the full image becomes a patch dataset.

### What extraction means

Extraction means:

- take one full image
- take its MAP2 mask
- define sampling regions relative to that mask
- crop many local multi-channel image patches
- save those patches in a structured dataset

### Why extraction exists

The later ML does not learn directly from whole `1024 x 1024` images.

It learns from local image neighborhoods.

This is important because the scientific question is local:

- punctate signal
- dendritic texture
- nearby background
- ambiguous edge regions

### Sampling groups

The extractor samples four patch groups:

- `in_mask`
- `boundary`
- `near_mask_outside`
- `far_background`

These are not labels of truth.

They are region categories defined by geometry relative to the MAP2 mask.

### Why these groups matter

This is where a lot of the framework design lives.

If you only sampled obvious puncta-like spots, the system would not learn the full decision boundary you care about.

By sampling several MAP2-relative regions, the model is exposed to:

- likely dendrite-associated regions
- ambiguous border regions
- nearby outside-mask regions
- far-background regions

This gives the later latent analysis a richer comparison space.

### What each extracted patch contains

Each patch stores:

- MAP2 crop
- FLAG crop
- HA crop
- SHANK2 crop
- local MAP2 mask crop
- metadata such as condition, sample id, coordinates, z-center, patch group

### Shard files

Originally, the pipeline saved one file per patch.

That was too slow.

So we changed the storage model to shard files:

- one file per `sample + patch_group`
- each shard contains many patch tensors stacked together
- the manifest points to:
  - `shard_path`
  - `shard_index`

That means the manifest still refers to an individual patch, but the filesystem stores them in batches.

### Performance/caching changes

We also added:

- per-sample channel caching to local `.npy`
- sample-level resume using sample manifests
- uncompressed patch shard writing by default
- vectorized `z_center` computation

This does not change the scientific meaning.

It only changes efficiency.

### Where user input is embedded here

You defined or influenced:

- patch size: `48 x 48`
- axial context: `11` slices
- stride: `24 px`
- the use of MAP2-relative groups instead of candidate puncta labels
- the condition labels for each sample
- the channel schema


## 6. What the machine learning is doing before the latent report

Up to this point, no model has learned latent structure yet.

The pipeline has only:

- standardized the files
- built a MAP2 ROI
- extracted local patches
- summarized metadata

This distinction matters.

Patch extraction is not the discovery step.

It is the dataset construction step.


## 7. Stage 4: Feature representation for latent analysis

The first latent report uses `engineered features`, not a neural network encoder yet.

### What this means

For each patch, the system computes numerical descriptors such as:

- per-channel projection mean
- per-channel projection max
- per-channel projection standard deviation
- mean inside vs outside the mask crop
- inside/outside ratios
- z-profile peak
- z-profile spread
- cross-channel summary ratios like `FLAG/HA`
- joint features like:
  - `flag_ha_mean_sum`
  - `flag_ha_mean_product`
  - `flag_ha_mean_absdiff`

### Why this counts as latent analysis support

Even though these features are hand-engineered, the grouping of patches is still not hand-labeled.

The feature extraction defines a numerical space.

The latent report then discovers structure inside that space.

### What the ML is actually doing here

At this stage, the "ML" is:

- standardizing the feature matrix
- reducing dimensionality with PCA
- clustering in that transformed feature space

So this first report is an `unsupervised or weakly supervised structure discovery` stage, not a deep learning stage.


## 8. Stage 5: Dimensionality reduction

### Why reduce dimensionality

Each patch has many feature columns.

Those features are often correlated.

For example:

- high MAP2 mean and high MAP2 max often go together
- HA and FLAG summaries may co-vary in some patch classes
- z statistics can partly overlap with intensity or texture summaries

### What PCA does

`PCA` finds new axes that capture the largest sources of variation in the feature matrix.

This means:

- the first few PCA components summarize the strongest structured differences between patches
- correlated raw features get compressed into fewer dimensions
- clustering becomes easier and less noisy

### Important clarification

PCA does not know biology.

It only finds dominant variance directions.

Those directions may correspond to:

- real biology
- imaging artifacts
- staining intensity differences
- condition-specific signal structure

That is why interpretation later matters.


## 9. Stage 6: Clustering

### What clustering is doing

Once each patch has coordinates in feature/PCA space, clustering asks:

- which patches sit near each other?
- what dense groups exist in this space?

The initial pipeline uses `GMM` or `KMeans`.

This groups the patches into latent classes.

### Why these are latent classes

These clusters are latent because you did not define them as:

- punctum
- dendrite
- background
- artifact

The clustering algorithm creates groups based on structure in the data itself.

### What a cluster can mean

A cluster may correspond to:

- compact puncta-like patches
- diffuse dendritic texture
- nonspecific outside-mask haze
- boundary-associated mixed structure
- unusual co-expression patches

The cluster meaning is not assigned by the algorithm.

It is inferred afterward by inspecting:

- representative patches
- channel summaries
- condition enrichment
- patch-group enrichment


## 10. How latent patterns are pulled out of the data

This is the core conceptual point.

### Step 1: convert patches into vectors

Each patch becomes a feature vector.

This vector summarizes:

- morphology
- intensity
- z-behavior
- channel relationships
- mask-relative contrast

### Step 2: place all patches into one feature space

Once every patch is a vector, patches that are numerically similar sit near each other.

### Step 3: compress the space

PCA reduces the space to the strongest structure-bearing axes.

### Step 4: identify dense regions

Clustering finds groups of patches that recur.

Those groups are your first latent patterns.

### Step 5: interpret cluster composition

A latent pattern becomes biologically meaningful only when you inspect:

- what images belong to it
- whether it is mostly positive or negative
- whether it occurs mostly inside or outside MAP2
- whether it is HA-rich, FLAG-rich, co-expressed, diffuse, compact, etc.

So a latent pattern is not "a hidden neuron concept" in the abstract.

It is:

`a recurring structure in patch feature space that was not manually specified beforehand`


## 11. Where user input still shapes the latent output

Even though the clusters are not hand-labeled, user choices still strongly shape what can be discovered.

The most important user-controlled inputs are:

### Spatial prior

- using MAP2 as the only hard prior

### Sampling design

- what groups are sampled
- how wide the boundary zone is
- how far outside the mask counts as near vs far background

### Patch geometry

- patch width and height
- z-window
- stride

### Channel schema

- which channels are included
- their order
- which channels are always present vs condition-dependent

### Cohort design

- which positive and negative images are paired in the analysis
- whether mixed channel schemas are allowed

### Representation choice

- which engineered features are computed
- how many PCA dimensions are kept
- how many clusters are requested

So the latent pipeline is not free of human design.

It is `unsupervised inside a user-designed analysis space`.


## 12. What the latent report output means

The latent report writes outputs such as:

- patch-level latent report table
- cluster-by-condition counts
- cluster feature means
- representative patch galleries
- PCA plot

### How to read the outputs

You should think of these outputs as answering questions like:

- what recurring patch types exist?
- which are mostly inside MAP2?
- which are mostly outside?
- which are enriched in positive samples?
- which are present in both positives and negatives?
- which are rare or unusual?

This is where different kinds of puncta and different kinds of background can emerge naturally.


## 13. What this first latent report is not yet doing

It is not yet:

- a deep neural network encoder
- a self-supervised embedding model
- a formal novelty detector against the negative distribution
- a final biological classifier

This first latent report is the baseline discovery stage.


## 14. Why this baseline stage still matters

This stage tells us whether the patch dataset contains usable structure at all.

If the baseline report already separates:

- in-mask punctate classes
- diffuse shaft/background classes
- positive-enriched subclasses
- negative-like subclasses

then the pipeline is already working as a discovery tool.

If it fails, that tells us:

- the features need improvement
- the sampling design needs improvement
- the mask prior may need tuning
- or a learned encoder is needed


## 15. What would come after this latent report

The next future stage would be a learned embedding model.

That would likely mean:

- train an autoencoder or self-supervised encoder on the extracted patches
- replace or augment the engineered features with learned embeddings
- cluster the learned embeddings
- compare positive patches to the negative distribution
- add novelty or background-likeness scoring

At that point the "ML" becomes deeper in the strict sense.

But it is still built on top of the same patch dataset and MAP2-guided sampling framework.


## 16. Final conceptual summary

The full workflow is:

1. convert raw ND2 files to OME-TIFF
2. generate a MAP2 connectivity-derived analysis mask
3. sample local patches relative to the MAP2 ROI
4. store those patches as a dataset with metadata
5. compute numerical features for each patch
6. reduce and cluster that feature space
7. inspect which latent groups recur and how they differ by condition and MAP2 context

The most important point is this:

The system does not discover latent patterns by "looking for puncta directly."

It discovers latent patterns by:

- defining a biologically meaningful local sampling space
- converting each local patch into a numerical representation
- grouping similar patches without hand-labeled punctum classes
- then interpreting those groups biologically

That is the framework we have built.
