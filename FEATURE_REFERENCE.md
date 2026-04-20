# Feature Reference

This file documents the feature outputs defined in the code under `src/map2_patch_discovery/`, with a focus on:

- what patches and shards are in this pipeline
- patch-manifest summary fields written during extraction
- engineered features used by the latent report
- how the final latent-report feature set is selected

Primary source files:

- `src/map2_patch_discovery/summaries.py`
- `src/map2_patch_discovery/patches.py`
- `src/map2_patch_discovery/features.py`
- `src/map2_patch_discovery/latent_report.py`

## 1. What A Patch Is

In this pipeline, a patch is a small local crop extracted around a sampled `(x, y)` center from a microscopy volume.

Each patch contains:

- one 3D crop per required channel, shaped roughly as `Z x Y x X`
- one 2D MAP2 mask crop, shaped as `Y x X`
- metadata describing where that crop came from

Patch geometry is controlled by the dataset config:

- `width_px`
- `height_px`
- `z_window`

The `(x, y)` center is chosen from one of the configured sampling groups:

- `in_mask`
- `boundary`
- `near_mask_outside`
- `far_background`

For each chosen center:

- the z-center is estimated from the MAP2 channel by taking the z index with maximal intensity at that `(x, y)` location
- every required channel is cropped around the same `(x, y, z)` center
- a matching 2D MAP2 mask crop is extracted over the same `Y x X` footprint

So when the code later computes features, it is not working from whole images. It is working from small per-patch channel crops plus the local MAP2 mask crop.

## 2. What A Shard Is

Patches are not stored one file at a time. They are grouped into `.npz` shard files during extraction.

Each shard contains:

- `patch_ids`: the patch IDs stored in that shard
- `map2_mask`: stacked 2D mask crops for those patches
- `channel_<NAME>`: stacked 3D crops for each requested channel

If sharding by group is enabled, the code writes one shard per sample and patch group, for example:

- `sampleA_in_mask.npz`
- `sampleA_boundary.npz`

If sharding by group is disabled, all patches for the sample go into one shard:

- `sampleA_all_patches.npz`

The manifest then records:

- which shard a patch lives in
- that patch's index inside the shard
- metadata about the patch center and sample

Later, feature extraction loads the shard, slices out one patch by `shard_index`, and computes features from that in-memory patch payload.

## 3. Patch Payload Used For Feature Extraction

When `extract_engineered_features()` reads a patch, it reconstructs a payload like this:

- `map2_mask`: one 2D boolean mask crop
- `channel_MAP2`: one 3D crop if MAP2 is requested
- `channel_FLAG`: one 3D crop if FLAG is requested
- `channel_HA`: one 3D crop if HA is requested
- `channel_SHANK2`: one 3D crop if SHANK2 is requested

From each 3D channel crop, the code typically derives:

- `max_proj = max over z`
- `mean_proj = mean over z`
- a bright-pixel mask from the max projection
- a z-profile from averaging over `Y` and `X`

The local `map2_mask` is used to define:

- inside-mask pixels
- outside-mask pixels
- MAP2-aware spatial context
- in-mask cross-channel correlation

## 4. Patch Manifest Summary Features

During patch extraction, each required channel crop is summarized by `summarize_channel_crop()` in `src/map2_patch_discovery/summaries.py`.

For each channel, the following fields are added to the manifest with a lowercase channel prefix, for example `map2_mean`, `flag_std`, `ha_z_peak_index`.

- `mean`: mean intensity of the full crop
- `max`: maximum intensity of the full crop
- `sum`: sum of all voxel intensities in the crop
- `std`: standard deviation of all voxel intensities in the crop
- `z_peak_index`: z-slice index whose mean intensity is maximal
- `z_occupancy`: number of z-slices with mean intensity greater than zero

These are written in `src/map2_patch_discovery/patches.py` when building each manifest row.

The manifest also includes non-engineered metadata such as:

- sample identifiers and experimental metadata
- `patch_group`
- `x`, `y`, `z_center`
- `map2_overlap_fraction`
- `distance_to_mask_px`
- patch dimensions and pixel sizes

## 5. Engineered Features Used by the Latent Report

The latent report calls `extract_engineered_features()` in `src/map2_patch_discovery/features.py`.

For each patch:

1. the requested channel crop is loaded from the shard
2. a per-channel feature block is computed
3. MAP2-aware spatial features are added
4. cross-channel pair features are added
5. a few extra hand-built ratios and combinations are added

The final columns from `feature_df` are the candidate features for PCA and clustering, except `patch_id`.

## 6. Per-Channel Engineered Feature Block

For each requested channel, `_channel_feature_block()` computes features from:

- `max_proj = np.max(crop, axis=0)`
- `mean_proj = np.mean(crop, axis=0)`
- `bright_mask = max_proj >= bright_threshold`
- `inside` = pixels where the local 2D MAP2 mask is `True`
- `outside` = pixels where the local 2D MAP2 mask is `False`
- `center` = the middle box of the 2D patch
- `surround` = everything else in the 2D patch outside that middle box
- in the code, the `center` box starts one quarter of the way in from each edge, so `center` and `surround` together cover the full patch

Important clarification:

- `inside`/`outside` and `center`/`surround` are region masks, not pixel-by-pixel comparisons
- the features summarize groups of 2D mean-projection pixels: one group for `inside` or `center`, and one group for `outside` or `surround`

The feature names are emitted as `<channel>_<feature_name>` in lowercase, such as `flag_proj_mean` or `map2_glcm_contrast`.

### 6.1 Intensity and Contrast Features

- `proj_mean`: mean of the mean projection
- `proj_max`: max of the max projection
- `proj_std`: standard deviation of the mean projection
- `inside_mean`: mean intensity across all pixels inside the MAP2 mask region
- `outside_mean`: mean intensity across all pixels outside the MAP2 mask region
- `inside_outside_ratio`: ratio of the mean intensity of all inside-mask pixels to the mean intensity of all outside-mask pixels
- `local_contrast`: `std(mean_proj) / mean(mean_proj)`
- `dynamic_range`: `max(mean_proj) - min(mean_proj)`
- `bright_pixel_fraction`: fraction of pixels above the bright threshold
- `top_percentile_intensity`: mean of max-projection pixels in the top 1 percent
- `intensity_entropy`: entropy of a 32-bin histogram of the normalized mean projection

### 6.2 Center-Surround Features

- `center_surround_diff`: difference between the mean intensity of all center-region pixels and the mean intensity of all surround-region pixels
- `center_surround_ratio`: ratio of the mean intensity of all center-region pixels to the mean intensity of all surround-region pixels
- `center_mean`: mean intensity across all pixels in the center region
- `surround_mean`: mean intensity across all pixels in the surround region
- `local_background_estimate`: median intensity across all pixels in the surround region
- `local_snr`: `(mean(center region) - mean(surround region)) / std(surround region)`

### 6.3 Spotness, Edge, and High-Frequency Features

- `estimated_punctum_area`: number of bright pixels
- `log_response`: max absolute Laplacian-of-Gaussian response on the mean projection
- `dog_response`: max absolute Difference-of-Gaussians response using sigma 1 and 2
- `gradient_mean`: mean Sobel gradient magnitude
- `gradient_std`: standard deviation of Sobel gradient magnitude
- `high_frequency_power`: mean squared residual after subtracting a sigma-1 Gaussian blur
- `neighborhood_variance`: variance of the mean projection

### 6.4 Bright-Inside-vs-Outside Features

- `inside_bright_fraction`: fraction of inside-mask pixels that are bright
- `outside_bright_fraction`: fraction of outside-mask pixels that are bright
- `inside_bright_outside_bright_ratio`: ratio of the bright-pixel fraction inside the mask region to the bright-pixel fraction outside the mask region

### 6.5 Component Geometry Features

These come from connected components of `bright_mask`, using the largest component when needed.

These morphology features are measured on the largest bright connected object in the 2D patch, not on single pixels and not on the whole patch as one shape.

So the thing being measured here is:

- not a pixel
- not the whole patch uniformly
- specifically the largest bright connected object found in that channel's 2D projected patch

If the patch has several puncta, this block mostly summarizes the largest one, while:

- `component_count` tells you how many bright objects there were
- `largest_area` tells you how big the largest one was
- `dominant_object_fraction` tells you how much of the total bright area belonged to that largest object

- `component_count`: number of connected components
- `largest_area`: area of the largest connected component
- `largest_area_fraction`: largest component area divided by total patch area
- `eccentricity`: ellipse-like eccentricity from the component covariance eigenvalues
- `circularity`: `4 * pi * area / perimeter^2`
- `compactness`: largest component area divided by convex-hull area
- `boundary_irregularity`: `perimeter^2 / area`; larger for more ragged shapes
- `elongation`: major-axis scale divided by minor-axis scale
- `roundness`: equivalent-diameter divided by major-axis length; smaller for more stretched objects
- `radial_symmetry`: correlation between largest-component intensity and a radial weighting centered on the component
- `microstructure_density`: component count divided by patch area
- `dominant_object_fraction`: largest component area divided by total bright area

If there are no bright pixels, these return zero.

### 6.6 Texture Features

Gray-Level_co-occurnace Matrix (GLCM) measures how often pixel values occur next to eachother in an image. This captures smoothness or graininess which is not captured by brightness alone. Direction and distance between pixels dependent. GLCM features from the mean projection:

- `glcm_contrast` (variation)
- `glcm_homogeneity` (smoothness)
- `glcm_energy` (uniformity)
- `glcm_correlation`(pattern relationships)

These are computed from a 16-level quantized image over four angles. If `scikit-image` is unavailable, they default to zero.

Local Binary Pattern (LBP) is a pixel by pixel method that describes image texture by comparing each pixel to its neighbors. This captures patterns like edges, spots and textures that brightness alone cannot describe. LBP features from the mean projection:

- `lbp_entropy`
- `lbp_uniform_fraction`

If `scikit-image` is unavailable, they default to zero.

### 6.7 Multi-Scale Features

Wavelet-like features from the mean projection:

- `wavelet_energy_scale_1`
- `wavelet_energy_scale_2`
- optionally `wavelet_energy_scale_3`

If `pywt` is installed, these are derived from Haar wavelet detail energies. If not, a Gaussian-blur fallback is used, and the fallback includes `wavelet_energy_scale_3`.

### 6.8 Z-Profile Features

These are computed from the z-profile `crop.mean(axis=(1, 2))`.

- `z_peak`: index of the z-slice with maximum mean intensity
- `z_std`: standard deviation of the z-profile
- `z_width_halfmax`: number of slices at or above half of the peak value
- `z_slices_above_halfmax`: same quantity as above in the current code
- `z_center_of_mass`: intensity-weighted center of mass along z
- `z_skewness`: skewness of the z-profile
- `z_kurtosis`: kurtosis of the z-profile
- `z_peak_count`: number of detected peaks
- `z_multi_peak_score`: `max(0, peak_count - 1)`
- `z_peak_symmetry`: left/right symmetry around the main peak

### 6.9 LoG 2.5D Puncta Features

These use a hybrid 2D-detect / 3D-measure approach:

- first, multi-scale LoG detects candidate puncta on the 2D mean projection
- then, each detected punctum is measured back in the original 3D crop
- each punctum keeps one XY center and one LoG-derived radius, then uses that XY region through z to build a local z-profile and local background estimate

So these are not single-pixel measurements and not full segmentation masks. They are summaries of LoG-detected puncta measured in 3D.

The per-punctum measurements are then summarized across all accepted detections in the patch:

- `log_puncta_radius_px_*`: LoG-derived XY punctum radius
- `log_puncta_z_width_*`: number of z slices above half of the punctum's peak axial signal
- `log_puncta_anisotropy_*`: axial width divided by XY diameter
- `log_puncta_peak_signal_*`: peak value of the punctum's z-profile
- `log_puncta_bgsub_peak_*`: peak punctum signal after subtracting local annulus background
- `log_puncta_integrated_bgsub_*`: total positive background-subtracted punctum signal in the 3D local region
- `log_puncta_snr_*`: local punctum SNR from background-subtracted peak divided by local background standard deviation
- `log_puncta_sbr_*`: local punctum signal-to-background ratio

For each of those feature families, the code records:

- `*_mean`
- `*_median`
- `*_max`
- `*_cv`

And it also records:

- `log_puncta_dominance`: largest punctum integrated background-subtracted signal divided by the sum across detected puncta in the patch

## 7. MAP2-Aware Spatial Features

These come from `_map2_spatial_features()` and are always based on the MAP2 mask and MAP2 mean projection.

- `map2_mask_fraction`: fraction of the patch covered by the MAP2 mask
- `distance_to_mask_boundary_px`: signed distance carried in from the manifest
- `center_of_patch_map2_intensity`: MAP2 intensity at the patch center pixel
- `map2_local_thickness_proxy`: if the center lies inside the mask, twice the distance-transform value there

## 8. Cross-Channel Pair Features

For each of these pairs:

- `FLAG` vs `HA`
- `FLAG` vs `SHANK2`
- `HA` vs `SHANK2`
- `MAP2` vs `FLAG`
- `MAP2` vs `HA`
- `MAP2` vs `SHANK2`

the code computes:

- `<pair>_pixel_corr`: pixelwise correlation of the mean projections
- `<pair>_pixel_corr_in_mask`: same correlation restricted to the MAP2 mask
- `<pair>_bright_overlap_jaccard`: intersection over union of the bright masks
- `<pair>_bright_overlap_coef`: overlap coefficient of the bright masks
- `<pair>_manders_m1`: Manders-like overlap from channel A into B
- `<pair>_manders_m2`: Manders-like overlap from channel B into A
- `<pair>_com_offset`: Euclidean distance between intensity-weighted centers of mass
- `<pair>_compactness_ratio`: ratio of the channels' component compactness values
- `<pair>_spotness_ratio`: ratio of the channels' LoG spotness values
- `<pair>_mean_ratio`: ratio of mean intensities

These are only emitted when both channels in the pair are present in `features.channels`.

## 9. Extra Hand-Built Combination Features

After the per-channel and pairwise features, the code adds:

- `flag_to_ha_mean_ratio`
- `ha_to_flag_mean_ratio`
- `flag_ha_mean_sum`
- `flag_ha_mean_product`
- `flag_ha_mean_absdiff`
- `shank2_to_map2_mean_ratio`
- `flag_plus_ha_to_shank2_sum_ratio`

## 10. What Actually Gets Used in a Latent Report

The candidate feature set starts as every engineered feature column in `feature_df`, meaning all columns except `patch_id`.

That candidate set may then be reduced in three steps inside `src/map2_patch_discovery/latent_report.py`:

1. Feature-variance cluster filter, if `features.feature_variance_csv` and `features.feature_variance_cluster` are configured.
2. MAP2 feature policy.
   - `full`: keep all candidate features
   - `prior_only`: keep only `map2_mask_fraction`, `distance_to_mask_boundary_px`, and `map2_local_thickness_proxy` among MAP2-named features
   - `exclude_spatial`: drop the MAP2-aware spatial family while keeping other MAP2-derived features
   - `exclude_spatial_and_dendrite`: drop both the MAP2-aware spatial family and the post-hoc `posthoc_map2_*` dendrite-geometry family
   - `exclude_all_map2`: drop all MAP2-derived signal features, all MAP2-aware spatial features, and all post-hoc `posthoc_map2_*` dendrite-geometry features
   - `mask_internal_only`: keep the MAP2 mask only for internal extraction/sampling logic and drop all MAP2-derived signal features plus all mask-context analysis features such as inside/outside, center/surround, and `*_pixel_corr_in_mask`
3. Pre-clustering variance audit.
   - drops all-zero features
   - drops constant features
   - drops near-constant features

So the exact feature set used for PCA and clustering depends on:

- which channels are requested in `features.channels`
- whether feature-variance filtering is enabled
- which MAP2 policy is configured
- which features survive the variance audit for that specific dataset

## 11. Important Implementation Notes

- `MAP2` must be present in `features.channels`. The config validator now enforces this.
- Several texture features depend on optional libraries.
  - without `scikit-image`, GLCM and LBP features become zero
- without `pywt`, wavelet features fall back to blur-based multi-scale energies
- `z_width_halfmax` and `z_slices_above_halfmax` are currently identical in the code.

## 12. Quick Reading Guide

If you want the shortest mental model:

- a patch is a local multi-channel 3D crop plus a matching 2D MAP2 mask crop
- a shard is a `.npz` file holding many such patches together
- manifest summary features are simple per-channel intensity summaries
- per-channel engineered features describe intensity, texture, morphology, center-surround structure, and z-profile shape
- MAP2-aware features describe mask context
- pairwise features describe overlap and correlation across channels
- latent-report PCA and clustering use a filtered subset of those engineered features, not the raw manifest summary fields
