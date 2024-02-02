## ENCODE Masks

Masks are boolean 1D arrays which can be used to select a subset of tasks, i.e. eCLIP experiments, on the fly. This is handy, as we don't need to generate seperate datasets for different subset of experiemnts. Mask arrays have the same size as the number of tasks in the reference dataset, e.g. 223 eCLIP experiments in the ENCODE dataset. 

### Example: Subset eCLIP experiments in the HepG2 cell line

Say we want to focus model training to HepG2 experiments. Rather then generating a new TFDS dataset, we define a mask of the form `[True, False, ..., True]` where the i'th position is true if the i'th experiment (in alphabetical order `{Protein}_{Cell}`) is corresponding to an eCLIP experiment in the HepG2 cell line. We then save this boolean mask as a pytorch (`.pt`) file `masks/mask.ENCODE.HepG2.pt`. 

At runtime, we use the `MaskedTFDSDataset` dataset and provide our mask file via `mask_filepaths=['masks/mask.ENCODE.HepG2.pt']`. During sample loading, only rows which were flagged as `True` in our mask array will be selected from the full eCLIP count matrices. 

Note: `mask_filepaths` accepts multiple mask files. This allows us to combine different masks for even finer selection options, e.g. we may select eCLIP data of splice factors in K562 by providing masks for splice factors and the K562 cell line. 