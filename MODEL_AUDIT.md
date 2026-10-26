# Model architecture audit

All tensor shapes below use `(batch, feature maps, electrodes, time)`. The actual input is `(B,1,2,750)`. Dataset-specific electrode count, duration and class count necessarily differ from the original multi-channel studies. These implementations do not assert numerical equivalence with Keras or reproduce published accuracy.

## Source hierarchy

1. The original paper establishes the model design.
2. Author-maintained code resolves implementation details where available.
3. Any remaining choice is stated below, rather than described as an exact paper match.

Sources were read at the URLs below; the changing upstream branches are not claimed to be pinned releases. The program fingerprints the delivered local source files for each new experiment.

## EEGNet

Source: [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c), [authors' implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py), function `EEGNet`.

Defaults follow EEGNet-8,2: eight `(1,64)` temporal filters, spatial `(C,1)` depthwise filtering with multiplier two, sixteen-feature separable `(1,16)` temporal plus `(1,1)` pointwise convolution, pooling 4 then 8, ELU, and ordinary dropout 0.5. Spatial and classifier max-norm limits are 1 and 0.25. BatchNorm epsilon is 0.001; PyTorch momentum 0.01 corresponds to Keras retaining 0.99 of its previous running statistic. Glorot initialization and channels-last flatten order are used.


## DeepConvNet

Sources: [Schirrmeister et al., 2017](https://doi.org/10.1002/hbm.23730), [paper description](https://arxiv.org/html/1703.05051v5), [author-maintained Deep4Net](https://github.com/braindecode/braindecode/blob/master/braindecode/models/deep4.py), [ARL reference](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).

The `paper` profile uses temporal kernels of length 10, spatial `(C,1)` filtering, feature counts 25/50/100/200, max pooling of width/stride 3, ELU and dropout 0.5. The affine final classifier is equivalent to a full-width final convolution for one prediction per trial. First temporal convolution has a bias; spatial and subsequent convolutions preceding BatchNorm omit bias. BatchNorm uses epsilon 0.00001, momentum 0.1. Max-norm limits 2 for convolutional filters and 0.5 for the classifier follow the cited reference constraint convention.


## ShallowConvNet

Sources: the same Schirrmeister paper; [author-maintained ShallowFBCSPNet](https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py) and ARL reference linked above.

The `paper` profile uses 40 temporal filters of length 25, 40 spatial filters spanning C, BatchNorm, square, average pooling width 75/stride 15, safe logarithm, dropout 0.5 and a classifier. The temporal bias is enabled and spatial bias is disabled before BatchNorm. Max-norm conventions follow the reference implementation. Safe log clamps to 0.000001 for numerical stability.


## SCCNet

Source: [Wei et al., 2019, paper and Figure 1](https://www.merl.com/publications/docs/TR2018-210.pdf).

The implementation uses the paper's heuristic Nu=22, Nt=1, twenty spatiotemporal filters, temporal width 12, square, dropout 0.5, pooling width 62/stride 12 and log. Both convolutions receive an L2 penalty of 0.0001. Replacing a permuted `(Nu,12)` convolution by `Conv2d(Nu,20,(1,12))` is algebraically equivalent feature mixing, not a depthwise convolution. The original paper contains a discrepancy between a textual filter-count sentence and the diagram; this implementation follows Figure 1's twenty filters.

## MBEEGSE

Sources: [Altuwaijri et al., 2022, publisher table](https://www.mdpi.com/2075-4418/12/4/995), [coauthor's reference implementation](https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py), functions `MBEEG_SENet` and `EEGNet`; [SE reference](https://github.com/Altaheri/EEG-ATCNet/blob/main/attention_models.py).

The publisher table confirms branch temporal filters 4/8/16, kernels 16/32/64, dropout 0/0.1/0.2 and SE ratios 4/4/2. Remaining details follow the coauthor code: D=2, F2=2F1 per branch, separable kernel 16, pooling 8 then 8, SE biased dense layers with He-normal initialization, concatenation of flattened SE features, then one classifier with max-norm 0.25. The spatial constraint follows that code's actual default axis on its time-first kernel, not an assumed spatial-vector norm.


## Shared adaptations and experiment variants

- All models output logits for PyTorch cross-entropy. `predict_proba` supplies softmax; call `eval()` first. This preserves the classification interpretation while avoiding a second softmax in the loss.
- Each classifier has two outputs. Input is not resampled, and literal sample widths retain different physical durations from studies recorded at other sampling rates.
- PyTorch BatchNorm running variance, dropout masks, random draws and numerical kernels can differ from TensorFlow/Keras even with matching hyperparameters. No cross-framework weight-transfer equivalence test was performed.
- Adam and a fixed 3000-epoch search are this project's common experiment design. They are not a claim of every paper's optimizer, preprocessing, validation, subject scheme or training schedule.
- Grid uses native nonlinearities/dropout; ablations deliberately vary them. SCCNet/ShallowConvNet square replacement is an architecture change. MBEEGSE common-dropout ablation differs from its native branch-specific probabilities.
- Constraints run **after** each optimizer update. SCCNet's explicit L2 penalty is separate from optional optimizer weight decay.
- Train-fitted missing-value imputation replaces the older split-specific means. This preprocessing change must be included when describing differences from old scores.

