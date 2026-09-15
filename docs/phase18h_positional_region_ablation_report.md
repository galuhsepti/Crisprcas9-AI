# Phase 18H - Locked Positional-Region Ablation

Status: **COMPLETED**

This experiment measures predictive contribution and associated positional information. It makes no biological-causality or mechanistic claim.

## Reused Full Controls

| Domain | Spearman | RMSE | Pearson | R2 | MAE |
|---|---:|---:|---:|---:|---:|
| deepspcas9 | 0.70804937 | 0.15501222 | 0.71457965 | 0.50746017 | 0.12277021 |
| crispron_xiang_luo | 0.75578064 | 15.51928110 | 0.75077910 | 0.55667142 | 12.15414820 |

## Regional Results

| Domain | Region | Spearman | RMSE | Pearson | R2 | MAE | Delta Spearman | Delta RMSE | Relative RMSE degradation | Adjusted Spearman CI | Adjusted RMSE CI | Class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| deepspcas9 | REGION_1_PAM_DISTAL | 0.69623001 | 0.15748596 | 0.70317541 | 0.49161446 | 0.12585329 | -0.01181935 | 0.00247374 | 0.01595837 | [-0.0335046006304294, 0.008950500234231822] | [-0.001275602856043695, 0.00641638101759504] | LITTLE_UNIQUE_REGIONAL_CONTRIBUTION |
| deepspcas9 | REGION_2_MID_DISTAL | 0.70325865 | 0.15579194 | 0.71070885 | 0.50249267 | 0.12411457 | -0.00479071 | 0.00077972 | 0.00503008 | [-0.020995638506208517, 0.01106229491595835] | [-0.0020295377656588875, 0.003604638786213124] | LITTLE_UNIQUE_REGIONAL_CONTRIBUTION |
| deepspcas9 | REGION_3_MID_PROXIMAL | 0.67573857 | 0.16099392 | 0.68595925 | 0.46871389 | 0.12841351 | -0.03231080 | 0.00598170 | 0.03858860 | [-0.054553379308721645, -0.009690424560923139] | [0.0021606418399797447, 0.00986664164241936] | STRONG_REGIONAL_CONTRIBUTOR |
| deepspcas9 | REGION_4_PAM_PROXIMAL | 0.52402303 | 0.18688596 | 0.53310503 | 0.28408271 | 0.15197611 | -0.18402634 | 0.03187374 | 0.20562083 | [-0.23055844212146456, -0.136687389343985] | [0.02465606426753948, 0.03886273083599962] | STRONG_REGIONAL_CONTRIBUTOR |
| crispron_xiang_luo | REGION_1_PAM_DISTAL | 0.74061721 | 15.86491023 | 0.73642179 | 0.53670483 | 12.51533126 | -0.01516343 | 0.34562913 | 0.02227095 | [-0.0329185825134452, 0.0020190890942554853] | [-0.0409822314887899, 0.7234808453698652] | LITTLE_UNIQUE_REGIONAL_CONTRIBUTION |
| crispron_xiang_luo | REGION_2_MID_DISTAL | 0.74622636 | 15.75853407 | 0.74049702 | 0.54289690 | 12.35240848 | -0.00955428 | 0.23925297 | 0.01541650 | [-0.022600280219783605, 0.003310961075706497] | [-0.04470200649281122, 0.530465683186273] | LITTLE_UNIQUE_REGIONAL_CONTRIBUTION |
| crispron_xiang_luo | REGION_3_MID_PROXIMAL | 0.73987435 | 15.96540879 | 0.73118341 | 0.53081662 | 12.53196816 | -0.01590629 | 0.44612770 | 0.02874667 | [-0.03391118606603233, 0.0009664863899243359] | [0.05591638972531797, 0.8438388763337002] | MODERATE_REGIONAL_CONTRIBUTOR |
| crispron_xiang_luo | REGION_4_PAM_PROXIMAL | 0.56823100 | 19.19002368 | 0.57079087 | 0.32215007 | 15.73317229 | -0.18754964 | 3.67074258 | 0.23652788 | [-0.2356874866756421, -0.14272917659537532] | [2.86172475962209, 4.438155531522327] | STRONG_REGIONAL_CONTRIBUTOR |

## Preregistered Questions

- deepspcas9: strongest descriptive region is REGION_4_PAM_PROXIMAL; primary delta-Spearman then delta-RMSE order is REGION_4_PAM_PROXIMAL, REGION_3_MID_PROXIMAL, REGION_1_PAM_DISTAL, REGION_2_MID_DISTAL.
- deepspcas9: PAM-proximal versus PAM-distal descriptive comparison is PAM-proximal removal is descriptively stronger.
- deepspcas9: positional information classification is BROADLY_DISTRIBUTED.
- crispron_xiang_luo: strongest descriptive region is REGION_4_PAM_PROXIMAL; primary delta-Spearman then delta-RMSE order is REGION_4_PAM_PROXIMAL, REGION_3_MID_PROXIMAL, REGION_1_PAM_DISTAL, REGION_2_MID_DISTAL.
- crispron_xiang_luo: PAM-proximal versus PAM-distal descriptive comparison is PAM-proximal removal is descriptively stronger.
- crispron_xiang_luo: positional information classification is BROADLY_DISTRIBUTED.
- REGION_1_PAM_DISTAL: cross-domain class is WEAK_IN_BOTH.
- REGION_2_MID_DISTAL: cross-domain class is WEAK_IN_BOTH.
- REGION_3_MID_PROXIMAL: cross-domain class is CROSS_DOMAIN_CONSISTENT.
- REGION_4_PAM_PROXIMAL: cross-domain class is CROSS_DOMAIN_CONSISTENT.
- deepspcas9: REGION_4 confirmatory result is STRONG_REGIONAL_CONTRIBUTOR; contributor support is True.
- deepspcas9: REGION_4-versus-broad localization answer is NOT_CONCENTRATED_IN_REGION_4_BROAD.
- crispron_xiang_luo: REGION_4 confirmatory result is STRONG_REGIONAL_CONTRIBUTOR; contributor support is True.
- crispron_xiang_luo: REGION_4-versus-broad localization answer is NOT_CONCENTRATED_IN_REGION_4_BROAD.

## Whole-Family Context

- deepspcas9: reused Phase 18F drop-all class is ESSENTIAL_OR_STRONG_CONTRIBUTOR with delta Spearman -0.24985138 and delta RMSE 0.03889015. Contributor evidence appears across multiple regions within the whole-family signal. Whether regional effects account for most of that signal is not identifiable from these conditional, non-additive ablations, and the effects are not summed.
- crispron_xiang_luo: reused Phase 18F drop-all class is ESSENTIAL_OR_STRONG_CONTRIBUTOR with delta Spearman -0.29311055 and delta RMSE 5.06043853. Contributor evidence appears across multiple regions within the whole-family signal. Whether regional effects account for most of that signal is not identifiable from these conditional, non-additive ablations, and the effects are not summed.

Regional effects are qualitative, conditional, and non-additive; they are not summed to reconstruct the whole-family effect.

## Execution And Inference Boundaries

- Exactly 8 new fits and 8 new prediction bundles were produced; 4 Phase 18F references were reused and none was refit.
- XGBoost used seed 42 with no tuning, early stopping, sample weights, validation fit, or validation refit.
- Paired bootstrap used 10,000 replicates per domain, sorted unique forward spacer20 groups, all observations in sampled groups, one PCG64(42) stream shared by four comparisons, paired draws, and linear percentile quantiles.
- Nominal 95% intervals cover all three effects. Bonferroni 99.6875% intervals at quantiles (0.0015625, 0.9984375) cover delta Spearman and delta RMSE across 16 confirmatory intervals.
- Bridge, quarantine, and validation were excluded. Moreno-Mateos was not accessed. Domains retain native label scales.
- Cross-region rankings and PAM-proximal versus PAM-distal comparisons are descriptive, not superiority tests.
- Recovery additional fits: 0; recovery additional predictions: 0.
