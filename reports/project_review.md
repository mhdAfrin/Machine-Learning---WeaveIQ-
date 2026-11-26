# WeaveIQ project review and continuation

## Verified stopping point

The uploaded project ends after tuning, selection, refitting on development records, and final test evaluation in notebook 04. Feature importance is not present in the uploaded code or reports. There is no prediction demo, final README, or submission report in this archive.

## Checks completed

44 independent data/report consistency checks passed. See project_review_checks.json for individual results.

- Raw CSV: 121,148 records, 18 columns, 1,009 distinct IDs.
- Original Excel and CSV align; original target blanks reproduce the CSV zeros.
- Eligibility (original target recorded, original production positive) reconstructs the saved 19,969 candidate rows exactly.
- Candidates cover 946 IDs. Their target values and production amounts match the original Excel.
- Development: 16,139 records, 756 IDs. Test: 3,830 records, 190 IDs.
- Development and test are disjoint by ID and source row, and together cover the candidates.
- The split manifest agrees with the saved partition.
- All five saved folds are nonempty, with each development ID in exactly one fold.
- Original CSV ID, Construction, and three planning fields agree with the forward-filled original Excel fields.
- Every stored tuning search has six candidates; ten searches represent 300 five-fold fits.
- Initial comparison contains 55 fold results.
- The frozen model family and feature version agree with the development-CV winner.
- Test predictions align with test records and targets. Baseline predictions equal the development target median (17).
- Recalculated MAE, RMSE, R2, mean-order MAE, and negative prediction counts reproduce both saved metric rows to numerical precision.
- Residuals equal actual minus predicted. The high-amount threshold of 297 comes from the development 95th percentile.
- All Python code cells in the four uploaded notebooks parse, and no saved error outputs were found.

## Frozen result

Selected configuration: Histogram Gradient Boosting, engineered features, absolute-error loss, learning rate 0.03, 200 iterations, 15 maximum leaves, minimum 100 records per leaf, L2 regularization 1.0, random state 42, automatic early stopping disabled.

| Metric | Selected model | Median baseline |
|---|---:|---:|
| Test MAE | 44.648047453845365 | 46.13159268929504 |
| Test RMSE | 236.4862138611679 | 242.59945391357437 |
| Test R2 | 0.03391476566924112 | -0.01667799947967752 |
| Test mean-order MAE | 44.241402530325054 | 44.49381337275989 |
| Negative predictions | 67 | 0 |

The gain is modest (about 3.22% in row-weighted MAE and 0.57% in mean-order MAE). Strong underprediction of large recorded amounts and poor zero-outcome performance remain. R2 is not an accuracy percentage. The high-amount test subgroup consists of 103 records from only eight IDs.

## Research limitations to retain

- The target is the recorded combined rejection-and-cut-piece amount, not a rejection percentage. A last-row final completed-order interpretation has not been established.
- Repeated reporting records were retained. Row-level metrics weight orders with more records more heavily; the saved additional mean-order MAE helps expose this.
- The 31 IDs without original recorded targets and subsequent eligibility exclusions were documented. They must not be labelled verified zero-rejection orders.
- Positive production is an eligibility rule; its use for selection does not make it a predictor. Same-record production, actual shrinkage, identifiers and tracking fields are excluded from model inputs.
- The bounded search tests six configurations for each model/feature version, not an exhaustive parameter optimization.
- Correlated planning quantities, encoded construction, and associated derived inputs make individual permutation rankings conditional and potentially unstable.
- Development folds were used for model selection. Interpretation on those folds is exploratory, not another independent generalization estimate.
- Preserve the reported raw test predictions. Changing clipping, settings, features or exclusions after seeing the test changes the evaluated procedure.

## Reproducibility and Git state

The archive's last commit and recorded origin/main are 5bac335. Notebook 04 has substantive later edits; tuning reports, frozen configuration and final-test reports/plot are not yet committed. Other tracked text differences in the extracted copy are line endings only.

The original environment records Python 3.14.0, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4 and joblib 1.6.0. The inspection environment has different versions (including scikit-learn 1.8.0). All saved metric checks above use the stored predictions and do not require unpickling or retraining the original model.

The new standalone interpretation code was executed on all five development folds in a separate portability run. Four refit validation scores differed from the original tuning scores, so those portability importance values are NOT included as authoritative project results. Run the continuation in the original project environment. Its metadata records versions and its validation-check CSV compares refit scores against stored tuning scores.

The new notebook passed notebook v4.5 schema validation. The new Python modules and notebook code passed syntax checks. Byte comparisons confirm the original saved model, frozen configuration, final test outputs and split files were unchanged during inspection.

## Next step

Run the feature-importance continuation, review its output, commit tuning/evaluation/interpretation, then build the prediction demo and complete submission documentation. No further parameter search or final-test reevaluation is required for this interpretation step.
