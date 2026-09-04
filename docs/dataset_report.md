# Dataset Report: CRISPR-Cas9 sgRNA Activity Prediction

## 1. Dataset Selection

### Selected Dataset: DeepSpCas9 (Kim et al. 2019)

**Paper:** Kim, H.K. et al. "SpCas9 activity prediction by DeepSpCas9, a deep learning-based model with high generalization performance." Science Advances 5, eaax9249 (2019).

**Source:** Benchmarking-CRISPR-on-tools repository

**URL:** https://github.com/HaoDK12/Benchmarking-CRISPR-on-tools

**File:** `Training/DeepSpCas9 (Library).csv`

**Description:**
- 12,832 unique sgRNAs targeting human genes
- Experimentally validated SpCas9 activity (modification frequency)
- 30-mer context sequences (standard format for CRISPR prediction)
- Activity measured as indel frequency from deep sequencing

**License:** Available through public GitHub repository

---

## 2. Dataset Characteristics

### 2.1 Sequence Format

| Property | Value |
|----------|-------|
| Total samples | 12,832 |
| Unique sequences | 12,832 |
| Sequence length (30mer) | 30 bp |
| Sequence length (23mer) | 23 bp |
| PAM type | NGG (SpCas9) |
| Alphabet | A, C, G, T only |

### 2.2 Sequence Structure (30-mer)

```
Position:  0-3    4-23     24-26    27-29
           [context][guide][PAM][context]
           4 bp    20 bp   3 bp   3 bp
```

- **Positions 0-3:** 5' flanking context (4 bp)
- **Positions 4-23:** Guide RNA target sequence (20 bp)
- **Positions 24-26:** PAM sequence (NGG)
- **Positions 27-29:** 3' flanking context (3 bp)

### 2.3 Target Variable (Activity Score)

| Statistic | Value |
|-----------|-------|
| Name | modFreq (modification frequency) |
| Type | Continuous (regression) |
| Range | 0.0 to 0.951 |
| Mean | 0.403 |
| Median | 0.411 |
| Std | 0.230 |
| Missing values | 0 |

### 2.4 Nucleotide Composition

| Nucleotide | Frequency |
|------------|-----------|
| A | 22.1% |
| C | 25.5% |
| G | 29.9% |
| T | 22.4% |

### 2.5 Guide Sequence Properties

| Property | Value |
|----------|-------|
| Mean GC content | 53.6% |
| GC content std | 15.0% |
| PAM validation | 100% NGG |

---

## 3. Independent Test Set: Moreno-Mateos (2015)

**Paper:** Moreno-Mateos, M.A. et al. "CRISPRscan: designing highly efficient sgRNAs for CRISPR-Cas9 targeting in vivo." Nature Methods 12, 982-988 (2015).

**Source:** Same repository

**File:** `Training/Moreno-Mateos.csv`

**Description:**
- 1,020 unique sgRNAs
- Independent from DeepSpCas9 (no sequence overlap)
- Can be used as external test set

**Properties:**
- Mean activity: 0.499
- Std activity: 0.290
- Range: 0.0 to 1.0

---

## 4. Data Quality Assessment

### 4.1 Positive Quality Indicators

✅ **No duplicate sequences** - All 12,832 sequences are unique

✅ **No missing values** - All sequences and activity scores are present

✅ **Valid nucleotides** - Only A, C, G, T (no invalid characters)

✅ **Consistent PAM** - 100% of sequences have valid NGG PAM

✅ **Proper sequence length** - All 30-mers and 23-mers are correct length

✅ **Experimental validation** - Activity scores from deep sequencing experiments

✅ **Standard format** - 30-mer context format used by major prediction tools

### 4.2 Potential Considerations

⚠️ **Single cell line** - Data from specific experimental conditions

⚠️ **Activity range** - Scores don't reach 1.0 (max 0.951)

⚠️ **No gene annotations** - Gene information not included in this dataset version

---

## 5. Data Leakage Analysis

### 5.1 Sequence-Level Leakage

- **Duplicate sequences:** 0 (no leakage)
- **Reverse complement duplicates:** Need to check (but 30-mer format makes this unlikely)

### 5.2 Gene-Level Leakage

- **Gene information:** Not available in current dataset
- **Mitigation:** Random train/val/test split acceptable

### 5.3 Recommended Split Strategy

| Split | Ratio | Samples (approx) |
|-------|-------|-------------------|
| Train | 70% | 8,982 |
| Validation | 15% | 1,925 |
| Test | 15% | 1,925 |

**Strategy:** Random split with fixed seed (reproducible)

**Alternative:** Use Moreno-Mateos as independent test set

---

## 6. Preprocessing Requirements

### 6.1 Sequence Processing

1. **Extract guide sequence:** Positions 4-23 from 30-mer
2. **Validate PAM:** Positions 24-26 must be NGG
3. **Normalize sequences:** Convert to uppercase (already done)

### 6.2 Activity Score Processing

1. **Raw scores:** Keep as modFreq (0-1 range)
2. **Optional normalization:** Min-max or z-score if needed
3. **Log-transform:** Consider for skewed distribution

### 6.3 Feature Engineering

For traditional ML models (RF, XGBoost):
- GC content (guide sequence)
- Nucleotide composition (A, C, G, T frequencies)
- k-mer features (2-mer, 3-mer)
- Position-specific nucleotide features

For CNN:
- One-hot encoding of 30-mer or 20-mer guide

---

## 7. Dataset Files

### 7.1 Primary Dataset

**File:** `data/raw/DeepSpCas9.csv`
**Source:** `data/external/Benchmarking-CRISPR-on-tools/Training/DeepSpCas9 (Library).csv`
**Size:** 12,832 samples

### 7.2 Independent Test Set

**File:** `data/raw/Moreno-Mateos.csv`
**Source:** `data/external/Benchmarking-CRISPR-on-tools/Training/Moreno-Mateos.csv`
**Size:** 1,020 samples

---

## 8. Citation Requirements

If using this dataset, cite:

1. **Kim et al. 2019** (DeepSpCas9 data):
   Kim, H.K. et al. "SpCas9 activity prediction by DeepSpCas9, a deep learning-based model with high generalization performance." Science Advances 5, eaax9249 (2019).

2. **Benchmarking repository:**
   Yuan, H. et al. "A comparative analysis of CRISPR-Cas9 editing efficiency prediction tools." (2024)

---

## 9. Report Metadata

- **Report Created:** 2026-09-04
- **Last Updated:** 2026-09-04
- **Author:** AI Assistant
- **Status:** Dataset selected and validated

---

## 10. Next Steps

1. [x] Dataset identified and downloaded
2. [x] Data quality validated
3. [ ] Create preprocessing pipeline
4. [ ] Split data into train/val/test
5. [ ] Implement feature engineering
6. [ ] Build and train models
