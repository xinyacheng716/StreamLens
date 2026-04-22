# StreamLens — Project Context

---

## Project Summary

- **Project name:** StreamLens — Genre Fairness Audit for Streaming Recommendations
- **One-line description:** Detects systematic popularity bias in streaming recommendation systems by analyzing the gap between genre quality (ratings) and genre exposure (rating volume), and proposes data-driven curation strategies.
- **Core problem being solved:** Streaming platforms like Netflix tend to over-recommend popular content, systematically under-exposing high-quality niche genres. This leads to user frustration, poor content discovery, and potential churn — especially among users with non-mainstream taste.
- **Personal motivation:** As a Foreign Literature student and arthouse film viewer, I experienced this problem firsthand. I built this project to validate whether my personal experience reflects a systemic pattern in the data.
- **Target roles:** BA / DA / AI PM at US tech and entertainment companies.

---

## Technical Stack

- **Language:** Python
- **Data:** MovieLens ml-latest (full dataset, ~33.8M ratings, 86,537 movies)
- **Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib / Seaborn
- **ML:** scikit-learn (Logistic Regression + Random Forest)
- **Dashboard:** Streamlit (localhost running, not yet deployed)
- **Version control:** GitHub — https://github.com/xinyacheng716/StreamLens

---

## Current Phase Status

- [x] Phase 0 — Environment setup + dataset exploration
- [x] Phase 1 — Data cleaning + genre-level aggregation
- [x] Phase 2 — Bias quantification + visualization
- [x] Phase 3 — ML layer (Logistic Regression + Random Forest)
- [ ] Phase 4 — Dashboard + Write-up + Deployment + README ← CURRENT

---

## Dataset

**Switched from ml-latest-small to ml-latest (full dataset) during Phase 4.**

Reason: ml-latest-small had rating_count median = 1 for underserved films,
making avg_rating statistically unreliable (CLT requires n ≥ 30).
ml-latest resolves this — after applying rating_count ≥ 30 threshold,
1,662 statistically valid underserved films remain.

| Metric | ml-latest-small | ml-latest (current) |
|--------|----------------|---------------------|
| Total ratings | 100,836 | 33,832,162 |
| Movies | 9,742 | 86,537 |
| Users | 610 | 330,975 |
| Genres | 18 | 18 |

---

## Key Findings (Confirmed with ml-latest)

### Phase 2 — Bias Score Analysis
- Pearson r = **−0.266** (updated from −0.365 in small dataset)
- Moderate negative correlation between genre quality and exposure
- Most underserved genres: Film-Noir (+0.94), Documentary (+0.78), War (+0.67)
- Most overpromoted genres: Comedy (−0.78), Action (−0.67)
- Bias Score formula: `Quality Percentile (avg_rating) − Exposure Percentile (rating_count)`

### Phase 3 — ML Results (updated)

| Model | Accuracy | Class 1 Recall | Class 1 F1 |
|-------|----------|----------------|------------|
| Logistic Regression | 0.749 | 0.67 | 0.69 |
| **Random Forest** | **0.861** | **0.88** | **0.84** |

- `avg_rating` feature importance: **88%** (up from 77% in small dataset)
- Genre features combined: ~12%
- Structural reason: high avg_rating → high rating_pct → high bias_score → is_underserved = 1
- This is not data leakage, but reflects definitional structure

### Film-level underserved list
- `data/processed/film_underserved.csv` contains all films with rating_count ≥ 30
- 1,662 underserved films (is_underserved = 1, rating_count ≥ 30)
- Used in Dashboard Section 4 Interactive Film Explorer

---

## Dashboard Status (app.py)

**Completed sections:**
- [x] Section 1: The Problem
- [x] Section 2: The Data
- [x] Section 3: Bias Evidence (Bias Score bar chart, Correlation scatter, Quadrant chart)
- [x] Section 4: ML Insights (Model comparison, Feature importance, Interactive Film Explorer)
- [ ] Section 5: Limitations & Future Work ← NEXT
- [ ] Section 6: Business Recommendation

**File locations:**
- Dashboard: `streamlit_app/app.py`
- Run command: `streamlit run streamlit_app/app.py`
- Figures: `outputs/figures/`
  - phase2_bias_score.png
  - phase2_quadrant.png
  - phase2_correlation.png
  - phase3_feature_importance.png

---

## Section 5: Limitations & Future Work (NOT YET WRITTEN)

### Confirmed limitations to include:

**Limitation 1 — Exposure Proxy**
`rating_count` mixes three sources: algorithmic recommendations,
organic audience search, and historical content volume.
r = −0.266 and Bias Scores cannot be attributed solely to algorithmic bias.
Film-Noir's +0.94 gap is large enough that audience size alone unlikely
explains it, but the alternative explanation cannot be fully ruled out.

**Limitation 2 — avg_rating Structural Dominance**
avg_rating accounts for 88% of RF predictive power partly because of
its direct mathematical link to is_underserved definition, not purely
external causal factors.

**Limitation 3 — Class Imbalance**
Overpromoted: 9,038 test samples vs Underserved: 6,267 test samples.
May cause model to identify overpromoted films slightly better (F1: 0.88 vs 0.84).
Future fix: oversampling or class_weight parameter.

**Limitation 4 — Dataset Recency**
MovieLens ml-latest ratings have a cutoff date, not reflecting
recent streaming platform algorithm changes.

### Future Work to include:
Replace rating_count with real platform data:
- **Impression count** → pure algorithmic exposure
- **Click-through rate (CTR)** → separates "algorithm didn't push" from "audience didn't want"
- **Completion rate** → more objective quality signal than avg_rating (no selection bias)

---

## Section 6: Business Recommendation (NOT YET WRITTEN)

### Confirmed narrative direction:
Phase 2 found genre-level bias → Phase 3 found avg_rating is stronger
signal than genre at film level → Business recommendation should reflect both:

1. Genre-based curation is insufficient alone
2. Stronger intervention: target high avg_rating + low rating_count films directly
3. Genre used as secondary signal to prioritise niche content

### Three recommended interventions (to be written as cards):
1. **Direct film-level intervention** — use avg_rating ≥ threshold + rating_count ≤ threshold
   as algorithmic trigger for boosted recommendation
2. **Genre fairness audit cadence** — periodic review of genre-level Bias Scores
   to catch drift
3. **Human-in-the-loop editorial layer** — algorithm finds candidates,
   editors curate final list to avoid pure algorithmic blind spots

---

## Design Decisions Log

| Date | Decision | Options Considered | Reason |
|------|----------|--------------------|--------|
| 2026-04 | Use MovieLens dataset | Kaggle Netflix vs MovieLens | MovieLens has explicit rating + volume data |
| 2026-04 | Genre-level analysis | Film-level vs genre-level | Genre-level avoids selection bias of individual niche films |
| 2026-04 | Streamlit for dashboard | React + FastAPI vs Streamlit | Faster to build, sufficient for BA/PM portfolio |
| 2026-04 | Explode multi-genre rows | Explode vs Fractional vs Primary genre only | Preserves relative exposure differences consistently |
| 2026-04 | Switch to ml-latest | Keep small vs switch full | Small dataset rating_count median = 1, statistically unreliable |
| 2026-04 | rating_count ≥ 30 threshold | Various thresholds | CLT requires n ≥ 30 for avg_rating to be statistically meaningful |

---

## Things I Do NOT Want

- Do not build everything for me without explanation
- Do not use advanced libraries or patterns I haven't learned yet without flagging
- Do not skip the "why" when making technical suggestions
- Do not assume I understand something just because I haven't asked about it
- Do not let me keep second-guessing my own correct instincts — redirect me to commit
- Always look at actual data before making recommendations
- Always explain ALL new functions, variables, and background knowledge before showing code
- Always explain concept/logic before showing code