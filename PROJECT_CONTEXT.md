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
- **Dashboard:** Streamlit — deployed at https://streamlens.streamlit.app
- **Version control:** GitHub — https://github.com/xinyacheng716/StreamLens

---

## Current Phase Status

- [x] Phase 0 — Environment setup + dataset exploration
- [x] Phase 1 — Data cleaning + genre-level aggregation
- [x] Phase 2 — Bias quantification + visualization
- [x] Phase 3 — ML layer (Logistic Regression + Random Forest)
- [x] Phase 4 — Dashboard + Deployment
- [ ] README ← CURRENT

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

### Section 3 — Bias Score Analysis
- Pearson r = **−0.266** (updated from −0.365 in small dataset)
- Moderate negative correlation between genre quality and exposure
- Most underserved genres: Film-Noir (+0.94), Documentary (+0.78), War (+0.67)
- Most overpromoted genres: Comedy (−0.78), Action (−0.67)
- Bias Score formula: `Quality Percentile (avg_rating) − Exposure Percentile (rating_count)`
- With sample sizes ranging from 318,917 to 14,377,237, significance testing loses discriminating power. Effect size (Bias Score) is used instead.

### Section 4 — ML Results

| Model | Accuracy | Class 1 Recall | Class 1 F1 |
|-------|----------|----------------|------------|
| Logistic Regression | 0.749 | 0.67 | 0.69 |
| **Random Forest** | **0.861** | **0.88** | **0.84** |

- `avg_rating` feature importance: **88%**
- Genre features combined: ~12%
- Structural reason: high avg_rating → high rating_pct → high bias_score → is_underserved = 1
- This is not data leakage, but reflects definitional structure

### Film-level underserved list
- `data/processed/film_underserved.csv` contains all films with rating_count ≥ 30
- 1,662 underserved films (is_underserved = 1, rating_count ≥ 30)

---

## Section 6 — Business Recommendation (Finalised)

### Three interventions with justified thresholds:

**Intervention 1 — Film-Level Algorithmic Trigger**
- Flag films with `avg_rating >= 3.8` AND `rating_count <= 60`
- Why 3.8: mean avg_rating of underserved films is 3.77. 3.8 targets above-average quality within the underserved population.
- Why 60: median rating_count of underserved films is 58.5. Films below median represent the lower half by exposure, where suppression signal is strongest.
- Alt threshold: rating_count <= 120 (75th percentile), expands candidate pool to ~1,247 films. Right threshold depends on platform editorial capacity.

**Intervention 2 — Genre Fairness Audit Cadence**
- Run quarterly Bias Score analysis. Flag genres with `bias_score > 0.3`.
- Why 0.3: natural gap between Animation (+0.33) and Mystery (+0.28). Not arbitrary — reflects where data clusters into two groups.
- Conservative alt: threshold 0.5, flags only top 3 (Film-Noir, Documentary, War). Suitable for platforms with limited editorial capacity.

**Intervention 3 — Human-in-the-Loop Editorial Layer**
- Algorithm surfaces candidates; editorial team makes final decisions before any boost goes live.
- Rationale: fixing one algorithmic bias can introduce another. rating_count cannot fully separate algorithmic suppression from small audience size — human judgment bridges this gap.

---

## Dashboard Status

- [x] Section 1: The Problem
- [x] Section 2: The Data
- [x] Section 3: Bias Evidence (Bias Score bar chart, correlation scatter, quadrant chart, effect size explanation, t-test justification)
- [x] Section 4: ML Insights (model comparison, feature importance, interactive film explorer)
- [x] Section 5: Limitations & Future Work (4 limitations with impact/severity color coding)
- [x] Section 6: Business Recommendation (3 interventions with justified thresholds, distribution charts for Int 1, dot plot for Int 2)
- [x] Deployed: https://streamlens.streamlit.app

**File locations:**
- Dashboard: `streamlit_app/app.py`
- Run command: `streamlit run streamlit_app/app.py`
- Figures: `outputs/figures/`
  - phase2_bias_score.png
  - phase2_quadrant.png
  - phase2_correlation.png
  - phase3_feature_importance.png

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
| 2026-04 | Intervention 1 rating_count threshold = 60 | 60 vs 100 vs 120 | 60 = median of underserved films, suppression signal strongest below median |
| 2026-04 | Intervention 2 bias_score threshold = 0.3 | Various | Natural gap between Animation (+0.33) and Mystery (+0.28) |
| 2026-04 | Effect size over significance testing | t-test vs effect size | Large n makes all differences statistically significant; effect size more meaningful |

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
- Do not use -- in writing (too AI-sounding)
- Do not use emoji unless functionally justified
- Do not refer to sections as "Phase X" in dashboard — use formal section names
- Do not make mistakes on section numbers or formal names