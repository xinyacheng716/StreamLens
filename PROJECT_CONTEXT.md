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
- [x] Phase 4 — Dashboard (all 6 sections complete)
- [x] Deployment — https://streamlens.streamlit.app
- [x] README — complete, pushed to GitHub root
- [ ] Dashboard polish — CURRENT (4 known issues to fix, see below)

---

## Dataset

**Switched from ml-latest-small to ml-latest (full dataset) during Phase 4.**

Reason: ml-latest-small had rating_count median = 1 for underserved films,
making avg_rating statistically unreliable (CLT requires n >= 30).
ml-latest resolves this — after applying rating_count >= 30 threshold,
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
- Pearson r = **-0.266** (updated from -0.365 in small dataset)
- Moderate negative correlation between genre quality and exposure
- Most underserved genres: Film-Noir (+0.94), Documentary (+0.78), War (+0.67)
- Most overpromoted genres: Comedy (-0.78), Action (-0.67)
- Bias Score formula: `Quality Percentile (avg_rating) - Exposure Percentile (rating_count)`
- With sample sizes ranging from 318,917 to 14,377,237, significance testing loses discriminating power. Effect size (Bias Score) is used instead.

### Section 4 — ML Results

| Model | Accuracy | Class 1 Recall | Class 1 F1 |
|-------|----------|----------------|------------|
| Logistic Regression | 0.749 | 0.67 | 0.69 |
| **Random Forest** | **0.861** | **0.88** | **0.84** |

- `avg_rating` feature importance: **88%**
- Genre features combined: ~12%
- Structural reason: high avg_rating raises quality percentile, raises bias_score, raises is_underserved probability. Not data leakage, but reflects definitional structure.

### Film-level underserved list
- `data/processed/film_underserved.csv` contains all films with rating_count >= 30
- 1,662 underserved films (is_underserved = 1, rating_count >= 30)
- Used in Section 4 Interactive Film Explorer

---

## Dashboard Status

All 6 sections complete and deployed.

- [x] Section 1: The Problem
- [x] Section 2: The Data
- [x] Section 3: Bias Evidence (Bias Score bar chart, correlation scatter, quadrant chart, effect size explanation, t-test justification)
- [x] Section 4: ML Insights (model comparison, feature importance, interactive film explorer)
- [x] Section 5: Limitations & Future Work (4 limitations with impact/severity color coding)
- [x] Section 6: Business Recommendation (3 interventions with justified thresholds, distribution charts for Int 1, dot plot for Int 2)

**File locations:**
- Dashboard: `streamlit_app/app.py`
- Run command: `streamlit run streamlit_app/app.py`
- Figures: `outputs/figures/`
  - phase2_bias_score.png
  - phase2_quadrant.png
  - phase2_correlation.png
  - phase3_feature_importance.png

### Dashboard visual improvements completed today
- Replaced default Streamlit header with custom HTML header (blue left accent line)
- Added stats row (33.8M / 18 / 1,662) below title
- Replaced all st.header() and st.subheader() with custom HTML h2/h3 (font-weight: 500) for visual consistency
- Removed duplicate st.metric() stats from Section 2 (already shown in header)
- Wrapped bias score chart and other large images in st.columns([0.5, 2, 0.5]) to reduce size
- Removed non-functional section navigation pills

---

## Known Issues To Fix (Next Session)

These were identified from a recruiter-perspective review. Fix in order of priority:

**Priority 1 — Interactive Film Explorer missing context (Section 4)**
After the slider filters films, the result table appears with no explanation of what to do with it.
Fix: add one line above the table: "These films are the candidate pool for Intervention 1. A platform editorial team would review this list before applying any algorithmic boost."

**Priority 2 — Key Findings stated too confidently given n=18**
Pearson r = -0.266 based on 18 genre-level data points is a thin statistical basis.
The findings section says the pattern is "systematic" without caveat.
Fix: add a one-sentence hedge near the correlation finding: "Note: this correlation is based on 18 genre-level aggregates. The film-level ML analysis (Section 4) provides a stronger basis for the individual-film conclusions."

**Priority 3 — ML accuracy not translated into business impact (Section 4)**
86.1% accuracy and 88% recall are stated as numbers but never converted into real-world meaning.
Fix: add one line after the recall figure: "In practical terms, an 88% recall rate on 1,662 underserved films means the model correctly identifies approximately 1,462 of them, missing around 200."

**Priority 4 — Exposure proxy limitation should appear earlier**
Limitations are in Section 5, but rating_count is used as exposure proxy from Section 2 onward.
Fix: add a short footnote or info box in Section 2 when rating_count is first introduced, saying it is a proxy and linking to the full limitation in Section 5.

**Priority 5 — Section 2 feels thin after removing st.metric()**
The section now only has a methodology note about exploding rows.
Fix: either add a small data preview (st.dataframe of genre_summary.csv head), or merge Section 2 content into Section 3 introduction.

---

## Business Recommendations (Section 6)

**Intervention 1 — Film-Level Algorithmic Trigger**
- Flag films with avg_rating >= 3.8 AND rating_count <= 60 as boost candidates
- Why 3.8: mean avg_rating of underserved films is 3.77
- Why 60: median rating_count of underserved films is 58.5
- Current candidate pool at these thresholds: 831 films
- Alt threshold: rating_count <= 120 (75th percentile) expands pool to ~1,247 films

**Intervention 2 — Quarterly Genre Fairness Audit**
- Flag genres with bias_score > 0.3 for editorial review
- Why 0.3: natural gap in data between Animation (+0.33) and Mystery (+0.28)
- Conservative alt: threshold 0.5, flags only Film-Noir, Documentary, War
- Currently 5 genres qualify: Film-Noir, Documentary, War, Animation, Western

**Intervention 3 — Human-in-the-Loop Editorial Layer**
- Algorithm surfaces candidates; editorial team makes final decisions
- Rationale: fixing one bias can introduce another; rating_count cannot fully separate algorithmic suppression from small audience size

---

## README Status

Complete and pushed to GitHub root (README.md).

Structure:
1. Title + one-line pitch
2. The Problem
3. Key Findings (with images)
4. Business Recommendations
5. How It Works (analytical pipeline + methodological decisions)
6. Technical Stack
7. Project Structure
8. How to Run Locally
9. Limitations
10. About

Note: README uses the version Claude drafted, not self-written. Decision made because the drafted version was stronger and the goal of writing a self-version (familiarity with project) was already achieved through building the project itself.

Streamlit sleep disclaimer added below badges:
"The dashboard is hosted on Streamlit Community Cloud. If you see a loading screen, please wait about 60 seconds for the app to wake up."

---

## Design Decisions Log

| Date | Decision | Options Considered | Reason |
|------|----------|--------------------|--------|
| 2026-04 | Use MovieLens dataset | Kaggle Netflix vs MovieLens | MovieLens has explicit rating + volume data |
| 2026-04 | Genre-level analysis | Film-level vs genre-level | Genre-level avoids selection bias of individual niche films |
| 2026-04 | Streamlit for dashboard | React + FastAPI vs Streamlit | Faster to build, sufficient for BA/PM portfolio |
| 2026-04 | Explode multi-genre rows | Explode vs Fractional vs Primary genre only | Preserves relative exposure differences consistently |
| 2026-04 | Switch to ml-latest | Keep small vs switch full | Small dataset rating_count median = 1, statistically unreliable |
| 2026-04 | rating_count >= 30 threshold | Various thresholds | CLT requires n >= 30 for avg_rating to be statistically meaningful |
| 2026-04 | Intervention 1 rating_count threshold = 60 | 60 vs 100 vs 120 | 60 = median of underserved films, suppression signal strongest below median |
| 2026-04 | Intervention 2 bias_score threshold = 0.3 | Various | Natural gap between Animation (+0.33) and Mystery (+0.28) |
| 2026-04 | Effect size over significance testing | t-test vs effect size | Large n makes all differences statistically significant; effect size more meaningful |
| 2026-04 | Keep dashboard as single scrollable page | Tabs vs scroll | Tabs break narrative flow; recruiters may not click through all tabs |
| 2026-04 | Remove section navigation pills | Keep vs remove | Non-functional buttons look broken; removed for cleanliness |
| 2026-04 | Custom HTML header instead of st.title() | Default vs custom | Default Streamlit header font-weight too heavy, clashes with body text style |

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
- Do not use em dash (--) in writing (AI-sounding)
- Do not use emoji unless functionally justified
- Do not refer to sections as "Phase X" in dashboard — use formal section names
- Do not make mistakes on section numbers or formal names
- When writing new code steps, ALWAYS explain: (1) concept/logic, (2) every new function with parameters, (3) every new variable, (4) background knowledge needed. Never assume understanding.