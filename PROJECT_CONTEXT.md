# Project Context — For AI Assistant

> This file gives the AI assistant full context about who I am and what I'm building.
> Paste this at the start of every new chat session.

---

## About Me

- **Name:** Sophie Cheng (Cheng, Xin-Ya)
- **Background:** Dual major in Economics & Foreign Language and Literature, National Taiwan University (GPA 3.91/4.30). Exchange at Singapore Management University.
- **Current status:** Graduating NTU June 2025. Starting Columbia MSBA in August 2026.
- **Work experience:** Klook (Platform Operations, SQL + A/B testing), Eastspring Investment (compliance automation), Porsche Taiwan (finance reconciliation), NTU Data Analytics Club (Python data cleaning + ML project participation)
- **Programming level:** Beginner-intermediate. Familiar with SQL, Excel VBA, basic Python (pandas, data cleaning). Has worked on real datasets (30,000+ rows) with AI assistance. Wants to improve Python independently through this project.
- **ML level:** Exposure through NTU club project (feature engineering, coordination). No independent ML modeling experience yet. Willing to learn from scratch.
- **Other tools:** Tableau, Stata, R (econometrics), Power Automate, SAP
- **Hours available per day:** ~5 hours
- **Timeline:** ~3.5 months before Columbia starts (deadline: August 2026)

---

## Career Goal

- **Target roles:** Business Analyst / Data Analyst / Operations Analyst / AI PM
- **Target companies:** Tech and entertainment companies (e.g. Netflix, Disney, Spotify, Klook-type platforms), US market preferred
- **Why this project:** To demonstrate business analytical thinking, data-driven decision making, and basic ML understanding. The project should tell a clear story that resonates with BA/PM interviewers — not just technical output.

---

## How I Want to Work With You

- I am learning as I build. **Explain concepts when you write code**, don't just give me the answer.
- When I ask "how do I do X", **show me the logic first**, then the code.
- If I'm about to make a bad design decision, **tell me honestly** instead of going along with it.
- I want to be able to **explain every design decision** in my own words during a job interview.
- **Challenge me** if my understanding seems wrong or shallow.
- **Don't let me seek confirmation constantly** — push me to back my own judgment.
- Default language: **Traditional Chinese (zh-TW)** for explanations, English for code and documentation.

---

## Project Summary

- **Project name:** StreamLens — Genre Fairness Audit for Streaming Recommendations
- **One-line description:** Detects systematic popularity bias in streaming recommendation systems by analyzing the gap between genre quality (ratings) and genre exposure (rating volume), and proposes data-driven curation strategies.
- **Core problem being solved:** Streaming platforms like Netflix tend to over-recommend popular content, systematically under-exposing high-quality niche genres. This leads to user frustration, poor content discovery, and potential churn — especially among users with non-mainstream taste.
- **Personal motivation:** As a Foreign Literature student and arthouse film viewer, I experienced this problem firsthand. I built this project to validate whether my personal experience reflects a systemic pattern in the data.
- **What makes it non-trivial:** Popularity bias quantification, genre-level fairness analysis, selection bias awareness, business recommendation framing, hybrid curation proposal (algorithmic + editorial)

---

## Technical Stack (Planned)

- **Language:** Python
- **Data:** MovieLens dataset (public, ~100k+ ratings)
- **Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib / Seaborn → Streamlit dashboard
- **ML (Phase 3):** scikit-learn — predicting underexposed but high-potential genres
- **Deployment:** Streamlit Cloud or Render
- **Version control:** GitHub

---

## Project Phases

- [ ] Phase 0 — Environment setup + explore MovieLens dataset structure
- [ ] Phase 1 — Data cleaning + genre-level aggregation (ratings vs. exposure volume)
- [ ] Phase 2 — Bias quantification + visualization (which genres are systematically underexposed?)
- [ ] Phase 3 — ML layer (can we predict which genres deserve more exposure based on rating patterns?)
- [ ] Phase 4 — Dashboard + business recommendations write-up + deployment + README

**Current phase:** Phase 0
**Last thing completed:** Project direction finalized. PROJECT_CONTEXT.md created.
**Current blocker / question:** None — ready to start Phase 0.

---

## Core Analytical Argument (The Story)

1. **Observation:** I noticed Netflix rarely surfaces niche foreign/arthouse films despite their strong reviews.
2. **Hypothesis:** Certain genres consistently receive high ratings but low exposure (measured by number of ratings), suggesting a systemic recommendation bias.
3. **Analysis:** Use MovieLens data to quantify the rating score vs. rating volume gap across genres. Identify genres that fall in the "high quality, low exposure" quadrant.
4. **Awareness:** Account for selection bias — niche genres attract self-selected audiences, inflating average ratings. Adjust interpretation accordingly.
5. **Business recommendation:** Propose a hybrid curation model — algorithmic detection of underexposed genres + periodic editorial intervention (fairness auditing cadence).
6. **PM angle:** Why not just fix the algorithm? Because every algorithm has blind spots, and fixing one may introduce others. Human-in-the-loop curation is lower risk and more adaptable.

---

## Design Decisions Log

| Date | Decision | Options Considered | Reason for Choice |
|------|----------|--------------------|-------------------|
| 2026-04 | Use MovieLens dataset | Kaggle Netflix dataset vs MovieLens | MovieLens has explicit rating + volume data needed for bias analysis; Netflix dataset lacks rating counts |
| 2026-04 | Focus on genre-level analysis (not individual films) | Film-level vs genre-level | Genre-level avoids selection bias of individual niche films; more actionable for business recommendations |
| 2026-04 | Streamlit for dashboard | React + FastAPI vs Streamlit | Streamlit is faster to build and sufficient for BA/PM portfolio demo; React adds complexity without proportional value |
| 2026-04 | Explode multi-genre rows | Explode vs Fractional counting vs Primary genre only | Fractional shrinks the gap between popular and niche genres by penalizing genres that co-occur with many others — hurting the analysis goal. Explode preserves relative exposure differences across genres consistently. |


---

## Things I Do NOT Want

- Do not build everything for me without explanation
- Do not use advanced libraries or patterns I haven't learned yet without flagging it
- Do not skip the "why" when making technical suggestions
- Do not assume I understand something just because I haven't asked about it
- Do not let me keep second-guessing my own correct instincts — redirect me to commit to my judgment

---

**Current phase:** Phase 4
**Last thing completed:** Phase 3 — Logistic Regression (0.760) and Random 
Forest (0.835) built. Feature importance reveals avg_rating accounts for 77% 
of predictive power, partially challenging the hypothesis that genre 
composition is the primary driver of underexposure at film level.
**Current blocker / question:** None — ready to start Phase 4.