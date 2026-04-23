import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Page config (must be first Streamlit call) ──────────────────────────
# st.set_page_config() 設定網頁基本屬性（標題、icon、版面寬度）必須是第一個streamlit指令
st.set_page_config(
    page_title="StreamLens",
    page_icon="🎬",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/genre_summary.csv")

# Load film-level data for interactive explorer
film_df = pd.read_csv("data/processed/film_underserved.csv")

# ── Header ───────────────────────────────────────────────────────────────
st.title("🎬 StreamLens")
st.markdown("**Genre Fairness Audit for Streaming Recommendation Systems**")
st.markdown("*Detecting systematic popularity bias using MovieLens data*") # *斜體*

st.divider() # 顯示水平分割線

# ════════════════════════════════════════════════════════════════════════
# SECTION 1: THE PROBLEM
# ════════════════════════════════════════════════════════════════════════
st.header("1. The Problem")

st.markdown("""
Streaming platforms like Netflix rely on recommendation algorithms to surface content. 
These algorithms tend to favour **popular genres** — those with high rating volumes — 
creating a feedback loop that systematically **underexposes high-quality niche content**.

This audit quantifies that bias using MovieLens data, identifying which genres are 
being systematically overlooked relative to their quality.
""")

# Key finding callout
st.markdown("""
> 🔍 **Key example:** Film-Noir ranks **#1 in quality** (avg rating percentile: 0.99) 
> but sits in the **bottom 10% of exposure** (rating count percentile: 0.04). 
> Bias score: **+0.94** — the largest gap of any genre.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
# SECTION 2: THE DATA
# ════════════════════════════════════════════════════════════════════════
st.header("2. The Data")

st.markdown("""
**Dataset:** [MovieLens ml-latest](https://grouplens.org/datasets/movielens/) 
— a public benchmark dataset widely used in recommendation system research.
""")

# Key statistics
col1, col2, col3 = st.columns(3) # st.columns() 把版面切成 n 欄位，讓元件並排顯示而不是從上到下堆疊

with col1:
    st.metric("Total Ratings", "33,832,162") # st.metrics() 顯示數字卡片，有標題和數值，適合展示統計相關文字
with col2:
    st.metric("Movies", "86,537")
with col3:
    st.metric("Genres Analysed", str(len(df))) # len(df) 回傳 dataframe的列述（幾筆資料）

st.markdown("""
**Methodology note:** Movies tagged with multiple genres (e.g. *Action | Adventure*) 
were **exploded** — each genre receives a full copy of the film's rating data. 
This preserves relative exposure differences across genres consistently, 
rather than penalising genres that frequently co-occur with others.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
# SECTION 3: BIAS EVIDENCE
# ════════════════════════════════════════════════════════════════════════
st.header("3. Bias Evidence")

st.markdown("""
We define **Bias Score** as:

> `Bias Score = Quality Percentile (avg rating) − Exposure Percentile (rating count)`

A **positive** score means a genre is higher quality than its exposure suggests — it is **underserved**.  
A **negative** score means a genre receives more exposure than its quality warrants — it is **overpromoted**.
""")

# ── Bias Score bar chart ─────────────────────────────────────────────────
st.subheader("Bias Score by Genre")

st.image(
    "outputs/figures/phase2_bias_score.png",
    caption="Green = underserved (quality > exposure) | Red = overpromoted (exposure > quality)",
    use_container_width=True
)

st.markdown("""
**Most underserved genres:**
- 🎬 **Film-Noir** — Bias Score: **+0.94** 
- 📽️ **Documentary** — Bias Score: **+0.78**
- ⚔️ **War** — Bias Score: **+0.67**

**Most overpromoted genres:**
- 😂 **Comedy** — Bias Score: **−0.83**
- 💥 **Action** — Bias Score: **−0.67**

The Bias Score tells us *how large* the gap is — but not *why* it exists. 
To understand whether a genre is underserved because of genuinely high quality, 
or overpromoted despite low quality, we need to look at both dimensions separately.
""")

st.subheader("Quality vs. Exposure: Genre-Level Correlation")

st.image(
    "outputs/figures/phase2_correlation.png",
    caption="Each dot = one genre. Pearson r = −0.266: higher-rated genres tend to receive less exposure.",
    use_container_width=True
)

st.markdown("""
**Pearson r = −0.266** indicates a moderate negative correlation between genre quality 
and exposure. This means the pattern is **systemic**, not coincidental — 
higher-rated genres consistently appear in the lower half of exposure.

Note: r measures the *direction and strength* of the linear relationship between 
the two variables. It is not a regression slope and carries no unit.
""")

st.divider()

# ── Quadrant chart ───────────────────────────────────────────────────────
st.subheader("Quality vs. Exposure Quadrant")

st.markdown("""
The quadrant plots each genre on **two axes simultaneously** — quality percentile (Y) 
and exposure percentile (X). Unlike the Bias Score which collapses both into a single 
number, this view shows *where* a genre sits in quality-exposure space, and *why* 
it received its Bias Score.

For example, Film-Noir's Bias Score of +0.94 is explained here: it sits at the 
**top-left corner** — maximum quality, minimum exposure.
""")

st.image(
    "outputs/figures/phase2_quadrant.png",
    caption="Top-left quadrant = systematically underserved genres",
    use_container_width=True
)

_, col_center, _ = st.columns([0.5, 3, 0.5])
with col_center:
    st.markdown("""
| Quadrant | Quality | Exposure | Interpretation | Example Genres |
|----------|---------|----------|----------------|----------------|
| 🟢 Top-left — Underserved | High | Low | Deserve more visibility | Film-Noir, Documentary, War |
| 🔵 Top-right — Dominant | High | High | Well-served by the system | Drama, Crime |
| ⬜ Bottom-left — Niche | Low | Low | Small audience, lower-rated | Horror |
| 🔴 Bottom-right — Overpromoted | Low | High | Overexposed relative to quality | Comedy, Action, Thriller |
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════
# SECTION 4: ML INSIGHTS
# ════════════════════════════════════════════════════════════════════════
st.header("4. ML Insights")

# ── Segue from Bias Score Analysis ───────────────────────────────────────────────────
st.markdown("""
Bias Score Analysis established a genre-level finding: higher-rated genres tend to receive 
systematically less exposure (Pearson r = −0.266). However, this analysis has 
an inherent limitation — **18 data points** (one per genre) is a thin basis for 
drawing strong conclusions.

This raises two questions:

> **Q1:** Does this bias pattern hold at the individual film level — or is it 
> an artifact of genre-level aggregation?

> **Q2:** If it does hold, how much of the predictive signal comes from genre 
> itself — versus other film characteristics?

To investigate, we build a machine learning model at the **film level**, 
predicting whether an individual film is underserved (`is_underserved = 1`) 
based on its features.
""")

st.divider()

# ── Model comparison ─────────────────────────────────────────────────────
st.subheader("Model Comparison")

st.markdown("""
Two models were trained and compared. Since our goal is to **minimise missed 
underserved films** (false negatives), **Recall** is the primary metric — 
we care more about catching every underserved film than avoiding false alarms.
""")

st.markdown("""
| Model | Accuracy | Class 1 Recall | Class 1 F1 |
|-------|----------|----------------|------------|
| Logistic Regression | 0.749 | 0.67 | 0.69 |
| **Random Forest** | **0.861** | **0.88** | **0.84** |
""")

st.markdown("""
**Random Forest outperforms Logistic Regression** across all metrics — 
11.2% higher accuracy, and Recall improving from 0.67 to **0.88**. 
This means the model successfully identifies **88% of truly underserved films**.
""")

st.divider()

# ── Feature importance ───────────────────────────────────────────────────
st.subheader("What Drives Underexposure? — Feature Importance")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.image(
        "outputs/figures/phase3_feature_importance.png",
        caption="Random Forest feature importance — avg_rating dominates at 88%",
        use_container_width=True
    )

with col_right:
    st.markdown("""
**Q1 answer:** Yes — the bias pattern holds at the film level.  
Random Forest achieves 86.1% accuracy, confirming that underserved 
films follow a learnable, systematic pattern. This means the genre-level 
finding from Bias Score Analysis is not merely an artifact of aggregation — 
it reflects a real, film-level phenomenon.

---

**Q2 answer:** Genre is not the main driver.

| Feature Group | Importance |
|---------------|------------|
| `avg_rating` | **88%** |
| Genre flags + genre count | ~12% |

`avg_rating` dominates because of its structural relationship with how 
`is_underserved` was defined:

- A high `avg_rating` raises a film's quality percentile
- A high quality percentile raises its Bias Score
- A high Bias Score makes `is_underserved = 1` more likely

This is not data leakage — `avg_rating` is a legitimate predictor with 
real business meaning. But it does mean the model is partly picking up 
on the *structure of our definition*, not purely on external causal factors.

---
    """)

st.markdown("""
**What this means:**  
The data suggests platforms are not systematically ignoring specific genres 
— the stronger pattern is that **high-quality content tends to be underserved, 
regardless of genre**.

However, this conclusion should be read carefully: Drama and Comedy appear 
heavily in *both* underserved and overpromoted classes, which means genre 
alone has low discriminating power at the film level. Genre is a secondary 
signal — useful, but not sufficient on its own. (⚠️ These findings are subject to important limitations — see Section 5.)
""")



st.divider()


st.subheader("Interactive Film Explorer: Underserved Films")

st.info("""
**Bias Score formula:**  Quality Percentile (avg_rating) − Exposure Percentile (rating_count) >> A higher Bias Score = higher quality relative to exposure = more underserved.
""")

st.markdown("""
Based on the ML findings, the most actionable way to identify underserved content 
is to filter directly by **high quality + low exposure** at the film level.
Adjust the sliders below to explore which films are being systematically overlooked.
""")

col_s1, col_s2 = st.columns(2)

with col_s1:
    # slider for minimum avg_rating
    # min_value and max_value define the range of the slider
    # value sets the default starting position
    # step defines the smallest increment when dragging
    min_rating = st.slider(
        "Minimum Average Rating",
        min_value=3.5,
        max_value=4.5,
        value=3.8,
        step=0.1
    )

with col_s2:
    # slider for maximum rating_count (exposure)
    max_exposure = st.slider(
        "Maximum Rating Count (Exposure)",
        min_value=30,
        max_value=500, # 為什麼 max_value 設 500 而不是 3,015： 3,015 是極端值，如果滑桿拉到 3,015，幾乎所有電影都會顯示，失去篩選意義。500 已經涵蓋 75% 以上的電影，讓使用者可以有意義地探索。
        value=100,
        step=10
    )

# Filter film_df based on slider values
# Only show underserved films (is_underserved=1) that match the criteria
filtered = film_df.query(
    "is_underserved == 1 and avg_rating >= @min_rating and rating_count <= @max_exposure"
).sort_values("bias_score", ascending=False)

# @ prefix inside query() means "use this Python variable"
# without @, pandas would look for a column named min_rating

st.markdown(f"**{len(filtered)} films** match these criteria.")

st.dataframe(
    filtered[['title', 'avg_rating', 'rating_count', 'bias_score']],
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# SECTION 5: Limitations & Future Work
# ============================================================
st.header("5. Limitations & Future Work")

st.markdown(
    "This analysis rests on several assumptions. "
    "Understanding where they break down is necessary before acting on the findings."
)

# Legend as inline tokens on the right side.
# st.markdown() with unsafe_allow_html=True lets us write raw HTML inside Streamlit.
# This is necessary here because Streamlit has no native "small inline badge" component.
# The outer div uses text-align:right to push the tokens to the right side.
# Each span is a colored dot (the circle unicode character) followed by small label text.
st.markdown(
    """
    <div style="text-align: right; margin-bottom: 8px;">
        <span style="
            background-color: #fff3cd;
            border: 1px solid #f0ad4e;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 12px;
            color: #856404;
            margin-right: 8px;
        ">High impact</span>
        <span style="
            background-color: #cfe2ff;
            border: 1px solid #9ec5fe;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 12px;
            color: #084298;
        ">Medium impact</span>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Limitation 1 — Exposure Proxy", expanded=True):
    st.warning(
        """
        **Problem:** `rating_count` mixes three signals: algorithmic recommendations, 
        organic audience search, and historical content volume. It is not a clean measure 
        of algorithmic exposure alone.

        **Impact:** The r = −0.266 correlation and Bias Scores cannot be attributed solely 
        to platform suppression. Film-Noir's +0.94 gap is large enough that audience size 
        alone is unlikely to explain it — but that alternative cannot be fully ruled out.

        **Fix with real data:** Replace `rating_count` with impression counts (pure 
        algorithmic exposure) and click-through rates (separates supply-side suppression 
        from demand-side disinterest).
        """
    )

with st.expander("Limitation 2 — avg_rating Structural Dominance"):
    st.warning(
        """
        **Problem:** `avg_rating` accounts for 88% of Random Forest predictive power — 
        partly because `is_underserved` was defined using `avg_rating` via `bias_score`. 
        The model partially learns back the definition.

        **Impact:** The 86.1% accuracy overstates how well genre features alone predict 
        underexposure. Genre contributes only ~12% of predictive signal.

        **Fix with real data:** Define `is_underserved` using impression data, which breaks 
        the structural link between `avg_rating` and the target variable.
        """
    )

with st.expander("Limitation 3 — Class Imbalance"):
    st.info(
        """
        **Problem:** Test set contains 9,038 overpromoted films vs. 6,267 underserved films 
        (~1.44:1 ratio). The model identifies overpromoted films slightly better 
        (F1: 0.88) than underserved ones (F1: 0.84).

        **Impact:** In production, the model may slightly undercount underserved films — 
        the direction we most want to get right.

        **Fix:** Apply `class_weight='balanced'` in scikit-learn, or use SMOTE oversampling 
        on the minority class before training.
        """
    )

with st.expander("Limitation 4 — Dataset Recency"):
    st.info(
        """
        **Problem:** MovieLens ml-latest has a data cutoff and does not reflect recent 
        changes in streaming platform algorithms, content libraries, or user behavior.

        **Impact:** Genre bias patterns observed here may have shifted as platforms have 
        updated their recommendation systems.

        **Fix with real data:** Access live recommendation logs with timestamps to enable 
        temporal drift analysis.
        """
    )

st.divider()

# ============================================================
# SECTION 6: Business Recommendation
# ============================================================
st.header("6. Business Recommendation")

st.markdown(
    """
    Section 3 - Bias Evidence confirmed genre-level bias is real: Genres such as Film-Noir, Documentary, and War are 
    systematically underexposed relative to their quality (r = −0.266).

    Section 4 - ML Insights revealed that at the film level, `avg_rating` accounts for 88% of predictive 
    power — genre features contribute only ~12%. Genre-based curation alone is therefore 
    insufficient as an intervention.

    The data supports a three-layer strategy:
    """
)


# Each intervention uses a left border bar instead of a full colored box.
# This gives visual structure and hierarchy without the heavy background of st.success().
# It matches the cleaner aesthetic of the rest of the dashboard.
# unsafe_allow_html=True is required to render custom HTML/CSS in st.markdown().
st.markdown(
    """
    <div style="
        border-left: 4px solid #198754;
        background-color: #f6fdf9;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-radius: 0 6px 6px 0;
    ">
        <p style="font-weight: 600; margin-bottom: 6px;">
            Intervention 1 — Film-Level Algorithmic Trigger
        </p>
        <p style="margin: 0;">
            Flag films with <code>avg_rating &gt;= 3.8</code> and 
            <code>rating_count &lt;= 100</code> as candidates for boosted recommendation.
            Section 4 shows avg_rating is the dominant signal at the film level. 
            Targeting films directly — regardless of genre — captures more underserved 
            content than genre-filtering alone. 1,662 statistically valid films 
            (rating_count &gt;= 30) qualify under these thresholds in the current dataset.
        </p>
    </div>

    <div style="
        border-left: 4px solid #198754;
        background-color: #f6fdf9;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-radius: 0 6px 6px 0;
    ">
        <p style="font-weight: 600; margin-bottom: 6px;">
            Intervention 2 — Genre Fairness Audit Cadence
        </p>
        <p style="margin: 0;">
            Run quarterly Bias Score analysis across all genres. Flag any genre where 
            <code>bias_score &gt; 0.3</code> for content team review. Section 3 shows 
            genre-level patterns are real and measurable. A recurring audit catches drift — 
            new content additions can shift Bias Scores over time. Five of 18 genres 
            currently exceed the +0.3 threshold: Film-Noir (+0.94), Documentary (+0.78), 
            War (+0.67), Western (+0.44), Animation (+0.33).
        </p>
    </div>

    <div style="
        border-left: 4px solid #198754;
        background-color: #f6fdf9;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-radius: 0 6px 6px 0;
    ">
        <p style="font-weight: 600; margin-bottom: 6px;">
            Intervention 3 — Human-in-the-Loop Editorial Layer
        </p>
        <p style="margin: 0;">
            Algorithm surfaces candidates; editorial team makes final curation decisions 
            before any boost goes live. Every algorithm has blind spots. Limitation 1 
            confirms that <code>rating_count</code> cannot fully separate algorithmic 
            suppression from small audience size — human judgment bridges this gap until 
            cleaner exposure data is available.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.caption(
    "StreamLens — Genre Fairness Audit for Streaming Recommendations | "
    "Data: MovieLens ml-latest (33.8M ratings) | "
    "Built by Sophie Cheng"
)