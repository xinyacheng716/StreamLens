import streamlit as st
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StreamLens",
    page_icon="🎬",
    layout="wide"
)

# ── Load data (needed for genre count in stats) ──────────────────────────
df = pd.read_csv("data/processed/genre_summary.csv")

# ── Header ───────────────────────────────────────────────────────────────
# Left blue accent line + project label using HTML
# unsafe_allow_html=True is required to render custom HTML in Streamlit
st.markdown(
    """
    <div style="
        border-left: 4px solid #378add;
        padding: 0.15rem 0 0.15rem 1rem;
        margin-bottom: 0.5rem;
    ">
        <div style="
            font-size: 11px;
            font-weight: 500;
            color: #378add;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 6px;
        ">Genre fairness audit</div>
        <div style="
            font-size: 28px;
            font-weight: 500;
            color: var(--text-color);
            margin-bottom: 4px;
        ">🎬 StreamLens</div>
        <div style="
            font-size: 14px;
            color: gray;
            line-height: 1.6;
        ">Investigating whether niche genres, despite higher average ratings,
        receive systematically lower exposure on streaming platforms.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Stats row ─────────────────────────────────────────────────────────────
# st.columns([1,1,1]) splits the layout into 3 equal columns
# We use a light gray background on the stats row via a container div
st.markdown(
    f"""
    <div style="
        background-color: #f7f8fa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    ">
        <div>
            <div style="font-size: 11px; color: gray; margin-bottom: 3px;">Ratings analysed</div>
            <div style="font-size: 22px; font-weight: 500;">33.8M</div>
            <div style="font-size: 11px; color: gray;">MovieLens ml-latest</div>
        </div>
        <div>
            <div style="font-size: 11px; color: gray; margin-bottom: 3px;">Genres compared</div>
            <div style="font-size: 22px; font-weight: 500;">{len(df)}</div>
            <div style="font-size: 11px; color: gray;">Across 86,537 films</div>
        </div>
        <div>
            <div style="font-size: 11px; color: gray; margin-bottom: 3px;">Underserved films</div>
            <div style="font-size: 22px; font-weight: 500;">1,662</div>
            <div style="font-size: 11px; color: gray;">Rating count ≥ 30</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()

# ── Placeholder so you can see the header in context ─────────────────────
st.markdown("*(rest of dashboard goes here)*")