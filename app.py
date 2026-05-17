import streamlit as st
import asyncio
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
from Models.groq import GroqModel
from Models.gemini import GeminiModel
from Models.huggingface import HuggingFaceModel
from pipeline.ensemble import EnsemblePipeline

st.set_page_config(page_title="Artemis-Consensus", layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #0052cc; }
    .subtitle   { color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .score-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
    .badge-high   { background: #d4edda; color: #155724; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-medium { background: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-low    { background: #f8d7da; color: #721c24; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_pipeline():
    models = [GroqModel(), GeminiModel(), HuggingFaceModel()]
    return EnsemblePipeline(models)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_raw      = st.checkbox("Show raw model responses", value=True)
    show_critique = st.checkbox("Show critique scores",    value=True)
    show_metrics  = st.checkbox("Show latency & tokens",  value=True)

    st.markdown("---")
    st.markdown("**API Keys Status**")
    st.write("🔑 Groq:", "✅" if os.getenv("GROQ_API_KEY")        else "❌ Missing")
    st.write("🔑 Gemini:", "✅" if os.getenv("GEMINI_API_KEY")    else "❌ Missing")
    st.write("🔑 HuggingFace:", "✅" if os.getenv("HUGGINGFACE_API_KEY") else "❌ Missing")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Artemis Consensus: A Multi-LLM Ensemble System for Reliable Answer Generation.")

st.markdown('<div class="main-title">Artemis Consensus</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-LLM Ensemble · Cross-Critique Synthesis · Research-Grade Evaluation</div>', unsafe_allow_html=True)

question = st.text_area("Ask a question", height=100, placeholder="Type your question here...")
run_btn  = st.button("Run Ensemble", type="primary", width="stretch")


if run_btn and question.strip():
    pipeline = load_pipeline()
    with st.spinner("Running ensemble pipeline..."):
        result = asyncio.run(pipeline.run(question))

    tabs=st.tabs(["Final Answer", "Model Responses", "Critique" , "Scores", "Metrics"])

    with tabs[0]:
        if result.final_answer:
            label=result.final_score.confidence_label if result.final_score else "Unknown"
            st.markdown(f"Confidence: <span class='badge-{label.lower()}'>{label}</span>", unsafe_allow_html=True)
        if result.disagreement_detected:
            st.warning("⚠️ Disagreement detected between models!")
        st.markdown(f"### Final Answer:\n{result.final_answer}")
        st.write(result.final_score.to_dict() if result.final_score else "No score available.")


    with tabs[1]:
        for r in result.model_responses:
            with st.expander(f"{'✅' if r.success else '❌'} {r.model_name} — {r.latency_ms}ms"):
                if r.success:
                    st.write(r.answer)
                else:
                    st.error(f"Error: {r.error}")

    with tabs[2]:
        if result.critique_feedback:
            for model_name, feedback_list in result.critique_feedback.items():
                st.markdown(f"**{model_name} Critique Feedback:**")
                for fb in feedback_list:
                    st.write(f"- {fb}")

        else:
            st.info("No critique feedback available.")


    with tabs[3]:
        if result.critique_scores:
            score_data = {
                name: score.to_dict() for name, score in result.critique_scores.items()}
            df_scores = pd.DataFrame(score_data).T
            numeric_cols = ["factuality", "confidence", "completeness", "consistency", "reasoning", "weighted_total"]
            st.dataframe(df_scores[numeric_cols].style.format("{:.2f}").background_gradient(cmap="RdYlGn"), width="stretch")

            fig,ax=plt.subplots(figsize=(8,4))
            dims=["factuality", "confidence", "completeness", "consistency", "reasoning"]
            for name,score in result.critique_scores.items():
                values=[getattr(score, dim) for dim in dims]
                ax.plot(dims, values, marker='o', label=name)

            ax.set_ylim(0,1)
            ax.legend(fontsize='small')
            ax.set_title("Critique Scores by Dimension")
            plt.xticks(rotation=45,fontsize=10)
            st.pyplot(fig)
        else:
            st.info("No critique scores available.")
    with tabs[4]:
        col1,col2,col3=st.columns(3)
        col1.metric("Total Latency (ms)", f"{result.total_latency_ms:.2f}")
        col2.metric("Total Tokens", f"{result.total_tokens:,}")
        col3.metric("Models Used", len([r for r in result.model_responses if r.success]))


        model_metrics=pd.DataFrame([
            {
                "Model": r.model_name,
                "Latency (ms)": f"{r.latency_ms:.2f}",
                "Tokens Used": f"{r.tokens_used:,}",
                "Success": "✅" if r.success else "❌"
            }
            for r in result.model_responses
        ])
        st.dataframe(model_metrics, width="stretch")
elif run_btn:
    st.warning("Please enter a question to run the ensemble pipeline.")
    