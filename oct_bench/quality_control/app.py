import os
import sys
import json
import glob
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path so oct_bench can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Auto-load API keys from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # dotenv not installed, keys must be set manually
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib
import oct_bench.evaluation.models
importlib.reload(oct_bench.evaluation.models)
from oct_bench.evaluation.models import get_evaluator
from oct_bench.evaluation.metrics import compute_all_metrics, bleu_score, rouge_l, hallucination_rate
from oct_bench.data.augmentor import AUGMENTATIONS, get_augmentation_names, apply_augmentation

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="OCT-Bench Benchmark Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Light Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1A73E8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #5F6368;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F1F3F4;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #DADCE0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #137333;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #3C4043;
    }
    .cot-box {
        background-color: #F8F9FA;
        border-left: 4px solid #1A73E8;
        padding: 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.92rem;
        color: #202124;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load dataset
DATASET_PATH = "./data/generated_qa/dataset.json"

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_dataset(data):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)

# Page header
st.markdown('<div class="main-title">👁️ OCT-Bench Evaluation & Quality Control System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hierarchical benchmark platform for Multimodal Large Language Models on Optical Coherence Tomography (OCT).</div>', unsafe_allow_html=True)

# Load questions
questions = load_dataset()

# Top level tabs
tab_qc, tab_analytics, tab_live, tab_metrics, tab_robust = st.tabs([
    "👁️ Expert Quality Control",
    "📊 Benchmark Leaderboard & Heatmap",
    "⚡ Live Model Inference",
    "🔬 Advanced Metrics",
    "🛡️ Robustness Testing",
])

# ==============================================================================
# TAB 1: EXPERT QUALITY CONTROL (HUMAN-IN-THE-LOOP)
# ==============================================================================
with tab_qc:
    st.sidebar.header("Dataset Overview")
    
    if not questions:
        st.sidebar.warning("No questions generated yet.")
        st.info("No generated questions found. Please execute the pipeline to generate mock/LLM questions first.")
        if st.button("Generate Default Mock Dataset Now"):
            with st.spinner("Creating mock database..."):
                from mock_generator import main as gen_mock
                from oct_bench.data.adapters import UnifiedJSONAdapter
                from oct_bench.generation.generator import VQAGenerator
                
                gen_mock()
                adapter = UnifiedJSONAdapter("./data/raw")
                samples = adapter.load_samples()
                
                tasks = ["T01", "T06", "T09", "T12", "T17"]
                generator = VQAGenerator()
                generator.generate_benchmark(samples, tasks)
                st.rerun()
    else:
        # Sidebar stats
        total_q = len(questions)
        perception_count = len([q for q in questions if q['task_id'].startswith(('T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08'))])
        cognition_count = len([q for q in questions if q['task_id'].startswith(('T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16'))])
        reasoning_count = total_q - perception_count - cognition_count
        
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{perception_count}</div><div class="metric-label">Perception</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cognition_count}</div><div class="metric-label">Cognition</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{reasoning_count}</div><div class="metric-label">Reasoning</div></div>', unsafe_allow_html=True)
            
        st.sidebar.markdown("---")
        
        # Select question dropdown
        q_ids = [q["question_id"] for q in questions]
        selected_qid = st.sidebar.selectbox("Select Question to Review:", q_ids)
        
        # Locate question
        q_index = next(i for i, q in enumerate(questions) if q["question_id"] == selected_qid)
        q_data = questions[q_index]

        # Two column layout
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.subheader("Annotated Scan Image")
            img_path = q_data.get("metadata", {}).get("rendered_image_path", "")
            if not img_path:
                img_path = q_data.get("rendered_image_path", "")
                
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption=f"Task Image for {q_data['question_id']}")
            else:
                st.error(f"Image not found at path: {img_path}")
                
            st.info(f"**Anatomical Details & Context:**\n"
                    f"- Original Dataset: {q_data.get('metadata', {}).get('dataset_source', 'N/A')}\n"
                    f"- Image ID: {q_data.get('image_id', 'N/A')}\n"
                    f"- Disease diagnosis: {q_data.get('metadata', {}).get('disease_label', 'N/A')}")
                    
            # Med-CoT Clinical Reasoning Preview Button
            if st.button("🧠 Preview Clinical CoT for this Scan", help="Simulate step-by-step clinical chain of thought"):
                from oct_bench.evaluation.models import get_evaluator
                ev = get_evaluator("random")
                if hasattr(ev, "predict_with_reasoning"):
                    ans_dummy, cot_preview = ev.predict_with_reasoning(img_path, q_data["question"], q_data["options"], cot=True)
                else:
                    ans_dummy = ev.predict(img_path, q_data["question"], q_data["options"])
                    cot_preview = "1. Visual Perception: Focal optical reflectivity disruption.\n2. Anatomical Localization: Retinal layers (ILM to RPE).\n3. Clinical Correlation: Corresponds with verified guideline features."
                st.markdown("##### 🔬 Clinical Chain-of-Thought (Med-CoT) Steps:")
                st.markdown(f'<div class="cot-box">{cot_preview.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                    
        with right_col:
            st.subheader("Edit & Validate MCQ Details")
            
            with st.form(key="mcq_edit_form"):
                task_id = st.text_input("Task ID (T01 - T20)", value=q_data["task_id"], disabled=True)
                question_text = st.text_area("Question Text", value=q_data["question"], height=80)
                
                # Options A-D
                opt_a = st.text_input("Option A", value=q_data["options"].get("A", ""))
                opt_b = st.text_input("Option B", value=q_data["options"].get("B", ""))
                opt_c = st.text_input("Option C", value=q_data["options"].get("C", ""))
                opt_d = st.text_input("Option D", value=q_data["options"].get("D", ""))
                
                # Correct Option
                correct_opt = st.selectbox("Correct Option", ["A", "B", "C", "D"], index=["A", "B", "C", "D"].index(q_data["correct_answer"]))
                
                # Explanation
                explanation_text = st.text_area("Clinical Explanation", value=q_data.get("explanation", ""), height=100)
                
                # Action buttons
                col_save, col_regen, col_del = st.columns(3)
                
                with col_save:
                    submit_button = st.form_submit_button(label="✅ Save & Approve")
                with col_regen:
                    regen_button = st.form_submit_button(label="🔄 Regenerate")
                with col_del:
                    delete_button = st.form_submit_button(label="🗑️ Reject & Delete")
                    
                if submit_button:
                    questions[q_index]["question"] = question_text
                    questions[q_index]["options"] = {
                        "A": opt_a,
                        "B": opt_b,
                        "C": opt_c,
                        "D": opt_d
                    }
                    questions[q_index]["correct_answer"] = correct_opt
                    questions[q_index]["explanation"] = explanation_text
                    
                    save_dataset(questions)
                    st.success(f"Successfully saved and updated question {selected_qid}!")
                    st.rerun()

                if regen_button:
                    with st.spinner(f"Regenerating question for {q_data.get('image_id', 'image')} (Task {q_data.get('task_id', '')})..."):
                        try:
                            from oct_bench.data.adapters import UnifiedJSONAdapter
                            from oct_bench.data.schemas import ImageSample
                            from oct_bench.generation.generator import VQAGenerator
                            
                            sample = None
                            try:
                                adapter = UnifiedJSONAdapter("./data/raw")
                                raw_samples = adapter.load_samples()
                                sample = next((s for s in raw_samples if s.image_id == q_data.get("image_id")), None)
                            except Exception:
                                pass
                                
                            if not sample:
                                meta = q_data.get("metadata", {})
                                orig_path = meta.get("original_image_path", q_data.get("rendered_image_path", ""))
                                sample = ImageSample(
                                    image_id=q_data.get("image_id", "img_sample"),
                                    image_path=orig_path,
                                    dataset_source=meta.get("dataset_source", "OCT"),
                                    width=512,
                                    height=512,
                                    disease_label=meta.get("disease_label", "NORMAL")
                                )
                            
                            generator = VQAGenerator()
                            new_mcq = generator.generate_question(q_data["task_id"], sample)
                            new_dict = new_mcq.model_dump()
                            new_dict["question_id"] = selected_qid
                            
                            questions[q_index] = new_dict
                            save_dataset(questions)
                            st.success(f"Successfully regenerated question {selected_qid}!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"Failed to regenerate question: {err}")
                    
                if delete_button:
                    questions.pop(q_index)
                    save_dataset(questions)
                    st.warning(f"Deleted question {selected_qid} from benchmark database.")
                    st.rerun()

# ==============================================================================
# TAB 2: BENCHMARK LEADERBOARD & HEATMAP
# ==============================================================================
with tab_analytics:
    st.subheader("📊 MLLM Evaluation Leaderboard & Cognitive Heatmap")
    st.write("Comprehensive benchmarking results comparing multimodal foundation models across Perception, Cognition, and Clinical Reasoning.")

    paper_results_path = os.path.join(PROJECT_ROOT, "data", "reference", "oct_bench_paper_results.csv")
    paper_df = pd.read_csv(paper_results_path)
    task_keys = [f"T{i:02d}" for i in range(1, 21)]
    reference_models = {
        row["model"]: {
            "overall": float(row["overall"]),
            "category": row["category"],
            "Perception": float(row["perception"]),
            "Cognition": float(row["cognition"]),
            "Reasoning": float(row["reasoning"]),
            "tasks": {task: float(row[task]) for task in task_keys},
        }
        for _, row in paper_df.iterrows()
    }

    # Keep local runs separate from the published benchmark; they may use different data.
    local_models = {}
    for csv_file in glob.glob("./reports/*_predictions.csv"):
        try:
            df_rep = pd.read_csv(csv_file)
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as err:
            st.warning(f"Could not read report {os.path.basename(csv_file)}: {err}")
            continue

        if {"prediction", "correct_answer"}.issubset(df_rep.columns):
            prediction = df_rep["prediction"].astype("string").str.strip().str.upper()
            answer = df_rep["correct_answer"].astype("string").str.strip().str.upper()
            valid_rows = prediction.notna() & answer.notna()
            df_rep = df_rep.loc[valid_rows].copy()
            df_rep["_is_correct"] = prediction.loc[valid_rows].eq(answer.loc[valid_rows])
        elif "is_correct" in df_rep.columns:
            correctness = df_rep["is_correct"].astype("string").str.strip().str.lower()
            df_rep["_is_correct"] = correctness.map({"true": True, "false": False, "1": True, "0": False})
            df_rep = df_rep.loc[df_rep["_is_correct"].notna()].copy()
        else:
            st.warning(f"Skipping report {os.path.basename(csv_file)}: no prediction/answer or is_correct columns.")
            continue

        if df_rep.empty:
            st.warning(f"Skipping report {os.path.basename(csv_file)}: no valid prediction rows.")
            continue

        m_slug = os.path.basename(csv_file).replace("_predictions.csv", "")
        m_name = (
            "Random Baseline (Local)"
            if "random" in m_slug.lower()
            else m_slug.replace("-", " ").replace("_", " ").title()
        )
        d_acc = (
            df_rep.groupby("dimension")["_is_correct"].mean() * 100
            if "dimension" in df_rep.columns
            else pd.Series(dtype=float)
        )
        t_acc = (
            df_rep.groupby("task_id")["_is_correct"].mean() * 100
            if "task_id" in df_rep.columns
            else pd.Series(dtype=float)
        )
        local_models[m_name] = {
            "overall": float(df_rep["_is_correct"].mean() * 100),
            "category": "Evaluated Local Model",
            "Perception": float(d_acc["Perception"]) if "Perception" in d_acc else None,
            "Cognition": float(d_acc["Cognition"]) if "Cognition" in d_acc else None,
            "Reasoning": float(d_acc["Reasoning"]) if "Reasoning" in d_acc else None,
            "tasks": {
                task: float(t_acc[task])
                for task in task_keys
                if task in t_acc
            },
        }

    # Top KPI Metrics Row
    best_model, best_scores = max(reference_models.items(), key=lambda item: item[1]["overall"])
    published_model_count = len(reference_models)
    local_model_count = len(local_models)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("🏆 Top Measured Model", best_model, f"{best_scores['overall']:.2f}% accuracy")
    with kpi2:
        st.metric("📄 Published Models", f"{published_model_count}")
    with kpi3:
        st.metric("🎯 Chance Baseline", "25.0%", "4-Option Random Guess")
    with kpi4:
        st.metric("🧪 Local Models", f"{local_model_count}")

    st.caption(
        "Published results: Fu et al., *Can Multimodal Large Language Models Understand OCT?*, "
        "Tables 2–3 (zero-shot): https://arxiv.org/html/2607.16609. "
        "Local results use the separate dataset in ./reports and are not directly comparable to published scores."
    )

    st.markdown("---")

    st.info("Published OCT-Bench results use zero-shot evaluation. CoT comparisons are omitted because no paired CoT results are available.")

    leaderboard_rows = []
    for m, d in reference_models.items():
        leaderboard_rows.append({
            "Model": m,
            "Accuracy (%)": d["overall"],
            "Category": d["category"],
            "Perception": d["Perception"],
            "Cognition": d["Cognition"],
            "Reasoning": d["Reasoning"]
        })
    df_lb = pd.DataFrame(leaderboard_rows).sort_values("Accuracy (%)", ascending=True)

    fig_bar = px.bar(
        df_lb,
        x="Accuracy (%)",
        y="Model",
        orientation="h",
        color="Category",
        text_auto=".1f",
        title="<b>Published OCT-Bench Overall Accuracy (Zero-Shot)</b>",
        color_discrete_map={
            "Closed-Source Models": "#4285F4",
            "Open-Source Models": "#00C853",
            "Medical-Domain Models": "#AA00FF",
        }
    )
    fig_bar.add_vline(x=25.0, line_dash="dash", line_color="#FF5252", annotation_text="Random Guess Baseline (25%)", annotation_position="top right")
    fig_bar.update_layout(
        template="plotly_white",
        height=max(380, len(df_lb) * 42),
        xaxis_range=[0, 105],
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Side-by-side: Cognitive Dimension Breakdown + Radar Chart
    col_dim, col_radar = st.columns([1, 1])

    with col_dim:
        dim_melted = df_lb[["Model", "Category", "Perception", "Cognition", "Reasoning"]].melt(
            id_vars=["Model", "Category"],
            value_vars=["Perception", "Cognition", "Reasoning"],
            var_name="Cognitive Dimension",
            value_name="Dimension Accuracy (%)"
        )
        fig_dim = px.bar(
            dim_melted,
            x="Model",
            y="Dimension Accuracy (%)",
            color="Cognitive Dimension",
            barmode="group",
            text_auto=".1f",
            title="<b>Accuracy Drop Across Cognitive Pathway</b>",
            color_discrete_sequence=["#29B6F6", "#AB47BC", "#FFA726"]
        )
        fig_dim.update_layout(
            template="plotly_white",
            height=420,
            yaxis_range=[0, 105],
            xaxis_tickangle=-30,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    with col_radar:
        all_radar_avail = [
            model for model, scores in reference_models.items()
            if all(scores[dimension] is not None for dimension in ("Perception", "Cognition", "Reasoning"))
        ]
        default_radar = [
            model for model in ["GPT-5.4-mini", "Gemini-2.5-flash", "Hulu-Med-32B"]
            if model in all_radar_avail
        ]
        selected_radar = st.multiselect(
            "Select Models to Compare on Radar:",
            options=all_radar_avail,
            default=default_radar,
            key="radar_models_multiselect"
        )

        fig_radar = go.Figure()
        categories = ["Perception", "Cognition", "Reasoning"]
        radar_colors = px.colors.qualitative.Plotly

        for idx, rm in enumerate(selected_radar):
            if rm in reference_models:
                p_val = reference_models[rm]["Perception"]
                c_val = reference_models[rm]["Cognition"]
                r_val = reference_models[rm]["Reasoning"]
                vals = [p_val, c_val, r_val]
                vals_closed = vals + [vals[0]]
                cats_closed = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed,
                    theta=cats_closed,
                    name=rm,
                    fill="toself",
                    opacity=0.15,
                    line=dict(color=radar_colors[idx % len(radar_colors)], width=2)
                ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
            title="<b>Cognitive Dimension Triad Radar</b>",
            template="plotly_white",
            height=420,
            margin=dict(l=30, r=30, t=50, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Interactive Task-Level Heatmap (T01 - T20)
    st.markdown("### 🔥 Task-Wise Accuracy Heatmap across 20 Clinical Tasks")
    st.write("Detailed breakdown identifying model strengths and failure modes across each fine-grained task.")

    task_labels = [
        "T01 Modality", "T02 Annot", "T03 Morph", "T04 Boundary", "T05 Reflect", "T06 Count", "T07 Scale", "T08 Spatial",
        "T09 Region", "T10 Layer", "T11 Inter-Layer", "T12 Lesion", "T13 Structure", "T14 Location", "T15 Disease", "T16 Deficit",
        "T17 Diagnosis", "T18 Stage", "T19 Treatment", "T20 Followup"
    ]
    task_keys = [f"T{i:02d}" for i in range(1, 21)]

    heatmap_models = list(reference_models.keys())
    heatmap_matrix = []

    for m in heatmap_models:
        row = [reference_models[m]["tasks"].get(tk, float("nan")) for tk in task_keys]
        heatmap_matrix.append(row)

    fig_heat = px.imshow(
        heatmap_matrix,
        x=task_labels,
        y=heatmap_models,
        text_auto=".1f",
        aspect="auto",
        zmin=0,
        zmax=100,
        color_continuous_scale="Viridis",
        title="<b>Granular Task Accuracy (%) Matrix [Perception (T01-08) | Cognition (T09-16) | Reasoning (T17-20)]</b>",
        labels=dict(x="Clinical Task", y="Model", color="Accuracy (%)")
    )
    fig_heat.update_layout(
        template="plotly_white",
        height=max(380, len(heatmap_models) * 38 + 120),
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 4. Filterable Table
    st.markdown("### 📋 Full Benchmark Leaderboard Table")
    df_display = pd.DataFrame(leaderboard_rows)[["Model", "Category", "Accuracy (%)", "Perception", "Cognition", "Reasoning"]].sort_values("Accuracy (%)", ascending=False)
    st.dataframe(df_display, use_container_width=True)

    if local_models:
        st.markdown("### Local Evaluation Results")
        st.caption("These runs use the local dataset and are shown separately; they are not directly comparable to OCT-Bench paper scores.")
        local_rows = [
            {
                "Model": model,
                "Accuracy (%)": scores["overall"],
                "Perception": scores["Perception"],
                "Cognition": scores["Cognition"],
                "Reasoning": scores["Reasoning"],
            }
            for model, scores in local_models.items()
        ]
        df_local = pd.DataFrame(local_rows).sort_values("Accuracy (%)", ascending=True)
        st.dataframe(df_local, use_container_width=True)

        col_local_acc, col_local_dim = st.columns(2)
        with col_local_acc:
            fig_local_acc = px.bar(
                df_local,
                x="Accuracy (%)",
                y="Model",
                orientation="h",
                text_auto=".1f",
                title="<b>Local-Run Accuracy</b>",
                color_discrete_sequence=["#FF9100"],
            )
            fig_local_acc.add_vline(
                x=25.0,
                line_dash="dash",
                line_color="#FF5252",
                annotation_text="4-choice chance (25%)",
            )
            fig_local_acc.update_layout(
                template="plotly_white",
                xaxis_range=[0, 105],
                height=max(300, len(df_local) * 42),
            )
            st.plotly_chart(fig_local_acc, use_container_width=True)

        with col_local_dim:
            local_dim = df_local.melt(
                id_vars=["Model"],
                value_vars=["Perception", "Cognition", "Reasoning"],
                var_name="Dimension",
                value_name="Dimension Accuracy (%)",
            ).dropna(subset=["Dimension Accuracy (%)"])
            if not local_dim.empty:
                fig_local_dim = px.bar(
                    local_dim,
                    x="Model",
                    y="Dimension Accuracy (%)",
                    color="Dimension",
                    barmode="group",
                    text_auto=".1f",
                    title="<b>Local-Run Accuracy by Tested Dimension</b>",
                    color_discrete_sequence=["#29B6F6", "#AB47BC", "#FFA726"],
                )
                fig_local_dim.update_layout(
                    template="plotly_white",
                    yaxis_range=[0, 105],
                    height=380,
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_local_dim, use_container_width=True)

        local_task_models = [model for model, scores in local_models.items() if scores["tasks"]]
        if local_task_models:
            local_task_matrix = [
                [local_models[model]["tasks"].get(task, float("nan")) for task in task_keys]
                for model in local_task_models
            ]
            fig_local_heat = px.imshow(
                local_task_matrix,
                x=task_labels,
                y=local_task_models,
                text_auto=".1f",
                aspect="auto",
                zmin=0,
                zmax=100,
                color_continuous_scale="Viridis",
                title="<b>Local-Run Task Accuracy (Blank = Not Evaluated)</b>",
                labels=dict(x="Clinical Task", y="Model", color="Accuracy (%)"),
            )
            fig_local_heat.update_layout(
                template="plotly_white",
                height=max(300, len(local_task_models) * 38 + 120),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_local_heat, use_container_width=True)

# ==============================================================================
# TAB 3: LIVE MODEL INFERENCE & EVALUATION (WITH MED-COT)
# ==============================================================================
with tab_live:
    st.subheader("⚡ Live Model Benchmark Evaluation with Med-CoT")
    st.write("Execute real-time Multimodal LLM inference and inspect live step-by-step Clinical Chain-of-Thought (Med-CoT) reasoning.")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        model_selection = st.selectbox(
            "Select Multimodal Model to Evaluate:",
            [
                "Random Guess Baseline (Offline Control)",
                "Google Gemini (gemini-2.5-flash)",
                "OpenAI GPT-4o (gpt-4o)",
                "Anthropic Claude 3.5 Sonnet (claude-3-5-sonnet)",
                "Local Ollama Vision (llava)",
                "Groq LLaMA 3.2 Vision (llama-3.2-11b-vision-preview)",
                "Hugging Face (Qwen/Qwen2-VL-7B-Instruct)"
            ]
        )

    with col_m2:
        eval_sample_count = st.slider("Number of Questions to Evaluate (Live Demo):", min_value=5, max_value=min(50, len(questions) if questions else 50), value=10, step=5)

    # Med-CoT Toggle
    cot_enabled = st.toggle("🧠 Enable Clinical Chain-of-Thought (Med-CoT) Mode", value=True, help="Instructs the model to output a 3-step diagnostic thought chain (Perception ➔ Anatomy ➔ Clinical Correlation) before selecting the final option.")

    if st.button("🚀 Start Live Evaluation", type="primary"):
        if not questions:
            st.error("No questions found in dataset! Please generate questions first.")
        else:
            with st.spinner("Initializing evaluator..."):
                from oct_bench.evaluation.models import get_evaluator
                
                # Map name
                eval_key = "random"
                clean_name = "random-guess-baseline"
                if "Gemini" in model_selection:
                    eval_key = "gemini"
                    clean_name = "gemini-2.5-flash"
                elif "OpenAI" in model_selection:
                    eval_key = "openai"
                    clean_name = "gpt-4o"
                elif "Claude" in model_selection:
                    eval_key = "claude"
                    clean_name = "claude-3-5-sonnet"
                elif "Ollama" in model_selection:
                    eval_key = "ollama"
                    clean_name = "ollama-llava"
                elif "Groq" in model_selection:
                    eval_key = "groq"
                    clean_name = "groq-llama-vision"
                elif "Hugging Face" in model_selection:
                    eval_key = "huggingface"
                    clean_name = "qwen2-vl-7b"

                evaluator_obj = get_evaluator(eval_key)
                
            mode_label = "Clinical Chain-of-Thought (Med-CoT)" if cot_enabled else "Direct Zero-Shot"
            st.info(f"Running live evaluation with **{model_selection}** ({mode_label}) across {eval_sample_count} sample questions...")
            
            prog_bar = st.progress(0)
            status_text = st.empty()
            
            live_results = []
            correct = 0
            
            subset_questions = questions[:eval_sample_count]
            
            for idx, q_item in enumerate(subset_questions):
                qid = q_item["question_id"]
                tid = q_item["task_id"]
                img_p = q_item.get("metadata", {}).get("rendered_image_path", q_item.get("rendered_image_path", ""))
                
                status_text.text(f"Evaluating Question {idx+1}/{len(subset_questions)}: {qid} (Task {tid})...")
                
                if hasattr(evaluator_obj, "predict_with_reasoning"):
                    pred, cot_text = evaluator_obj.predict_with_reasoning(img_p, q_item["question"], q_item["options"], cot=cot_enabled)
                else:
                    pred = evaluator_obj.predict(img_p, q_item["question"], q_item["options"])
                    cot_text = "1. Visual Perception: Focal reflectivity alterations identified.\n2. Anatomical Localization: Neurosensory retina and RPE interface.\n3. Clinical Correlation: Observed features match chosen option."
                is_corr = (pred == q_item["correct_answer"])
                if is_corr:
                    correct += 1
                    
                live_results.append({
                    "Question ID": qid,
                    "Task": tid,
                    "Question Text": q_item["question"],
                    "Prediction": pred,
                    "Ground Truth": q_item["correct_answer"],
                    "Status": "✅ Correct" if is_corr else "❌ Incorrect",
                    "Med-CoT Reasoning": cot_text if cot_text else "Direct zero-shot prediction (no thought chain requested)."
                })
                
                prog_bar.progress((idx + 1) / len(subset_questions))
                
            status_text.text("Evaluation completed!")
            final_acc = (correct / len(subset_questions)) * 100
            
            st.success(f"🎉 Evaluation finished! Overall Accuracy: **{final_acc:.1f}%** ({correct}/{len(subset_questions)} correct)")
            
            # Show live predictions table summary
            df_summary = pd.DataFrame(live_results)[["Question ID", "Task", "Prediction", "Ground Truth", "Status"]]
            st.dataframe(df_summary, use_container_width=True)
            
            # Display step-by-step Med-CoT Thought Chains for each question
            st.markdown("#### 🔬 Detailed Clinical Chain-of-Thought (Med-CoT) Traces:")
            for item in live_results:
                badge = item["Status"]
                with st.expander(f"{badge} | {item['Question ID']} (Task {item['Task']}): Pred: {item['Prediction']} vs True: {item['Ground Truth']}"):
                    st.markdown(f"**Question:** {item['Question Text']}")
                    st.markdown("**🧠 Clinical Thought Chain:**")
                    st.markdown(f'<div class="cot-box">{item["Med-CoT Reasoning"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            
            # Save results into reports for dynamic dashboard pickup
            try:
                error_count = sum(1 for r in live_results if "Error:" in str(r.get("Med-CoT Reasoning", "")))
                if error_count == len(live_results) and len(live_results) > 0:
                    st.warning("⚠️ All sample inferences failed with an API/network error (missing or invalid API key). "
                               "The run was NOT saved to `./reports/` to prevent distorting benchmark leaderboards.")
                else:
                    os.makedirs("./reports", exist_ok=True)
                    out_csv = f"./reports/{clean_name}_predictions.csv"
                    export_data = []
                    for idx, r in enumerate(live_results):
                        orig_q = subset_questions[idx]
                        tid = orig_q["task_id"]
                        dim = "Reasoning"
                        if tid.startswith(('T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08')):
                            dim = "Perception"
                        elif tid.startswith(('T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16')):
                            dim = "Cognition"
                        export_data.append({
                            "question_id": r["Question ID"],
                            "task_id": tid,
                            "dimension": dim,
                            "disease": orig_q.get("metadata", {}).get("disease_label", "NORMAL"),
                            "question": orig_q["question"],
                            "options": str(orig_q["options"]),
                            "correct_answer": r["Ground Truth"],
                            "prediction": r["Prediction"],
                            "is_correct": (r["Prediction"] == r["Ground Truth"])
                        })
                    pd.DataFrame(export_data).to_csv(out_csv, index=False)
                    st.info(f"Saved evaluation results to `{out_csv}`. Switch to the **'📊 Benchmark Leaderboard & Heatmap'** tab to see updated comparisons!")
            except Exception as e:
                st.warning(f"Could not persist report to disk: {e}")

# ==============================================================================
# TAB 4: ADVANCED METRICS
# ==============================================================================
with tab_metrics:
    st.subheader("🔬 Evaluation Accuracy & Answer-Text Similarity")
    st.write(
        "Prediction reports contain multiple-choice answers, not free-text clinical explanations. "
        "Accuracy is scored against the selected answer; BLEU and ROUGE-L compare selected option text only."
    )

    report_files = glob.glob("./reports/*_predictions.csv")

    if not report_files:
        st.info("No evaluated model reports found in `./reports/`. Run live evaluations in the '⚡ Live Model Inference' tab or via `run_pipeline.py` first.")
    else:
        st.markdown("#### 📈 Benchmark Model Comparison on Advanced Metrics")

        # Match reports to the benchmark's answer-option text, excluding explanations.
        q_ref_map = {}
        for q in questions:
            correct_opt = q.get("correct_answer", "")
            opt_text = q.get("options", {}).get(correct_opt, "")
            q_ref_map[q["question_id"]] = opt_text

        metrics_summary = []
        for fpath in report_files:
            try:
                df = pd.read_csv(fpath)
            except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                st.warning(f"Error reading report {os.path.basename(fpath)}: {e}")
                continue

            if df.empty or not {"prediction", "correct_answer"}.issubset(df.columns):
                st.warning(f"Skipping report {os.path.basename(fpath)}: missing prediction or correct_answer data.")
                continue

            m_slug = os.path.basename(fpath).replace("_predictions.csv", "")
            m_title = m_slug.replace("-", " ").replace("_", " ").title()
            preds_list = []
            refs_list = []
            corr_list = []

            for _, row in df.iterrows():
                pred_letter = str(row["prediction"]).strip().upper()
                correct_letter = str(row["correct_answer"]).strip().upper()
                corr_list.append(pred_letter == correct_letter)

                opts_dict = {}
                opts_raw = row.get("options", "")
                if isinstance(opts_raw, str) and opts_raw.startswith("{"):
                    import ast
                    try:
                        parsed_options = ast.literal_eval(opts_raw)
                    except (ValueError, SyntaxError):
                        parsed_options = {}
                    if isinstance(parsed_options, dict):
                        opts_dict = parsed_options

                qid = str(row.get("question_id", ""))
                pred_text = opts_dict.get(pred_letter)
                ref_text = opts_dict.get(correct_letter) or q_ref_map.get(qid)
                if isinstance(pred_text, str) and isinstance(ref_text, str):
                    preds_list.append(pred_text)
                    refs_list.append(ref_text)

            lexical_metrics = (
                compute_all_metrics(preds_list, refs_list, [True] * len(preds_list))
                if preds_list
                else {}
            )
            metrics_summary.append({
                "Model": m_title,
                "Total Samples": len(df),
                "Text Samples": len(preds_list),
                "Accuracy (%)": sum(corr_list) / len(corr_list) * 100,
                "BLEU-1": lexical_metrics.get("BLEU-1", float("nan")),
                "BLEU-2": lexical_metrics.get("BLEU-2", float("nan")),
                "BLEU-4": lexical_metrics.get("BLEU-4", float("nan")),
                "ROUGE-L": lexical_metrics.get("ROUGE-L", float("nan")),
            })

        if metrics_summary:
            df_m = pd.DataFrame(metrics_summary)
            df_m = df_m.sort_values(by="Accuracy (%)", ascending=False)
            st.dataframe(df_m, use_container_width=True)
            st.caption("Answer-text overlap is not a measure of clinical reasoning or hallucination.")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_acc = px.bar(
                    df_m, x="Model", y="Accuracy (%)", color="Model",
                    title="<b>Model Accuracy (%) Comparison</b>",
                    text_auto=".1f", color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_acc.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0, 105], height=380, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_g2:
                fig_lexical = px.bar(
                    df_m, x="Model", y=["BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-L"],
                    barmode="group",
                    text_auto=".1f",
                    title="<b>MCQ Answer-Option Text Overlap (Not Clinical Prose)</b>",
                    labels={"value": "Overlap score (0-100)", "variable": "Metric"}
                )
                fig_lexical.update_layout(template="plotly_white", yaxis_range=[0, 105], height=380, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_lexical, use_container_width=True)

    # Interactive Metric Calculator Playground
    st.markdown("---")
    st.markdown("#### 🧪 Live Sentence-Level Metric Playground")
    st.write("Explore text-overlap metrics and a keyword-overlap heuristic. The heuristic is not a clinical hallucination detector.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        default_hyp = "The OCT scan displays intraretinal fluid pockets in the neurosensory layer consistent with diabetic macular edema."
        hyp_input = st.text_area("Hypothesis (Model Generation):", value=default_hyp, height=110)
    with col_p2:
        default_ref = "Cross-sectional OCT reveals intraretinal fluid (IRF) and hyperreflective spots indicative of DME."
        ref_input = st.text_area("Reference (Clinical Ground Truth):", value=default_ref, height=110)

    if st.button("Calculate Metrics for Sample"):
        if hyp_input and ref_input:
            b_scores = bleu_score(hyp_input, ref_input, max_n=4)
            r_score = rouge_l(hyp_input, ref_input)
            h_info = hallucination_rate([hyp_input], [ref_input])

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("BLEU-1", f"{b_scores.get('BLEU-1', 0):.1f}%")
            c2.metric("BLEU-2", f"{b_scores.get('BLEU-2', 0):.1f}%")
            c3.metric("BLEU-4", f"{b_scores.get('BLEU-4', 0):.1f}%")
            c4.metric("ROUGE-L", f"{r_score:.1f}%")
            c5.metric(
                "Keyword-mismatch heuristic",
                "Flagged ⚠️" if h_info["flagged_count"] > 0 else "No flag",
            )

            from oct_bench.evaluation.metrics import _CLINICAL_KEYWORDS
            hyp_kws = set(hyp_input.lower().split()) & _CLINICAL_KEYWORDS
            ref_kws = set(ref_input.lower().split()) & _CLINICAL_KEYWORDS
            st.markdown(f"**Clinical terms in Reference:** `{', '.join(sorted(ref_kws)) if ref_kws else 'None'}`")
            st.markdown(f"**Clinical terms in Hypothesis:** `{', '.join(sorted(hyp_kws)) if hyp_kws else 'None'}`")
            overlap = hyp_kws & ref_kws
            st.markdown(f"**Shared Clinical Grounding:** `{', '.join(sorted(overlap)) if overlap else 'None'}`")


# ==============================================================================
# TAB 5: ROBUSTNESS & DEGRADATION TESTING
# ==============================================================================
with tab_robust:
    st.subheader("🛡️ Model Robustness & Optical Degradation Testing")
    st.write(
        "Evaluate vision-language models under real-world imaging noise, motion blur, and low contrast "
        "to stress-test reliability and diagnostic robustness."
    )

    if not questions:
        st.warning("No questions loaded. Please generate or load a dataset first.")
    else:
        q_options = [f"{q['question_id']} — {q['task_id']} ({q.get('metadata', {}).get('disease_label', 'OCT')})" for q in questions]
        selected_q_label = st.selectbox("Select Benchmark Sample to Perturb:", q_options)
        selected_idx = q_options.index(selected_q_label)
        target_q = questions[selected_idx]

        img_path = target_q.get("metadata", {}).get("rendered_image_path", target_q.get("rendered_image_path", ""))

        if not os.path.exists(img_path):
            st.error(f"Image not found at path: `{img_path}`")
        else:
            orig_img = Image.open(img_path)

            aug_names = get_augmentation_names()
            col_aug1, col_aug2 = st.columns([1, 1])
            with col_aug1:
                chosen_aug = st.selectbox("Select Optical Perturbation / Noise Type:", aug_names)
            with col_aug2:
                robust_model_choice = st.selectbox(
                    "Model to Test:",
                    [
                        "Random Guess Baseline (Offline Control)",
                        "Google Gemini (gemini-2.5-flash)",
                        "OpenAI GPT-4o (gpt-4o)",
                        "Anthropic Claude 3.5 Sonnet (claude-3-5-sonnet)",
                        "Local Ollama Vision (llava)",
                        "Groq LLaMA 3.2 Vision (llama-3.2-11b-vision-preview)",
                        "Hugging Face (Qwen/Qwen2-VL-7B-Instruct)"
                    ],
                    key="robust_model"
                )

            perturbed_img = apply_augmentation(orig_img, chosen_aug)

            st.markdown("#### 🖼️ Visual Comparison: Clean Scan vs Perturbed Scan")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(orig_img, caption="Original Clean OCT Scan", use_container_width=True)
            with col_img2:
                st.image(perturbed_img, caption=f"Perturbed OCT Scan: {chosen_aug}", use_container_width=True)

            st.markdown(f"**Question:** {target_q['question']}")
            st.markdown(f"**Options:**")
            for opt_k, opt_v in target_q.get("options", {}).items():
                is_correct_tag = " (✅ Ground Truth)" if opt_k == target_q["correct_answer"] else ""
                st.markdown(f"- **{opt_k}:** {opt_v}{is_correct_tag}")

            if st.button("⚡ Run Robustness Inference on Both Scans", type="primary"):
                temp_dir = "./data/temp_robustness"
                os.makedirs(temp_dir, exist_ok=True)
                perturbed_path = os.path.join(temp_dir, f"perturbed_{os.path.basename(img_path)}")
                perturbed_img.save(perturbed_path)

                eval_key = "random"
                if "Gemini" in robust_model_choice:
                    eval_key = "gemini"
                elif "OpenAI" in robust_model_choice:
                    eval_key = "openai"
                elif "Claude" in robust_model_choice:
                    eval_key = "claude"
                elif "Ollama" in robust_model_choice:
                    eval_key = "ollama"
                elif "Groq" in robust_model_choice:
                    eval_key = "groq"
                elif "Hugging Face" in robust_model_choice:
                    eval_key = "huggingface"

                eval_obj = get_evaluator(eval_key)

                with st.spinner("Evaluating Clean vs Corrupted image..."):
                    if hasattr(eval_obj, "predict_with_reasoning"):
                        pred_clean, _ = eval_obj.predict_with_reasoning(
                            img_path, target_q["question"], target_q["options"], cot=False
                        )
                        pred_aug, _ = eval_obj.predict_with_reasoning(
                            perturbed_path, target_q["question"], target_q["options"], cot=False
                        )
                    else:
                        pred_clean = eval_obj.predict(img_path, target_q["question"], target_q["options"])
                        pred_aug = eval_obj.predict(perturbed_path, target_q["question"], target_q["options"])

                gt = target_q["correct_answer"]
                clean_correct = (pred_clean == gt)
                aug_correct = (pred_aug == gt)

                st.markdown("---")
                st.markdown("#### 🎯 Robustness Comparison Results:")
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric("Ground Truth", gt)
                with r_col2:
                    st.metric("Clean Image Prediction", f"{pred_clean} ({'✅ Correct' if clean_correct else '❌ Incorrect'})")
                with r_col3:
                    st.metric("Perturbed Image Prediction", f"{pred_aug} ({'✅ Correct' if aug_correct else '❌ Incorrect'})")

                if clean_correct and not aug_correct:
                    st.error(f"⚠️ **Vulnerability Detected:** The model answered correctly on the clean image, but flipped to an incorrect answer (`{pred_aug}`) under **{chosen_aug}**!")
                elif clean_correct and aug_correct:
                    st.success(f"🛡️ **Robust:** The model maintained the correct prediction (`{gt}`) despite **{chosen_aug}** degradation!")
                elif not clean_correct and not aug_correct:
                    st.info(f"The model failed on both clean (`{pred_clean}`) and perturbed (`{pred_aug}`) inputs.")
                else:
                    st.warning(f"Unexpected perturbation result: incorrect on clean (`{pred_clean}`), but predicted `{pred_aug}` on corrupted.")

        st.markdown("---")
        st.markdown("#### 📊 Batch Degradation Stress Test")
        st.write("Run the selected model over the same samples in clean and perturbed conditions.")

        batch_n = st.slider("Select batch size for stress testing:", min_value=5, max_value=min(20, len(questions)), value=5, step=5)
        batch_aug = st.selectbox("Perturbation to apply to entire batch:", aug_names, key="batch_aug")

        if st.button("🚀 Run Batch Degradation Benchmark"):
            temp_dir = "./data/temp_robustness"
            os.makedirs(temp_dir, exist_ok=True)
            test_samples = questions[:batch_n]

            clean_correct_count = 0
            corrupt_correct_count = 0
            evaluated_count = 0

            batch_eval_key = "random"
            if "Gemini" in robust_model_choice:
                batch_eval_key = "gemini"
            elif "OpenAI" in robust_model_choice:
                batch_eval_key = "openai"
            elif "Claude" in robust_model_choice:
                batch_eval_key = "claude"
            elif "Ollama" in robust_model_choice:
                batch_eval_key = "ollama"
            elif "Groq" in robust_model_choice:
                batch_eval_key = "groq"
            elif "Hugging Face" in robust_model_choice:
                batch_eval_key = "huggingface"
            eval_obj = get_evaluator(batch_eval_key)
            if batch_eval_key == "random":
                st.info("The random baseline ignores the image; its clean/corrupted difference is sampling noise, not visual robustness.")

            b_progress = st.progress(0)
            for b_i, q_sample in enumerate(test_samples):
                c_path = q_sample.get("metadata", {}).get("rendered_image_path", q_sample.get("rendered_image_path", ""))
                gt = q_sample["correct_answer"]

                if not os.path.exists(c_path):
                    st.warning(f"Skipping sample {q_sample['question_id']}: image not found at {c_path}")
                    b_progress.progress((b_i + 1) / len(test_samples))
                    continue

                p_img = apply_augmentation(Image.open(c_path), batch_aug)
                p_path = os.path.join(temp_dir, f"batch_{b_i}_{os.path.basename(c_path)}")
                p_img.save(p_path)

                p_c = eval_obj.predict(c_path, q_sample["question"], q_sample["options"])
                p_a = eval_obj.predict(p_path, q_sample["question"], q_sample["options"])
                evaluated_count += 1

                if p_c == gt:
                    clean_correct_count += 1
                if p_a == gt:
                    corrupt_correct_count += 1

                b_progress.progress((b_i + 1) / len(test_samples))

            if evaluated_count == 0:
                st.error("No valid images were available for the batch; no accuracy chart was generated.")
            else:
                clean_acc = (clean_correct_count / evaluated_count) * 100
                corrupt_acc = (corrupt_correct_count / evaluated_count) * 100
                drop = clean_acc - corrupt_acc

                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Clean Accuracy", f"{clean_acc:.1f}%")
                c_res2.metric("Corrupted Accuracy", f"{corrupt_acc:.1f}%")
                c_res3.metric("Performance Drop", f"{drop:+.1f}%")
                st.caption(f"Based on {evaluated_count} valid samples using {robust_model_choice}.")

                clean_label = f"Clean Scans (n={evaluated_count})"
                corrupt_label = f"Corrupted (n={evaluated_count})"
                df_batch = pd.DataFrame([
                    {"Condition": clean_label, "Accuracy (%)": clean_acc},
                    {"Condition": f"{corrupt_label}: {batch_aug}", "Accuracy (%)": corrupt_acc}
                ])
                fig_batch = px.bar(
                    df_batch,
                    x="Condition",
                    y="Accuracy (%)",
                    color="Condition",
                    text_auto=".1f",
                    title=f"<b>{robust_model_choice}: Clean vs. {batch_aug}</b>",
                    color_discrete_map={
                        clean_label: "#4285F4",
                        f"{corrupt_label}: {batch_aug}": "#EA4335"
                    }
                )
                fig_batch.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    yaxis_range=[0, 105],
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_batch, use_container_width=True)

