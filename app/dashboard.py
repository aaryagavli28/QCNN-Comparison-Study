import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="QCNN Comparison Study",
    page_icon="⚛️",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("metrics_final.csv")

try:
    with open("confusion_matrices.json", "r") as f:
        confusion_data = json.load(f)
except:
    confusion_data = {}

# ==========================================
# HEADER
# ==========================================

st.title("⚛️ Quantum CNN Comparison Study")

st.markdown("""
### Comparative Analysis of Hybrid Quantum Convolutional Neural Networks

This dashboard presents a comparative study of multiple Quantum CNN
architectures implemented using PennyLane and PyTorch on the MNIST-179 dataset.
""")

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("QCNN Dashboard")

st.sidebar.markdown("""
### Project Information

**Dataset**
- MNIST-179

**Frameworks**
- PennyLane
- PyTorch
- Streamlit

**Models**
- Single Encoding
- Multi Encoding
- Multi Noisy
- Inception
""")

st.sidebar.divider()

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["All"] + list(df["model"].unique())
)

# ==========================================
# FILTER
# ==========================================

if selected_model != "All":
    display_df = df[df["model"] == selected_model]
else:
    display_df = df

# ==========================================
# KPI SECTION
# ==========================================

best_model = df.loc[df["val_acc"].idxmax()]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🏆 Best Model",
        best_model["model"]
    )

with col2:
    st.metric(
        "📈 Best Accuracy",
        f"{best_model['val_acc']*100:.2f}%"
    )

with col3:
    st.metric(
        "📊 Dataset",
        "MNIST-179"
    )

st.divider()

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Results",
    "⏱ Runtime",
    "🔲 Confusion Matrix",
    "🧠 Findings",
    "⚙️ Methodology",
    "📚 References"
])

# ==========================================
# RESULTS TAB
# ==========================================

with tab1:

    st.subheader("Experiment Results")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.download_button(
        label="⬇ Download Results CSV",
        data=df.to_csv(index=False),
        file_name="metrics_final.csv",
        mime="text/csv"
    )

    st.subheader("Validation Accuracy Comparison")

    fig_acc = px.bar(
        display_df,
        x="model",
        y="val_acc",
        text="val_acc",
        color="val_acc",
        title="Validation Accuracy"
    )

    fig_acc.update_layout(height=500)

    st.plotly_chart(
        fig_acc,
        use_container_width=True
    )

    st.subheader("Validation Loss Comparison")

    fig_loss = px.bar(
        display_df,
        x="model",
        y="val_loss",
        text="val_loss",
        color="val_loss",
        title="Validation Loss"
    )

    fig_loss.update_layout(height=500)

    st.plotly_chart(
        fig_loss,
        use_container_width=True
    )

    st.subheader("🏆 Model Leaderboard")

    ranking = df.sort_values(
        "val_acc",
        ascending=False
    )

    st.dataframe(
        ranking[
            [
                "model",
                "val_acc",
                "val_loss"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# RUNTIME TAB
# ==========================================

with tab2:

    runtime_df = pd.DataFrame({
        "model": [
            "single_encoding",
            "multi_encoding",
            "multi_noisy",
            "inception"
        ],
        "runtime_minutes": [
            65,
            34,
            230,
            37
        ],
        "accuracy": [
            0.7958,
            0.8333,
            0.8292,
            0.9208
        ]
    })

    st.subheader("Training Runtime Comparison")

    fig_runtime = px.bar(
        runtime_df,
        x="model",
        y="runtime_minutes",
        color="runtime_minutes",
        text="runtime_minutes"
    )

    fig_runtime.update_layout(height=500)

    st.plotly_chart(
        fig_runtime,
        use_container_width=True
    )

    st.subheader("Accuracy vs Runtime")

    fig_scatter = px.scatter(
        runtime_df,
        x="runtime_minutes",
        y="accuracy",
        size="accuracy",
        color="model",
        hover_name="model"
    )

    fig_scatter.update_layout(height=600)

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

# ==========================================
# CONFUSION MATRIX TAB
# ==========================================

with tab3:

    st.subheader("Confusion Matrix")

    if confusion_data:

        dataset_key = list(confusion_data.keys())[0]

        model_options = list(
            confusion_data[dataset_key].keys()
        )

        selected_cm_model = st.selectbox(
            "Select Model",
            model_options
        )

        cm = confusion_data[
            dataset_key
        ][selected_cm_model]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig_cm.update_layout(
            title=f"Confusion Matrix - {selected_cm_model}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=600
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

    else:

        st.warning(
            "No confusion matrix file found."
        )

# ==========================================
# FINDINGS TAB
# ==========================================

with tab4:

    st.success("""
    Best Performing Model:
    Inception QCNN
    (Validation Accuracy = 92.08%)
    """)

    st.markdown("""
    ## Key Findings

    - Inception achieved the highest validation accuracy.
    - Multi Encoding achieved strong performance with low runtime.
    - Multi Noisy produced competitive accuracy but required substantial computational resources.
    - Single Encoding provided the simplest quantum architecture.
    - Quantum circuit simulation remains computationally expensive.
    - Hybrid quantum-classical architectures show strong potential for image classification tasks.
    """)

# ==========================================
# METHODOLOGY TAB
# ==========================================

with tab5:

    st.markdown("""
    ## Experimental Methodology

    Dataset
    ↓

    MNIST-179

    ↓

    Train / Validation Split

    ↓

    Quantum Feature Encoding

    ↓

    QCNN Architecture

    ↓

    Training using PennyLane + PyTorch

    ↓

    Performance Evaluation

    ↓

    Comparative Analysis
    """)

# ==========================================
# REFERENCES TAB
# ==========================================

with tab6:

    st.markdown("""
    ## Research References

    ### Quantum Convolutional Neural Networks
    Cong, Choi, Lukin (2019)

    ### Quanvolutional Neural Networks
    Henderson et al. (2019)

    ### Hybrid Quantum-Classical Convolutional Neural Networks

    ### Quantum CNNs for Image Classification

    ### Frameworks Used

    - PennyLane
    - PyTorch
    - Streamlit
    - Plotly
    """)
