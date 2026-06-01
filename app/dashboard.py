import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="QCNN Comparison Study",
    page_icon="⚛️",
    layout="wide"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv("metrics_final.csv")

# ----------------------------------
# HEADER
# ----------------------------------

st.title("⚛️ Quantum CNN Comparison Study")

st.markdown("""
Comparative analysis of hybrid Quantum Convolutional Neural Networks (QCNNs)
implemented using PennyLane and PyTorch on the MNIST-179 dataset.
""")

st.divider()

# ----------------------------------
# KPI SECTION
# ----------------------------------

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

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("QCNN Dashboard")

st.sidebar.info("""
### Project Information

Dataset:
MNIST-179

Framework:
- PennyLane
- PyTorch
- Streamlit

Models:
- Single Encoding
- Multi Encoding
- Multi Noisy
- Inception
""")

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Results",
        "⏱ Runtime",
        "🧠 Findings",
        "📚 References"
    ]
)

# ==================================
# RESULTS TAB
# ==================================

with tab1:

    st.subheader("Experiment Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Validation Accuracy")

    fig_acc = px.bar(
        df,
        x="model",
        y="val_acc",
        text="val_acc",
        color="val_acc",
        title="Validation Accuracy Comparison"
    )

    fig_acc.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_acc,
        use_container_width=True
    )

    st.subheader("Validation Loss")

    fig_loss = px.bar(
        df,
        x="model",
        y="val_loss",
        text="val_loss",
        color="val_loss",
        title="Validation Loss Comparison"
    )

    fig_loss.update_layout(
        height=500
    )

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

# ==================================
# RUNTIME TAB
# ==================================

with tab2:

    runtime_df = pd.DataFrame(
        {
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
            ]
        }
    )

    st.subheader(
        "Training Runtime Comparison"
    )

    fig_runtime = px.bar(
        runtime_df,
        x="model",
        y="runtime_minutes",
        color="runtime_minutes",
        text="runtime_minutes"
    )

    fig_runtime.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_runtime,
        use_container_width=True
    )

# ==================================
# FINDINGS TAB
# ==================================

with tab3:

    st.subheader("Research Findings")

    st.success("""
    Inception achieved the highest validation accuracy
    among all tested QCNN architectures.
    """)

    st.markdown("""
    ### Key Observations

    - Inception achieved the best performance (92.08%)
    - Multi Encoding achieved competitive performance
      with significantly lower runtime
    - Multi Noisy required the highest computational cost
    - Quantum circuit simulation remains expensive
    - Hybrid quantum-classical architectures show
      promising classification performance
    """)

# ==================================
# REFERENCES TAB
# ==================================

with tab4:

    st.subheader("Research Papers")

    st.markdown("""
    ### Quantum Convolutional Neural Networks

    Cong, Choi, Lukin (2019)

    ### Quanvolutional Neural Networks

    Henderson et al. (2019)

    ### Hybrid Quantum-Classical CNN Models

    Fan et al.

    ### Quantum CNNs for Image Classification

    Lü et al.
    """)
