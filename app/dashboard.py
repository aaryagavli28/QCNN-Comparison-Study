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

    st.header("📚 Research References")

    references = [
        "[1] H. P. Kanna, P. Goriparthi, K. S. Rajasekhar, and A. Kolusu, 'Performance Evaluation of Quantum Encoding Techniques in QCNN for Bone Marrow Cell Classification,' in Proc. Int. Conf. Computing, Communication, Control and Cyber-Physical Systems (I5CPS), IEEE, 2026.",

        "[2] M. E. Sahin, E. Altamura, O. Wallis, S. P. Wood, A. Dekusar, D. A. Millar, T. Imamichi, A. Matsuo, and S. Mensa, 'Qiskit Machine Learning: An Open-Source Library for Quantum Machine Learning Tasks at Scale on Quantum Hardware and Classical Simulators,' arXiv preprint arXiv:2505.17756, 2025.",

        "[3] K. Zaman, A. Marchisio, M. A. Hanif, and M. Shafique, 'A Survey on Quantum Machine Learning: Basics, Current Trends, Challenges, Opportunities, and the Road Ahead,' arXiv preprint arXiv:2310.10315, 2025.",

        "[4] F. Fan, Y. Shi, T. Guggenmos, and X. X. Zhu, 'Hybrid Quantum-Classical Convolutional Neural Network Model for Image Classification,' IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 12, pp. 18145–18159, Dec. 2024.",

        "[5] P. Easom-McCaldin, A. Bouridane, A. Belatreche, R. Jiang, and S. Al-Maadeed, 'Efficient Quantum Image Classification Using Single-Qubit Encoding,' IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 2, pp. 1472–1486, Feb. 2024.",

        "[6] Y. Song, J. Li, Y. Wu, S. Qin, Q. Wen, and F. Gao, 'A Resource-Efficient Quantum Convolutional Neural Network,' Frontiers in Physics, vol. 12, Art. no. 1362690, 2024.",

        "[7] K. Zaman, T. Ahmed, M. A. Hanif, A. Marchisio, and M. Shafique, 'A Comparative Analysis of Hybrid-Quantum Classical Neural Networks,' in Proc. 3rd Int. Conf. Emergent Quantum Technologies (ICEQT), 2024.",

        "[8] E. H. Houssein, Z. Abohashima, M. Elhoseny, and W. M. Mohamed, 'Hybrid Quantum-Classical Convolutional Neural Network Model for COVID-19 Prediction Using Chest X-Ray Images,' Journal of Computational Design and Engineering, vol. 9, no. 2, pp. 343–363, 2022.",

        "[9] V. Bergholm, J. Izaac, M. Schuld, C. Gogolin, S. Ahmed, V. Ajith, N. Killoran, et al., 'PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations,' arXiv preprint arXiv:1811.04968, 2022.",

        "[10] M. Broughton, G. Verdon, T. McCourt, A. J. Martinez, J. H. Yoo, S. V. Isakov, P. Massey, et al., 'TensorFlow Quantum: A Software Framework for Quantum Machine Learning,' arXiv preprint arXiv:2003.02989, 2021.",

        "[11] H. Y. Huang, R. Kueng, and J. Preskill, 'Information-Theoretic Bounds on Quantum Advantage in Machine Learning,' Physical Review Letters, vol. 126, no. 19, Art. no. 190505, 2021.",

        "[12] M. Henderson, S. Shakya, S. Pradhan, and T. Cook, 'Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits,' Quantum Machine Intelligence, vol. 2, no. 2, pp. 1–9, 2020.",

        "[13] S. Oh, J. Choi, and J. Kim, 'A Tutorial on Quantum Convolutional Neural Networks (QCNN),' arXiv preprint arXiv:2009.09423, 2020."
    ]

    st.info(
        "The following references were used during the literature review and implementation of the QCNN comparative analysis study."
    )

    for ref in references:
        st.markdown(f"**{ref}**")
        st.divider()

    st.success("Total References Used: 13")
