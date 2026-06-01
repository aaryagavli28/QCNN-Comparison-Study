import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="QCNN Comparison Study",
    layout="wide"
)

st.title("Quantum CNN Comparison Study")

st.markdown("""
Comparison of hybrid quantum-classical CNN architectures on the MNIST-179 dataset.
""")

df = pd.read_csv("metrics_final.csv")

st.header("Results Table")
st.dataframe(df)

st.header("Validation Accuracy Comparison")

fig_acc = px.bar(
    df,
    x="model",
    y="val_acc",
    text="val_acc",
    title="Validation Accuracy"
)

st.plotly_chart(fig_acc, use_container_width=True)

st.header("Validation Loss Comparison")

fig_loss = px.bar(
    df,
    x="model",
    y="val_loss",
    text="val_loss",
    title="Validation Loss"
)

st.plotly_chart(fig_loss, use_container_width=True)

st.header("Model Ranking")

ranking = df.sort_values(
    "val_acc",
    ascending=False
)

for idx, row in enumerate(
    ranking.itertuples(),
    start=1
):
    st.write(
        f"{idx}. {row.model} — {row.val_acc:.4f}"
    )

best = ranking.iloc[0]

st.success(
    f"Best Model: {best['model']} "
    f"(Validation Accuracy = {best['val_acc']:.4f})"
)
