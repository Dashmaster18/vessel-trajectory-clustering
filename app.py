import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AIS Vessel Data Analysis Dashboard",
    layout="wide"
)

st.title("🚢 AIS Vessel Data Analysis Dashboard")

uploaded_file = st.file_uploader(
    "Upload AIS CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Missing Values
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    ))

    # Remove Missing Values
    df = df.dropna()

    # Remove Duplicates
    duplicates = df.duplicated().sum()

    st.subheader("Duplicate Records")
    st.write(f"Duplicates Found: {duplicates}")

    df = df.drop_duplicates()

    # Summary Statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # Numeric Columns
    numerical_df = df.select_dtypes(include=["number"])

    if len(numerical_df.columns) > 0:

        # Histograms
        st.subheader("Feature Distributions")

        fig = numerical_df.hist(
            figsize=(15, 10),
            bins=20
        )

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            numerical_df.corr(),
            cmap="coolwarm",
            annot=False,
            ax=ax
        )

        st.pyplot(fig)

        # Boxplots
        st.subheader("Outlier Detection")

        selected_column = st.selectbox(
            "Select Column",
            numerical_df.columns
        )

        fig, ax = plt.subplots(figsize=(10, 3))

        sns.boxplot(
            x=df[selected_column],
            ax=ax
        )

        ax.set_title(
            f"Boxplot of {selected_column}"
        )

        st.pyplot(fig)

        # PCA
        st.subheader("PCA Visualization")

        X_scaled = StandardScaler().fit_transform(
            numerical_df
        )

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            X_pca,
            columns=["PC1", "PC2"]
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            ax=ax
        )

        ax.set_title(
            "PCA Projection of AIS Features"
        )

        st.pyplot(fig)

        st.subheader("Explained Variance Ratio")

        st.write(
            pd.DataFrame({
                "Component": ["PC1", "PC2"],
                "Explained Variance":
                pca.explained_variance_ratio_
            })
        )

    else:
        st.warning(
            "No numerical columns found for analysis."
        )

else:
    st.info(
        "Please upload an AIS CSV file to begin analysis."
    )
