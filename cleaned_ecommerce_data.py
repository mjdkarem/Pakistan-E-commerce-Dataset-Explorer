import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# --- Page Config ---
st.set_page_config(page_title="Pakistan E-commerce EDA", layout="wide")

# --- Title ---
st.title("Pakistan E-commerce Dataset Explorer")

# --- Read Dataset ---
zip_path = "cleaned_ecommerce_data.zip"  # اسم ملف المضغوط
csv_filename = "cleaned_ecommerce_data.csv"  # اسم ملف CSV داخل المضغوط

@st.cache_data
def get_clean_data():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(csv_filename) as file:
            df = pd.read_csv(file, low_memory=True, engine="python", encoding='utf-8', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['order_month'] = df['created_at'].dt.to_period('M')
        df['order_date'] = df['created_at'].dt.date
    if 'grand_total' in df.columns:
        df['grand_total'] = pd.to_numeric(df['grand_total'].astype(str).str.replace(',', ''), errors='coerce')
    return df

# --- Load Data ---
df = get_clean_data()

# --- Styling Function ---
def minimalist_dark_chart(ax, title=""):
    ax.set_facecolor("#111")
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.title.set_color('white')
    ax.title.set_fontsize(12)
    ax.set_title(title, pad=10)
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')

# --- Tabs ---
tabs = st.tabs(["Overview & KPIs", "Visual Analysis", "Deeper Analysis", "Conclusion & Insights", "Credits"])

# --- Tab 1: Overview ---
with tabs[0]:
    st.subheader("Overview & Key Performance Indicators (KPIs)")
    st.markdown("""
    This project explores over 500,000 e-commerce transactions in Pakistan (March 2016 - August 2018).

    **Target audience**: Business analysts, data scientists, e-commerce professionals, and startups.
    """)

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"{df['grand_total'].sum():,.0f}" if 'grand_total' in df.columns else "N/A")
        with col2:
            st.metric("Total Orders", f"{df['increment_id'].nunique():,}" if 'increment_id' in df.columns else "N/A")
        with col3:
            st.metric("Total Customers", f"{df['customer_id'].nunique():,}" if 'customer_id' in df.columns else "N/A")
        with col4:
            avg_order = df.groupby('increment_id')['grand_total'].sum().mean() if 'grand_total' in df.columns and 'increment_id' in df.columns else np.nan
            st.metric("Avg. Order Value", f"{avg_order:,.2f}" if not np.isnan(avg_order) else "N/A")

        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("Dataset is empty or not loaded properly.")

# --- Tab 2: Visual Analysis ---
with tabs[1]:
    st.subheader("Visual Exploration")

    if not df.empty and 'created_at' in df.columns:
        year_filter = st.selectbox("Select Year", options=sorted(df['created_at'].dt.year.dropna().unique().astype(int)))
        filtered_df = df[df['created_at'].dt.year == year_filter]

        with plt.style.context('dark_background'):
            charts = []

            # Monthly Revenue Trend
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            filtered_df.groupby('order_month')['grand_total'].sum().plot(ax=ax1)
            minimalist_dark_chart(ax1, "Monthly Revenue Trend")
            charts.append((fig1, "Seasonal revenue trend."))

            # Orders by Status
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            filtered_df['status'].value_counts().plot(kind='bar', ax=ax2)
            minimalist_dark_chart(ax2, "Orders by Status")
            charts.append((fig2, "Distribution of order statuses."))

            # Payment Methods
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            filtered_df['payment_method'].value_counts().plot(kind='bar', ax=ax3)
            minimalist_dark_chart(ax3, "Payment Methods")
            charts.append((fig3, "Payment method breakdown."))

            # Product Categories
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            filtered_df['category_name_1'].value_counts().head(10).plot(kind='bar', ax=ax4)
            minimalist_dark_chart(ax4, "Top 10 Product Categories")
            charts.append((fig4, "Top product categories."))

            # Orders by Day of Week
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            filtered_df['day_of_week'] = filtered_df['created_at'].dt.day_name()
            filtered_df['day_of_week'].value_counts().loc[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].plot(kind='bar', ax=ax5)
            minimalist_dark_chart(ax5, "Orders by Day of the Week")
            charts.append((fig5, "Which days drive most sales?"))

            # Revenue by Payment Method
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            filtered_df.groupby('payment_method')['grand_total'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax6)
            minimalist_dark_chart(ax6, "Revenue by Payment Method")
            charts.append((fig6, "Revenue by each payment option."))

            # Display charts 2 by 2
            for i in range(0, len(charts), 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(charts[i][0])
                    st.caption(charts[i][1])
                if i+1 < len(charts):
                    with col2:
                        st.pyplot(charts[i+1][0])
                        st.caption(charts[i+1][1])

# --- Tab 3: Deeper Analysis ---
with tabs[2]:
    st.subheader("Deeper Analysis")

    if not df.empty and 'category_name_1' in df.columns and 'order_month' in df.columns:
        heatmap_df = df[df['category_name_1'].notna()]
        heatmap_df = heatmap_df.groupby(['order_month', 'category_name_1'])['grand_total'].sum().unstack().fillna(0)

        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap_df.T, cmap="coolwarm", ax=ax, cbar_kws={'shrink': 0.5}, linewidths=0.1)
            minimalist_dark_chart(ax, "Monthly Revenue Heatmap by Category")
            st.pyplot(fig)
            st.caption("Shows monthly revenue distribution across categories.")
    else:
        st.warning("Required columns not found for deeper analysis.")

# --- Tab 4: Conclusion ---
with tabs[3]:
    st.subheader("Conclusion & Insights")
    st.markdown("""
    **Key Takeaways:**

    - Sales vary seasonally, peaking around major holidays.
    - A few product categories dominate revenue.
    - Cash on Delivery remains the most preferred payment method.
    - Orders are concentrated around midweek and weekends.
    """)

# --- Tab 5: Credits ---
with tabs[4]:
    st.subheader("About This Project")
    st.markdown("""
    This project explores over 500,000 e-commerce transactions in Pakistan between March 2016 and August 2018.  
    It reveals emerging patterns in digital retail, uncovering trends in customer behavior, category preferences, and order timing.

    The dataset provides rare access to real-world data in a developing e-commerce market.  
    Insights from this analysis can guide business strategies and startup innovations.

    **Target audience**: Business analysts, data scientists, e-commerce professionals, and startups exploring Pakistan’s digital commerce potential.

    ---

    **Streamlit App created by**: Majd Sidawi  
    **Email**: majdawi@outlook.com  
    **LinkedIn**: [Majd Sidawi](https://linkedin.com/in/majd-sidawi-6541701a0/)  

    **Dataset Source**: [Pakistan’s Largest E-commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset)

    ---

    ### 📊 Bonus Visualization: Tableau Dashboard
    Dive deeper into the data with an interactive Tableau dashboard here:  
    👉 [View on Tableau Public](https://public.tableau.com/app/profile/majd.si/viz/PakistanE-commerceDatasetExplorer/Dashboard1)
    """)
