# crop_trend_prediction_app_multi_model_fixed.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import base64

# --- Utility to get image base64 ---
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64(r"D:\Guvi\Guvi projects\Crop production project\agriculture.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(127, 0, 255, 0.1); /* Light purple transparent background */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱Crops Prediction🌱")
st.markdown(f'<h1 style="color:#7F00FF;font-size:24px;"><center>{"Explore Different Models"}</h1>', unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel(r"D:\Guvi\Guvi projects\Crop production project\Afterdataclean.xlsx", index_col=0, engine='openpyxl')
    return df

df = load_data()
df = df[df['flag_description'] == 'Official figure']

# --- Pivot Table ---
df_pivot = df.pivot_table(index=['area', 'item', 'year'], columns='element', values='value').reset_index()

# Expected columns
expected_cols = ['area harvested', 'yield', 'production']
existing_cols = [col for col in expected_cols if col in df_pivot.columns]

if not existing_cols:
    st.error("⚠️ Expected element columns not found in pivot table. Please check your data.")
    st.stop()
else:
    df_pivot.dropna(subset=expected_cols, inplace=True)

# Rename columns
df_pivot = df_pivot.rename(columns={
    'area harvested': 'area_harvested',
    'yield': 'yield_kg_ha',
    'production': 'production_tonnes'
})

# --- Feature Engineering ---
model_df = df_pivot.copy()
X = model_df[['area', 'item', 'year', 'area_harvested', 'yield_kg_ha']].copy()
y = model_df['production_tonnes']

# Encode
le_area = LabelEncoder()
le_item = LabelEncoder()
X['area'] = le_area.fit_transform(X['area'])
X['item'] = le_item.fit_transform(X['item'])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Sidebar Model Selection ---
st.sidebar.header("🔍 Model Selection")
model_option = st.sidebar.selectbox(
    "Choose a model:", 
    ["Select a model", 'Linear Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest']
)

if model_option == "Select a model":
    st.warning("Please select a model from the sidebar to train and view results.")
else:
    if model_option == 'Linear Regression':
        model = LinearRegression()
    elif model_option == 'K-Nearest Neighbors':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_option == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)
    elif model_option == 'Random Forest':
        model = RandomForestRegressor(random_state=42)

    # --- Train ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"""
    #### 📊 **Model Results: {model_option}**
    - 📌 **Root Mean Squared Error (RMSE):** `{rmse:,.2f}`
    - 📌 **R² Score:** `{r2:.4f}`
    ---
    """)

    # Optional debug
    # if st.sidebar.checkbox("Show debug info"):
    #     st.write("✅ Columns after pivot:", df_pivot.columns)

    # --- Actual vs Predicted Plot ---
    st.markdown("### 🌟 Actual vs Predicted Production")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, color='teal', alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("Actual Production")
    ax1.set_ylabel("Predicted Production")
    ax1.set_title("Actual vs Predicted")
    st.pyplot(fig1)

    # --- Feature Importance ---
    if model_option in ['Decision Tree', 'Random Forest']:
        st.markdown("### 🌳 Feature Importance")
        feature_names = ['area', 'item', 'year', 'area_harvested', 'yield_kg_ha']
        importance = model.feature_importances_
        fig2, ax2 = plt.subplots()
        ax2.barh(feature_names, importance, color='slateblue')
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Feature Importance")
        st.pyplot(fig2)

    # --- Year-wise Trend ---
    st.markdown("### 📅 Year-wise Production Trend (Average)")
    test_df = pd.DataFrame(X_test, columns=['area', 'item', 'year', 'area_harvested', 'yield_kg_ha'])
    test_df['Actual'] = y_test.values
    test_df['Predicted'] = y_pred
    year_trend = test_df.groupby('year')[['Actual', 'Predicted']].mean().reset_index()

    fig3, ax3 = plt.subplots()
    ax3.plot(year_trend['year'], year_trend['Actual'], label='Actual', marker='o')
    ax3.plot(year_trend['year'], year_trend['Predicted'], label='Predicted', marker='s')
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Average Production")
    ax3.set_title("Year-wise Average Production")
    ax3.legend()
    st.pyplot(fig3)

    # --- Top 10 Countries ---
    st.markdown("### 🌍 Top 10 Countries by Average Production")
    top_countries = model_df.groupby('area')['production_tonnes'].mean().sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots()
    top_countries.plot(kind='bar', color='mediumseagreen', ax=ax4)
    ax4.set_ylabel("Average Production (tonnes)")
    ax4.set_title("Top 10 Countries by Average Production")
    st.pyplot(fig4)

    # --- Top 10 Crops ---
    st.markdown("### 🥦 Crop-wise Production Distribution")
    crop_dist = model_df.groupby('item')['production_tonnes'].sum().sort_values(ascending=False).head(10)
    fig5, ax5 = plt.subplots()
    crop_dist.plot(kind='barh', color='coral', ax=ax5)
    ax5.set_xlabel("Total Production (tonnes)")
    ax5.set_title("Top 10 Crops by Total Production")
    st.pyplot(fig5)

    # --- Custom Prediction ---
    st.sidebar.header("🔮 Predict Your Own")
    area_input = st.sidebar.number_input("🌾 Area harvested (ha):", min_value=0.0, value=1000.0)
    yield_input = st.sidebar.number_input("📈 Yield (kg/ha):", min_value=0.0, value=2000.0)
    year_input = st.sidebar.number_input("📅 Year:", min_value=1960, max_value=2050, value=2020)
    area_cat = st.sidebar.selectbox("🌍 Country:", sorted(model_df['area'].unique()))
    item_cat = st.sidebar.selectbox("🥕 Crop:", sorted(model_df['item'].unique()))

    if st.sidebar.button("🚀 Predict Production"):
        area_encoded = le_area.transform([area_cat])[0]
        item_encoded = le_item.transform([item_cat])[0]
        custom_input = scaler.transform([[area_encoded, item_encoded, year_input, area_input, yield_input]])
        prediction = model.predict(custom_input)
        st.sidebar.success(f"Estimated Production: {prediction[0]:,.2f} tons")
        st.success(f"Estimated Production for your input: {prediction[0]:,.2f} tons")
