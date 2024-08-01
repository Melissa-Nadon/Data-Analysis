import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Set page config with custom icon
st.set_page_config(page_title="CSV Data Analysis", page_icon="logo.png")

# Add custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #FFFACD; /* Snow */
        color: #333;
    }
    .stApp {
        background-color: #FFFFFF;    
    }
    .stButton>button {
        background-color: #87CEFA; /* LightSkyBlue */
        color: white;
    }
    .stDownloadButton>button {
        background-color: #87CEFA; /* LightSkyBlue */
        color: white;
    }
    .stSelectbox, .stMultiSelect {
        background-color: #e6f7ff; /* Light Blue */
        color: #333;
    }
    .stMultiSelect>div>div {
        background-color: #FFFFFF; /* Snow */
        color: white;
    }
    .stMultiSelect>div>div>div {
        background-color: #FFFFFF !important; /* Snow */
        color: white !important;
    }
    .stMultiSelect>div>div>div:focus {
        background-color: #D3D3D3 !important; /* LightGray */
        color: white !important;
    }
    .stMultiSelect>div>div>div:active {
        background-color: #D3D3D3 !important; /* LightGray */
        color: white !important;
    }
    .stCheckbox>div>div {
        background-color: #1E90FF; /* Dark Blue */
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color:#2F4F4F; /* DarkSlateGray */
    }
    .stMarkdown p {
        color: #333;
    }
    .stDataFrame {
        background-color: #fff;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("CSV Data Analysis 📊")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)

    # Convert Date column to DateTime
    if 'DateTime' in df.columns:
        # Try parsing the DateTime with multiple formats
        date_formats = ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M"]
        for fmt in date_formats:
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'], format=fmt)
                break
            except ValueError:
                continue
        else:
            st.error("The 'DateTime' column format does not match any of the expected formats. Please ensure it is in a consistent format.")
            st.stop()

        df = df.dropna(subset=['DateTime'])

        if df.empty:
            st.error("The DataFrame is empty after processing the 'DateTime' column.")
        else:
            tabs = st.tabs(["Data Analysis", "Count and Resample Analysis"])

            with tabs[0]:
                show_data = st.checkbox("Show uploaded data")
                if show_data:
                    st.write("Here's the data you uploaded:")
                    st.write(df)

                # Set DateTime as index
                df.set_index('DateTime', inplace=True)

                # Select columns for calculations
                numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                selected_columns = st.multiselect("Select columns to calculate", numeric_columns)
                if selected_columns:
                    # Convert minute data to hourly if needed
                    df_hourly = df[selected_columns].resample('H').mean()

                    period = st.selectbox("Select period for calculation", ['hourly', 'daily', 'weekly', 'monthly'])

                    # Handle NaN values by filling them with zeros
                    df_hourly.fillna(0, inplace=True)

                    # Resample data based on the selected period
                    if period == 'hourly':
                        df_resampled = df_hourly.resample('H').agg(['mean', 'std', 'max', 'min'])
                    elif period == 'daily':
                        df_resampled = df_hourly.resample('D').agg(['mean', 'std', 'max', 'min'])
                    elif period == 'weekly':
                        df_resampled = df_hourly.resample('W').agg(['mean', 'std', 'max', 'min'])
                    elif period == 'monthly':
                        df_resampled = df_hourly.resample('M').agg(['mean', 'std', 'max', 'min'])

                    df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]

                    st.write(f"**Calculations for selected columns over {period} period:**")
                    styled_df = df_resampled.style.format("{:.2f}").set_table_styles(
                        [{
                            'selector': 'thead th',
                            'props': [('background-color', '#FF1493'), ('color', '#333'), ('font-weight', 'bold'), ('text-align', 'center')]
                        },
                        {
                            'selector': 'tbody tr:nth-child(even)',
                            'props': [('background-color', '#ffe6f7')]
                        },
                        {
                            'selector': 'tbody tr:nth-child(odd)',
                            'props': [('background-color', '#ffffff')]
                        },
                        {
                            'selector': 'td',
                            'props': [('text-align', 'center')]
                        }]
                    )
                    st.write(styled_df, unsafe_allow_html=True)

                    # Select columns for graph
                    columns_to_plot = st.multiselect("Select columns to plot", df_resampled.columns)
                    if columns_to_plot:
                        # Plot combined graph using Plotly
                        st.write(f"**Combined graph for selected columns over {period} period:**")
                        fig = go.Figure()
                        for column_to_plot in columns_to_plot:
                            fig.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled[column_to_plot], mode='lines+markers', name=column_to_plot))

                        fig.update_layout(
                            title=f"Selected Columns over {period} period",
                            xaxis_title="DateTime",
                            yaxis_title="Value",
                            legend_title="Columns"
                        )

                        st.plotly_chart(fig)

                    csv = df_resampled.to_csv(index=True)
                    st.download_button(
                        label="Download Resampled Data as CSV",
                        data=csv,
                        file_name='resampled_data.csv',
                        mime='text/csv',
                    )

            with tabs[1]:
                st.write("Count and Resample Analysis")

                # Select columns for count analysis
                selected_columns_for_count = st.multiselect("Select columns to count", numeric_columns)
                if selected_columns_for_count:
                    count_data = df[selected_columns_for_count].count()
                    st.write("Count of non-NA/null values for each column:")
                    st.write(count_data)

                # Select columns for resample analysis
                selected_columns_for_resample = st.multiselect("Select columns to resample", numeric_columns)
                if selected_columns_for_resample:
                    resample_period = st.selectbox("Select period for resample", ['hourly', 'daily', 'weekly', 'monthly'])

                    # Resample data based on the selected period
                    if resample_period == 'hourly':
                        df_resampled_for_resample = df[selected_columns_for_resample].resample('H').mean()
                    elif resample_period == 'daily':
                        df_resampled_for_resample = df[selected_columns_for_resample].resample('D').mean()
                    elif resample_period == 'weekly':
                        df_resampled_for_resample = df[selected_columns_for_resample].resample('W').mean()
                    elif resample_period == 'monthly':
                        df_resampled_for_resample = df[selected_columns_for_resample].resample('M').mean()

                    # Handle NaN values by filling them with zeros
                    df_resampled_for_resample.fillna(0, inplace=True)

                    st.write(f"**Resampled data for selected columns over {resample_period} period:**")
                    st.write(df_resampled_for_resample)

                    csv_resample = df_resampled_for_resample.to_csv(index=True)
                    st.download_button(
                        label="Download Resampled Data as CSV",
                        data=csv_resample,
                        file_name='resampled_resample_data.csv',
                        mime='text/csv',
                    )

                    # Select columns for graph in tab 2
                    columns_to_plot_tab2 = st.multiselect("Select columns to plot", df_resampled_for_resample.columns)
                    if columns_to_plot_tab2:
                        st.write(f"**Combined graph with linear regression for selected columns over {resample_period} period in tab 2:**")
                        fig = go.Figure()
                        for column_to_plot in columns_to_plot_tab2:
                            fig.add_trace(go.Scatter(x=df_resampled_for_resample.index, y=df_resampled_for_resample[column_to_plot], mode='lines+markers', name=column_to_plot))

                            # Add linear regression line
                            X = np.array(df_resampled_for_resample.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
                            y = df_resampled_for_resample[column_to_plot].values
                            model = LinearRegression()
                            model.fit(X, y)
                            trend = model.predict(X)
                            fig.add_trace(go.Scatter(x=df_resampled_for_resample.index, y=trend, mode='lines', name=f"{column_to_plot}_regression", line=dict(dash='dash')))

                        fig.update_layout(
                            title=f"Selected Columns over {resample_period} period",
                            xaxis_title="DateTime",
                            yaxis_title="Value",
                            legend_title="Columns"
                        )

                        st.plotly_chart(fig)

    else:
        st.error("The CSV file does not contain a 'DateTime' column. Please upload a CSV file with a 'DateTime' column.")
else:
    st.write("Please upload a CSV file to proceed with calculations.")
