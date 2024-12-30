import pandas as pd

private_2024 = pd.read_csv('Private_2023_2024.csv.xls')
private_1995 = pd.read_csv('Private_1995_2023-05-17.csv')
postcodes = pd.read_csv('postcodes.csv', encoding='ISO-8859-1')
private_2024 = private_2024[private_2024['property_type'].isin(['Condominium', 'Apartment'])]
private_1995 = private_1995[private_1995['property_type'].isin(['Condominium', 'Apartment'])]
private_1995 = private_1995.drop(columns = ['purchaser_address_indicator', 'area_type','area_sqft', 'nett_price_dollars', 'unit_price_psm_dollars', 'unit_price_psf_dollars','postal_district'])
private_1995['area_sqm'] = pd.to_numeric(private_1995['area_sqm'], errors='coerce')
private_1995['area_sqft'] = private_1995['area_sqm'] * 10.7639
private_1995['area_sqft'] = private_1995['area_sqft'].round(2)
private_1995.rename(columns={' transaction_price_dollars ': 'transaction_price_dollars'}, inplace=True)
private_1995['transaction_price_dollars'] = private_1995['transaction_price_dollars'].str.replace(',', '', regex=False)
private_1995['transaction_price_dollars'] = pd.to_numeric(private_1995['transaction_price_dollars'], errors='coerce')
private_1995['unit_price_psm_dollars'] = private_1995['transaction_price_dollars']/private_1995['area_sqm']
private_1995['unit_price_psf_dollars'] = private_1995['transaction_price_dollars']/private_1995['area_sqft']
private_1995['unit_price_psf_dollars'] = private_1995['unit_price_psf_dollars'].round(0)
private_1995 = private_1995.dropna(subset=['property_tenure'])
private_1995 = private_1995[private_1995['num_of_units'] == 1]
private_1995 = private_1995.drop(columns = ['num_of_units'])
private_1995['sale_date'] = pd.to_datetime(private_1995['sale_date'], dayfirst=True, errors='coerce')
private_1995['sale_date'] = private_1995['sale_date'].dt.strftime('%d/%m/%Y')
private_2024 = private_2024.drop(columns = ['nett_price', 'latitude', 'longitude', 'ingested_date', 
                                           'purchaser_address_indicator', 'postal_sector', 'postal_district',
                                           'id', 'number_of_units', 'enbloc_ind', 'type_of_area'])
private_2024 = private_2024[private_2024['multi_unit'] != True]
private_2024 = private_2024.drop(columns = ['multi_unit'])
private_2024.rename(columns={'transacted_price': 'transaction_price_dollars'}, inplace=True)
private_2024.rename(columns={'unit_price_psf': 'unit_price_psf_dollars'}, inplace=True)
private_2024.rename(columns={'unit_price_psm': 'unit_price_psm_dollars'}, inplace=True)
private_2024.rename(columns={'type_of_sale': 'sale_type'}, inplace=True)
private_2024.rename(columns={'completion_date': 'completion_top_date'}, inplace=True)
private_2024.rename(columns={'tenure': 'property_tenure'}, inplace=True)
private_2024['sale_date'] = pd.to_datetime(private_2024['sale_date'])  # Ensure it's in datetime format
private_2024['sale_date'] = private_2024['sale_date'].dt.strftime('%d/%m/%Y')
private_1995['planning_region'] = pd.NA
private_1995['planning_area'] = pd.NA
private_1995 = private_1995[private_2024.columns]
private = pd.concat([private_1995, private_2024], ignore_index=True)
duplicates = private[private.duplicated(subset=['address', 'project_name', 'sale_date', 'transaction_price_dollars'], keep= False)]
private = private.drop_duplicates(subset=['address', 'project_name', 'sale_date', 'transaction_price_dollars'], keep='first')
private['sale_date'] = pd.to_datetime(private['sale_date'])
new_sales_dates = private[private['sale_type'] == 'New Sale'].groupby('project_name')['sale_date'].min()
dropped_rows = set()  # Using a set to avoid duplicates
def filter_resales(row):
    # If the project has a "New Sale", check if the "Resale" is before the first "New Sale" date
    if row['project_name'] in new_sales_dates:
        earliest_new_sale = new_sales_dates[row['project_name']]
        # If the row is a "Resale" and occurred before the "New Sale" date, it's a dropped row
        if row['sale_type'] == 'Resale' and row['sale_date'] < earliest_new_sale:
            # Using a tuple (address, project_name, sale_date, sale_type) to ensure uniqueness
            dropped_rows.add((row['address'], row['project_name'], row['sale_date'], row['sale_type']))
            return False
    return True
private = private[private.apply(filter_resales, axis=1)]
dropped_df = pd.DataFrame(list(dropped_rows), columns=['address', 'project_name', 'sale_date', 'sale_type'])
private.reset_index(drop=True, inplace=True)
private['sale_date'] = pd.to_datetime(private['sale_date'])  # Ensure it's in datetime format
private['sale_date'] = private['sale_date'].dt.strftime('%d/%m/%Y')
private.rename(columns={'postal_code': 'PostalCode'}, inplace=True)
private['PostalCode'] = private['PostalCode'].astype(str)
postcodes['PostalCode'] = postcodes['PostalCode'].astype(str)
private = private.merge(postcodes[['PostalCode', 'Planning Area']], on='PostalCode', how='left')
private = private.drop(columns=['planning_area'])
private['Planning Area'] = private['Planning Area'].str.title()
private['area_sqft'] = private['area_sqft'].round().astype(int)

import numpy as np

# Assuming `private` is your private dataset
private['sale_date'] = pd.to_datetime(private['sale_date'], errors='coerce')  # Handle invalid dates

# Drop rows where sale_date or transaction_price_dollars are missing
private = private.dropna(subset=['sale_date', 'transaction_price_dollars'])

# Sort by address and sale_date
private = private.sort_values(by=['address', 'sale_date']).reset_index(drop=True)

# Create a new DataFrame for the desired output
rows = []

# Group by address to identify transaction pairs
for address, group in private.groupby('address'):
    if len(group) < 2:
        continue  # Skip if there are less than two transactions
    
    # Iterate over the group to create transaction pairs
    for i in range(len(group) - 1):
        held_from = group.iloc[i]
        sold_at = group.iloc[i + 1]
        
        # Validate that sale_date is not NaT (Not a Time)
        if pd.isna(held_from['sale_date']) or pd.isna(sold_at['sale_date']):
            continue
        
        # Calculate holding period (rounded to nearest half year)
        holding_days = (sold_at['sale_date'] - held_from['sale_date']).days
        holding_years = holding_days / 365.0
        if np.isnan(holding_years):
            continue  # Skip if holding_years is invalid
        holding_period = round(holding_years * 2) / 2  # Round to nearest 0.5
        
        # Calculate ROI
        roi = sold_at['transaction_price_dollars'] - held_from['transaction_price_dollars']
        
        # Append the data
        rows.append({
            'address': address,  # Include address
            'held_from': held_from['sale_date'],
            'sold_at': sold_at['sale_date'],
            'holding_period': holding_period,
            'Gain/Loss': roi,
            'Planning Area': sold_at['Planning Area'],
            'property_tenure': sold_at['property_tenure'],
            'transaction_price_dollars': sold_at['transaction_price_dollars'],
            'project_name': held_from['project_name'],  # Same for both
            'area_sqft': held_from['area_sqft'],        # Same for both
            'sale_type': sold_at['sale_type'],
        })

# Create the new DataFrame
result_private = pd.DataFrame(rows)

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Assuming your 'result_private' dataset is loaded
# Example: result_private = pd.read_csv('your_data.csv')

# Define floor area ranges (this is a static list that will be updated dynamically)
floor_area_ranges = {
    "< 600 sqft": (0, 600),
    "600-850 sqft": (600, 850),
    "850-1100 sqft": (850, 1100),
    "1100-1500 sqft": (1100, 1500),
    "1500 sqft+": (1500, float('inf')),
    "All Resales": (0, float('inf'))
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Condominium and Apartment Resale Gain/Loss Dashboard", style={"textAlign": "center"}),

    # Dropdown for project name
    html.Div([
        html.Label("Select Project Name: (Searchable)"),
        dcc.Dropdown(
            id="project-dropdown",
            options=[{"label": name, "value": name} for name in result_private["project_name"].unique()],
            value=result_private["project_name"].unique()[0],  # Default value
            searchable=True,
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto"}),

    # Dropdown for floor area range
    html.Div([
        html.Label("Select Floor Area Range:"),
        dcc.Dropdown(
            id="floor-area-dropdown",
            value="< 600 sqft",  # Default value
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto", "marginTop": "10px"}),

    # Display number of entries
    html.Div(id="num-entries", style={"textAlign": "center", "fontSize": "20px", "marginTop": "20px"}),

    # Scatter plot
    dcc.Graph(id="gain-loss-plot"),
])

# Callback to update the scatterplot, number of entries, and floor area options
@app.callback(
    [Output("gain-loss-plot", "figure"),
     Output("num-entries", "children"),
     Output("floor-area-dropdown", "options")],
    [Input("project-dropdown", "value"),
     Input("floor-area-dropdown", "value")]
)
def update_plot(selected_project, selected_floor_area):
    # Filter data by project name
    filtered_df = result_private[result_private["project_name"] == selected_project]

    # Get unique floor area values for the selected project
    min_area = filtered_df["area_sqft"].min()
    max_area = filtered_df["area_sqft"].max()

    # Filter floor area ranges based on available data
    available_floor_ranges = {key: (min_val, max_val) for key, (min_val, max_val) in floor_area_ranges.items()
                              if min_area <= max_val and max_area >= min_val}

    # Update floor area dropdown options
    floor_area_options = [{"label": key, "value": key} for key in available_floor_ranges.keys()]

    # Filter data by floor area range
    floor_min, floor_max = available_floor_ranges.get(selected_floor_area, (0, float('inf')))
    filtered_df = filtered_df[
        (filtered_df["area_sqft"] >= floor_min) &
        (filtered_df["area_sqft"] <= floor_max)
    ]
    
    # Calculate the number of entries
    num_entries = len(filtered_df)

    # Convert 'sold_at' and 'held_from' to datetime for extracting year
    filtered_df['sold_at'] = pd.to_datetime(filtered_df['sold_at'])
    filtered_df['held_from'] = pd.to_datetime(filtered_df['held_from'])
    
    # Format the 'sold_at' and 'held_from' columns to only show the date
    filtered_df['Buy Date'] = filtered_df['held_from'].dt.strftime('%Y-%m-%d')
    filtered_df['Sell Date'] = filtered_df['sold_at'].dt.strftime('%Y-%m-%d')
    
    # Extract year sold
    filtered_df['Year Sold'] = filtered_df['sold_at'].dt.year
    filtered_df['Address'] = filtered_df['address']
    filtered_df['Area (sqft)'] = filtered_df['area_sqft']
    
    # Format transaction price and gain/loss with commas
    filtered_df['Transacted Price (SGD)'] = filtered_df['transaction_price_dollars'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Area (sqft)'] = filtered_df['area_sqft'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Gain/Loss (SGD)'] = filtered_df['Gain/Loss'].apply(lambda x: f"{x:,.0f}")

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="Year Sold",
        y="Gain/Loss",  # This is the actual 'Gain/Loss' for the scatter plot (without commas)
        hover_data={
            "Gain/Loss (SGD)": True,
            "Buy Date": True,
            "Sell Date": True,
            "Area (sqft)": True,
            "Address": True,
            "Transacted Price (SGD)": True,
            "Gain/Loss" : False
        },
        title=f"Gain/Loss for {selected_project} ({selected_floor_area})",
        color="Gain/Loss",  # Color by gain/loss for easier visualization
        color_continuous_scale="RdYlGn"  # Red to yellow to green
    )

    # Additional tweaks
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Gain/Loss (SGD)",
            tickformat=",.0f"  # Add commas to tick labels
        ),
        coloraxis_showscale=True,  # Ensure color bar is visible
        transition_duration=500,
        width = 1200,
        height = 800,
        yaxis = dict(
            tickformat = ',.0f'
        )
    )

    # Return figure, entry count, and updated floor area dropdown options
    return fig, f"Number of resales: {num_entries}", floor_area_options

if __name__ == "__main__":
    app.run_server(debug=True)
    
