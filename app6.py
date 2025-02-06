import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import cv2
import time
from ultralytics import YOLO
import base64
import io
import joblib
import numpy as np
import easyocr
import re
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer



# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

# Load YOLO model for irrigation
model = YOLO('yolov8n.pt')

###### PEST DETECTION AND MAPPING
#load model for pest detection
model_pest = load_model('ccmt5gb.h5')
pest_csv_path = 'pest_mapping.csv' #pest mapping input
pest_solutions_df = pd.read_csv(pest_csv_path)
class_names = [
    'anthracnose-cashew', 'bacterial blight-cassava', 'brown spot-cassava',
    'cashew-mosaic-cassava', 'fall armyworm-maize', 'grasshopper-maize',
    'green mite-cassava', 'gumosis-cashew', 'healthy-cashew',
    'healthy-cassava', 'healthy-maize', 'healthy-tomato', 'leaf beetle-maize',
    'leaf blight-maize', 'leaf blight-tomato', 'leaf curl-tomato',
    'leaf miner-cashew', 'leaf spot-maize', 'red rust-cashew',
    'septoria leaf spot-tomato', 'streak virus-maize', 'verticillium wilt-tomato'
]

# Function to preprocess the uploaded image
def pest_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the pest/disease class
def pest_predict_pest(image_path):
    img = pest_preprocess_image(image_path)
    predictions = model_pest.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the class with the highest probability
    return predicted_class[0]

# Function to map the predicted class to a solution from the CSV
def get_solution_for_class(class_index):
    class_name = class_names[class_index]

    # Assuming your CSV has 'Class' column for the class names and 'Solution' column for corresponding solutions
    solution_row = pest_solutions_df[pest_solutions_df['Diseases'] == class_name]
    if not solution_row.empty:
        solution = solution_row['OrganicPrevention'].values[0]
        return solution
    else:
        return "Solution not found for this class."

# Function to process user-uploaded image and return class and solution
def pest_process_user_image(image_path):
    if os.path.exists(image_path):
        # Predict the pest/disease class
        predicted_class = pest_predict_pest(image_path)

        # Get the corresponding pest/disease name
        class_name = class_names[predicted_class]

        # Get the solution for the predicted class
        solution = get_solution_for_class(predicted_class)

        result = {
            "predicted_class": class_name,
            "suggested_solution": solution
        }

        return result
    else:
        return {"error": "Image not found. Please upload a valid image."}



######SOIL HEALTH ASSESSMENT
model_path = 'soil_health_model.pkl'
rf_classifier = joblib.load(model_path)

default_feature_values = {
    'EC': 0.35, 'OC': 1.5, 'OM': 2.0, 'Zn': 0.5, 'Fe': 5.0, 'Cu': 0.3, 'Mn': 1.0, 
    'Sand': 55, 'Silt': 20, 'Clay': 25, 'CaCO3': 3.0, 'CEC': 12, 
    'Temperature': 25, 'Humidity': 60, 'Rainfall': 500
}

def predict_from_csv(soil_data):
    try:
        rf_predictions = rf_classifier.predict(soil_data)
        label_mapping = {1: 'Non Fertile', 0: 'Fertile'}
        predicted_labels = [label_mapping[pred] for pred in rf_predictions]
        return predicted_labels
    except Exception as e:
        return [f"Error processing the data: {str(e)}"]

def predict_from_user_input(n_value, p_value, k_value, ph_value):
    try:
        user_input = [ph_value, default_feature_values['EC'], default_feature_values['OC'], default_feature_values['OM'],
                      n_value, p_value, k_value, default_feature_values['Zn'], default_feature_values['Fe'],
                      default_feature_values['Cu'], default_feature_values['Mn'], default_feature_values['Sand'],
                      default_feature_values['Silt'], default_feature_values['Clay'], default_feature_values['CaCO3'],
                      default_feature_values['CEC'], default_feature_values['Temperature'],
                      default_feature_values['Humidity'], default_feature_values['Rainfall']]
        user_input_array = np.array(user_input).reshape(1, -1)
        prediction = rf_classifier.predict(user_input_array)
        label_mapping = {0: 'Non Fertile', 1: 'Fertile'}
        return label_mapping[prediction[0]]
    except Exception as e:
        return f"Error processing the input: {str(e)}"

####FARMER"S INFO
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    scale_percent = 150  
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_id_details(image_path):
    preprocessed_image = preprocess_image(image_path)
    temp_image_path = "temp_preprocessed_image.jpg"
    cv2.imwrite(temp_image_path, preprocessed_image)
    results = reader.readtext(temp_image_path, detail=0)
    details_text = "\n".join(results)
    print("\nExtracted Text: \n", details_text)
    details = {}
    name_match = re.search(r'Name[:\s]*([A-Za-z\s]+)', details_text, re.IGNORECASE)
    if name_match:
        details['Name'] = name_match.group(1).strip()
    id_match = re.search(r'ID[:\s]*([A-Za-z0-9]+)', details_text, re.IGNORECASE)
    if id_match:
        details['ID'] = id_match.group(1).strip()
    address_match = re.search(r'Address[:\s](.*)', details_text, re.IGNORECASE)
    if address_match:
        details['Address'] = address_match.group(1).strip()
    return details_text

def save_image(contents, filename):
    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image_path = os.path.join('temp_images', filename)
    with open(image_path, 'wb') as f:
        f.write(decoded)
    return image_path


app.layout = html.Div(
    style={'backgroundColor': '#121212', 'minHeight': '100vh', 'padding': '40px', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.Div(
            children=[
                html.H1(
                    "Agricultural Smart Dashboard",
                    style={
                        'color': 'white', 'textAlign': 'center', 'fontSize': '3.5rem', 
                        'background': 'linear-gradient(90deg, #00C9FF, #92FE9D)',
                        '-webkit-background-clip': 'text', 'background-clip': 'text',
                        'color': 'transparent', 'marginBottom': '40px', 'fontWeight': 'bold'
                    }
                ),
            ],
        ),

        # Tabs for different sections
        dcc.Tabs(
            id='tabs',
            value='tab-1',
            children=[
                dcc.Tab(label='Soil Health Analysis', value='tab-1', style={'fontSize': '20px', 'padding': '12px 0','color':'#fff','backgroundColor': '#7B3F00','fontWeight': 'bold'}),
                dcc.Tab(label='Crop & Pest Prediction', value='tab-2', style={'fontSize': '20px', 'padding': '12px 0','color': '#fff','backgroundColor': '#006400','fontWeight': 'bold'}),
                dcc.Tab(label='Smart Irrigation System', value='tab-3', style={ 'fontSize': '20px', 'padding': '12px 0','color': '#fff','backgroundColor': '#00C9FF','fontWeight': 'bold'}),
                dcc.Tab(label="Farmer's Information Hub", value='tab-4', style={'fontSize': '20px', 'padding': '12px 0','color': '#fff','backgroundColor': '#FFCF6E','fontWeight': 'bold'}),
            ],
            colors={
                'border': '#2C2C2C',
                'primary': '#00C9FF',
                'background': '#333'
            }
        ),

        # Content will be rendered based on tab selection
        html.Div(id='tabs-content', style={'marginTop': '30px'})
    ]
)

# Callback to switch between tabs
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            id='soil-health-section',
            style={'marginTop': '50px'},
            children=[
                html.H4("Soil Health Analysis", style={'color': 'white', 'textAlign': 'center', 'fontSize': '2rem'}),
                html.Div([
                    html.Label("Nitrogen (N):", style={'color': 'white', 'fontSize': '16px'}),
                    dcc.Input(id='input-n', type='number', placeholder='Enter N value', 
                              style={'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px', 'backgroundColor': '#1f1f1f', 'color': 'white'}),
                    html.Label("Phosphorus (P):", style={'color': 'white', 'marginTop': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='input-p', type='number', placeholder='Enter P value', 
                              style={'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px', 'backgroundColor': '#1f1f1f', 'color': 'white'}),
                    html.Label("Potassium (K):", style={'color': 'white', 'marginTop': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='input-k', type='number', placeholder='Enter K value', 
                              style={'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px', 'backgroundColor': '#1f1f1f', 'color': 'white'}),
                    html.Label("pH:", style={'color': 'white', 'marginTop': '10px', 'fontSize': '16px'}),
                    dcc.Input(id='input-ph', type='number', placeholder='Enter pH value', 
                              style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#1f1f1f', 'color': 'white'}),
                    dbc.Button(
                        "Submit", id="submit-manual-btn", color="success", 
                        style={'marginTop': '20px', 'padding': '10px 0', 'fontSize': '16px', 'backgroundColor': '#00c9ff', 'fontWeight': 'bold'}
                    ),
                ], style={
                    'display': 'flex', 'flexDirection': 'column', 'gap': '10px', 
                    'maxWidth': '400px', 'margin': '0 auto', 'padding': '20px', 
                    'border': '1px solid #2C2C2C', 'borderRadius': '10px', 'backgroundColor': '#333'
                }),
                dcc.Upload(
                    id='upload-soil-csv',
                    children=html.Div(['Drag and Drop or ', html.A('Select CSV File', style={'color': '#007bff'})]),
                    style={
                        'width': '50%', 'height': '60px', 'lineHeight': '60px', 
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#666', 
                        'textAlign': 'center', 'margin': 'auto', 'color': '#F0F0F0'
                    }
                ),
                html.Div(id='soil-health-results', style={'textAlign': 'center', 'color': 'white', 'marginTop': '20px'}),
            ]
        )
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Pest Prediction and Prevention', style={'color': 'white', 'textAlign': 'center', 'fontSize': '2rem'}),
            dcc.Upload(
                id='upload-image-2',
                children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px', 'color' : 'white'
                },
                
            ),
            html.Div(id='output-image-upload')
        ])
    elif tab == 'tab-3':
        return html.Div(
            id='irrigation-section',
            style={'marginTop': '50px'},
            children=[
                html.H4("Smart Irrigation System", style={'color': 'white', 'textAlign': 'center', 'fontSize': '2rem'}),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div(['Drag and Drop or ', html.A('Select Image', style={'color': '#007bff'})]),
                    style={
                        'width': '50%', 'height': '60px', 'lineHeight': '60px', 
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#666', 
                        'textAlign': 'center', 'margin': 'auto', 'color': '#F0F0F0'
                    }
                ),
                dcc.Upload(
                    id='upload-csv',
                    children=html.Div(['Drag and Drop or ', html.A('Select CSV File', style={'color': '#007bff'})]),
                    style={
                        'width': '50%', 'height': '60px', 'lineHeight': '60px', 
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#666', 
                        'textAlign': 'center', 'margin': 'auto', 'marginTop': '20px', 'color': '#F0F0F0'
                    }
                ),
                html.Div(id='output-image', style={'textAlign': 'center', 'marginTop': '20px', 'color': '#F0F0F0'}),
                html.Div(id='output-csv-name', style={'textAlign': 'center', 'color': '#F0F0F0', 'marginTop': '10px'}),
                html.Div(id='irrigation-results', style={'textAlign': 'center', 'color': '#F0F0F0', 'marginTop': '20px'}),
            ]
        )
    elif tab == 'tab-4':
        return html.Div(
            id='farmer-info-section',
            style={'marginTop': '50px', 'color': 'white', 'textAlign': 'center'},
            children=[
                html.H4("Upload ID Card for Farmer Details Extraction", style={'fontSize': '2rem'}),
                dcc.Upload(
                    id='upload-image-1',
                    children=html.Div(['Drag and Drop or ', html.A('Select Image', style={'color': '#007bff'})]),
                    style={
                        'width': '50%', 'height': '60px', 'lineHeight': '60px', 
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#666', 
                        'textAlign': 'center', 'margin': 'auto', 'color': '#FFF'
                    }
                ),
                html.Div(id='output-image-1', style={'textAlign': 'center', 'marginTop': '20px', 'color': '#F0F0F0'}),
                html.Div(id='output-id-details', style={'marginTop': '30px', 'color': 'white', 'fontSize': '18px'}),
            ]
        )

# Callback to display Soil Health section when button is clicked
@app.callback(
    Output('soil-health-section', 'style'),
    [Input('show-soil-health-btn', 'n_clicks')],
    prevent_initial_call=True
)
def show_soil_health_section(n_clicks):
    return {'display': 'block'}

# Callback to handle manual input and CSV upload for soil health prediction
@app.callback(
    Output('soil-health-results', 'children'),
    [Input('submit-manual-btn', 'n_clicks'), Input('upload-soil-csv', 'contents')],
    [State('input-n', 'value'), State('input-p', 'value'), State('input-k', 'value'), State('input-ph', 'value'), State('upload-soil-csv', 'filename')]
)
##SOIL HEALTH ANALYSIS
def analyze_soil_health(n_clicks, csv_content, n_value, p_value, k_value, ph_value, csv_filename):
    if csv_content:
        # Decode the CSV content and read it into a DataFrame
        csv_data = base64.b64decode(csv_content.split(',')[1])
        soil_data = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
        predictions = predict_from_csv(soil_data)
        return html.Div([
            html.H5(f"Predicted Labels from {csv_filename}:", style={'color': 'white'}),
            html.P(", ".join(predictions), style={'color': 'white'})
        ])
    elif n_clicks:
        # If manual input is provided
        if all([n_value, p_value, k_value, ph_value]):
            prediction = predict_from_user_input(n_value, p_value, k_value, ph_value)
            return html.H5(f"The soil is predicted to be: {prediction}", style={'color': 'white'})
        else:
            return html.H5("Please provide valid N, P, K, and pH values.", style={'color': 'red'})
    return ""


###IRRIGATION
# Function to detect pests
def detect_pests(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image)
    pests_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [75, 14]:
                pests_detected = True
                break
        if pests_detected:
            break
    return pests_detected

# Function to analyze soil health
def analyze_soil(soil_data):
    avg_humidity = soil_data['Humidity'].mean()
    avg_temperature = soil_data['Temperature'].mean()
    avg_rainfall = soil_data['Rainfall'].mean()

    if avg_humidity < 30 and avg_rainfall < 5:  # Example thresholds
        return True  # Need irrigation non-fertile
    else:
        return False  # No irrigation needed fertile

# Simulated function to control irrigation system
def water_plants(duration=10):
    time.sleep(duration)  # Simulate watering duration
    return "Watering complete."


# Callback to display Smart Irrigation section when button is clicked
@app.callback(
    Output('irrigation-section', 'style'),
    [Input('show-irrigation-btn', 'n_clicks')],
    prevent_initial_call=True
)
def show_irrigation_section(n_clicks):
    return {'display': 'block'}

# Callback to handle image and CSV upload and process the irrigation system
@app.callback(
    [Output('output-image', 'children'), Output('output-csv-name', 'children'), Output('irrigation-results', 'children')],
    [Input('upload-image', 'contents'), Input('upload-csv', 'contents')],
    [State('upload-image', 'filename'), State('upload-csv', 'filename')]
)
def handle_inputs(image_content, csv_content, image_filename, csv_filename):
    if image_content and csv_content:
        # Decode the image content
        image_data = base64.b64decode(image_content.split(',')[1])
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_data)

        # Decode the CSV content and read it into a DataFrame
        csv_data = base64.b64decode(csv_content.split(',')[1])
        soil_data = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))

        # Pest detection
        pests_detected = detect_pests(image_path)
        irrigation_needed = analyze_soil(soil_data)

        if pests_detected:
            irrigation_results = "Pests detected. Skipping irrigation."
        elif irrigation_needed:
            irrigation_results = "Soil health requires irrigation. " + water_plants()
        else:
            irrigation_results = "No irrigation needed based on soil health."

        return (
            html.Img(src=image_content, style={'width': '50%'}),
            f"CSV File: {csv_filename}",
            irrigation_results
        )
    return "", "", "Please upload both image and CSV."



# Show Farmer's Information Hub when the button is clicked
@app.callback(
    Output('farmer-info-section', 'style'),
    Input('show-farmer-info-btn', 'n_clicks')
)
def toggle_farmer_info_section(n_clicks):
    if n_clicks:
        return {'display': 'block', 'marginTop': '50px', 'color': 'white'}
    return {'display': 'none'}

# Callback to handle image upload and extract ID details
@app.callback(
    Output('output-id-details', 'children'),
    Input('upload-image-1', 'contents'),
    State('upload-image-1', 'filename')
)
def update_output(content, filename):
    if content is not None:
        # Save the image temporarily and extract details
        image_path = save_image(content, filename)
        details_text = extract_id_details(image_path)
        details_text_line=details_text.splitlines()
        # Convert the image back to base64 for displaying in Dash
        _, img_str = content.split(',')
        img_data = base64.b64decode(img_str)
        encoded_image = base64.b64encode(img_data).decode('utf-8')

        # Display the uploaded image and extracted details
        return html.Div([
            html.Img(src=f"data:image/jpeg;base64,{encoded_image}", style={'width': '50%', 'height': 'auto', 'marginBottom': '20px'}),
            #html.P(f"{details_text_line}")
            #html.P(f"ID: {details.get('ID', 'Not Found')}"),
            #html.P(f"Address: {details.get('Address', 'Not Found')}"),
            dbc.Row(
            dbc.Col(
                html.Div([html.Li(item) for item in details_text_line]) ))
        ])

    return "No image uploaded."


###PEST DETECTION
# Callback to handle image upload and prediction
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image-2', 'contents')],
              [State('upload-image-2', 'filename')])
def update_output_image(contents, filename):
    if contents is not None:
        # Save the uploaded image to a temporary location
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image_path = f'temp_{filename}'
        with open(image_path, 'wb') as f:
            f.write(decoded)

        # Perform pest prediction on the uploaded image
        result = pest_process_user_image(image_path)
        
        # Remove the temporary image after prediction
        os.remove(image_path)

        if 'error' in result:
            return html.Div([
                html.H4('Error: {}'.format(result['error']))
            ])
        else:
            return html.Div([
                html.H4(f"Predicted Class: {result['predicted_class']}", style={'color': 'white', 'textAlign': 'center', 'fontSize': '2rem'}),
                html.H4(f"Suggested Solution: {result['suggested_solution']}", style={'color': 'white', 'textAlign': 'center', 'fontSize': '2rem'})
            ])

if __name__ == '__main__':
    app.run_server(debug=True)
