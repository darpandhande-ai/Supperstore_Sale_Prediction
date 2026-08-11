from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Supperstore_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Exact feature names extracted from model
FEATURES = [
    'Ship Mode', 'Customer Name', 'Segment', 'Country', 'City',
    'State', 'Region', 'Category', 'Sub-Category', 'Product Name',
    'Quantity', 'Discount', 'Profit'
]

# Form options for dropdowns
OPTIONS = {
    'Ship Mode': ['Standard Class', 'Second Class', 'First Class', 'Same Day'],
    'Segment': ['Consumer', 'Corporate', 'Home Office'],
    'Region': ['East', 'West', 'Central', 'South'],
    'Category': ['Furniture', 'Office Supplies', 'Technology'],
    'Sub-Category': ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper', 'Accessories', 'Envelopes', 'Fasteners', 'Supplies', 'Machines', 'Copiers']
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Superstore Sales Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .glass {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4 sm:p-8">

    <div class="w-full max-w-4xl glass bg-slate-900/60 border border-slate-700/50 rounded-2xl shadow-2xl p-6 sm:p-10 backdrop-blur-xl">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-3">
                AI Powered Analytics
            </div>
            <h1 class="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">Superstore Sales Predictor</h1>
            <p class="text-slate-400 text-sm sm:text-base mt-2">Enter order details below to estimate expected sales output.</p>
        </div>

        <!-- Form -->
        <form id="predictionForm" action="/predict" method="POST" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                
                <!-- Categorical Dropdowns -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Ship Mode</label>
                    <select name="Ship Mode" class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        {% for item in options['Ship Mode'] %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Segment</label>
                    <select name="Segment" class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        {% for item in options['Segment'] %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Region</label>
                    <select name="Region" class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        {% for item in options['Region'] %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Category</label>
                    <select name="Category" class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        {% for item in options['Category'] %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Sub-Category</label>
                    <select name="Sub-Category" class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        {% for item in options['Sub-Category'] %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <!-- Text Inputs -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Customer Name</label>
                    <input type="text" name="Customer Name" value="Claire Gute" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Country</label>
                    <input type="text" name="Country" value="United States" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">City</label>
                    <input type="text" name="City" value="Henderson" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">State</label>
                    <input type="text" name="State" value="Kentucky" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Product Name</label>
                    <input type="text" name="Product Name" value="Bush Somerset Bookcase" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <!-- Numeric Inputs -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Quantity</label>
                    <input type="number" step="1" name="Quantity" value="2" min="1" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Discount (0 to 1)</label>
                    <input type="number" step="0.01" name="Discount" value="0.00" min="0" max="1" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div class="md:col-span-2 lg:col-span-1">
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-2">Expected Profit ($)</label>
                    <input type="number" step="0.01" name="Profit" value="41.91" required class="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

            </div>

            <!-- Submit Button -->
            <div class="pt-4">
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3.5 px-6 rounded-xl transition duration-200 shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    Calculate Prediction
                </button>
            </div>
        </form>

        <!-- Result Container -->
        {% if prediction is not none %}
        <div class="mt-8 p-6 bg-indigo-950/60 border border-indigo-500/40 rounded-xl text-center animate-fade-in">
            <span class="text-xs uppercase tracking-widest text-indigo-300 font-semibold">Predicted Output</span>
            <div class="text-4xl font-extrabold text-white mt-1">
                ${{ "%.2f"|format(prediction) }}
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="mt-8 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-center text-red-300 text-sm">
            {{ error }}
        </div>
        {% endif %}

    </div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, options=OPTIONS, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, options=OPTIONS, prediction=None, error="Model pickle file not loaded.")

    try:
        # Construct input dict matching model features
        input_data = {}
        for feature in FEATURES:
            val = request.form.get(feature)
            if feature in ['Quantity', 'Discount', 'Profit']:
                input_data[feature] = [float(val)]
            else:
                input_data[feature] = [str(val)]

        df = pd.DataFrame(input_data)

        # Convert string columns to categorical codes if raw strings are provided
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes

        # Model prediction
        pred = model.predict(df)[0]
        return render_template_string(HTML_TEMPLATE, options=OPTIONS, prediction=pred)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, options=OPTIONS, prediction=None, error=f"Prediction Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
