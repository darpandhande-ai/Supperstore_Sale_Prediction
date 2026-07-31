import os
import pickle
import sys
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Model loading with robust exception handling to prevent startup crashes
MODEL_PATH = "Supperstore_model.pkl"
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model successfully loaded!", file=sys.stderr)
    except Exception as e:
        model_error = f"Failed to load model pickle: {str(e)}"
        print(f"CRITICAL ERROR: {model_error}", file=sys.stderr)
else:
    model_error = f"Model file '{MODEL_PATH}' was not found in the root directory."
    print(f"WARNING: {model_error}", file=sys.stderr)

# HTML UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Superstore Sales Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #10b981;
            --error: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group input {
            background: #0f172a;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .form-group input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }

        .btn-submit:hover {
            background: var(--accent-hover);
        }

        .btn-submit:disabled {
            background: #475569;
            cursor: not-allowed;
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            border-radius: 8px;
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .result-box .prediction-val {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--success);
        }

        .error-box {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: #fca5a5;
            border-radius: 8px;
            text-align: center;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Superstore Sales Predictor</h1>
        <p>Enter order details below to estimate model predictions</p>
    </div>

    {% if model_error %}
    <div class="error-box">
        <strong>Warning:</strong> {{ model_error }}
    </div>
    {% endif %}

    {% if error %}
    <div class="error-box">
        {{ error }}
    </div>
    {% endif %}

    <form method="POST" action="/predict" class="grid-form">
        <div class="form-group">
            <label>Ship Mode</label>
            <input type="number" step="any" name="Ship Mode" placeholder="e.g. 0" required>
        </div>
        <div class="form-group">
            <label>Customer Name</label>
            <input type="number" step="any" name="Customer Name" placeholder="e.g. 12" required>
        </div>
        <div class="form-group">
            <label>Segment</label>
            <input type="number" step="any" name="Segment" placeholder="e.g. 1" required>
        </div>
        <div class="form-group">
            <label>Country</label>
            <input type="number" step="any" name="Country" placeholder="e.g. 0" required>
        </div>
        <div class="form-group">
            <label>City</label>
            <input type="number" step="any" name="City" placeholder="e.g. 45" required>
        </div>
        <div class="form-group">
            <label>State</label>
            <input type="number" step="any" name="State" placeholder="e.g. 10" required>
        </div>
        <div class="form-group">
            <label>Region</label>
            <input type="number" step="any" name="Region" placeholder="e.g. 2" required>
        </div>
        <div class="form-group">
            <label>Category</label>
            <input type="number" step="any" name="Category" placeholder="e.g. 1" required>
        </div>
        <div class="form-group">
            <label>Sub-Category</label>
            <input type="number" step="any" name="Sub-Category" placeholder="e.g. 5" required>
        </div>
        <div class="form-group">
            <label>Product Name</label>
            <input type="number" step="any" name="Product Name" placeholder="e.g. 102" required>
        </div>
        <div class="form-group">
            <label>Quantity</label>
            <input type="number" step="any" name="Quantity" placeholder="e.g. 3" required>
        </div>
        <div class="form-group">
            <label>Discount</label>
            <input type="number" step="0.01" name="Discount" placeholder="e.g. 0.20" required>
        </div>
        <div class="form-group">
            <label>Profit</label>
            <input type="number" step="0.01" name="Profit" placeholder="e.g. 41.91" required>
        </div>

        <button type="submit" class="btn-submit" {% if model_error and not model %}disabled{% endif %}>
            Predict Value
        </button>
    </form>

    {% if prediction %}
    <div class="result-box">
        <h2>Predicted Output</h2>
        <div class="prediction-val">{{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, model_error=model_error)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        err = model_error or "Model pickle file is not available."
        return render_template_string(HTML_TEMPLATE, error=err, model_error=model_error)
    
    try:
        feature_order = [
            "Ship Mode", "Customer Name", "Segment", "Country", 
            "City", "State", "Region", "Category", 
            "Sub-Category", "Product Name", "Quantity", "Discount", "Profit"
        ]
        
        input_data = [float(request.form.get(f, 0)) for f in feature_order]
        input_df = pd.DataFrame([input_data], columns=feature_order)
        
        prediction_value = model.predict(input_df)[0]
        formatted_prediction = f"{prediction_value:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction=formatted_prediction, model_error=model_error)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Prediction Error: {str(e)}", model_error=model_error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
