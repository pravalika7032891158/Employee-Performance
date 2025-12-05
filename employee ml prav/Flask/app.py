from flask import Flask, render_template, request
import numpy as np
import pickle
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path

app = Flask(__name__)

# Locate model files relative to this script (workspace root/IBM Files)
MODEL_DIR = Path(__file__).resolve().parent.parent / "IBM Files"
model = None
for fname in ("model_xgb.pkl", "model_rf.pkl", "model_lr.pkl"):
    p = MODEL_DIR / fname
    if p.exists():
        with open(p, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model: {p}")
        break

if model is None:
    raise FileNotFoundError(
        f"No model file found in {MODEL_DIR}. Place one of: model_xgb.pkl, model_rf.pkl, model_lr.pkl"
    )


@app.route("/")
def about():
    return render_template('home.html')

@app.route("/about")
def home():
    return render_template('about.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/submit")
def home2():
    return render_template('submit.html')

@app.route("/pred", methods=['POST'])
def predict():
    try:
        # Collecting form data
        quarter = request.form.get('quarter', '')
        department = request.form.get('department', '')
        day = request.form.get('day', '')
        team = request.form.get('team', '')
        targeted_productivity = request.form.get('targeted_productivity', '')
        smv = request.form.get('smv', '')
        over_time = request.form.get('over_time', '')
        incentive = request.form.get('incentive', '')
        idle_time = request.form.get('idle_time', '')
        idle_men = request.form.get('idle_men', '')
        no_of_style_change = request.form.get('no_of_style_change', '')
        no_of_workers = request.form.get('no_of_workers', '')
        month = request.form.get('month', '')

        # Validate all fields are filled
        if not all([quarter, department, day, team, targeted_productivity, smv, over_time, 
                    incentive, idle_time, idle_men, no_of_style_change, no_of_workers, month]):
            return render_template('submit.html', 
                                 prediction_text='Error: All fields are required!', 
                                 graphs=[])

        # Preparing input for model
        total = [[int(quarter), int(department), int(day), int(team),
                  float(targeted_productivity), float(smv), int(over_time), int(incentive),
                  float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)]]
        
        # Prediction
        prediction = model.predict(total)[0]
        print(f"Prediction: {prediction}")
        
        if prediction <= 0.3:
            text = 'The employee is Averagely Productive.'
            category = 'AVERAGE'
            color = '#FFA500'  # Orange
        elif 0.3 < prediction <= 0.8:
            text = 'The employee is Medium Productive.'
            category = 'MEDIUM'
            color = '#4CAF50'  # Green
        else:
            text = 'The employee is Highly Productive.'
            category = 'HIGHLY PRODUCTIVE'
            color = '#00AA00'  # Dark Green

        graphs = []

        # Bar chart
        plt.figure()
        categories = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
        values = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
        plt.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Employee Productivity Parameters')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        # Scatter plot
        plt.figure()
        plt.scatter([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)])
        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Scatter Plot of Employee Parameters')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        # Line plot
        plt.figure()
        plt.plot([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)], marker='o')
        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Line Plot of Employee Parameters')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        # Pie chart
        plt.figure()
        sizes = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
        labels = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Pie Chart of Employee Parameters')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        # Histogram
        plt.figure()
        data = np.random.randn(100)
        plt.hist(data, bins=20, color='purple')
        plt.xlabel('Random Data')
        plt.ylabel('Frequency')
        plt.title('Histogram of Random Data')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        # Boxplot
        plt.figure()
        data = np.random.rand(100, 5)
        plt.boxplot(data)
        plt.title('Boxplot of Random Data')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        graphs.append(graph_image)
        plt.close()

        return render_template('submit.html', prediction_text=text, graphs=graphs, 
                             prediction_value=f"{prediction:.4f}", category=category, color=color)
    
    except ValueError as e:
        print(f"ValueError in predict: {e}")
        return render_template('submit.html', 
                             prediction_text=f'Error: Invalid input format. Please enter numbers only. Details: {str(e)}', 
                             graphs=[], prediction_value='N/A', category='ERROR', color='#FF0000')
    except Exception as e:
        print(f"Unexpected error in predict: {e}")
        import traceback
        traceback.print_exc()
        return render_template('submit.html', 
                             prediction_text=f'Error: An unexpected error occurred: {str(e)}', 
                             graphs=[], prediction_value='N/A', category='ERROR', color='#FF0000')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)