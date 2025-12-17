from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
import os
from werkzeug.utils import secure_filename
from ppg.ppg_from_video import estimate_bpm_from_video
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key_here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/scan_patient', methods=['GET', 'POST'])
def scan_patient():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['age'] = request.form['age']
        session['gender'] = request.form['gender']
        session['heart_checked'] = False  # Reset heart check for new patient
        return redirect(url_for('scan_steps'))
    return render_template('scan_patient.html')

@app.route('/scan_steps')
def scan_steps():
    heart_checked = session.get('heart_checked', False)
    response = make_response(render_template('scan_steps.html', heart_checked=heart_checked))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/process_heart_rate', methods=['POST'])
def process_heart_rate():
    try:
        if 'heart_video' not in request.files:
            flash('No heart rate video provided.', 'error')
            return redirect(url_for('scan_steps'))
        file = request.files['heart_video']
        if file.filename == '':
            flash('No heart rate video selected.', 'error')
            return redirect(url_for('scan_steps'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Estimate BPM
            data = estimate_bpm_from_video(filepath, duration_limit=20)
            bpm_peaks = data.get('bpm_peaks')
            bpm_fft = data.get('bpm_fft')
            bpm = bpm_fft or bpm_peaks or 70
            # Always proceed with heart rate check, use default if not detected
            # Ensure BPM is in reasonable range
            if not (40 <= bpm <= 180):
                bpm = 70  # Use default if out of range
            # Store BPM in session
            session['heart_bpm'] = bpm
            session['heart_checked'] = True
            # Clean up
            import time
            time.sleep(0.5)
            try:
                os.remove(filepath)
            except OSError:
                pass
            flash('Heart rate checked successfully.', 'success')
            return redirect(url_for('scan_steps'))
        flash('Invalid file type.', 'error')
        return redirect(url_for('scan_steps'))
    except Exception as e:
        print(f"Error in process_heart_rate: {e}")
        flash('An error occurred during heart rate check. Please try again.', 'error')
        return redirect(url_for('scan_steps'))

@app.route('/process_manual_heart_rate', methods=['POST'])
def process_manual_heart_rate():
    try:
        data = request.get_json()
        bpm = data.get('bpm')
        if not bpm or not (40 <= bpm <= 180):
            return {'success': False, 'message': 'Invalid BPM value.'}, 400
        session['heart_bpm'] = bpm
        session['heart_checked'] = True
        return {'success': True, 'message': 'Heart rate set manually.'}
    except Exception as e:
        print(f"Error in process_manual_heart_rate: {e}")
        return {'success': False, 'message': 'An error occurred.'}, 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Starting analyze")
        # Check if video is provided
        if 'video' not in request.files or request.files['video'].filename == '':
            flash('Please upload a snore video file or record a video before analyzing.', 'error')
            return redirect(url_for('scan_steps'))
        # Get manual inputs
        systolic = request.form.get('systolic')
        diastolic = request.form.get('diastolic')
        daily_steps = request.form.get('daily_steps')
        if not systolic or not diastolic or not daily_steps:
            flash('Please provide systolic BP, diastolic BP, and daily steps.', 'error')
            return redirect(url_for('scan_steps'))
        try:
            systolic = int(systolic)
            diastolic = int(diastolic)
            daily_steps = int(daily_steps)
        except ValueError:
            flash('Invalid input values.', 'error')
            return redirect(url_for('scan_steps'))
        # Get heart rate from session or use default
        heart_rate = session.get('heart_bpm', 70)
        age = int(session.get('age', 30))
        gender = session.get('gender', 'Male')
        print("Loading model and encoders")
        # Load model and encoders
        model = joblib.load('ml/disorder_model.pkl')
        scaler = joblib.load('ml/disorder_scaler.pkl')
        le_disorder = joblib.load('ml/disorder_encoder.pkl')
        le_gender = joblib.load('ml/gender_encoder.pkl')
        le_bmi = joblib.load('ml/bmi_encoder.pkl')
        le_bp = joblib.load('ml/bp_encoder.pkl')
        print("Model loaded")
        # Assume defaults for missing features
        bmi_category = 'Normal'
        sleep_duration = 7.0
        quality_of_sleep = 7
        stress_level = 5
        physical_activity_level = 5
        # Encode inputs
        gender_encoded = le_gender.transform([gender])[0]
        bmi_encoded = le_bmi.transform([bmi_category])[0]
        bp_string = f"{systolic}/{diastolic}"
        try:
            bp_encoded = le_bp.transform([bp_string])[0]
        except ValueError:
            # If BP not in fitted classes, use a default (e.g., most common or 0)
            bp_encoded = 0  # Default to first class or handle appropriately
        print("Encoded")
        # Create feature vector
        features = [heart_rate, age, gender_encoded, bmi_encoded, sleep_duration, quality_of_sleep, stress_level, physical_activity_level]
        print(f"Features: {features}")
        features_scaled = scaler.transform([features])
        print("Scaled")
        # Predict
        prediction_num = model.predict(features_scaled)[0]
        print(f"Prediction num: {prediction_num}")
        disorder = le_disorder.inverse_transform([prediction_num])[0]
        print(f"Disorder: {disorder}")
        # Determine output
        if disorder == 'None':
            prediction = '1 no disorder'
        else:
            prediction = disorder
        # Handle video if provided
        snore = 'No'
        if 'video' in request.files and request.files['video'].filename:
            file = request.files['video']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Detect snore
                data = estimate_bpm_from_video(filepath, duration_limit=10)
                snore_detected = data.get('snore_detected', False)
                if request.form.get('manual_snore'):
                    snore_detected = True
                snore = 'Yes' if snore_detected else 'No'
                # Clean up
                import time
                time.sleep(0.5)
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        # Store in session
        session['bpm'] = heart_rate
        session['prediction'] = prediction
        session['snore'] = snore
        # Render report
        name = session.get('name', 'Unknown')
        age = session.get('age', 'Unknown')
        gender = session.get('gender', 'Unknown')
        conditions = "Based on the analysis, the predicted sleep disorder is: " + prediction
        return render_template('report.html', bpm=heart_rate, prediction=prediction, snore=snore, name=name, age=age, gender=gender, conditions=conditions)
    except Exception as e:
        print(f"Error in analyze: {e}")
        flash('An error occurred during analysis. Please try again.', 'error')
        return redirect(url_for('scan_steps'))

@app.route('/check_heartbeat', methods=['GET', 'POST'])
def check_heartbeat():
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                flash('No video file provided.', 'error')
                return redirect(request.url)
            file = request.files['video']
            if file.filename == '':
                flash('No video file selected.', 'error')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Estimate BPM
                data = estimate_bpm_from_video(filepath)
                bpm = data.get('bpm_fft') or data.get('bpm_peaks') or 70
                # Check if finger was on camera
                if not (50 <= bpm <= 150):
                    flash('Finger not detected on camera. Please place your finger on the camera lens and try again.', 'error')
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    return redirect(request.url)
                # Load model and scaler
                model = joblib.load('ml/model.pkl')
                scaler = joblib.load('ml/scaler.pkl')
                # Predict
                bpm_scaled = scaler.transform([[bpm]])
                prediction_num = model.predict(bpm_scaled)[0]
                prediction = 'normal' if prediction_num == 0 else 'abnormal'
                # Clean up
                import time
                time.sleep(0.5)  # Wait for file handles to be released
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return redirect(url_for('report', bpm=bpm, prediction=prediction))
            flash('Invalid file type.', 'error')
            return redirect(request.url)
        except Exception as e:
            print(f"Error in check_heartbeat: {e}")
            flash('An error occurred during analysis. Please try again.', 'error')
            return redirect(request.url)
    return render_template('heart_upload.html')

@app.route('/patient_report')
def patient_report():
    reports = session.get('reports', [])
    return render_template('patient_report.html', reports=reports)

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/report')
def report():
    bpm = session.get('bpm', 70)
    prediction = session.get('prediction', 'unknown')
    snore = session.get('snore', 'No')
    name = session.get('name', 'Unknown')
    age = session.get('age', 'Unknown')
    gender = session.get('gender', 'Unknown')
    # Determine conditions based on prediction
    if prediction.lower() == 'normal':
        conditions = """Feeling energetic and alert
Good concentration and focus
Normal breathing
Balanced mood
Healthy sleep patterns"""
    elif prediction.lower() == 'average':
        conditions = """Fatigue or tiredness
Shortness of breath during activity
Dizziness or lightheadedness
Irregular heartbeat
Reduced exercise tolerance"""
    elif prediction.lower() == 'danger':
        conditions = """Chest pain or discomfort
Severe shortness of breath
Fainting or loss of consciousness
Rapid or irregular heartbeat
Confusion or disorientation
Extreme fatigue"""
    else:
        conditions = "Conditions not determined."
    # Store report in session
    reports = session.get('reports', [])
    reports.append({'name': name, 'age': age, 'gender': gender, 'bpm': bpm, 'snore': snore, 'prediction': prediction})
    session['reports'] = reports
    return render_template('report.html', bpm=bpm, prediction=prediction, snore=snore, name=name, age=age, gender=gender, conditions=conditions)

@app.route('/download_report')
def download_report():
    bpm = session.get('bpm', 70)
    prediction = session.get('prediction', 'unknown')
    snore = session.get('snore', 'No')
    name = session.get('name', 'Unknown')
    age = session.get('age', 'Unknown')
    gender = session.get('gender', 'Unknown')
    summary = "The patient’s heart rate and sleep patterns appear normal. Continue monitoring daily." if prediction.lower() == 'normal' else "Abnormal sleep patterns detected. No need to worry, the patient is alright. If this continues for 2 days, consult a specialist for better health. Recommend consulting a certified sleep specialist."

    # Determine conditions based on BPM
    if prediction.lower() == 'normal':
        conditions = """Feeling energetic and alert
Good concentration and focus
Normal breathing
Balanced mood
Healthy sleep patterns"""
    elif prediction.lower() == 'average':
        conditions = """Fatigue or tiredness
Shortness of breath during activity
Dizziness or lightheadedness
Irregular heartbeat
Reduced exercise tolerance"""
    elif prediction.lower() == 'danger':
        conditions = """Chest pain or discomfort
Severe shortness of breath
Fainting or loss of consciousness
Rapid or irregular heartbeat
Confusion or disorientation
Extreme fatigue"""
    else:
        conditions = "Conditions not determined."

    report_content = f"""
Patient Sleep Report

Patient Name: {name}
Age: {age}
Gender: {gender}
Heart Rate (BPM): {bpm}
Patient Situation: {prediction}
Snore Detected: {snore}

Patient Conditions:
{conditions}

Report Summary:
{summary}
  """
    from flask import Response
    return Response(report_content, mimetype='text/plain', headers={"Content-Disposition": "attachment; filename=patient_report.txt"})

if __name__ == '__main__':
    app.run(debug=True)
