from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import joblib
import numpy as np
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Create necessary directories
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('anemia_app.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  gender INTEGER,
                  hemoglobin REAL,
                  mch REAL,
                  mchc REAL,
                  mcv REAL,
                  result INTEGER,
                  probability REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_password = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                  ('admin', 'admin@anemia.com', admin_password, 'admin'))
    
    conn.commit()
    conn.close()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('anemia_app.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                      (username, email, hashed_password, 'user'))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('anemia_app.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[4]
            flash(f'Welcome back, {username}!', 'success')
            
            if user[4] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('predict'))
        else:
            flash('Invalid credentials!', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            gender = int(request.form['gender'])
            hemoglobin = float(request.form['hemoglobin'])
            mch = float(request.form['mch'])
            mchc = float(request.form['mchc'])
            mcv = float(request.form['mcv'])
            
            # Load model and predict
            model = joblib.load('xgboost_anemia_model.joblib')
            features = np.array([[gender, hemoglobin, mch, mchc, mcv]])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] * 100
            
            # Save prediction to database
            conn = sqlite3.connect('anemia_app.db')
            c = conn.cursor()
            c.execute("""INSERT INTO predictions 
                         (user_id, gender, hemoglobin, mch, mchc, mcv, result, probability)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (session['user_id'], gender, hemoglobin, mch, mchc, mcv, 
                       int(prediction), probability))
            conn.commit()
            conn.close()
            
            result = {
                'prediction': int(prediction),
                'probability': round(probability, 2),
                'status': 'Anemia Detected' if prediction == 1 else 'No Anemia',
                'color': 'danger' if prediction == 1 else 'success'
            }
            
            return render_template('predict.html', result=result)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

@app.route('/analytics')
@login_required
def analytics():
    plots_dir = 'static/plots'
    plots = []
    
    plot_files = [
        'target_distribution.png',
        'feature_distributions.png',
        'correlation_heatmap.png',
        'confusion_matrix.png',
        'roc_curve.png',
        'shap_summary_plot.png',
        'shap_feature_importance.png'
    ]
    
    for plot_file in plot_files:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            plots.append({
                'name': plot_file.replace('_', ' ').replace('.png', '').title(),
                'path': f'/static/plots/{plot_file}'
            })
    
    return render_template('analytics.html', plots=plots)

@app.route('/base-paper')
def base_paper():
    return render_template('base_paper.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = sqlite3.connect('anemia_app.db')
    c = conn.cursor()
    
    # Get all users
    c.execute("SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC")
    users = c.fetchall()
    
    # Get statistics
    c.execute("SELECT COUNT(*) FROM users WHERE role='user'")
    total_users = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM predictions WHERE result=1")
    anemia_cases = c.fetchone()[0]
    
    conn.close()
    
    stats = {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'anemia_cases': anemia_cases,
        'healthy_cases': total_predictions - anemia_cases
    }
    
    return render_template('admin_dashboard.html', users=users, stats=stats)

@app.route('/admin/user/delete/<int:user_id>')
@admin_required
def delete_user(user_id):
    if user_id == session['user_id']:
        flash('You cannot delete your own account!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    conn = sqlite3.connect('anemia_app.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/user/edit/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    conn = sqlite3.connect('anemia_app.db')
    c = conn.cursor()
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        role = request.form['role']
        
        c.execute("UPDATE users SET username=?, email=?, role=? WHERE id=?",
                  (username, email, role, user_id))
        conn.commit()
        conn.close()
        
        flash('User updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    
    return render_template('edit_user.html', user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
