from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import sqlite3
import hashlib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import random
import joblib

app = Flask(__name__)
app.secret_key = 'movieai_secret_key_2024'

# LOAD ML MODEL & DATA AT STARTUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    models = {}
    try:
        models['rf_model']      = joblib.load(os.path.join(BASE_DIR, 'best_recommendation_model.pkl'))
        models['user_encoder']  = joblib.load(os.path.join(BASE_DIR, 'user_encoder.pkl'))
        models['movie_encoder'] = joblib.load(os.path.join(BASE_DIR, 'movie_encoder.pkl'))
        models['scaler']        = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
        print("ML models loaded successfully.")
        return models
    except Exception as e:
        print(f"Model load error: {e}")
        return None

def load_movie_data():
    meta_path    = os.path.join(BASE_DIR, 'movies_metadata.csv')
    mapping_path = os.path.join(BASE_DIR, 'movie_mapping.csv')
    metadata = None
    mapping  = None
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path)
        metadata['year'] = pd.to_numeric(metadata['year'], errors='coerce').fillna(0).astype(int)
        metadata['genres'] = metadata['genres'].fillna('Unknown')
        print(f"movies_metadata.csv loaded: {len(metadata)} rows")
    else:
        print("movies_metadata.csv not found — place it in the app folder")
    if os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path)
        print(f"movie_mapping.csv loaded: {len(mapping)} rows")
    else:
        print("movie_mapping.csv not found — place it in the app folder")
    return metadata, mapping

ML_MODELS, MOVIES_METADATA, MOVIE_MAPPING = None, None, None

def init_app_data():
    global ML_MODELS, MOVIES_METADATA, MOVIE_MAPPING, GENRES
    ML_MODELS = load_models()
    MOVIES_METADATA, MOVIE_MAPPING = load_movie_data()
    if MOVIES_METADATA is not None:
        all_genres = set()
        for g in MOVIES_METADATA['genres'].dropna():
            for part in g.split('|'):
                p = part.strip()
                if p and p != '(no genres listed)':
                    all_genres.add(p)
        GENRES = sorted(list(all_genres))
    else:
        GENRES = []

GENRES = []
init_app_data()

# DATABASE SETUP

def get_db():
    db = sqlite3.connect(os.path.join(BASE_DIR, 'movie_app.db'))
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            avatar_color TEXT DEFAULT '#d4a843'
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_title TEXT,
            movie_id INTEGER,
            genre TEXT,
            year INTEGER,
            predicted_rating REAL,
            recommendation TEXT,
            confidence REAL,
            model_used TEXT DEFAULT 'HybridRF',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.commit()

    # Migration: add columns missing in older DB versions
    existing_cols = [row[1] for row in db.execute('PRAGMA table_info(predictions)').fetchall()]
    if 'movie_id' not in existing_cols:
        db.execute('ALTER TABLE predictions ADD COLUMN movie_id INTEGER')
        print('DB migrated: added movie_id column')
    if 'model_used' not in existing_cols:
        db.execute('ALTER TABLE predictions ADD COLUMN model_used TEXT DEFAULT HybridRF')
        print('DB migrated: added model_used column')
    db.commit()
    db.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# PREDICTION ENGINE — USES SAVED RF MODEL

def get_encoded_movie(movie_id):
    if MOVIE_MAPPING is None:
        return None
    row = MOVIE_MAPPING[MOVIE_MAPPING['movieId'] == int(movie_id)]
    if len(row) == 0:
        return None
    return int(row.iloc[0]['movie_encoded'])

def predict_with_model(movie_id, year, user_id_raw=1):

    if ML_MODELS is None:
        return None, "ML model files not found. Place .pkl files in the app directory."

    rf_model      = ML_MODELS['rf_model']
    user_encoder  = ML_MODELS['user_encoder']
    scaler        = ML_MODELS['scaler']

    # --- Encode user ---
    known_users = list(user_encoder.classes_)
    if user_id_raw in known_users:
        user_enc = int(user_encoder.transform([user_id_raw])[0])
    else:
        # Anonymous / new user: use middle of the known user range
        median_user = known_users[len(known_users) // 2]
        user_enc = int(user_encoder.transform([median_user])[0])

    # --- Encode movie ---
    movie_enc = get_encoded_movie(movie_id)
    if movie_enc is None:
        if MOVIE_MAPPING is not None:
            movie_enc = int(MOVIE_MAPPING['movie_encoded'].median())
        else:
            movie_enc = 0

    features = pd.DataFrame(
        [[user_enc, movie_enc, int(year)]],
        columns=['user_encoded', 'movie_encoded', 'year']
    )

    try:
        features_scaled = scaler.transform(features)
        pred_label      = int(rf_model.predict(features_scaled)[0])
        pred_proba      = rf_model.predict_proba(features_scaled)[0]
        confidence      = float(max(pred_proba)) * 100

        # Training: label 1 = rating >= 3.5, label 0 = rating < 3.5
        if pred_label == 1:
            predicted_rating = round(3.5 + (confidence / 100) * 1.45, 2)
        else:
            predicted_rating = round(0.5 + ((100 - confidence) / 100) * 2.85, 2)

        predicted_rating = min(5.0, max(0.5, predicted_rating))
        recommendation   = "Recommended" if pred_label == 1 else "Not Recommended"

        return {
            "predicted_rating": round(predicted_rating, 2),
            "recommendation":   recommendation,
            "confidence":       round(confidence, 1),
            "model_used":       "Hybrid RF (Best Model)"
        }, None

    except Exception as e:
        return None, str(e)


def get_genre_recommendations(genres_str, exclude_movie_id=None, limit=6):
    """Return top movies from metadata CSV matching the given genre string"""
    if MOVIES_METADATA is None:
        return []
    input_genres = set(g.strip() for g in str(genres_str).split('|') if g.strip())
    scored = []
    for _, row in MOVIES_METADATA.iterrows():
        if exclude_movie_id and int(row['movieId']) == int(exclude_movie_id):
            continue
        movie_genres = set(g.strip() for g in str(row['genres']).split('|'))
        overlap = len(input_genres & movie_genres)
        if overlap > 0:
            scored.append((overlap, row['movieId'], row['title'], row['genres'], int(row['year'])))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    seen = set()
    for item in scored:
        if item[1] not in seen:
            seen.add(item[1])
            results.append({
                "movieId": int(item[1]),
                "title":   item[2],
                "genres":  item[3],
                "year":    item[4]
            })
        if len(results) >= limit:
            break
    return results

# AUTH ROUTES

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        if not all([username, email, password, confirm]):
            flash('All fields are required.', 'error'); return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'error'); return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error'); return render_template('register.html')
        colors = ['#d4a843','#c0392b','#43b89c','#a29bfe','#fd79a8','#74b9ff']
        try:
            db = get_db()
            db.execute('INSERT INTO users (username,email,password,avatar_color) VALUES (?,?,?,?)',
                       (username, email, hash_password(password), random.choice(colors)))
            db.commit(); db.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db   = get_db()
        user = db.execute('SELECT * FROM users WHERE username=? AND password=?',
                          (username, hash_password(password))).fetchone()
        db.close()
        if user:
            session['user_id']      = user['id']
            session['username']     = user['username']
            session['avatar_color'] = user['avatar_color']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# DASHBOARD

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db()
    predictions = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10',
        (session['user_id'],)).fetchall()
    total = db.execute('SELECT COUNT(*) as c FROM predictions WHERE user_id=?',
                       (session['user_id'],)).fetchone()['c']
    recommended = db.execute(
        'SELECT COUNT(*) as c FROM predictions WHERE user_id=? AND recommendation="Recommended"',
        (session['user_id'],)).fetchone()['c']
    db.close()
    avg_rating = round(sum(p['predicted_rating'] for p in predictions) / len(predictions), 2) if predictions else 0

    stats = {"total": total, "recommended": recommended,
             "not_recommended": total - recommended, "avg_rating": avg_rating}

    model_info = {
        "model_loaded":    ML_MODELS is not None,
        "data_loaded":     MOVIES_METADATA is not None,
        "mapping_loaded":  MOVIE_MAPPING is not None,
        "total_movies":    len(MOVIES_METADATA) if MOVIES_METADATA is not None else 0,
        "total_genres":    len(GENRES)
    }
    return render_template('dashboard.html', predictions=predictions, stats=stats,
                           genres=GENRES, model_info=model_info)

# SEARCH MOVIES — FROM REAL CSV DATA

@app.route('/search_movies')
def search_movies():
    query = request.args.get('q', '').strip().lower()
    if len(query) < 2 or MOVIES_METADATA is None:
        return jsonify([])
    mask = MOVIES_METADATA['title'].str.lower().str.contains(query, na=False, regex=False)
    matches = MOVIES_METADATA[mask][['movieId','title','genres','year']].head(12)
    return jsonify(matches.to_dict(orient='records'))

# PREDICT API

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data     = request.get_json()
    movie_id = data.get('movie_id')
    title    = data.get('title', '').strip()
    genres   = data.get('genres', '')
    year     = int(data.get('year', 2000))

    if not movie_id:
        return jsonify({'error': 'Please select a movie from the dropdown suggestions.'}), 400

    result, error = predict_with_model(movie_id, year, user_id_raw=session['user_id'])
    if error:
        return jsonify({'error': error}), 500

    genre_recs = get_genre_recommendations(genres, exclude_movie_id=movie_id, limit=6)

    db = get_db()
    db.execute(
        '''INSERT INTO predictions (user_id,movie_title,movie_id,genre,year,
           predicted_rating,recommendation,confidence,model_used)
           VALUES (?,?,?,?,?,?,?,?,?)''',
        (session['user_id'], title, movie_id, genres, year,
         result['predicted_rating'], result['recommendation'],
         result['confidence'], result['model_used'])
    )
    db.commit(); db.close()

    return jsonify({'prediction': result, 'genre_recommendations': genre_recs,
                    'movie_title': title, 'year': year, 'genres': genres})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db()
    predictions = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
        (session['user_id'],)).fetchall()
    db.close()
    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    init_db()
    print(f"\n{'='*55}")
    print("  CineAI — AI-Based Personalized Recommendation App")
    print(f"{'='*55}")
    if ML_MODELS:
        print(f"  ML Model:      LOADED ✓")
    else:
        print(f"  ML Model:      MISSING")
        print(f"  >>> FIX: In your venv terminal run:")
        print(f"      pip install scikit-learn numpy pandas joblib")
    print(f"  Movie Data:    {'LOADED ('+str(len(MOVIES_METADATA))+' movies)' if MOVIES_METADATA is not None else 'MISSING — add movies_metadata.csv'}")
    print(f"  Movie Mapping: {'LOADED ('+str(len(MOVIE_MAPPING))+' entries)' if MOVIE_MAPPING is not None else 'MISSING — add movie_mapping.csv'}")
    print(f"  Genres:        {len(GENRES)} extracted from CSV data")
    print(f"{'='*55}\n")
    app.run(debug=True, port=5000)