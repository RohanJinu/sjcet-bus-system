"""
Bus Route Optimization — Flask Backend
Adds role-based access: admin (localhost), student, driver (via ngrok/login).
All AHP, ML, and GPS logic unchanged.
"""

from flask import Flask, jsonify, request, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import csv
from math import radians, cos, sin, asin, sqrt
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "sjcet-bus-demo-2026")

# ── LOAD ASSETS ONCE AT STARTUP ────────────────────────────────
bundle     = joblib.load("risk_model.pkl")
risk_model = bundle["model"]
le_weather = bundle["weather_encoder"]
le_road    = bundle["road_encoder"]
le_light   = bundle["light_encoder"]

weather_df = pd.read_csv("indian_weather_data.csv")
weather_df.columns = weather_df.columns.str.lower().str.strip()

# ── UTILITIES ──────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return 2 * R * asin(sqrt(a))

def _detect_col(candidates):
    for c in candidates:
        if c in weather_df.columns:
            return c
    return None

_col_lat  = _detect_col(["lat", "latitude", "Lat", "Latitude"])
_col_lon  = _detect_col(["lon", "longitude", "lng", "Lon", "Longitude", "Lng"])
_col_prec = _detect_col(["precip", "precipitation", "rainfall", "rain",
                          "prcp", "Precip", "Precipitation", "Rainfall"])
_col_vis  = _detect_col(["visibility", "vis", "Visibility", "Vis"])

if _col_prec is None or _col_vis is None:
    num_cols = weather_df.select_dtypes(include="number").columns.tolist()

def nearest_weather(lat, lon):
    if _col_lat and _col_lon:
        dists = weather_df.apply(
            lambda r: haversine(lat, lon, float(r[_col_lat]), float(r[_col_lon])), axis=1
        )
        row = weather_df.iloc[dists.idxmin()]
    else:
        row = weather_df.iloc[0]
    return row

def _get_precip(row):
    if _col_prec and _col_prec in row.index:
        v = row[_col_prec]
        return float(v) if not pd.isna(v) else 0.0
    for c in row.index:
        if any(kw in str(c).lower() for kw in ["rain", "precip", "prcp"]):
            try: return float(row[c])
            except: pass
    return 0.0

def _get_visibility(row):
    if _col_vis and _col_vis in row.index:
        v = row[_col_vis]
        return float(v) if not pd.isna(v) else 10.0
    for c in row.index:
        if any(kw in str(c).lower() for kw in ["vis", "sight"]):
            try: return float(row[c])
            except: pass
    return 10.0

_RISK_CACHE = {}
def _build_risk_cache():
    for wcat in ["clear", "cloudy", "rainy"]:
        for rcat in ["dry", "wet"]:
            for lcat in ["day", "night"]:
                try:
                    X = pd.DataFrame({
                        "weather_enc": [le_weather.transform([wcat])[0]],
                        "road_enc":    [le_road.transform([rcat])[0]],
                        "light_enc":   [le_light.transform([lcat])[0]],
                    })
                    prob = float(risk_model.predict_proba(X)[0][1])
                except Exception:
                    prob = 0.35
                _RISK_CACHE[(wcat, rcat, lcat)] = prob

_build_risk_cache()

def compute_risk_prob(lat, lon):
    w = nearest_weather(lat, lon)
    rainfall   = _get_precip(w)
    visibility = _get_visibility(w)
    weather_cat = "rainy"  if rainfall   > 20 else ("cloudy" if rainfall > 5 else "clear")
    road_cat    = "wet"    if rainfall   > 10 else "dry"
    light_cat   = "night"  if visibility <  3  else "day"
    raw_prob = _RISK_CACHE.get((weather_cat, road_cat, light_cat), 0.35)
    cache_vals = list(_RISK_CACHE.values())
    v_min, v_max = min(cache_vals), max(cache_vals)
    if v_max > v_min:
        normalised = (raw_prob - v_min) / (v_max - v_min)
        calibrated = 0.25 + normalised * 0.20
    else:
        calibrated = 0.35
    return {
        "risk_prob":   round(calibrated, 4),
        "raw_prob":    round(raw_prob, 4),
        "rainfall_mm": round(rainfall, 1),
        "visibility":  round(visibility, 1),
        "weather":     weather_cat,
        "road":        road_cat,
        "lighting":    light_cat,
    }

# ── IN-MEMORY GPS TRACKING ────────────────────────────────────
import time as _time
active_buses = {}

# ── IN-MEMORY ASSIGNMENTS ─────────────────────────────────────
# Populated when admin saves assignments. No database.
driver_assignments  = {}   # { driver_email: bus_id }
student_assignments = {}   # { student_email: bus_id }  set when admin publishes
routes_published    = False

# Also store driver_name so /api/get-drivers can return it
_driver_name_map    = {}   # { driver_email: driver_name }
route_geometry      = {}   # { bus_id: [{lat, lng, name}, ...] } — persisted to JSON

def _load_bus_assignments():
    """Re-populate assignments + route geometry from disk on startup."""
    global driver_assignments, routes_published, _driver_name_map, route_geometry
    # Load driver assignments CSV
    if os.path.exists("bus_assignments.csv"):
        try:
            with open("bus_assignments.csv", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bus_id = row.get("Bus ID", "").strip()
                    name   = row.get("Driver",  "").strip()
                    email  = row.get("Email",   "").strip().lower()
                    if bus_id and email:
                        driver_assignments[email] = bus_id
                        _driver_name_map[email]   = name
            if driver_assignments:
                routes_published = True
        except Exception:
            pass
    # Load route geometry JSON
    if os.path.exists("route_geometry.json"):
        try:
            with open("route_geometry.json", encoding="utf-8") as f:
                route_geometry = json.load(f)
        except Exception:
            pass

import json as _json
_load_bus_assignments()   # run once at startup

# ── IN-MEMORY ROLE STORE ───────────────────────────────────────
# Loaded from CSV files on demand. No database.
# drivers.csv is NEVER passed to AHP/ML — used only for login.
def _load_emails(filepath):
    """Return a dict of lowercased email -> name from a CSV file."""
    result = {}
    if not os.path.exists(filepath):
        return result
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find the email column (case-insensitive)
                email_val = None
                name_val  = None
                for k, v in row.items():
                    kl = k.lower().strip()
                    if kl == 'email' and v:
                        email_val = v.strip().lower()
                    if kl == 'name' and v:
                        name_val = v.strip()
                if email_val:
                    result[email_val] = name_val or email_val
    except Exception:
        pass
    return result

ADMIN_EMAIL = "admin@sjcetpalai.ac.in"

def _get_role(email):
    """Return ('admin'|'student'|'driver'|None, name)"""
    email = email.strip().lower()
    if email == ADMIN_EMAIL:
        return "admin", "Admin"
    drivers  = _load_emails("drivers.csv")
    students = _load_emails("students.csv")
    if email in drivers:
        return "driver", drivers[email]
    if email in students:
        return "student", students[email]
    return None, None

# ── ACCESS CONTROL DECORATORS ─────────────────────────────────
def admin_only(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "role" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── ROUTES ─────────────────────────────────────────────────────

@app.route("/")
def index():
    if "role" not in session:
        return redirect(url_for("login"))
    role  = session["role"]
    name  = session.get("name", "")
    email = session.get("email", "")
    # For students: tell the frontend which bus they're assigned to
    user_bus = student_assignments.get(email, "") if role == "student" else ""
    return render_template("index.html", role=role, user_name=name, user_bus=user_bus)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        role, name = _get_role(email)
        if role == "admin":
            session["role"]  = role
            session["email"] = email
            session["name"]  = name
            return redirect(url_for("index"))
        elif role in ("student", "driver"):
            if not routes_published:
                error = "Routes have not been published yet. Please check back later."
            elif role == "driver" and email not in driver_assignments:
                error = "You have not been assigned to a bus yet. Please contact the Transport Office."
            else:
                session["role"]  = role
                session["email"] = email
                session["name"]  = name
                if role == "driver":
                    return redirect(url_for("driver"))
                else:
                    return redirect(url_for("index"))
        else:
            error = "Invalid email address."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/risk", methods=["POST"])
@admin_only
def api_risk():
    data  = request.get_json(force=True)
    stops = data.get("stops", [])
    if not stops:
        return jsonify({"error": "No stops provided"}), 400
    results = []
    for s in stops:
        try:
            bd = compute_risk_prob(float(s["lat"]), float(s["lon"]))
            results.append({
                "lat":       s["lat"],
                "lon":       s["lon"],
                "risk_prob": bd["risk_prob"],
                "breakdown": {
                    "rainfall_mm": bd["rainfall_mm"],
                    "visibility":  bd["visibility"],
                    "weather":     bd["weather"],
                    "road":        bd["road"],
                    "lighting":    bd["lighting"],
                    "raw_prob":    bd["raw_prob"],
                }
            })
        except Exception as e:
            results.append({
                "lat":       s["lat"],
                "lon":       s["lon"],
                "risk_prob": 0.35,
                "breakdown": None,
                "error":     str(e)
            })
    return jsonify({"stops": results})


@app.route("/driver")
def driver():
    role = session.get("role")
    if role not in ("driver", "admin"):
        return redirect(url_for("login"))
    driver_name  = session.get("name", "Driver")
    driver_email = session.get("email", "")
    assigned_bus = driver_assignments.get(driver_email, None)
    return render_template("driver.html", driver_name=driver_name, assigned_bus=assigned_bus)


@app.route("/api/gps-update", methods=["POST"])
def api_gps_update():
    role = session.get("role")
    if role not in ("driver", "admin"):
        return jsonify({"error": "Driver access required"}), 403
    data = request.get_json(force=True)
    driver_email = session.get("email", "")
    role         = session.get("role")
    # Drivers: always use server-assigned bus_id (not client-provided)
    if role == "driver":
        bus_id = driver_assignments.get(driver_email, "")
        if not bus_id:
            return jsonify({"error": "No bus assigned to this driver"}), 403
    else:
        # Admin testing — accept client-provided bus_id
        bus_id = str(data.get("bus_id", "")).strip()
    if not bus_id:
        return jsonify({"error": "bus_id required"}), 400
    try:
        active_buses[bus_id] = {
            "bus_id":    bus_id,
            "lat":       float(data["lat"]),
            "lon":       float(data["lon"]),
            "speed":     float(data.get("speed", 0)),
            "timestamp": str(data.get("timestamp", "")),
            "last_seen": _time.time(),
        }
        return jsonify({"ok": True, "bus_id": bus_id})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/gps-stop", methods=["POST"])
def api_gps_stop():
    """
    POST /api/gps-stop
    Called immediately when the driver taps Stop Tracking.
    Deletes the bus from active_buses so the next poll shows it gone.
    """
    role = session.get("role")
    if role not in ("driver", "admin"):
        return jsonify({"error": "Driver access required"}), 403
    driver_email = session.get("email", "")
    if role == "driver":
        bus_id = driver_assignments.get(driver_email, "")
    else:
        data   = request.get_json(force=True, silent=True) or {}
        bus_id = str(data.get("bus_id", "")).strip()
    if bus_id and bus_id in active_buses:
        del active_buses[bus_id]
    return jsonify({"ok": True, "removed": bus_id})


@app.route("/api/live-status", methods=["GET"])
def api_live_status():
    role = session.get("role")
    if role not in ("student", "admin"):
        return jsonify({"error": "Access denied"}), 403
    now = _time.time()
    all_fresh = [
        {**b, "last_seen": round(now - b["last_seen"])}
        for b in active_buses.values()
        if now - b["last_seen"] <= 30   # 30s safety-net for missed stop signals
    ]
    if role == "student":
        # Return only the bus assigned to this student (looked up server-side)
        email  = session.get("email", "")
        my_bus = student_assignments.get(email)
        if my_bus:
            fresh = [b for b in all_fresh if b["bus_id"] == my_bus]
        else:
            fresh = all_fresh   # fallback: no map yet, show all
    else:
        fresh = all_fresh       # admin sees every active bus
    return jsonify({"buses": fresh, "count": len(fresh)})


@app.route("/api/send-notifications", methods=["POST"])
@admin_only
def api_send_notifications():
    import smtplib
    from email.mime.text import MIMEText

    data     = request.get_json(force=True)
    students = data.get("students", [])
    mode     = data.get("mode", "all")
    n        = int(data.get("n", len(students)))

    if mode == "limit":
        students = students[:n]

    sender   = os.environ.get("GMAIL_ADDRESS", "")
    password = os.environ.get("GMAIL_APP_PASSWORD", "")

    if not sender or not password:
        return jsonify({"error": "GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set in environment"}), 500

    sent, failed, errors = 0, 0, []

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
    except Exception as e:
        return jsonify({"error": f"SMTP login failed: {e}"}), 500

    for s in students:
        recipient = s.get("email", "").strip()
        if not recipient:
            failed += 1
            errors.append(f"No email for student: {s.get('name','?')}")
            continue
        try:
            stop_lat = s.get('stop_lat', '')
            stop_lon = s.get('stop_lon', '')
            maps_line = ""
            if stop_lat and stop_lon:
                maps_url  = f"https://maps.google.com/?q={stop_lat},{stop_lon}"
                maps_line = f"\n  📍 Stop Location : {maps_url}\n"
            body = (
                f"Dear {s.get('name', 'Student')},\n\n"
                f"Your bus route assignment for tomorrow is ready.\n\n"
                f"  Bus Number   : {s.get('bus', 'N/A')}\n"
                f"  Boarding Stop: {s.get('stop', 'N/A')}{maps_line}"
                f"  Est. Arrival : {s.get('eta', 'N/A')} minutes from college\n\n"
                f"Tap the link above to navigate to your boarding stop.\n"
                f"Please be at your stop 5 minutes before the scheduled time.\n\n"
                f"Regards,\n"
                f"SJCET Management\n"
                f"St. Joseph's College of Engineering and Technology, Palai"
            )
            msg = MIMEText(body)
            msg["Subject"] = f"Bus Route Assignment — {s.get('bus', 'N/A')}"
            msg["From"]    = sender
            msg["To"]      = recipient
            server.sendmail(sender, recipient, msg.as_string())
            sent += 1
        except Exception as e:
            failed += 1
            errors.append(f"{recipient}: {e}")

    server.quit()
    return jsonify({"sent": sent, "failed": failed, "errors": errors})


@app.route("/api/save-assignments", methods=["POST"])
@admin_only
def api_save_assignments():
    """
    POST /api/save-assignments
    Body: {
        "assignments":  [ {bus_id, driver_name, driver_email}, ... ],
        "student_map":  { student_email: bus_id, ... }   (optional)
    }
    Saves bus_assignments.csv, updates driver_assignments + student_assignments.
    Sets routes_published = True so students/drivers can now log in.
    """
    global driver_assignments, student_assignments, routes_published
    data = request.get_json(force=True)
    rows = data.get("assignments", [])
    if not rows:
        return jsonify({"error": "No assignments provided"}), 400

    # Write CSV
    try:
        with open("bus_assignments.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Bus ID", "Driver", "Email"])
            for r in rows:
                writer.writerow([r.get("bus_id",""), r.get("driver_name",""), r.get("driver_email","")])
    except Exception as e:
        return jsonify({"error": f"Could not write CSV: {e}"}), 500

    # Update driver lookup + name map
    driver_assignments = {
        r["driver_email"].strip().lower(): r["bus_id"]
        for r in rows if r.get("driver_email") and r.get("bus_id")
    }
    _driver_name_map = {
        r["driver_email"].strip().lower(): r.get("driver_name", "")
        for r in rows if r.get("driver_email")
    }
    # Update student lookup (sent from frontend after optimization)
    raw_map = data.get("student_map", {})
    student_assignments = {
        k.strip().lower(): v for k, v in raw_map.items() if k and v
    }
    # Save route geometry (stop lat/lng/name per bus) for map persistence
    raw_geo = data.get("route_geometry", {})
    if raw_geo:
        route_geometry.update(raw_geo)
        try:
            with open("route_geometry.json", "w", encoding="utf-8") as f:
                _json.dump(route_geometry, f)
        except Exception:
            pass

    routes_published = True
    return jsonify({"ok": True, "assigned": len(driver_assignments)})


@app.route("/api/get-drivers", methods=["GET"])
@admin_only
def api_get_drivers():
    """
    GET /api/get-drivers
    Returns the list of drivers from drivers.csv and current assignments.
    Used by the assignment modal in the frontend.
    """
    drivers = _load_emails("drivers.csv")
    driver_list = [{"name": name, "email": email} for email, name in drivers.items()]

    # Also return existing assignments so modal can pre-fill (include name for sidebar)
    existing = [
        {
            "bus_id":      bus,
            "driver_email": email,
            # Prefer name from _driver_name_map; fallback to scanning bus_assignments.csv directly
            "driver_name": _driver_name_map.get(email.strip().lower(), "")
        }
        for email, bus in driver_assignments.items()
    ]
    return jsonify({
        "drivers":     driver_list,
        "assignments": existing,
        "published":   routes_published
    })


@app.route("/api/route-geometry", methods=["GET"])
def api_route_geometry():
    """Return persisted route geometry for all buses. Available to any logged-in user."""
    if "role" not in session:
        return jsonify({"error": "Login required"}), 403
    return jsonify({"routes": route_geometry})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
