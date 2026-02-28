<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%9A%8C%20SJCET%20Bus%20Route%20Management%20System-1a4d2e?style=for-the-badge&logoColor=white" alt="SJCET Bus System" height="45"/>

<br/>
<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Leaflet.js-1.9-199900?style=flat-square&logo=leaflet&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p><i>AI-powered bus route optimisation, ML risk scoring, and live GPS tracking<br/>for St. Joseph's College of Engineering and Technology, Palai</i></p>

</div>

---

## 🗺️ What Is This?

The **SJCET Bus Route Management System** replaces manual transport planning with a fully automated, data-driven pipeline. The admin uploads a student CSV, clicks Optimise, and the system instantly builds the safest and most efficient bus routes using machine learning and multi-criteria decision analysis — then lets the admin track every bus live on a map as drivers make their way to college.

**No database. No cloud. Runs entirely on a laptop.**

---

## 👥 Three Roles, One System

| Role | How They Access It | What They See |
|---|---|---|
| 🟢 **Admin** (Transport Office) | `localhost:5000` | Full portal — upload data, optimise routes, assign drivers, monitor all buses live |
| 🔵 **Driver** | `ngrok-url/driver` on their phone | Mobile GPS console — broadcasts live position every 5 seconds |
| 🟡 **Student** | `ngrok-url` on their phone | Live map of their assigned bus, ETA, boarding stop, route overlay |

---

## ✨ Features

<details>
<summary><b>🖥️ Admin Portal</b></summary>
<br/>

- 📂 **CSV upload** with drag-and-drop student data
- 🌍 **Auto-geocoding** — missing stop coordinates resolved automatically via OpenStreetMap
- ⚡ **One-click optimisation** — AHP + ML + greedy routing in under 1 second for 200+ students
- 🗺️ **Interactive route map** — colour-coded polylines, stop markers, student manifests per stop
- 🎛️ **Live parameter sliders** — adjust bus capacity and average speed, routes rebuild instantly
- 👤 **Driver assignment modal** — assign named drivers to each route before publish
- 💾 **Save & Publish** — writes `bus_assignments.csv`, unlocks driver and student logins
- 📧 **Email notifications** — each student gets bus number, stop name, ETA, and a Google Maps link to their boarding stop
- 📡 **Multi-bus live tracking** — all active buses on one map, each in a distinct colour
- ⚠️ **Off-route detection** — marker and route line turn red when bus deviates more than 300 m from planned path

</details>

<details>
<summary><b>📱 Driver Console</b></summary>
<br/>

- Mobile-optimised single-page GPS tracker — works on any phone browser
- Large animated status circle (LOCATING → ACTIVE → STOPPED)
- Real-time diagnostic log showing GPS events and server ping confirmations
- Pings server every **5 seconds** with latitude, longitude, speed, and accuracy
- **Stop Tracking** immediately removes the bus from all maps via `POST /api/gps-stop`
- Proper **Logout button** that stops tracking and clears the session before redirecting

</details>

<details>
<summary><b>🎓 Student View</b></summary>
<br/>

- Live map showing **only their assigned bus** — filtered server-side
- Planned route overlay with named **stop pin markers** and hover tooltips
- Distance to college, ETA, and speed updated every 5 seconds
- Bus pass card with route, boarding stop, and driver details

</details>

---

## 🧠 How the Optimisation Works

The route-building pipeline runs in **4 phases**, executing in under a second:

```
Student CSV  ──►  Geocoding  ──►  ML Risk Score  ──►  AHP Priority  ──►  Greedy Routing  ──►  Routes
```

### Phase 1 — Geocoding
Stop names are resolved to coordinates using a built-in Kerala stop lookup table, with an OpenStreetMap Nominatim fallback for unknown stops.

### Phase 2 — ML Risk Scoring
A **scikit-learn Random Forest** classifier predicts accident risk probability for each stop based on local weather data:

| Input | Source |
|---|---|
| Weather (clear / cloudy / rainy) | Nearest row in `indian_weather_data.csv` |
| Road surface (dry / wet) | Inferred from rainfall > 10 mm |
| Lighting (day / night) | Inferred from visibility < 3 km |

### Phase 3 — AHP Priority Scoring
The **Analytic Hierarchy Process** scores each stop based on three weighted criteria:

```
ahpScore = 0.70 × (students at stop / max students)
         + 0.20 × (1 / distance to college)
         + 0.10   (base cluster constant)
```

### Phase 4 — Greedy Capacity-Constrained Routing
Builds routes one bus at a time using a composite cost function:

```
cost = 0.40 × distance
     − 0.60 × ahpScore           ← negative: pulls bus toward high-priority stops
     + RISK_W × (1 + riskScore)  ← penalises risky stops, avoids clustering them
```

The number of buses is determined **automatically**. No manual input needed.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/RohanJinu/sjcet-bus-system.git
cd sjcet-bus-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create your CSV files
```
students.csv  — name, email  (one row per student — for login auth)
drivers.csv   — name, email  (one row per driver  — for login auth)
```
> ⚠️ These files are in `.gitignore` and will never be committed — they contain personal data.

### 4. Set environment variables
```bash
# Windows
set GMAIL_ADDRESS=your@gmail.com
set GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# macOS / Linux
export GMAIL_ADDRESS=your@gmail.com
export GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```
[How to get a Gmail App Password →](https://myaccount.google.com/apppasswords)

### 5. Start the server
```bash
python app.py
```
Open **http://localhost:5000** and log in with `admin@sjcetpalai.ac.in`

---

## 📁 Project Structure

```
sjcet-bus-system/
│
├── app.py                      # Flask backend — all API routes and ML logic
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html              # Admin + student portal (single-page app)
│   ├── driver.html             # Driver GPS console
│   └── login.html              # Login page
│
├── risk_model.pkl              # Trained Random Forest + label encoders
└── indian_weather_data.csv     # Weather dataset (lat, lon, precipitation, visibility)
```

> Files **not** in the repo (generated at runtime or contain personal data):
> `students.csv` · `drivers.csv` · `bus_assignments.csv` · `route_geometry.json`

---

## 📡 ngrok Setup (Mobile Access for Drivers & Students)

Drivers and students access the system from their phones. ngrok creates a public HTTPS tunnel to your local server — no deployment needed.

```bash
# Install from https://ngrok.com, authenticate once
ngrok config add-authtoken <your-token>

# With app.py running, open a second terminal
ngrok http 5000
```

ngrok displays a URL like `https://a1b2-103-21-56.ngrok-free.app`

- **Drivers** open: `https://your-ngrok-url/driver`
- **Students** open: `https://your-ngrok-url`
- **Admin** stays on: `http://localhost:5000`

> The free ngrok tier generates a new URL on every restart — share it with drivers each morning before the trip.

---

## 📋 Daily Workflow

```
1.  python app.py              →  start Flask
2.  ngrok http 5000            →  get public URL, share with drivers
3.  Admin logs in              →  localhost:5000
4.  Upload student CSV         →  Student Data tab
5.  Run optimisation           →  Routes & Map tab → Optimise Routes
6.  Assign drivers & publish   →  Assign Drivers → Save & Publish
7.  Send email notifications   →  optional, one click
8.  Monitor live               →  Live AI Tracking tab
9.  Drivers tap Start          →  GPS broadcast begins on their phones
10. End of journey             →  Drivers tap Stop → Logout
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.0 |
| ML Model | scikit-learn (Random Forest), pandas, numpy, joblib |
| Frontend | Vanilla JavaScript (ES2020), Leaflet.js 1.9 |
| Maps | OpenStreetMap tiles via CartoDB Dark Matter |
| Geocoding | Built-in Kerala stop lookup + Nominatim OSM API |
| Email | Python smtplib + Gmail SMTP SSL |
| Tunnelling | ngrok |
| Storage | Flat CSV files + JSON — no database |

---

## 📄 CSV File Formats

### `students.csv` and `drivers.csv`
```csv
name,email
Arun Kumar,arun.kumar@sjcetpalai.ac.in
Priya Thomas,priya.thomas@sjcetpalai.ac.in
```

### Student data CSV (uploaded by admin in portal)
```csv
id,name,email,stop
1,Arun Kumar,arun.kumar@sjcetpalai.ac.in,Pala
2,Priya Thomas,priya.thomas@sjcetpalai.ac.in,Erattupetta
```
> Add optional `lat` and `lon` columns to skip geocoding for known stops.

---

<div align="center">

*Built for the Transport Office — St. Joseph's College of Engineering and Technology, Palai*

</div>
