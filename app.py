import os
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from datetime import datetime, timedelta
import random
import requests
from openai import OpenAI

app = Flask(__name__)

DB_PATH = "analytics.db"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()
# Basic language strings (demo: English, Telugu, Hindi)
LANG_STRINGS = {
    "en": {
        "home_title": "AI Powered Plant Disease Detection",
        "home_subtitle": "Upload a leaf image and get instant smart recommendations.",
        "start_analysis": "Start Analysis",
        "upload_title": "Upload a Leaf Image",
        "upload_subtitle": "Select a clear photo and your location to get weather‑aware recommendations.",
    },
    "te": {
        "home_title": "కృత్రిమ మేధస్సుతో మొక్కల వ్యాధి గుర్తింపు",
        "home_subtitle": "ఆకు ఫోటోను అప్‌లోడ్ చేసి వెంటనే సూచనలు పొందండి.",
        "start_analysis": "విశ్లేషణ ప్రారంభించండి",
        "upload_title": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_subtitle": "స్పష్టమైన ఆకు ఫోటో మరియు మీ ప్రాంతాన్ని ఇవ్వండి, వాతావరణ ఆధారిత సూచనలు పొందండి.",
    },
    "hi": {
        "home_title": "एआई आधारित पादप रोग पहचान",
        "home_subtitle": "पत्ती की तस्वीर अपलोड करें और तुरंत स्मार्ट सलाह पाएं।",
        "start_analysis": "विश्लेषण शुरू करें",
        "upload_title": "पत्ती की तस्वीर अपलोड करें",
        "upload_subtitle": "साफ पत्ती की फोटो और अपना स्थान दें, मौसम आधारित सलाह प्राप्त करें।",
    },
}


def get_lang():
    code = request.cookies.get("lang", "en")
    if code not in LANG_STRINGS:
        code = "en"
    return code, LANG_STRINGS[code]


def chat_system_prompt(lang_code: str, context: dict):
    if lang_code == "te":
        lang_line = "Reply in Telugu."
    elif lang_code == "hi":
        lang_line = "Reply in Hindi."
    else:
        lang_line = "Reply in English."

    disease = context.get("disease") or ""
    confidence = context.get("confidence") or ""
    rain_probability = context.get("rain_probability") or ""
    city = context.get("city") or ""

    return (
        "You are PlantCare AI, a friendly crop doctor for small farmers. "
        "Give practical, safe, low-cost advice. "
        "If you are unsure, say so and suggest consulting a local agriculture officer. "
        "Avoid dangerous chemical dosing instructions; recommend reading product labels and safety practices. "
        f"{lang_line} "
        f"Context: disease='{disease}', confidence='{confidence}', rain_probability='{rain_probability}', city='{city}'."
    )


CHAT_LOCAL_STRINGS = {
    "en": {
        "offline_title": "Offline Crop Care Assistant",
        "scan_line": "Based on your last scan: {disease} (confidence {confidence}%).",
        "low_conf": "Confidence is low. Please re-scan with a clear, close leaf photo in daylight (single leaf, plain background).",
        "rain_line": "Rain probability: {rain_probability}%.",
        "weather_high": "High rain chance: avoid spraying today; prefer a dry window and ensure drainage.",
        "weather_mid": "Moderate rain risk: if spraying is needed, prefer early morning/evening and avoid windy hours.",
        "weather_low": "Low rain risk: good window for treatment if needed.",
        "ask_more": "Ask me: treatment, prevention, fertilizer timing, or how to take a better photo.",
        "general": "Tell me the crop name and symptoms (spots / yellowing / curling / mold / mites) for better guidance.",
        "treatment": "General treatment guidance: remove heavily infected leaves, avoid overhead watering, improve airflow, and use a recommended product as per label directions.",
        "prevention": "Prevention: crop rotation, proper spacing, avoid overwatering, sanitize tools, and monitor weekly.",
        "photo": "Photo tips: take a close, sharp image of one leaf, avoid shadows, include the affected area, and keep background plain.",
    },
    "te": {
        "offline_title": "ఆఫ్‌లైన్ క్రాప్ కేర్ అసిస్టెంట్",
        "scan_line": "మీ చివరి స్కాన్ ఆధారంగా: {disease} (నమ్మకం {confidence}%).",
        "low_conf": "నమ్మకం తక్కువగా ఉంది. దయచేసి పగటి వెలుతురులో స్పష్టంగా (ఒకే ఆకు, సాధారణ బ్యాక్‌గ్రౌండ్) మళ్లీ స్కాన్ చేయండి.",
        "rain_line": "వర్షం అవకాశం: {rain_probability}%.",
        "weather_high": "వర్షం ఎక్కువ అవకాశం: ఈ రోజు స్ప్రే చేయకుండా, ఎండిన సమయంలో చేయండి. నీరు నిల్వ కాకుండా డ్రైనేజ్ చూసుకోండి.",
        "weather_mid": "మధ్యస్థ వర్షం ప్రమాదం: అవసరమైతే ఉదయం/సాయంత్రం స్ప్రే చేయండి; గాలివాన సమయంలో చేయకండి.",
        "weather_low": "వర్షం తక్కువ అవకాశం: అవసరమైతే చికిత్సకు అనుకూల సమయం.",
        "ask_more": "నన్ను అడగండి: చికిత్స, నివారణ, ఎరువు సమయం, మంచి ఫోటో ఎలా తీసుకోవాలి.",
        "general": "మొక్క పేరు మరియు లక్షణాలు (మచ్చలు / పసుపు / కర్లింగ్ / ఫంగస్ / పురుగులు) చెప్పండి, మెరుగైన సలహా ఇస్తాను.",
        "treatment": "సాధారణ చికిత్స: ఎక్కువగా ప్రభావిత ఆకులు తొలగించండి, పై నుంచి నీరు పోయడం తగ్గించండి, గాలి ప్రసరణ పెంచండి, ఉత్పత్తి లేబుల్ సూచనల ప్రకారం మందు వాడండి.",
        "prevention": "నివారణ: పంట మార్పిడి, సరైన అంతరం, అధిక నీరు పోయకుండా, పనిముట్ల శుభ్రత, వారానికి ఒకసారి పరిశీలన.",
        "photo": "ఫోటో సూచనలు: ఒకే ఆకును దగ్గరగా స్పష్టంగా తీయండి, నీడలు వద్దు, ప్రభావిత భాగం కనిపించేలా, సాధారణ బ్యాక్‌గ్రౌండ్ పెట్టండి.",
    },
    "hi": {
        "offline_title": "ऑफलाइन क्रॉप केयर असिस्टेंट",
        "scan_line": "आपके पिछले स्कैन के आधार पर: {disease} (विश्वास {confidence}%).",
        "low_conf": "विश्वास कम है। कृपया दिन की रोशनी में एक पत्ती की साफ, पास से फोटो लेकर दोबारा स्कैन करें (सादा बैकग्राउंड)।",
        "rain_line": "बारिश की संभावना: {rain_probability}%.",
        "weather_high": "बारिश की संभावना अधिक: आज स्प्रे न करें; सूखे समय में करें और जल निकासी सुनिश्चित करें।",
        "weather_mid": "मध्यम बारिश जोखिम: जरूरत हो तो सुबह/शाम करें और तेज़ हवा में स्प्रे न करें।",
        "weather_low": "कम बारिश जोखिम: जरूरत हो तो उपचार के लिए अच्छा समय।",
        "ask_more": "पूछें: उपचार, रोकथाम, खाद का समय, बेहतर फोटो कैसे लें।",
        "general": "फसल का नाम और लक्षण बताएं (धब्बे / पीला / मुड़ना / फफूंदी / कीट) ताकि मैं बेहतर सलाह दे सकूँ।",
        "treatment": "सामान्य उपचार: ज्यादा संक्रमित पत्ते हटाएं, ऊपर से पानी देना कम करें, हवा का प्रवाह बढ़ाएं, और उत्पाद के लेबल के अनुसार दवा/फंगीसाइड का उपयोग करें।",
        "prevention": "रोकथाम: फसल चक्र, सही दूरी, अधिक पानी से बचें, औजार साफ रखें, और साप्ताहिक निगरानी करें।",
        "photo": "फोटो टिप्स: एक पत्ती की नज़दीक, साफ और शार्प फोटो लें, छाया से बचें, प्रभावित हिस्सा दिखाएं, बैकग्राउंड सादा रखें।",
    },
}


def offline_chat_answer(question: str, lang_code: str, context: dict) -> str:
    s = CHAT_LOCAL_STRINGS.get(lang_code) or CHAT_LOCAL_STRINGS["en"]
    q = (question or "").lower()

    disease = (context.get("disease") or "Unknown").strip()
    confidence = str(context.get("confidence") or "N/A").strip()
    rain_probability = str(context.get("rain_probability") or "N/A").strip()

    lines = [f"{s['offline_title']}\n"]
    lines.append(s["scan_line"].format(disease=disease, confidence=confidence))

    try:
        c = float(confidence)
        if c < 60:
            lines.append(s["low_conf"])
    except Exception:
        pass

    # Weather line (if we have a numeric rain probability)
    try:
        rp = float(rain_probability)
        lines.append(s["rain_line"].format(rain_probability=rain_probability))
        if rp >= 70:
            lines.append(s["weather_high"])
        elif rp >= 40:
            lines.append(s["weather_mid"])
        else:
            lines.append(s["weather_low"])
    except Exception:
        # No weather
        pass

    # Intent routing
    if any(k in q for k in ["treatment", "medicine", "spray", "fungicide", "pesticide", "cure", "చికిత్స", "मात्रा", "उपचार"]):
        lines.append("\n" + s["treatment"])
    elif any(k in q for k in ["prevent", "prevention", "avoid", "future", "నివారణ", "रोकथाम"]):
        lines.append("\n" + s["prevention"])
    elif any(k in q for k in ["photo", "image", "scan", "camera", "ఫోటో", "चित्र", "तस्वीर"]):
        lines.append("\n" + s["photo"])
    else:
        lines.append("\n" + s["general"])
        lines.append(s["ask_more"])

    # If we have timeline context from the result page, add it (optional)
    care_today = context.get("care_today")
    care_next_days = context.get("care_next_days")
    care_next_weeks = context.get("care_next_weeks")
    if care_today or care_next_days or care_next_weeks:
        lines.append("\nCare timeline:")
        if care_today:
            lines.append(f"- Today: {care_today}")
        if care_next_days:
            lines.append(f"- Next 3–7 days: {care_next_days}")
        if care_next_weeks:
            lines.append(f"- Next 2 weeks: {care_next_weeks}")

    return "\n".join(lines).strip()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                disease TEXT NOT NULL,
                crop TEXT NOT NULL,
                healthy INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


# Simple demo catalog for e-commerce-style recommendations
PRODUCT_CATALOG = [
    {
        "id": 1,
        "name": "NeemGuard Bio Spray",
        "crop": "Tomato",
        "type": "Bio‑pesticide",
        "tag": "Organic",
        "price": "₹ 320 / L",
        "usage": "Dilute 3–5 ml per litre of water and spray on both sides of the leaf.",
        "targets": ["Bacterial_spot", "Early_blight", "Spider_mites"],
    },
    {
        "id": 2,
        "name": "CopperShield Fungicide",
        "crop": "Potato",
        "type": "Fungicide",
        "tag": "Chemical",
        "price": "₹ 450 / kg",
        "usage": "Apply as foliar spray at first sign of blight; follow label instructions.",
        "targets": ["Late_blight", "Early_blight"],
    },
    {
        "id": 3,
        "name": "GrowthPlus Organic Fertilizer",
        "crop": "All",
        "type": "Fertilizer",
        "tag": "Organic",
        "price": "₹ 280 / 5 kg",
        "usage": "Broadcast evenly around root zone and irrigate lightly.",
        "targets": ["healthy"],
    },
    {
        "id": 4,
        "name": "SpiderClean Miticide",
        "crop": "Tomato",
        "type": "Miticide",
        "tag": "Chemical",
        "price": "₹ 520 / 500 ml",
        "usage": "Use only when mite infestation is confirmed; avoid flowering stage.",
        "targets": ["Spider_mites"],
    },
]


def recommended_products_for_label(label: str):
    """Return a small list of recommended products for a predicted class label."""
    key = label.lower()
    results = []
    for p in PRODUCT_CATALOG:
        # Match by targets keywords inside label
        for t in p["targets"]:
            if t.lower() in key:
                results.append(p)
                break
    # Healthy case → general fertilizer
    if "healthy" in key:
        results.extend([p for p in PRODUCT_CATALOG if "healthy" in p["targets"] or p["crop"] == "All"])
    # De‑duplicate while preserving order
    seen = set()
    unique = []
    for p in results:
        if p["id"] not in seen:
            unique.append(p)
            seen.add(p["id"])
    # Fallback to generic catalog if nothing matched
    if not unique:
        unique = PRODUCT_CATALOG[:3]
    return unique


# Load trained model (retrained on Kaggle)
model = keras.models.load_model("plant_disease_mobilenet_v2.keras")

# Derive generic class names based on model output size
num_classes = model.output_shape[-1]
class_names = [f"Class_{i}" for i in range(num_classes)]

# Disease info dictionary + simple care timeline
disease_info = {
    "Tomato_healthy": {
        "description": "The plant is healthy with no visible disease.",
        "cause": "No infection detected.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper watering and sunlight.",
        "today": "Continue regular watering and monitor leaves for any new spots.",
        "next_days": "Keep an eye on weather and avoid over-watering after heavy rain.",
        "next_weeks": "Plan routine nutrition and periodic health checks for the crop."
    },
    "Potato___healthy": {
        "description": "The potato leaf is healthy.",
        "cause": "No infection.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper soil nutrition.",
        "today": "Inspect surrounding plants and clear any dead foliage from the soil.",
        "next_days": "Maintain balanced irrigation and avoid standing water.",
        "next_weeks": "Prepare for preventive fungicide only if disease pressure increases."
    },
    "Pepper__bell___healthy": {
        "description": "The pepper leaf is healthy.",
        "cause": "No infection.",
        "treatment": "No treatment required.",
        "prevention": "Regular monitoring is advised.",
        "today": "Check for pests on the underside of leaves and remove weeds around the plant.",
        "next_days": "Maintain consistent soil moisture and avoid water stress.",
        "next_weeks": "Plan light fertilizer application to support flowering and fruit set."
    }
}
def preprocess_image(img_path):
    # Match the image size used during training (e.g., 160x160)
    img = load_img(img_path, target_size=(160, 160))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def extract_crop_and_health(label: str):
    """
    Very lightweight parser to derive crop name and health flag
    from a class label string. This works best when labels contain
    the crop name and 'healthy' for healthy samples.
    """
    text = label.replace("__", "_").replace("___", "_")
    parts = text.split("_")
    crop = parts[0] if parts else "Unknown"

    is_healthy = 1 if "healthy" in label.lower() else 0
    return crop, is_healthy


def get_weather_for_city(city: str):
    """
    Fetch basic weather info and rain probability for the next hours
    using OpenWeatherMap (geocoding + One Call API).
    """
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY_HERE":
        return None

    try:
        # Geocoding: city -> lat/lon
        geo_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_resp = requests.get(
            geo_url,
            params={"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=5,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return None

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # One Call 3.0
        onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        resp = requests.get(
            onecall_url,
            params={
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,alerts",
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", [])[:12]
        if not hourly:
            return None

        pops = [h.get("pop", 0) for h in hourly]  # 0..1
        rain_probability = round(sum(pops) / len(pops) * 100, 1)

        current = data.get("current", {})
        temp = current.get("temp")
        humidity = current.get("humidity")

        return {
            "rain_probability": rain_probability,
            "temp": temp,
            "humidity": humidity,
        }
    except Exception:
        return None


@app.route("/")
def home():
    lang_code, ui = get_lang()
    return render_template("home.html", ui=ui, lang_code=lang_code)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join("static/uploads", file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)
            confidence = round(np.max(prediction) * 100, 2)
            predicted_class = class_names[np.argmax(prediction)]

            # Confidence-aware severity level
            if confidence >= 85:
                severity_level = "high"
            elif confidence >= 60:
                severity_level = "moderate"
            else:
                severity_level = "low"

            # Persist prediction for analytics
            crop, is_healthy = extract_crop_and_health(predicted_class)
            conn = sqlite3.connect(DB_PATH)
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO predictions (created_at, disease, crop, healthy) VALUES (?, ?, ?, ?)",
                    (datetime.utcnow().isoformat(timespec="seconds"), predicted_class, crop, is_healthy),
                )
                conn.commit()
            finally:
                conn.close()

            # Real-time weather integration
            city = request.form.get("city", "").strip()
            weather = get_weather_for_city(city) if city else None

            # Get disease info
            info = disease_info.get(predicted_class, {
                "description": "This disease affects plant growth and reduces yield.",
                "cause": "Fungal, bacterial, or viral infection.",
                "treatment": "Apply recommended fungicide or pesticide.",
                "prevention": "Ensure proper spacing and avoid overwatering.",
                "today": "Remove heavily infected leaves and avoid touching healthy plants immediately after.",
                "next_days": "Apply recommended treatment and repeat as advised on the product label.",
                "next_weeks": "Monitor for new symptoms and rotate crops in the next season to reduce disease pressure."
            })

            care_today = info.get("today")
            care_next_days = info.get("next_days")
            care_next_weeks = info.get("next_weeks")

            # E‑commerce: recommended inputs for this diagnosis
            recommended_products = recommended_products_for_label(predicted_class)

            # Fertilizer suggestion
            fertilizer_days = random.randint(20, 30)
            next_date = datetime.now() + timedelta(days=fertilizer_days)
            next_date = next_date.strftime("%Y-%m-%d")

            # Rain probability (real-time if possible, otherwise fallback)
            if weather:
                rain_probability = weather["rain_probability"]
                temp = weather["temp"]
                humidity = weather["humidity"]

                if rain_probability >= 70:
                    weather_advice = "High chance of rain. Avoid spraying today and protect soil from runoff."
                elif rain_probability >= 40:
                    weather_advice = "Moderate rain risk. Prefer early morning or evening applications."
                else:
                    weather_advice = "Low rain risk. Good window for treatment."
            else:
                rain_probability = random.randint(10, 80)
                temp = None
                humidity = None
                weather_advice = "Weather data unavailable. Use your local forecast as a reference."

            lang_code, ui = get_lang()
            return render_template("result.html",
                                   image_path=filepath,
                                   disease=predicted_class,
                                   confidence=confidence,
                                   description=info["description"],
                                   cause=info["cause"],
                                   treatment=info["treatment"],
                                   prevention=info["prevention"],
                                   severity_level=severity_level,
                                   is_uncertain=(confidence < 60),
                                   care_today=care_today,
                                   care_next_days=care_next_days,
                                   care_next_weeks=care_next_weeks,
                                   recommended_products=recommended_products,
                                   fertilizer_days=fertilizer_days,
                                   next_date=next_date,
                                   rain_probability=rain_probability,
                                   weather_advice=weather_advice,
                                   ui=ui,
                                   lang_code=lang_code)

    lang_code, ui = get_lang()
    return render_template("upload.html", ui=ui, lang_code=lang_code)


@app.route("/set-language/<code>")
def set_language(code):
    # Simple language toggle using a cookie
    if code not in LANG_STRINGS:
        code = "en"
    ref = request.referrer or url_for("home")
    resp = make_response(redirect(ref))
    resp.set_cookie("lang", code, max_age=60 * 60 * 24 * 365)
    return resp


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Please type a question."}), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    lang_code, _ = get_lang()
    context = {
        "disease": data.get("disease", ""),
        "confidence": data.get("confidence", ""),
        "rain_probability": data.get("rain_probability", ""),
        "city": data.get("city", ""),
        "care_today": data.get("care_today", ""),
        "care_next_days": data.get("care_next_days", ""),
        "care_next_weeks": data.get("care_next_weeks", ""),
    }

    # If no key, always use offline assistant.
    if not api_key:
        return jsonify({"answer": offline_chat_answer(question, lang_code, context)})

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": chat_system_prompt(lang_code, context)},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = "I couldn't generate an answer. Please try again."
        return jsonify({"answer": answer})
    except Exception as e:
        # Quota / billing / rate-limit → fallback to offline assistant
        msg = str(e).lower()
        if "insufficient_quota" in msg or "exceeded your current quota" in msg or "error code: 429" in msg:
            return jsonify({"answer": offline_chat_answer(question, lang_code, context)})
        return jsonify({"answer": f"Chat error: {str(e)}"}), 500


@app.route("/store")
def store():
    """Very simple e‑commerce-style catalog page."""
    crop_filter = request.args.get("crop", "").strip()
    type_filter = request.args.get("type", "").strip()

    items = PRODUCT_CATALOG
    if crop_filter:
        items = [p for p in items if p["crop"].lower() == crop_filter.lower() or p["crop"] == "All"]
    if type_filter:
        items = [p for p in items if p["type"].lower() == type_filter.lower()]

    lang_code, ui = get_lang()
    return render_template(
        "store.html",
        products=items,
        crop_filter=crop_filter,
        type_filter=type_filter,
        ui=ui,
        lang_code=lang_code,
    )


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Total predictions
        cur.execute("SELECT COUNT(*) AS c FROM predictions")
        row = cur.fetchone()
        total_predictions = row["c"] if row else 0

        # Most detected disease
        cur.execute(
            """
            SELECT disease, COUNT(*) AS c
            FROM predictions
            GROUP BY disease
            ORDER BY c DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        most_detected = row["disease"] if row else "N/A"

        # Healthy vs diseased
        cur.execute(
            """
            SELECT
                SUM(CASE WHEN healthy = 1 THEN 1 ELSE 0 END) AS healthy_count,
                SUM(CASE WHEN healthy = 0 THEN 1 ELSE 0 END) AS diseased_count
            FROM predictions
            """
        )
        row = cur.fetchone()
        healthy_count = row["healthy_count"] or 0
        diseased_count = row["diseased_count"] or 0

        # Last 7 days trend
        cur.execute(
            """
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM predictions
            WHERE date(created_at) >= date('now', '-6 day')
            GROUP BY date(created_at)
            ORDER BY d
            """
        )
        rows = cur.fetchall()
        counts_by_date = {r["d"]: r["c"] for r in rows}

        last_7_labels = []
        last_7_values = []
        for i in range(6, -1, -1):
            day = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
            last_7_labels.append(day[5:])  # MM-DD
            last_7_values.append(counts_by_date.get(day, 0))

        # Crop-wise stats (top 5)
        cur.execute(
            """
            SELECT crop, COUNT(*) AS c
            FROM predictions
            GROUP BY crop
            ORDER BY c DESC
            LIMIT 5
            """
        )
        crop_rows = cur.fetchall()
        crop_labels = [r["crop"] for r in crop_rows]
        crop_values = [r["c"] for r in crop_rows]

    finally:
        conn.close()

    lang_code, ui = get_lang()
    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        most_detected=most_detected,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        weekly_labels=last_7_labels,
        weekly_values=last_7_values,
        crop_labels=crop_labels,
        crop_values=crop_values,
        ui=ui,
        lang_code=lang_code,
    )


if __name__ == "__main__":
    app.run(debug=True)
