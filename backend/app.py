import base64
import os
import re
import sqlite3
from io import BytesIO
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from flask import abort
from flask_cors import CORS
from datetime import datetime, timedelta
import random
import requests
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
from langdetect import detect

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

load_dotenv(PROJECT_ROOT / ".env")

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
CORS(app)

DB_PATH = BACKEND_DIR / "analytics.db"
UPLOAD_DIR = STATIC_DIR / "uploads"
GRADCAM_DIR = STATIC_DIR / "gradcam"
HF_MODEL_API_URL = os.environ.get(
    "HF_MODEL_API_URL",
    "https://manasadn161-plant-disease-model.hf.space/predict",
).strip()
MODEL_API_TIMEOUT = int(os.environ.get("MODEL_API_TIMEOUT", "120"))
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip() or "openai/gpt-4o-mini"
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "PlantCareAI").strip()
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "").strip()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Basic language strings
LANG_STRINGS = {
    "en": {
        "home_title": "AI Powered Plant Disease Detection",
        "home_subtitle": "Upload a leaf image and get instant smart recommendations.",
        "start_analysis": "Start Analysis",
        "upload_title": "Upload a Leaf Image",
        "upload_subtitle": "Select a clear photo and your location to get weather-aware recommendations.",
    },
    "te": {},
    "hi": {},
    "kn": {},
}


UI_EXTRA_STRINGS = {
    "en": {
        "app_name": "PlantCare AI",
        "nav_home": "Home",
        "nav_detect": "Detect Disease",
        "nav_store": "Store",
        "nav_schemes": "Schemes",
        "nav_dashboard": "Dashboard",
        "nav_about": "About",
        "nav_contact": "Contact",
        "helpline_label": "Crop Disease Helpline (demo):",
        "helpline_number": "1800-180-1551",
        "how_title": "How it works",
        "how_step1": "Capture a clear photo of a single leaf in good lighting.",
        "how_step2": "Our AI analyzes the leaf for potential diseases in seconds.",
        "how_step3": "Get a confidence score, treatment tips, weather-aware advice, and fertilizer reminders.",
        "home_feat1_title": "Weather-aware insights",
        "home_feat1_body": "Rain probability and timing-aware advice so you spray and fertilize at the right time.",
        "home_feat2_title": "Smart care planner",
        "home_feat2_body": "Automatic care reminders with the next recommended follow-up date.",
        "home_feat3_title": "Farmer-friendly summaries",
        "home_feat3_body": "Simple explanation of disease, cause, and prevention without jargon.",
        "upload_city_placeholder": "Enter city / village name",
        "upload_analyze": "Analyze Leaf",
        "upload_drop_title": "Drag and drop a leaf photo",
        "upload_drop_subtitle": "or click to browse. Best results come from one leaf, plain background, and daylight.",
        "upload_no_image": "No image selected yet",
        "upload_help_link": "How it helps",
        "upload_checklist_title": "Smart scan checklist",
        "upload_check_1": "Use one clear leaf in the frame.",
        "upload_check_2": "Avoid shadows, fingers, and cluttered backgrounds.",
        "upload_check_3": "Capture both affected spots and healthy edges.",
        "upload_check_4": "Enter your city for weather-aware treatment advice.",
        "upload_highlight_label": "New:",
        "upload_highlight_body": "each result includes an AI attention heatmap, photo quality tips, weather context, and recommended crop inputs.",
        "chat_fab": "Chat",
        "chat_title": "Crop Care Assistant",
        "chat_subtitle": "Speak or type in English or Kannada.",
        "chat_placeholder": "Ask about symptoms, treatment, prevention, or photo tips...",
        "chat_send": "Send",
        "chat_mic_title": "Start voice input",
        "chat_voice_title": "Toggle spoken replies",
        "chat_listening": "Listening...",
        "chat_no_mic": "Mic unavailable",
        "chat_voice_on": "Voice on",
        "chat_voice_off": "Voice off",
        "chat_thinking": "Thinking...",
        "chat_net_error": "Network error. Please try again.",
        "chat_empty": "Please type a question.",
        "chat_welcome_generic": "Hi! Tell me your crop and what you are seeing on the leaf.",
        "chat_welcome_scan": "Based on your last scan: {disease} (confidence {confidence}%). Ask me about treatment, prevention, fertilizer, or photo tips.",
        "chat_quick_treatment": "Treatment",
        "chat_quick_prevention": "Prevention",
        "chat_quick_fertilizer": "Fertilizer",
        "chat_quick_photo": "Photo tips",
        "store_title": "CropCare Store",
        "store_subtitle": "Browse curated crop inputs, compare options, add them to cart, and complete a clean checkout flow.",
        "store_filter_crop": "Filter by crop",
        "store_filter_type": "Filter by input type",
        "store_apply_filters": "Apply Filters",
        "store_shop_item": "View item",
        "store_use_scan": "Use with new scan",
        "store_no_products": "No products match this filter.",
        "store_marketplace_label": "Marketplace experience",
        "store_pill_1": "Curated crop inputs",
        "store_pill_2": "Fast cart flow",
        "store_pill_3": "Clean checkout",
        "store_products_shown": "products shown",
        "store_current_focus": "current focus",
        "store_all_crops": "All crops",
        "store_search_label": "Search",
        "store_search_placeholder": "e.g., neem, copper",
        "store_recommended_for": "Recommended for",
        "store_clear_focus": "Clear focus",
        "store_crop_prefix": "Crop",
        "store_add_to_cart": "Add to cart",
        "store_buy_now": "Buy now",
        "store_cart_title": "Cart",
        "store_cart_empty": "Your cart is empty. Add products to begin checkout.",
        "store_cart_total": "Cart total",
        "store_cart_checkout": "Proceed to checkout",
        "store_cart_clear": "Clear cart",
        "store_cart_count": "items",
        "payment_title": "Secure Checkout",
        "payment_subtitle": "Review your products, delivery details, and payment mode in one place.",
        "payment_buy_now": "Buy now",
        "payment_secure_note": "Marketplace-style demo checkout",
        "payment_method_upi": "UPI",
        "payment_method_card": "Cards / Netbanking",
        "payment_method_cod": "Cash on delivery",
        "payment_name": "Farmer name",
        "payment_phone": "Mobile number",
        "payment_location": "Delivery address",
        "payment_place_order": "Place order",
        "payment_success": "Order placed successfully.",
        "payment_transaction": "Order ID",
        "payment_continue": "Continue shopping",
        "payment_landmark": "Landmark",
        "payment_summary": "Order summary",
        "payment_delivery": "Delivery details",
        "payment_remove": "Remove",
        "payment_qty": "Qty",
        "payment_pay_now": "Pay now",
        "payment_processing": "Redirecting to PhonePe...",
        "payment_popup_title": "Payment Successful",
        "payment_popup_body": "Your payment is complete and the order has been confirmed.",
        "payment_wait_message": "Please wait while we confirm your payment.",
        "payment_choose_method": "Choose payment mode",
        "payment_note_upi": "UPI selected for a faster checkout experience.",
        "payment_note_card": "Cards and netbanking selected for a familiar marketplace checkout.",
        "payment_note_cod": "Cash on delivery selected for doorstep payment.",
        "product_back": "Back to store",
        "product_buy": "Buy / Enquire",
        "result_title": "Analysis Result",
        "result_overview": "Overview",
        "result_treatment_advisor": "Treatment Advisor",
        "result_care_timeline": "Care Timeline",
        "result_recommended_inputs": "Recommended Inputs",
        "result_upload_another": "Upload another image",
        "result_open_store": "Open store",
        "result_confidence": "Confidence",
        "result_rain": "Rain",
        "result_weather_advice": "Weather Advice",
        "result_fertilizer_reminder": "Fertilizer Reminder",
        "result_next_date": "Next Date",
        "result_shop_fertilizer": "Shop fertilizer",
        "result_focus_map": "AI Focus Map",
        "result_focus_map_blurb": "This view highlights the part of the leaf that influenced the prediction the most.",
        "result_original_image": "Original image",
        "result_attention_overlay": "Disease highlight overlay",
        "result_overlay_unavailable": "Heatmap preview is not available for this prediction.",
        "result_low_confidence": "Prediction confidence is low. Please capture another clear image or consult an expert before taking strong action.",
        "result_photo_score": "Photo Quality Score",
        "result_crop": "Crop",
        "result_condition": "Detected condition",
        "result_weather_context": "Weather context",
        "result_rain_chance": "Rain chance {rain_probability}%",
        "result_description": "Description",
        "result_cause": "Cause",
        "result_recommended_steps": "Recommended next steps",
        "result_step_1": "Inspect nearby plants for similar symptoms.",
        "result_step_2": "Isolate heavily infected plants if possible.",
        "result_step_3": "Follow the suggested treatment and avoid overwatering.",
        "result_step_4": "Re-scan in a few days to track progress.",
        "result_treatment": "Treatment",
        "result_prevention": "Prevention",
        "result_photo_coach": "Photo coach",
        "result_tip": "Tip",
        "result_today": "Today",
        "result_next_days": "Next 3-7 days",
        "result_next_weeks": "Next 2 weeks",
        "result_severe_title": "Need expert support soon?",
        "result_severe_body": "This scan looks severe. You may want to file an agriculture support request for crop testing, sample guidance, or field follow-up.",
        "result_severe_cta": "Contact agriculture department",
        "footer_tagline": "AI-assisted plant disease detection built for farmers, gardeners, and everyday plant lovers.",
        "footer_note": "Use predictions as decision support, not as the only diagnosis for high-stakes crop treatment.",
        "footer_rights": "© 2026 PlantCare AI. All rights reserved.",
        "about_title": "About PlantCare AI",
        "about_intro": "PlantCare AI helps farmers, gardeners, and crop advisors understand plant leaf problems using AI-assisted detection, weather-aware guidance, and practical next-step recommendations.",
        "about_who": "Who it helps",
        "about_who_value": "Small farmers, agri students, nursery operators, and anyone caring for plants.",
        "about_offer": "What it offers",
        "about_offer_value": "Leaf disease detection, crop-care advice, AI attention maps, product suggestions, and weather context.",
        "about_note": "Important note",
        "about_note_value": "Use this as decision support. For serious crop loss risk, verify with a local agriculture expert.",
        "about_tutorial_title": "Tutorial",
        "about_tutorial_intro": "See the app flow from scan to action in a simple guided animation.",
        "about_tutorial_cta": "Open disease detection",
        "about_tutorial_step_1": "Upload one clear leaf photo with your village or city.",
        "about_tutorial_step_2": "Review the disease result, weather advice, and AI focus map.",
        "about_tutorial_step_3": "Open recommended products or file an agriculture support request for severe cases.",
        "about_tutorial_step_4": "Track care steps and revisit the app after treatment.",
        "contact_title": "Contact Us",
        "contact_intro": "Reach the PlantCare AI team or prepare an agriculture support application for crop testing, disease review, or sample submission guidance.",
        "contact_support": "Support",
        "contact_field": "Field helpline",
        "contact_best_for": "Best for",
        "contact_best_for_value": "Bug reports, crop-care questions, feature requests, and partnerships.",
        "contact_form_title": "Agriculture Department Assistance Application",
        "contact_form_intro": "Fill this form to prepare a clear request for disease testing, field support, or crop sample guidance.",
        "contact_form_name": "Farmer name",
        "contact_form_phone": "Phone number",
        "contact_form_email": "Email address",
        "contact_form_state": "State",
        "contact_form_district": "District",
        "contact_form_village": "Village / town",
        "contact_form_crop": "Crop",
        "contact_form_issue": "Need support for",
        "contact_form_severity": "Severity",
        "contact_form_area": "Affected area (acre / gunta / hectare)",
        "contact_form_message": "Describe the crop issue",
        "contact_form_submit": "Submit application",
        "contact_form_success": "Application saved successfully.",
        "contact_form_reference": "Reference ID",
        "contact_form_disclaimer": "This form saves your request inside the app and prepares the farmer's details for follow-up. It is not a direct Government of India filing portal.",
        "contact_resource_title": "Helpful official resources",
        "contact_resource_lab": "Soil health and sample support",
        "contact_resource_lab_body": "Find soil testing and sample-related guidance from the official Soil Health Card platform.",
        "contact_resource_call": "Kisan Call Center",
        "contact_resource_call_body": "Speak with an agriculture helpline representative for first-level guidance.",
        "contact_resource_insurance": "Crop risk support",
        "contact_resource_insurance_body": "Use PMFBY when disease or weather risk connects with crop insurance needs.",
        "schemes_title": "Government Schemes for Farmers",
        "schemes_intro": "Open official Government of India agriculture schemes connected to crop health, farm resilience, market access, and farmer support.",
        "schemes_open": "Visit official scheme",
        "schemes_verified": "Official portal",
    },
    "te": {},
    "hi": {},
    "kn": {},
}

LANG_STRINGS.update({
    "te": {
        "home_title": "AI ఆధారిత మొక్కల వ్యాధి గుర్తింపు",
        "home_subtitle": "ఆకు ఫోటోను అప్‌లోడ్ చేసి వెంటనే తెలివైన సూచనలు పొందండి.",
        "start_analysis": "విశ్లేషణ ప్రారంభించండి",
        "upload_title": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_subtitle": "స్పష్టమైన ఫోటో మరియు మీ ప్రాంతాన్ని ఇచ్చి వాతావరణ ఆధారిత సూచనలు పొందండి.",
    },
    "hi": {
        "home_title": "एआई आधारित पौध रोग पहचान",
        "home_subtitle": "पत्ती की फोटो अपलोड करें और तुरंत उपयोगी सलाह पाएं।",
        "start_analysis": "विश्लेषण शुरू करें",
        "upload_title": "पत्ती की फोटो अपलोड करें",
        "upload_subtitle": "साफ फोटो और अपना स्थान देकर मौसम आधारित सुझाव पाएं।",
    },
    "kn": {
        "home_title": "ಎಐ ಆಧಾರಿತ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ",
        "home_subtitle": "ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ತಕ್ಷಣ ಉಪಯುಕ್ತ ಸಲಹೆ ಪಡೆಯಿರಿ.",
        "start_analysis": "ವಿಶ್ಲೇಷಣೆ ಪ್ರಾರಂಭಿಸಿ",
        "upload_title": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_subtitle": "ಸ್ಪಷ್ಟವಾದ ಫೋಟೋ ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳದೊಂದಿಗೆ ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆ ಪಡೆಯಿರಿ.",
    },
})

LOCALIZED_UI_OVERRIDES = {
    "te": {
        "app_name": "PlantCare AI",
        "nav_home": "హోమ్",
        "nav_detect": "వ్యాధి గుర్తింపు",
        "nav_store": "స్టోర్",
        "nav_schemes": "పథకాలు",
        "nav_dashboard": "డాష్‌బోర్డ్",
        "nav_about": "గురించి",
        "nav_contact": "సంప్రదించండి",
        "how_title": "ఇది ఎలా పని చేస్తుంది",
        "how_step1": "మంచి వెలుతురులో ఒకే ఆకు ఫోటో తీయండి.",
        "how_step2": "మా AI కొన్ని సెకన్లలో ఆకు సమస్యను విశ్లేషిస్తుంది.",
        "how_step3": "నమ్మకం శాతం, చికిత్స సూచనలు, వాతావరణ సూచనలు మరియు గుర్తుచూపులు పొందండి.",
        "home_feat1_title": "వాతావరణ ఆధారిత సూచనలు",
        "home_feat1_body": "వర్ష అవకాశం తెలుసుకొని సరైన సమయంలో స్ప్రే చేయండి.",
        "home_feat2_title": "స్మార్ట్ కేర్ ప్లానర్",
        "home_feat2_body": "తదుపరి సంరక్షణ తేదీతో ఆటోమేటిక్ గుర్తుచూపులు.",
        "home_feat3_title": "సరళమైన వివరాలు",
        "home_feat3_body": "వ్యాధి, కారణం, నివారణను సులభమైన భాషలో చూపిస్తుంది.",
        "upload_city_placeholder": "పట్టణం / గ్రామం పేరు నమోదు చేయండి",
        "upload_analyze": "ఆకును విశ్లేషించండి",
        "upload_drop_title": "ఆకు ఫోటోను డ్రాగ్ చేసి వదలండి",
        "upload_drop_subtitle": "లేదా క్లిక్ చేసి ఫైల్ ఎంచుకోండి. ఒకే ఆకు, సాధారణ బ్యాక్‌గ్రౌండ్, పగటి వెలుతురు ఉత్తమం.",
        "upload_no_image": "ఇంకా చిత్రాన్ని ఎంచుకోలేదు",
        "upload_help_link": "ఇది ఎలా సహాయపడుతుంది",
        "upload_checklist_title": "స్మార్ట్ స్కాన్ చెక్‌లిస్ట్",
        "upload_check_1": "ఒకే ఆకు స్పష్టంగా కనిపించేలా తీసుకోండి.",
        "upload_check_2": "నెరಳು, వేళ్లు, గందరగోళమైన బ్యాక్‌గ్రౌండ్ తప్పించండి.",
        "upload_check_3": "దెబ్బతిన్న ప్రాంతం మరియు ఆరోగ్యమైన అంచులు రెండూ కనిపించాలి.",
        "upload_check_4": "వాతావరణ సూచనల కోసం మీ నగరాన్ని ఇవ్వండి.",
        "upload_highlight_label": "కొత్తది:",
        "upload_highlight_body": "ప్రతి ఫలితంలో AI హీట్‌మ్యాప్, ఫోటో నాణ్యత చిట్కాలు, వాతావరణ సమాచారం మరియు ఉత్పత్తి సూచనలు ఉంటాయి.",
        "chat_fab": "చాట్",
        "chat_title": "పంట సహాయకుడు",
        "chat_subtitle": "తెలుగు లేదా ఇంగ్లీష్‌లో టైప్ చేయండి లేదా మాట్లాడండి.",
        "chat_placeholder": "లక్షణాలు, చికిత్స, నివారణ లేదా ఫోటో చిట్కాలు అడగండి...",
        "chat_send": "పంపండి",
        "chat_mic_title": "వాయిస్ ఇన్‌పుట్ ప్రారంభించండి",
        "chat_voice_title": "శబ్ద సమాధానాలను మార్చండి",
        "chat_no_mic": "మైక్ లేదు",
        "chat_welcome_generic": "హాయ్! మీ పంట పేరు మరియు ఆకు మీద కనిపిస్తున్న సమస్య చెప్పండి.",
        "store_title": "స్మార్ట్ ఇన్‌పుట్ స్టోర్",
        "store_subtitle": "సూచించిన స్ప్రేలు మరియు ఎరువులను చూడండి.",
        "store_filter_crop": "పంట ఆధారంగా ఫిల్టర్ చేయండి",
        "store_filter_type": "ఉత్పత్తి రకం ద్వారా ఫిల్టర్ చేయండి",
        "store_apply_filters": "ఫిల్టర్లు అమలు చేయండి",
        "store_shop_item": "వివరాలు చూడండి",
        "store_use_scan": "కొత్త స్కాన్‌తో ఉపయోగించండి",
        "store_no_products": "ఈ ఫిల్టర్‌కు సరిపోయే ఉత్పత్తులు లేవు.",
        "product_back": "స్టోర్‌కు తిరిగి వెళ్లండి",
        "result_title": "విశ్లేషణ ఫలితం",
        "footer_tagline": "రైతులు, తోటమాలి మరియు మొక్కల ప్రేమికుల కోసం AI ఆధారిత మొక్కల వ్యాధి గుర్తింపు.",
        "footer_note": "ఈ ఫలితాలను సహాయక సూచనలుగా ఉపయోగించండి; తుది నిర్ణయం కోసం నిపుణుడిని సంప్రదించండి.",
        "footer_rights": "© 2026 PlantCare AI. అన్ని హక్కులు సురక్షితం.",
        "about_title": "PlantCare AI గురించి",
        "about_intro": "PlantCare AI రైతులు మరియు తోటమాలికి ఆకు సమస్యలను అర్థం చేసుకోవడానికి సహాయపడుతుంది.",
        "contact_title": "మమ్మల్ని సంప్రదించండి",
        "contact_intro": "PlantCare AI ఉపయోగంలో సహాయం కావాలా? మాతో సంప్రదించండి.",
        "dashboard_intro_title": "PlantCare AI డాష్‌బోర్డ్",
        "dashboard_intro_body": "ఇక్కడ మీరు మొత్తం గుర్తింపులు, ట్రెండ్లు, ఆరోగ్య స్థితి మరియు ఎక్కువగా స్కాన్ చేసిన పంటల వివరాలను చూడవచ్చు.",
        "dashboard_cta": "కొత్త విశ్లేషణ ప్రారంభించండి",
        "schemes_title": "రైతుల కోసం ప్రభుత్వ పథకాలు",
        "schemes_intro": "పంట ఆరోగ్యం, వ్యవసాయ స్థిరత్వం, మార్కెట్ ప్రాప్యత మరియు రైతు మద్దతుకు సంబంధించిన భారత ప్రభుత్వ అధికారిక వ్యవసాయ పథకాలను తెరవండి.",
        "schemes_open": "అధికారిక పథకాన్ని తెరవండి",
        "schemes_verified": "అధికారిక పోర్టల్",
        "store_marketplace_label": "మార్కెట్‌ప్లేస్ అనుభవం",
        "store_pill_1": "ఎంచుకున్న పంట ఇన్‌పుట్లు",
        "store_pill_2": "త్వరిత కార్ట్ ప్రవాహం",
        "store_pill_3": "శుభ్రమైన చెకౌట్",
        "store_products_shown": "ఉత్పత్తులు చూపబడ్డాయి",
        "store_current_focus": "ప్రస్తుత దృష్టి",
        "store_all_crops": "అన్ని పంటలు",
        "store_search_label": "శోధన",
        "store_search_placeholder": "ఉదా: neem, copper",
        "store_recommended_for": "ఇదికి సిఫారసు",
        "store_clear_focus": "ఫోకస్ తొలగించు",
        "store_crop_prefix": "పంట",
        "payment_choose_method": "చెల్లింపు విధానాన్ని ఎంచుకోండి",
        "payment_wait_message": "మీ చెల్లింపును నిర్ధారించే వరకు వేచి ఉండండి.",
        "payment_note_upi": "త్వరిత చెకౌట్ కోసం UPI ఎంచుకున్నారు.",
        "payment_note_card": "పరిచిత మార్కెట్‌ప్లేస్ చెకౌట్ కోసం కార్డ్ / నెట్‌బ్యాంకింగ్ ఎంచుకున్నారు.",
        "payment_note_cod": "డెలివరీ సమయంలో నగదు చెల్లింపు ఎంచుకున్నారు.",
    },
    "hi": {
        "app_name": "PlantCare AI",
        "nav_home": "होम",
        "nav_detect": "रोग पहचान",
        "nav_store": "स्टोर",
        "nav_schemes": "योजनाएं",
        "nav_dashboard": "डैशबोर्ड",
        "nav_about": "परिचय",
        "nav_contact": "संपर्क",
        "how_title": "यह कैसे काम करता है",
        "how_step1": "अच्छी रोशनी में एक साफ पत्ती की फोटो लें।",
        "how_step2": "हमारा AI कुछ सेकंड में पत्ती का विश्लेषण करता है।",
        "how_step3": "विश्वास प्रतिशत, उपचार सुझाव, मौसम सलाह और रिमाइंडर पाएं।",
        "home_feat1_title": "मौसम आधारित सुझाव",
        "home_feat1_body": "बारिश की संभावना देखकर सही समय पर स्प्रे और खाद की योजना बनाएं।",
        "home_feat2_title": "स्मार्ट केयर प्लानर",
        "home_feat2_body": "अगली देखभाल तारीख के साथ स्वचालित रिमाइंडर.",
        "home_feat3_title": "सरल जानकारी",
        "home_feat3_body": "रोग, कारण और बचाव को आसान भाषा में समझाता है।",
        "upload_city_placeholder": "शहर / गांव का नाम लिखें",
        "upload_analyze": "पत्ती का विश्लेषण करें",
        "upload_drop_title": "पत्ती की फोटो यहां छोड़ें",
        "upload_drop_subtitle": "या क्लिक करके चुनें। एक पत्ती, साफ बैकग्राउंड और दिन की रोशनी सबसे बेहतर है।",
        "upload_no_image": "अभी तक कोई फोटो नहीं चुनी गई",
        "upload_help_link": "यह कैसे मदद करता है",
        "upload_checklist_title": "स्मार्ट स्कैन चेकलिस्ट",
        "upload_check_1": "फ्रेम में एक ही साफ पत्ती रखें।",
        "upload_check_2": "छाया, उंगलियां और बिखरे बैकग्राउंड से बचें।",
        "upload_check_3": "प्रभावित हिस्सा और स्वस्थ किनारे दोनों दिखें।",
        "upload_check_4": "मौसम सलाह के लिए अपना शहर दर्ज करें।",
        "upload_highlight_label": "नया:",
        "upload_highlight_body": "हर परिणाम में AI हीटमैप, फोटो गुणवत्ता टिप्स, मौसम जानकारी और उत्पाद सुझाव मिलते हैं।",
        "chat_fab": "चैट",
        "chat_title": "फसल सहायक",
        "chat_subtitle": "हिंदी या अंग्रेज़ी में पूछें या बोलें।",
        "chat_placeholder": "लक्षण, उपचार, रोकथाम या फोटो टिप्स पूछें...",
        "chat_send": "भेजें",
        "chat_mic_title": "वॉइस इनपुट शुरू करें",
        "chat_voice_title": "बोली जाने वाली प्रतिक्रिया बदलें",
        "chat_no_mic": "माइक नहीं",
        "chat_welcome_generic": "नमस्ते! अपनी फसल और पत्ती पर दिख रही समस्या बताइए।",
        "store_title": "स्मार्ट इनपुट स्टोर",
        "store_subtitle": "सुझाए गए स्प्रे और खाद देखें।",
        "store_filter_crop": "फसल के अनुसार फ़िल्टर करें",
        "store_filter_type": "उत्पाद प्रकार के अनुसार फ़िल्टर करें",
        "store_apply_filters": "फ़िल्टर लागू करें",
        "store_shop_item": "विवरण देखें",
        "store_use_scan": "नए स्कैन के साथ उपयोग करें",
        "store_no_products": "इस फ़िल्टर के लिए कोई उत्पाद नहीं मिला।",
        "product_back": "स्टोर पर वापस जाएं",
        "result_title": "विश्लेषण परिणाम",
        "footer_tagline": "किसानों, माली और पौध प्रेमियों के लिए AI-सहायित पौध रोग पहचान.",
        "footer_note": "इन परिणामों का उपयोग सहायक सलाह के रूप में करें; गंभीर स्थिति में विशेषज्ञ से पुष्टि करें।",
        "footer_rights": "© 2026 PlantCare AI. सर्वाधिकार सुरक्षित।",
        "about_title": "PlantCare AI के बारे में",
        "about_intro": "PlantCare AI किसानों और बागवानी करने वालों को पत्ती की समस्याओं को समझने में मदद करता है।",
        "contact_title": "संपर्क करें",
        "contact_intro": "PlantCare AI के उपयोग में मदद चाहिए? हमसे संपर्क करें।",
        "dashboard_intro_title": "PlantCare AI डैशबोर्ड",
        "dashboard_intro_body": "यहां आप कुल पहचान, ट्रेंड, पौध स्वास्थ्य और सबसे अधिक स्कैन की गई फसलों का सार देख सकते हैं।",
        "dashboard_cta": "नया विश्लेषण शुरू करें",
        "schemes_title": "किसानों के लिए सरकारी योजनाएं",
        "schemes_intro": "फसल स्वास्थ्य, कृषि स्थिरता, बाज़ार पहुंच और किसान सहायता से जुड़ी भारत सरकार की आधिकारिक कृषि योजनाएं खोलें।",
        "schemes_open": "आधिकारिक योजना खोलें",
        "schemes_verified": "आधिकारिक पोर्टल",
        "store_marketplace_label": "मार्केटप्लेस अनुभव",
        "store_pill_1": "चुने हुए कृषि इनपुट",
        "store_pill_2": "तेज़ कार्ट फ्लो",
        "store_pill_3": "साफ चेकआउट",
        "store_products_shown": "उत्पाद दिखाए गए",
        "store_current_focus": "वर्तमान फोकस",
        "store_all_crops": "सभी फसलें",
        "store_search_label": "खोज",
        "store_search_placeholder": "उदा: neem, copper",
        "store_recommended_for": "इसके लिए सुझाया गया",
        "store_clear_focus": "फोकस हटाएं",
        "store_crop_prefix": "फसल",
        "payment_choose_method": "भुगतान तरीका चुनें",
        "payment_wait_message": "कृपया प्रतीक्षा करें, हम आपके भुगतान की पुष्टि कर रहे हैं।",
        "payment_note_upi": "तेज़ चेकआउट के लिए UPI चुना गया है।",
        "payment_note_card": "परिचित मार्केटप्लेस चेकआउट के लिए कार्ड / नेटबैंकिंग चुना गया है।",
        "payment_note_cod": "डिलीवरी पर नकद भुगतान चुना गया है।",
    },
    "kn": {
        "app_name": "PlantCare AI",
        "nav_home": "ಮುಖಪುಟ",
        "nav_detect": "ರೋಗ ಪತ್ತೆ",
        "nav_store": "ಸ್ಟೋರ್",
        "nav_schemes": "ಯೋಜನೆಗಳು",
        "nav_dashboard": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "nav_about": "ಬಗ್ಗೆ",
        "nav_contact": "ಸಂಪರ್ಕ",
        "how_title": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "how_step1": "ಒಂದು ಸ್ಪಷ್ಟ ಎಲೆಯ ಫೋಟೋವನ್ನು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "how_step2": "ನಮ್ಮ AI ಕೆಲವು ಕ್ಷಣಗಳಲ್ಲಿ ಎಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ.",
        "how_step3": "ವಿಶ್ವಾಸ ಮಟ್ಟ, ಚಿಕಿತ್ಸೆ ಸಲಹೆ, ಹವಾಮಾನ ಮಾಹಿತಿ ಮತ್ತು ನೆನಪುಗಾರಿಕೆಗಳನ್ನು ಪಡೆಯಿರಿ.",
        "home_feat1_title": "ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆಗಳು",
        "home_feat1_body": "ಮಳೆಯ ಸಾಧ್ಯತೆಯನ್ನು ಗಮನಿಸಿ ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ಸ್ಪ್ರೇ ಮಾಡಿ.",
        "home_feat2_title": "ಸ್ಮಾರ್ಟ್ ಕೇರ್ ಪ್ಲಾನರ್",
        "home_feat2_body": "ಮುಂದಿನ ಕಾಳಜಿ ದಿನಾಂಕದೊಂದಿಗೆ ಸ್ವಯಂ ನೆನಪುಗಾರಿಕೆಗಳು.",
        "home_feat3_title": "ಸರಳ ವಿವರಣೆಗಳು",
        "home_feat3_body": "ರೋಗ, ಕಾರಣ ಮತ್ತು ತಡೆಗಟ್ಟುವಿಕೆಯನ್ನು ಸುಲಭವಾಗಿ ವಿವರಿಸುತ್ತದೆ.",
        "upload_city_placeholder": "ನಗರ / ಗ್ರಾಮದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ",
        "upload_analyze": "ಎಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "upload_drop_title": "ಎಲೆಯ ಫೋಟೋವನ್ನು ಇಲ್ಲಿ ಬಿಡಿ",
        "upload_drop_subtitle": "ಅಥವಾ ಕ್ಲಿಕ್ ಮಾಡಿ ಆರಿಸಿ. ಒಂದು ಎಲೆ, ಸರಳ ಹಿನ್ನೆಲೆ ಮತ್ತು ಹಗಲಿನ ಬೆಳಕು ಅತ್ಯುತ್ತಮ.",
        "upload_no_image": "ಇನ್ನೂ ಯಾವುದೇ ಚಿತ್ರ ಆಯ್ಕೆ ಮಾಡಿಲ್ಲ",
        "upload_help_link": "ಇದು ಹೇಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ",
        "upload_checklist_title": "ಸ್ಮಾರ್ಟ್ ಸ್ಕ್ಯಾನ್ ಪರಿಶೀಲನಾ ಪಟ್ಟಿ",
        "upload_check_1": "ಫ್ರೇಮ್‌ನಲ್ಲಿ ಒಂದು ಸ್ಪಷ್ಟ ಎಲೆ ಇರಲಿ.",
        "upload_check_2": "ನೆರಳು, ಬೆರಳುಗಳು ಮತ್ತು ಅಸ್ತವ್ಯಸ್ತ ಹಿನ್ನೆಲೆ ತಪ್ಪಿಸಿ.",
        "upload_check_3": "ಬಾಧಿತ ಭಾಗವೂ ಆರೋಗ್ಯಕರ ಅಂಚೂ ಕಾಣಬೇಕು.",
        "upload_check_4": "ಹವಾಮಾನ ಸಲಹೆಗೆ ನಿಮ್ಮ ನಗರವನ್ನು ನಮೂದಿಸಿ.",
        "upload_highlight_label": "ಹೊಸದು:",
        "upload_highlight_body": "ಪ್ರತಿ ಫಲಿತಾಂಶದಲ್ಲೂ AI ಹೀಟ್‌ಮ್ಯಾಪ್, ಫೋಟೋ ಗುಣಮಟ್ಟ ಸೂಚನೆಗಳು, ಹವಾಮಾನ ಮಾಹಿತಿ ಮತ್ತು ಉತ್ಪನ್ನ ಸಲಹೆಗಳಿವೆ.",
        "chat_fab": "ಚಾಟ್",
        "chat_title": "ಬೆಳೆ ಸಹಾಯಕ",
        "chat_subtitle": "ಕನ್ನಡ ಅಥವಾ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ ಅಥವಾ ಮಾತನಾಡಿ.",
        "chat_placeholder": "ಲಕ್ಷಣಗಳು, ಚಿಕಿತ್ಸೆ, ತಡೆಗಟ್ಟುವಿಕೆ ಅಥವಾ ಫೋಟೋ ಸಲಹೆಗಳ ಬಗ್ಗೆ ಕೇಳಿ...",
        "chat_send": "ಕಳುಹಿಸಿ",
        "chat_mic_title": "ಧ್ವನಿ ಇನ್‌ಪುಟ್ ಪ್ರಾರಂಭಿಸಿ",
        "chat_voice_title": "ಧ್ವನಿ ಪ್ರತಿಕ್ರಿಯೆ ಬದಲಿಸಿ",
        "chat_no_mic": "ಮೈಕ್ ಇಲ್ಲ",
        "chat_welcome_generic": "ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಬೆಳೆ ಮತ್ತು ಎಲೆಯಲ್ಲಿ ಕಾಣುತ್ತಿರುವ ಸಮಸ್ಯೆಯನ್ನು ತಿಳಿಸಿ.",
        "store_title": "ಸ್ಮಾರ್ಟ್ ಇನ್‌ಪುಟ್ ಸ್ಟೋರ್",
        "store_subtitle": "ಸೂಚಿಸಲಾದ ಸ್ಪ್ರೇ ಮತ್ತು ರಸಗೊಬ್ಬರಗಳನ್ನು ನೋಡಿ.",
        "store_filter_crop": "ಬೆಳೆ ಆಧಾರವಾಗಿ ಫಿಲ್ಟರ್ ಮಾಡಿ",
        "store_filter_type": "ಉತ್ಪನ್ನ ಪ್ರಕಾರದಂತೆ ಫಿಲ್ಟರ್ ಮಾಡಿ",
        "store_apply_filters": "ಫಿಲ್ಟರ್ ಅನ್ವಯಿಸಿ",
        "store_shop_item": "ವಿವರ ನೋಡಿ",
        "store_use_scan": "ಹೊಸ ಸ್ಕ್ಯಾನ್ ಜೊತೆ ಬಳಸಿ",
        "store_no_products": "ಈ ಫಿಲ್ಟರ್‌ಗೆ ಹೊಂದುವ ಉತ್ಪನ್ನಗಳಿಲ್ಲ.",
        "product_back": "ಸ್ಟೋರ್‌ಗೆ ಹಿಂತಿರುಗಿ",
        "result_title": "ವಿಶ್ಲೇಷಣೆಯ ಫಲಿತಾಂಶ",
        "footer_tagline": "ರೈತರು, ತೋಟಗಾರರು ಮತ್ತು ಸಸ್ಯ ಪ್ರೇಮಿಗಳಿಗಾಗಿ AI ನೆರವಿನ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ.",
        "footer_note": "ಈ ಫಲಿತಾಂಶಗಳನ್ನು ಸಹಾಯಕ ಮಾರ್ಗದರ್ಶನವಾಗಿ ಬಳಸಿ; ಗಂಭೀರ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಪರಿಣಿತರ ಸಲಹೆ ಪಡೆಯಿರಿ.",
        "footer_rights": "© 2026 PlantCare AI. ಎಲ್ಲ ಹಕ್ಕುಗಳನ್ನು ಕಾಯ್ದಿರಿಸಲಾಗಿದೆ.",
        "about_title": "PlantCare AI ಬಗ್ಗೆ",
        "about_intro": "PlantCare AI ರೈತರು ಮತ್ತು ತೋಟಗಾರರಿಗೆ ಎಲೆಯ ಸಮಸ್ಯೆಗಳನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.",
        "contact_title": "ನಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಿ",
        "contact_intro": "PlantCare AI ಬಳಕೆಯಲ್ಲಿ ಸಹಾಯ ಬೇಕೆ? ನಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "dashboard_intro_title": "PlantCare AI ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "dashboard_intro_body": "ಇಲ್ಲಿ ನೀವು ಒಟ್ಟು ಪತ್ತೆಗಳು, ಪ್ರವೃತ್ತಿಗಳು, ಆರೋಗ್ಯದ ಸ್ಥಿತಿ ಮತ್ತು ಹೆಚ್ಚು ಸ್ಕ್ಯಾನ್ ಆದ ಬೆಳೆಗಳನ್ನು ನೋಡಬಹುದು.",
        "dashboard_cta": "ಹೊಸ ವಿಶ್ಲೇಷಣೆ ಪ್ರಾರಂಭಿಸಿ",
        "schemes_title": "ರೈತರಿಗೆ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "schemes_intro": "ಬೆಳೆ ಆರೋಗ್ಯ, ಕೃಷಿ ಸ್ಥಿರತೆ, ಮಾರುಕಟ್ಟೆ ಪ್ರವೇಶ ಮತ್ತು ರೈತರ ಬೆಂಬಲಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಭಾರತ ಸರ್ಕಾರದ ಅಧಿಕೃತ ಕೃಷಿ ಯೋಜನೆಗಳನ್ನು ತೆರೆಯಿರಿ.",
        "schemes_open": "ಅಧಿಕೃತ ಯೋಜನೆ ತೆರೆಯಿರಿ",
        "schemes_verified": "ಅಧಿಕೃತ ಪೋರ್ಟಲ್",
        "store_marketplace_label": "ಮಾರ್ಕೆಟ್‌ಪ್ಲೇಸ್ ಅನುಭವ",
        "store_pill_1": "ಆಯ್ದ ಬೆಳೆ ಇನ್‌ಪುಟ್‌ಗಳು",
        "store_pill_2": "ವೇಗವಾದ ಕಾರ್ಟ್ ಪ್ರವಾಹ",
        "store_pill_3": "ಸ್ವಚ್ಛ ಚೆಕ್‌ಔಟ್",
        "store_products_shown": "ಉತ್ಪನ್ನಗಳು ತೋರಿಸಲಾಗಿದೆ",
        "store_current_focus": "ಪ್ರಸ್ತುತ ಗಮನ",
        "store_all_crops": "ಎಲ್ಲಾ ಬೆಳೆಗಳು",
        "store_search_label": "ಹುಡುಕು",
        "store_search_placeholder": "ಉದಾ: neem, copper",
        "store_recommended_for": "ಇದಕ್ಕಾಗಿ ಶಿಫಾರಸು",
        "store_clear_focus": "ಗಮನ ತೆರವುಗೊಳಿಸಿ",
        "store_crop_prefix": "ಬೆಳೆ",
        "payment_choose_method": "ಪಾವತಿ ವಿಧಾನ ಆಯ್ಕೆಮಾಡಿ",
        "payment_wait_message": "ದಯವಿಟ್ಟು ಕಾಯಿರಿ, ನಿಮ್ಮ ಪಾವತಿಯನ್ನು ದೃಢೀಕರಿಸುತ್ತಿದ್ದೇವೆ.",
        "payment_note_upi": "ವೇಗವಾದ ಚೆಕ್‌ಔಟ್‌ಗಾಗಿ UPI ಆಯ್ಕೆಮಾಡಲಾಗಿದೆ.",
        "payment_note_card": "ಪರಿಚಿತ ಮಾರ್ಕೆಟ್‌ಪ್ಲೇಸ್ ಚೆಕ್‌ಔಟ್‌ಗಾಗಿ ಕಾರ್ಡ್ / ನೆಟ್‌ಬ್ಯಾಂಕಿಂಗ್ ಆಯ್ಕೆಮಾಡಲಾಗಿದೆ.",
        "payment_note_cod": "ಬಾಗಿಲಿಗೆ ನಗದು ಪಾವತಿ ಆಯ್ಕೆಮಾಡಲಾಗಿದೆ.",
    },
}


def get_lang():
    """Get current language code and UI strings from cookie or default."""
    from flask import request
    
    # Try to get language from cookie
    lang_code = request.cookies.get('lang', 'en')
    
    # Fallback to 'en' if invalid
    if lang_code not in LANG_STRINGS:
        lang_code = 'en'
    
    # Always fall back to English keys so partially translated packs
    # do not break shared templates that expect a complete UI dictionary.
    ui = {
        **LANG_STRINGS["en"],
        **LANG_STRINGS.get(lang_code, {}),
        **UI_EXTRA_STRINGS["en"],
        **UI_EXTRA_STRINGS.get(lang_code, {}),
        **LOCALIZED_UI_OVERRIDES.get(lang_code, {}),
    }
    
    return lang_code, ui


@app.context_processor
def inject_ui_globals():
    lang_code, ui = get_lang()
    return {"lang_code": lang_code, "ui": ui}


CHAT_LOCAL_STRINGS = {
    "en": {
        "offline_title": "PlantCare Assistant",
        "scan_line": "Current detection: {disease} (confidence {confidence}%).",
        "low_conf": "Confidence is low. Please retake a clear leaf photo before acting on the result.",
        "rain_line": "Rain probability: {rain_probability}%.",
        "weather_high": "Rain chance is high, so avoid spraying right now.",
        "weather_mid": "Rain risk is moderate. Prefer a dry window for treatment.",
        "weather_low": "Rain risk is low, so conditions look better for treatment.",
        "ask_more": "You can ask about treatment, prevention, fertilizer timing, this page, or where to navigate next.",
        "general": "Hello! I can explain the current page, help with the latest scan, or guide you around the website.",
        "treatment": "Treatment guidance: remove badly affected leaves, avoid overwatering, improve airflow, and follow product labels carefully.",
        "prevention": "Prevention: keep spacing healthy, avoid wet leaves for too long, sanitize tools, and monitor nearby plants.",
        "photo": "Photo tip: capture one leaf clearly in daylight with a plain background and both affected and healthy areas visible.",
    },
    "hi": {
        "offline_title": "PlantCare सहायक",
        "scan_line": "मौजूदा पहचान: {disease} (विश्वास {confidence}%).",
        "low_conf": "विश्वास कम है। कृपया स्पष्ट पत्ती की नई फोटो लेकर फिर जांच करें।",
        "rain_line": "बारिश की संभावना: {rain_probability}%.",
        "weather_high": "बारिश की संभावना अधिक है, अभी स्प्रे न करें।",
        "weather_mid": "बारिश का जोखिम मध्यम है। उपचार के लिए सूखा समय चुनें।",
        "weather_low": "बारिश का जोखिम कम है, इसलिए उपचार के लिए समय बेहतर दिख रहा है।",
        "ask_more": "आप उपचार, रोकथाम, खाद का समय, इस पेज के बारे में या वेबसाइट में कहां जाना है यह पूछ सकते हैं।",
        "general": "नमस्ते! मैं इस पेज को समझा सकता हूँ, स्कैन में मदद कर सकता हूँ और वेबसाइट पर मार्गदर्शन दे सकता हूँ।",
        "treatment": "उपचार: ज्यादा प्रभावित पत्तियां हटाएं, अधिक पानी न दें, हवा का प्रवाह अच्छा रखें और दवा का लेबल ध्यान से पढ़ें।",
        "prevention": "रोकथाम: सही दूरी रखें, पत्तियों पर लंबे समय तक पानी न रहने दें, औजार साफ रखें और आसपास के पौधों को देखते रहें।",
        "photo": "फोटो टिप: दिन की रोशनी में एक पत्ती की साफ फोटो लें और प्रभावित व स्वस्थ दोनों हिस्से दिखाएं।",
    },
    "te": {
        "offline_title": "PlantCare సహాయకుడు",
        "scan_line": "ప్రస్తుత గుర్తింపు: {disease} (నమ్మకం {confidence}%).",
        "low_conf": "నమ్మకం తక్కువగా ఉంది. దయచేసి స్పష్టమైన ఆకు ఫోటోతో మళ్లీ పరీక్షించండి.",
        "rain_line": "వర్ష అవకాశం: {rain_probability}%.",
        "weather_high": "వర్ష అవకాశం ఎక్కువగా ఉంది, ఇప్పుడే స్ప్రే చేయొద్దు.",
        "weather_mid": "మధ్యస్థ వర్ష ప్రమాదం ఉంది. పొడి సమయాన్ని ఎంచుకోండి.",
        "weather_low": "వర్ష ప్రమాదం తక్కువగా ఉంది, కాబట్టి చికిత్సకు సమయం బాగుంది.",
        "ask_more": "మీరు చికిత్స, నివారణ, ఎరువు సమయం, ఈ పేజీ గురించి లేదా వెబ్‌సైట్‌లో ఎక్కడికి వెళ్లాలో అడగవచ్చు.",
        "general": "హలో! నేను ఈ పేజీని వివరించగలను, స్కాన్ గురించి చెప్పగలను, వెబ్‌సైట్‌లో నడిపించగలను.",
        "treatment": "చికిత్స: బాగా దెబ్బతిన్న ఆకులు తొలగించండి, ఎక్కువ నీరు పోయవద్దు, గాలి ప్రసరణ ఉంచండి, మందు లేబుల్‌ను జాగ్రత్తగా అనుసరించండి.",
        "prevention": "నివారణ: సరైన దూరం ఉంచండి, ఆకులపై ఎక్కువసేపు తేమ నిల్వ కాకుండా చూడండి, పనిముట్లు శుభ్రంగా ఉంచండి.",
        "photo": "ఫోటో చిట్కా: పగటి వెలుతురులో ఒకే ఆకును స్పష్టంగా తీసి దెబ్బతిన్న భాగం కూడా కనిపించేలా చేయండి.",
    },
    "kn": {
        "offline_title": "PlantCare ಸಹಾಯಕ",
        "scan_line": "ಪ್ರಸ್ತುತ ಪತ್ತೆ: {disease} (ವಿಶ್ವಾಸ {confidence}%).",
        "low_conf": "ವಿಶ್ವಾಸ ಕಡಿಮೆ ಇದೆ. ದಯವಿಟ್ಟು ಇನ್ನೊಂದು ಸ್ಪಷ್ಟ ಎಲೆಯ ಫೋಟೋ ತೆಗೆದು ಮರುಪರಿಶೀಲನೆ ಮಾಡಿ.",
        "rain_line": "ಮಳೆಯ ಸಾಧ್ಯತೆ: {rain_probability}%.",
        "weather_high": "ಮಳೆಯ ಸಾಧ್ಯತೆ ಹೆಚ್ಚು ಇದೆ, ಈಗ ಸ್ಪ್ರೇ ಮಾಡಬೇಡಿ.",
        "weather_mid": "ಮಳೆಯ ಅಪಾಯ ಮಧ್ಯಮವಾಗಿದೆ. ಒಣ ಸಮಯದಲ್ಲಿ ಚಿಕಿತ್ಸೆ ಮಾಡಿ.",
        "weather_low": "ಮಳೆಯ ಅಪಾಯ ಕಡಿಮೆ ಇದೆ, ಚಿಕಿತ್ಸೆಗಾಗಿ ಸಮಯ ಉತ್ತಮವಾಗಿದೆ.",
        "ask_more": "ನೀವು ಚಿಕಿತ್ಸೆ, ತಡೆಗಟ್ಟುವಿಕೆ, ರಸಗೊಬ್ಬರ ಸಮಯ, ಈ ಪುಟದ ಮಾಹಿತಿ ಅಥವಾ ವೆಬ್‌ಸೈಟ್‌ನಲ್ಲಿ ಎಲ್ಲಿಗೆ ಹೋಗಬೇಕು ಎಂದು ಕೇಳಬಹುದು.",
        "general": "ನಮಸ್ಕಾರ! ನಾನು ಈ ಪುಟವನ್ನು ವಿವರಿಸಬಹುದು, ಸ್ಕ್ಯಾನ್ ಫಲಿತಾಂಶವನ್ನು ಹೇಳಬಹುದು, ಮತ್ತು ವೆಬ್‌ಸೈಟ್‌ನಲ್ಲಿ ನಿಮಗೆ ಮಾರ್ಗದರ್ಶನ ಕೊಡಬಹುದು.",
        "treatment": "ಚಿಕಿತ್ಸೆ: ಹೆಚ್ಚು ಹಾನಿಯಾದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಹೆಚ್ಚು ನೀರು ಹಾಕಬೇಡಿ, ಗಾಳಿಯ ಸಂಚಾರ ಇರಲಿ, ಮತ್ತು ಔಷಧದ ಲೇಬಲ್ ಅನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ಅನುಸರಿಸಿ.",
        "prevention": "ತಡೆಗಟ್ಟುವಿಕೆ: ಸರಿಯಾದ ಅಂತರ ಇರಲಿ, ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಹೆಚ್ಚು ಕಾಲ ಉಳಿಯದಂತೆ ನೋಡಿ, ಉಪಕರಣಗಳನ್ನು ಸ್ವಚ್ಛವಾಗಿರಿಸಿ.",
        "photo": "ಫೋಟೋ ಸಲಹೆ: ಹಗಲಿನ ಬೆಳಕಿನಲ್ಲಿ ಒಂದು ಸ್ಪಷ್ಟ ಎಲೆಯ ಫೋಟೋ ತೆಗೆದು ಬಾಧಿತ ಭಾಗವೂ ಕಾಣುವಂತೆ ಮಾಡಿ.",
    },
}

CHAT_META_LABELS = {
    "en": {
        "you_are_on": "You are on: {page_title}",
        "page_path": "Page path: {page_path}",
        "visible_sections": "Visible sections: {sections}",
        "page_summary": "Page summary: {summary}",
        "care_timeline": "Care timeline:",
        "today": "Today",
        "next_days": "Next 3-7 days",
        "next_weeks": "Next 2 weeks",
    },
    "hi": {
        "you_are_on": "आप यहाँ हैं: {page_title}",
        "page_path": "पेज पथ: {page_path}",
        "visible_sections": "दिखने वाले भाग: {sections}",
        "page_summary": "पेज सार: {summary}",
        "care_timeline": "देखभाल समयरेखा:",
        "today": "आज",
        "next_days": "अगले 3-7 दिन",
        "next_weeks": "अगले 2 हफ्ते",
    },
    "te": {
        "you_are_on": "మీరు ఉన్న పేజీ: {page_title}",
        "page_path": "పేజీ మార్గం: {page_path}",
        "visible_sections": "కనిపించే విభాగాలు: {sections}",
        "page_summary": "పేజీ సారాంశం: {summary}",
        "care_timeline": "సంరక్షణ కాలరేఖ:",
        "today": "ఈ రోజు",
        "next_days": "తదుపరి 3-7 రోజులు",
        "next_weeks": "తదుపరి 2 వారాలు",
    },
    "kn": {
        "you_are_on": "ನೀವು ಇರುವ ಪುಟ: {page_title}",
        "page_path": "ಪುಟ ಮಾರ್ಗ: {page_path}",
        "visible_sections": "ಕಾಣುವ ವಿಭಾಗಗಳು: {sections}",
        "page_summary": "ಪುಟ ಸಾರಾಂಶ: {summary}",
        "care_timeline": "ಕಾಳಜಿ ಸಮಯರೇಖೆ:",
        "today": "ಇಂದು",
        "next_days": "ಮುಂದಿನ 3-7 ದಿನಗಳು",
        "next_weeks": "ಮುಂದಿನ 2 ವಾರಗಳು",
    },
}


def chat_system_prompt(lang_code: str, context: dict) -> str:
    page_title = context.get("page_title") or ""
    disease = context.get("disease") or ""
    return f"Language={lang_code}; page={page_title}; disease={disease}"


def offline_chat_answer(question: str, lang_code: str, context: dict) -> str:
    s = CHAT_LOCAL_STRINGS.get(lang_code) or CHAT_LOCAL_STRINGS["en"]
    meta = CHAT_META_LABELS.get(lang_code) or CHAT_META_LABELS["en"]
    q = (question or "").lower()

    disease = (context.get("disease") or "Unknown").strip()
    confidence = str(context.get("confidence") or "N/A").strip()
    rain_probability = str(context.get("rain_probability") or "N/A").strip()
    page_title = (context.get("page_title") or "").strip()
    page_path = (context.get("page_path") or "").strip()
    page_headings = context.get("page_headings") or []
    page_summary = (context.get("page_summary") or "").strip()

    lines = [f"{s['offline_title']}\n"]
    has_scan = disease and disease.lower() != "unknown"
    if has_scan:
        lines.append(s["scan_line"].format(disease=disease, confidence=confidence))

    explain_page = any(k in q for k in ["this page", "current page", "what is on this page", "what is there in this page", "page help", "section", "explain", "about this page", "ಈ ಪುಟ", "ಈ ಪೇಜ್", "ಪುಟದಲ್ಲಿ ಏನು ಇದೆ", "ಈ ಪುಟವನ್ನು ವಿವರಿಸು", "ఈ పేజీ", "ఈ పేజీలో ఏముంది", "ఈ పేజీని వివరించు", "यह पेज", "इस पेज में क्या है", "इस पेज को समझाओ"])
    if explain_page and page_title:
        lines.append(meta["you_are_on"].format(page_title=page_title))
    if explain_page and page_path:
        lines.append(meta["page_path"].format(page_path=page_path))

    if explain_page:
        if page_headings:
            lines.append(meta["visible_sections"].format(sections=", ".join(str(h) for h in page_headings[:6])))
        if page_summary:
            lines.append(meta["page_summary"].format(summary=page_summary[:500]))

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
    elif explain_page:
        lines.append("\n" + s["ask_more"])
    else:
        lines.append("\n" + s["general"])
        lines.append(s["ask_more"])

    # If we have timeline context from the result page, add it (optional)
    care_today = context.get("care_today")
    care_next_days = context.get("care_next_days")
    care_next_weeks = context.get("care_next_weeks")
    if care_today or care_next_days or care_next_weeks:
        lines.append("\n" + meta["care_timeline"])
        if care_today:
            lines.append(f"- {meta['today']}: {care_today}")
        if care_next_days:
            lines.append(f"- {meta['next_days']}: {care_next_days}")
        if care_next_weeks:
            lines.append(f"- {meta['next_weeks']}: {care_next_weeks}")

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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS support_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                reference_id TEXT NOT NULL UNIQUE,
                farmer_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                village TEXT NOT NULL,
                crop TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                affected_area TEXT,
                message TEXT NOT NULL
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
    {
        "id": 5,
        "name": "MancoMax Protect",
        "crop": "All",
        "type": "Fungicide",
        "tag": "Chemical",
        "price": "Rs 390 / kg",
        "usage": "Broad-spectrum protection for blight/leaf spots; follow label directions and safety practices.",
        "targets": ["blight", "leaf_spot", "Leaf_Mold", "rust", "Apple_scab", "Black_rot", "Leaf_scorch", "Powdery_mildew", "Cercospora", "Target_Spot"],
    },
    {
        "id": 6,
        "name": "Bordeaux Copper Mix",
        "crop": "All",
        "type": "Bactericide/Fungicide",
        "tag": "Chemical",
        "price": "Rs 260 / kg",
        "usage": "Copper-based mix for bacterial and fungal leaf issues; avoid overuse and follow label directions.",
        "targets": ["Bacterial_spot", "Apple_scab", "Cedar_apple_rust"],
    },
    {
        "id": 7,
        "name": "SulfurSafe Mildew Care",
        "crop": "All",
        "type": "Fungicide",
        "tag": "Organic",
        "price": "Rs 240 / kg",
        "usage": "Helps manage powdery mildew; spray early and ensure good coverage as per label directions.",
        "targets": ["Powdery_mildew", "mildew"],
    },
    {
        "id": 8,
        "name": "YellowTrap Sticky Cards",
        "crop": "All",
        "type": "IPM",
        "tag": "Organic",
        "price": "Rs 180 / pack",
        "usage": "Reduces whiteflies/aphids that spread viruses; place near plants and replace when full.",
        "targets": ["Virus", "mosaic", "Yellow_Leaf_Curl", "Haunglongbing", "greening"],
    },
    {
        "id": 9,
        "name": "Seaweed Tonic Bio-Stimulant",
        "crop": "All",
        "type": "Biostimulant",
        "tag": "Organic",
        "price": "Rs 220 / L",
        "usage": "Supports recovery and stress tolerance; apply as per label directions (do not exceed recommended dose).",
        "targets": ["healthy", "recovery", "stress"],
    },
    {
        "id": 10,
        "name": "Balanced NPK Fertilizer",
        "crop": "All",
        "type": "Fertilizer",
        "tag": "Chemical",
        "price": "Rs 300 / 5 kg",
        "usage": "Apply small doses; avoid over-fertilizing during active disease. Follow label directions.",
        "targets": ["healthy", "nutrition"],
    },
]

GOVERNMENT_SCHEMES = [
    {
        "name": "PM-KISAN",
        "summary": "Income support for eligible farmer families to help with crop input expenses and farm continuity.",
        "benefit": "Direct income support and beneficiary services",
        "url": "https://pmkisan.gov.in/",
        "source": "Department of Agriculture and Farmers Welfare",
    },
    {
        "name": "PM Fasal Bima Yojana",
        "summary": "Crop insurance support for notified crops against yield loss and related production risks.",
        "benefit": "Risk coverage and crop insurance services",
        "url": "https://pmfby.gov.in/",
        "source": "PMFBY Official Portal",
    },
    {
        "name": "Soil Health Card",
        "summary": "Soil test and nutrient guidance support that helps farmers improve crop health decisions and sample tracking.",
        "benefit": "Soil testing and nutrient planning",
        "url": "https://soilhealth.dac.gov.in/soil-lab",
        "source": "Soil Health Card Portal",
    },
    {
        "name": "National Mission for Sustainable Agriculture",
        "summary": "Supports sustainable agriculture through rainfed development, soil health management, water efficiency, and climate resilience.",
        "benefit": "Sustainable farming and resource management",
        "url": "https://nmsa.dac.gov.in/",
        "source": "NMSA Official Portal",
    },
    {
        "name": "PM-RKVY",
        "summary": "Supports agriculture development projects, innovation, and state-led interventions for stronger farm systems.",
        "benefit": "Project-based agriculture development support",
        "url": "https://rkvy.da.gov.in/",
        "source": "PM-RKVY Official Portal",
    },
    {
        "name": "e-NAM",
        "summary": "National agriculture market platform for transparent price discovery, mandi connectivity, and digital trade support.",
        "benefit": "Market access and transparent trading",
        "url": "https://enam.gov.in/web/",
        "source": "e-NAM Official Portal",
    },
]

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

SUPPORT_ISSUES = [
    "Crop disease testing",
    "Need to send crop sample",
    "Field officer guidance",
    "Soil testing support",
    "Crop loss advisory",
]

SEVERITY_OPTIONS = ["Low", "Moderate", "High", "Critical"]

PRODUCT_TRANSLATIONS = {
    "hi": {
        1: {"name": "नीमगार्ड बायो स्प्रे", "type": "जैव कीटनाशक", "tag": "ऑर्गेनिक", "usage": "3–5 मि.ली. प्रति लीटर पानी में मिलाकर पत्ते के दोनों ओर छिड़कें।"},
        2: {"name": "कॉपरशील्ड फंगीसाइड", "type": "फंगीसाइड", "tag": "केमिकल", "usage": "ब्लाइट के शुरुआती संकेत पर पत्तियों पर छिड़काव करें; लेबल निर्देशों का पालन करें।"},
        3: {"name": "ग्रोथप्लस ऑर्गेनिक फर्टिलाइज़र", "type": "उर्वरक", "tag": "ऑर्गेनिक", "usage": "जड़ों के आसपास समान रूप से डालें और हल्की सिंचाई करें।"},
        4: {"name": "स्पाइडरक्लीन मिटीसाइड", "type": "मिटीसाइड", "tag": "केमिकल", "usage": "माइट संक्रमण की पुष्टि होने पर ही उपयोग करें; फूल आने के समय से बचें।"},
        5: {"name": "मैनकोमैक्स प्रोटेक्ट", "type": "फंगीसाइड", "tag": "केमिकल", "usage": "ब्लाइट और लीफ स्पॉट के लिए व्यापक सुरक्षा; लेबल निर्देशों और सुरक्षा नियमों का पालन करें।"},
        6: {"name": "बोर्डो कॉपर मिक्स", "type": "बैक्टीरिसाइड / फंगीसाइड", "tag": "केमिकल", "usage": "बैक्टीरियल और फंगल पत्ती रोगों के लिए कॉपर आधारित मिश्रण; अधिक उपयोग से बचें।"},
        7: {"name": "सल्फरसेफ मिल्ड्यू केयर", "type": "फंगीसाइड", "tag": "ऑर्गेनिक", "usage": "पाउडरी मिल्ड्यू प्रबंधन में मदद करता है; जल्दी छिड़काव करें और अच्छी कवरेज रखें।"},
        8: {"name": "येलो ट्रैप स्टिकी कार्ड्स", "type": "आईपीएम", "tag": "ऑर्गेनिक", "usage": "व्हाइटफ्लाई और एफिड्स कम करने में मदद; पौधों के पास लगाएँ और भरने पर बदलें।"},
        9: {"name": "सीवीड टॉनिक बायो-स्टिमुलेंट", "type": "बायोस्टिमुलेंट", "tag": "ऑर्गेनिक", "usage": "रिकवरी और तनाव सहनशीलता में मदद; अनुशंसित मात्रा से अधिक न दें।"},
        10: {"name": "बैलेंस्ड एनपीके फर्टिलाइज़र", "type": "उर्वरक", "tag": "केमिकल", "usage": "छोटी मात्रा में दें; सक्रिय रोग की स्थिति में अधिक खाद न डालें।"},
    },
    "te": {
        1: {"name": "నీమ్‌గార్డ్ బయో స్ప్రే", "type": "జైవిక కీటకనాశిని", "tag": "ఆర్గానిక్", "usage": "ఒక లీటర్ నీటికి 3–5 మి.లి. కలిపి ఆకుల రెండు వైపులా పిచికారీ చేయండి."},
        2: {"name": "కాపర్‌షీల్డ్ ఫంగిసైడ్", "type": "ఫంగిసైడ్", "tag": "కెమికల్", "usage": "బ్లైట్ మొదటి లక్షణం కనిపించగానే పిచికారీ చేయండి; లేబుల్ సూచనలు పాటించండి."},
        3: {"name": "గ్రోత్‌ప్లస్ ఆర్గానిక్ ఫర్టిలైజర్", "type": "ఎరువు", "tag": "ఆర్గానిక్", "usage": "వేర్ల చుట్టూ సమంగా చల్లి తేలికగా నీరు పోయండి."},
        4: {"name": "స్పైడర్‌క్లీన్ మిటిసైడ్", "type": "మిటిసైడ్", "tag": "కెమికల్", "usage": "మైట్ దాడి నిర్ధారించినప్పుడు మాత్రమే వాడండి; పుష్ప దశలో నివారించండి."},
        5: {"name": "మాంకోమ్యాక్స్ ప్రొటెక్ట్", "type": "ఫంగిసైడ్", "tag": "కెమికల్", "usage": "బ్లైట్ మరియు లీఫ్ స్పాట్‌కు విస్తృత రక్షణ; లేబుల్ సూచనలు పాటించండి."},
        6: {"name": "బోర్డో కాపర్ మిక్స్", "type": "బాక్టిరిసైడ్ / ఫంగిసైడ్", "tag": "కెమికల్", "usage": "బ్యాక్టీరియా మరియు ఫంగల్ ఆకురోగాలకు కాపర్ ఆధారిత మిశ్రమం; అధిక వాడకం చేయవద్దు."},
        7: {"name": "సల్ఫర్‌సేఫ్ మిల్డ్యూ కేర్", "type": "ఫంగిసైడ్", "tag": "ఆర్గానిక్", "usage": "పౌడరీ మిల్డ్యూ నియంత్రణకు సహాయం; త్వరగా పిచికారీ చేసి మంచి కవరేజ్ కల్పించండి."},
        8: {"name": "యెల్లో ట్రాప్ స్టిక్కీ కార్డ్స్", "type": "ఐపీఎం", "tag": "ఆర్గానిక్", "usage": "వైట్‌ఫ్లై మరియు ఆఫిడ్స్ తగ్గించడంలో సహాయం; మొక్కల దగ్గర అమర్చండి."},
        9: {"name": "సీవీడ్ టానిక్ బయో-స్టిములెంట్", "type": "బయోస్టిమ్యులెంట్", "tag": "ఆర్గానిక్", "usage": "పునరుద్ధరణ మరియు ఒత్తిడి తట్టుకునే శక్తికి సహాయం; సూచించిన మోతాదే వాడండి."},
        10: {"name": "బ్యాలెన్స్డ్ ఎన్‌పీకే ఫర్టిలైజర్", "type": "ఎరువు", "tag": "కెమికల్", "usage": "చిన్న మోతాదులుగా వాడండి; వ్యాధి చురుకుగా ఉన్నప్పుడు అధిక ఎరువు వేయవద్దు."},
    },
    "kn": {
        1: {"name": "ನೀಮ್‌ಗಾರ್ಡ್ ಬಯೋ ಸ್ಪ್ರೇ", "type": "ಜೈವ ಕೀಟನಾಶಕ", "tag": "ಆರ್ಗಾನಿಕ್", "usage": "ಒಂದು ಲೀಟರ್ ನೀರಿಗೆ 3–5 ಮಿ.ಲಿ. ಬೆರೆಸಿ ಎಲೆಯ ಎರಡೂ ಬದಿಗಳಿಗೆ ಸ್ಪ್ರೇ ಮಾಡಿ."},
        2: {"name": "ಕಾಪರ್‌ಶೀಲ್ಡ್ ಫಂಗಿಸೈಡ್", "type": "ಫಂಗಿಸೈಡ್", "tag": "ಕೆಮಿಕಲ್", "usage": "ಬ್ಲೈಟ್ ಮೊದಲ ಲಕ್ಷಣ ಕಂಡಾಗ ಫೋಲಿಯರ್ ಸ್ಪ್ರೇ ಮಾಡಿ; ಲೇಬಲ್ ಸೂಚನೆಗಳನ್ನು ಅನುಸರಿಸಿ."},
        3: {"name": "ಗ್ರೋತ್‌ಪ್ಲಸ್ ಆರ್ಗಾನಿಕ್ ಫರ್ಟಿಲೈಸರ್", "type": "ರಸಗೊಬ್ಬರ", "tag": "ಆರ್ಗಾನಿಕ್", "usage": "ಬೇರು ಸುತ್ತ ಸಮವಾಗಿ ಹರಿ ಮತ್ತು ಸ್ವಲ್ಪ ನೀರು ಹಾಕಿ."},
        4: {"name": "ಸ್ಪೈಡರ್‌ಕ್ಲೀನ್ ಮೈಟಿಸೈಡ್", "type": "ಮೈಟಿಸೈಡ್", "tag": "ಕೆಮಿಕಲ್", "usage": "ಮೈಟ್ ದಾಳಿ ದೃಢಪಟ್ಟಾಗ ಮಾತ್ರ ಬಳಸಿ; ಹೂಬಿಡುವ ಹಂತ ತಪ್ಪಿಸಿ."},
        5: {"name": "ಮ್ಯಾಂಕೋಮ್ಯಾಕ್ಸ್ ಪ್ರೊಟೆಕ್ಟ್", "type": "ಫಂಗಿಸೈಡ್", "tag": "ಕೆಮಿಕಲ್", "usage": "ಬ್ಲೈಟ್ ಮತ್ತು ಲೀಫ್ ಸ್ಪಾಟ್‌ಗಳಿಗೆ ವಿಶಾಲ ರಕ್ಷಣೆ; ಲೇಬಲ್ ಸೂಚನೆಗಳನ್ನು ಅನುಸರಿಸಿ."},
        6: {"name": "ಬೋರ್ಡೋ ಕಾಪರ್ ಮಿಕ್ಸ್", "type": "ಬ್ಯಾಕ್ಟಿರಿಸೈಡ್ / ಫಂಗಿಸೈಡ್", "tag": "ಕೆಮಿಕಲ್", "usage": "ಬ್ಯಾಕ್ಟೀರಿಯಾ ಮತ್ತು ಫಂಗಲ್ ಎಲೆ ಸಮಸ್ಯೆಗಳಿಗೆ ಕಾಪರ್ ಆಧಾರಿತ ಮಿಶ್ರಣ; ಅತಿಯಾಗಿ ಬಳಸದಿರಿ."},
        7: {"name": "ಸಲ್ಫರ್‌ಸೇಫ್ ಮಿಲ್ಡ್ಯೂ ಕೇರ್", "type": "ಫಂಗಿಸೈಡ್", "tag": "ಆರ್ಗಾನಿಕ್", "usage": "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ ನಿಯಂತ್ರಣಕ್ಕೆ ಸಹಾಯ; ಬೇಗ ಸ್ಪ್ರೇ ಮಾಡಿ ಉತ್ತಮ ಕವರೆಜ್ ಕೊಡಿರಿ."},
        8: {"name": "ಯೆಲ್ಲೋ ಟ್ರ್ಯಾಪ್ ಸ್ಟಿಕ್ಕಿ ಕಾರ್ಡ್ಸ್", "type": "ಐಪಿಎಂ", "tag": "ಆರ್ಗಾನಿಕ್", "usage": "ವೈಟ್‌ಫ್ಲೈ ಮತ್ತು ಆಫಿಡ್‌ಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಸಹಾಯ; ಸಸ್ಯಗಳ ಬಳಿ ಇಡಿ."},
        9: {"name": "ಸೀವೀಡ್ ಟಾನಿಕ್ ಬಯೋ-ಸ್ಟಿಮ್ಯುಲೆಂಟ್", "type": "ಬಯೋಸ್ಟಿಮ್ಯುಲೆಂಟ್", "tag": "ಆರ್ಗಾನಿಕ್", "usage": "ಪುನಶ್ಚೇತನ ಮತ್ತು ಒತ್ತಡ ಸಹನೆಗೆ ಸಹಾಯ; ಸೂಚಿಸಿದ ಪ್ರಮಾಣದಲ್ಲಿ ಬಳಸಿ."},
        10: {"name": "ಬ್ಯಾಲೆನ್ಸ್ಡ್ ಎನ್‌ಪಿಕೆ ಫರ್ಟಿಲೈಸರ್", "type": "ರಸಗೊಬ್ಬರ", "tag": "ಕೆಮಿಕಲ್", "usage": "ಸಣ್ಣ ಪ್ರಮಾಣಗಳಲ್ಲಿ ಬಳಸಿ; ರೋಗ ಸಕ್ರಿಯವಾಗಿರುವಾಗ ಹೆಚ್ಚು ರಸಗೊಬ್ಬರ ಬೇಡ."},
    },
}

SCHEME_TRANSLATIONS = {
    "hi": {
        "PM-KISAN": {"summary": "योग्य किसान परिवारों को कृषि इनपुट और खेती की निरंतरता के लिए आय सहायता देता है.", "benefit": "प्रत्यक्ष आय सहायता और लाभार्थी सेवाएं", "source": "कृषि एवं किसान कल्याण विभाग"},
        "PM Fasal Bima Yojana": {"summary": "अधिसूचित फसलों के लिए उपज हानि और संबंधित जोखिमों पर बीमा सहायता प्रदान करता है.", "benefit": "जोखिम कवरेज और फसल बीमा सेवाएं", "source": "पीएमएफबीवाई आधिकारिक पोर्टल"},
        "Soil Health Card": {"summary": "मिट्टी परीक्षण और पोषक तत्व मार्गदर्शन के जरिए फसल स्वास्थ्य निर्णयों में मदद करता है.", "benefit": "मिट्टी परीक्षण और पोषक योजना", "source": "सॉइल हेल्थ कार्ड पोर्टल"},
        "National Mission for Sustainable Agriculture": {"summary": "वर्षा आधारित विकास, मिट्टी स्वास्थ्य, जल दक्षता और जलवायु सहनशीलता को समर्थन देता है.", "benefit": "सतत खेती और संसाधन प्रबंधन", "source": "एनएमएसए आधिकारिक पोर्टल"},
        "PM-RKVY": {"summary": "राज्य आधारित कृषि विकास परियोजनाओं और नवाचार हस्तक्षेपों को समर्थन देता है.", "benefit": "परियोजना आधारित कृषि विकास सहायता", "source": "पीएम-आरकेवीवाई आधिकारिक पोर्टल"},
        "e-NAM": {"summary": "पारदर्शी मूल्य खोज और डिजिटल व्यापार समर्थन के लिए राष्ट्रीय कृषि बाजार मंच.", "benefit": "बाजार पहुंच और पारदर्शी व्यापार", "source": "ई-नाम आधिकारिक पोर्टल"},
    },
    "te": {
        "PM-KISAN": {"summary": "అర్హులైన రైతు కుటుంబాలకు వ్యవసాయ ఇన్‌పుట్లు మరియు సాగు కొనసాగింపుకు ఆదాయ మద్దతు అందిస్తుంది.", "benefit": "ప్రత్యక్ష ఆదాయ మద్దతు మరియు లబ్ధిదారు సేవలు", "source": "వ్యవసాయ & రైతు సంక్షేమ శాఖ"},
        "PM Fasal Bima Yojana": {"summary": "అధిసూచిత పంటలకు దిగుబడి నష్టం మరియు సంబంధిత ప్రమాదాలపై బీమా మద్దతు అందిస్తుంది.", "benefit": "ప్రమాద రక్షణ మరియు పంట బీమా సేవలు", "source": "పీఎంఎఫ్‌బీవై అధికారిక పోర్టల్"},
        "Soil Health Card": {"summary": "మట్టి పరీక్ష మరియు పోషక సలహాలతో పంట ఆరోగ్య నిర్ణయాలకు సహాయం చేస్తుంది.", "benefit": "మట్టి పరీక్ష మరియు పోషక ప్రణాళిక", "source": "సాయిల్ హెల్త్ కార్డ్ పోర్టల్"},
        "National Mission for Sustainable Agriculture": {"summary": "వర్షాధార అభివృద్ధి, మట్టి ఆరోగ్యం, నీటి సామర్థ్యం మరియు వాతావరణ సహనాన్ని మద్దతు ఇస్తుంది.", "benefit": "సుస్థిర వ్యవసాయం మరియు వనరుల నిర్వహణ", "source": "ఎన్‌ఎంఎస్‌ఏ అధికారిక పోర్టల్"},
        "PM-RKVY": {"summary": "రాష్ట్ర ఆధారిత వ్యవసాయ అభివృద్ధి ప్రాజెక్టులు మరియు వినూత్న చర్యలకు మద్దతు ఇస్తుంది.", "benefit": "ప్రాజెక్ట్ ఆధారిత వ్యవసాయ అభివృద్ధి మద్దతు", "source": "పీఎం-ఆర్‌కేవీవై అధికారిక పోర్టల్"},
        "e-NAM": {"summary": "పారదర్శక ధర నిర్ణయం మరియు డిజిటల్ వాణిజ్యానికి జాతీయ వ్యవసాయ మార్కెట్ వేదిక.", "benefit": "మార్కెట్ ప్రాప్యత మరియు పారదర్శక వాణిజ్యం", "source": "ఈ-నామ్ అధికారిక పోర్టల్"},
    },
    "kn": {
        "PM-KISAN": {"summary": "ಅರ್ಹ ರೈತ ಕುಟುಂಬಗಳಿಗೆ ಕೃಷಿ ಇನ್‌ಪುಟ್ ವೆಚ್ಚ ಮತ್ತು ಕೃಷಿ ನಿರಂತರತೆಗೆ ಆದಾಯ ಸಹಾಯ ಒದಗಿಸುತ್ತದೆ.", "benefit": "ನೇರ ಆದಾಯ ಬೆಂಬಲ ಮತ್ತು ಲಾಭಾರ್ಥಿ ಸೇವೆಗಳು", "source": "ಕೃಷಿ ಮತ್ತು ರೈತ ಕಲ್ಯಾಣ ಇಲಾಖೆ"},
        "PM Fasal Bima Yojana": {"summary": "ಅಧಿಸೂಚಿತ ಬೆಳೆಗಳಿಗೆ ಉತ್ಪಾದನಾ ನಷ್ಟ ಮತ್ತು ಸಂಬಂಧಿತ ಅಪಾಯಗಳ ವಿರುದ್ಧ ವಿಮೆ ಬೆಂಬಲ ಒದಗಿಸುತ್ತದೆ.", "benefit": "ಅಪಾಯ ರಕ್ಷಣೆ ಮತ್ತು ಬೆಳೆ ವಿಮೆ ಸೇವೆಗಳು", "source": "ಪಿಎಂಎಫ್‌ಬಿವೈ ಅಧಿಕೃತ ಪೋರ್ಟಲ್"},
        "Soil Health Card": {"summary": "ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮತ್ತು ಪೋಷಕ ಮಾರ್ಗದರ್ಶನದ ಮೂಲಕ ಬೆಳೆ ಆರೋಗ್ಯ ನಿರ್ಧಾರಗಳಿಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ.", "benefit": "ಮಣ್ಣು ಪರೀಕ್ಷೆ ಮತ್ತು ಪೋಷಕ ಯೋಜನೆ", "source": "ಸೊಯಿಲ್ ಹೆಲ್ತ್ ಕಾರ್ಡ್ ಪೋರ್ಟಲ್"},
        "National Mission for Sustainable Agriculture": {"summary": "ಮಳೆಆಧಾರಿತ ಅಭಿವೃದ್ಧಿ, ಮಣ್ಣು ಆರೋಗ್ಯ, ನೀರಿನ ದಕ್ಷತೆ ಮತ್ತು ಹವಾಮಾನ ಸಹನಶೀಲತೆಗೆ ಬೆಂಬಲ ನೀಡುತ್ತದೆ.", "benefit": "ಸತತ ಕೃಷಿ ಮತ್ತು ಸಂಪನ್ಮೂಲ ನಿರ್ವಹಣೆ", "source": "ಎನ್‌ಎಂಎಸ್‌ಎ ಅಧಿಕೃತ ಪೋರ್ಟಲ್"},
        "PM-RKVY": {"summary": "ರಾಜ್ಯಾಧಾರಿತ ಕೃಷಿ ಅಭಿವೃದ್ಧಿ ಯೋಜನೆಗಳು ಮತ್ತು ನವೀನ ಹಸ್ತಕ್ಷೇಪಗಳಿಗೆ ಬೆಂಬಲ ನೀಡುತ್ತದೆ.", "benefit": "ಯೋಜನೆ ಆಧಾರಿತ ಕೃಷಿ ಅಭಿವೃದ್ಧಿ ಬೆಂಬಲ", "source": "ಪಿಎಂ-ಆರ್‌ಕೆವಿವೈ ಅಧಿಕೃತ ಪೋರ್ಟಲ್"},
        "e-NAM": {"summary": "ಪಾರದರ್ಶಕ ಬೆಲೆ ಅನ್ವೇಷಣೆ ಮತ್ತು ಡಿಜಿಟಲ್ ವಾಣಿಜ್ಯಕ್ಕೆ ರಾಷ್ಟ್ರೀಯ ಕೃಷಿ ಮಾರುಕಟ್ಟೆ ವೇದಿಕೆ.", "benefit": "ಮಾರುಕಟ್ಟೆ ಪ್ರವೇಶ ಮತ್ತು ಪಾರದರ್ಶಕ ವಾಣಿಜ್ಯ", "source": "ಇ-ನಾಮ್ ಅಧಿಕೃತ ಪೋರ್ಟಲ್"},
    },
}


def get_product_by_id(product_id: int):
    for p in PRODUCT_CATALOG:
        if p.get("id") == product_id:
            return p
    return None


def recommended_products_for_label(label: str):
    """Return a small list of recommended products for a predicted class label."""
    key = (label or "").lower()
    crop, _ = extract_crop_and_health(label or "")
    crop_l = (crop or "").lower()
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

    # Prefer products that match the detected crop (or "All")
    if crop_l:
        preferred = [p for p in unique if p.get("crop", "").lower() == crop_l or p.get("crop") == "All"]
        if preferred:
            unique = preferred
    # Fallback to generic catalog if nothing matched
    if not unique:
        unique = PRODUCT_CATALOG[:3]
    return unique


def create_support_reference() -> str:
    return "AGRI-" + datetime.now().strftime("%Y%m%d") + f"-{random.randint(1000, 9999)}"


def localize_product(product: dict, lang_code: str) -> dict:
    localized = dict(product)
    overrides = PRODUCT_TRANSLATIONS.get(lang_code, {}).get(product.get("id"), {})
    localized.update(overrides)
    localized["crop"] = localize_crop_name(product.get("crop", ""), lang_code)
    return localized


def localize_products(products: list, lang_code: str) -> list:
    return [localize_product(product, lang_code) for product in products]


def localize_scheme(scheme: dict, lang_code: str) -> dict:
    localized = dict(scheme)
    overrides = SCHEME_TRANSLATIONS.get(lang_code, {}).get(scheme.get("name"), {})
    localized.update(overrides)
    return localized


PLANTVILLAGE_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

class_names = PLANTVILLAGE_CLASS_NAMES

# Disease info dictionary + simple care timeline
disease_info = {
    "Tomato___healthy": {
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
    "Pepper,_bell___healthy": {
        "description": "The pepper leaf is healthy.",
        "cause": "No infection.",
        "treatment": "No treatment required.",
        "prevention": "Regular monitoring is advised.",
        "today": "Check for pests on the underside of leaves and remove weeds around the plant.",
        "next_days": "Maintain consistent soil moisture and avoid water stress.",
        "next_weeks": "Plan light fertilizer application to support flowering and fruit set."
    }
}
def analyze_image_quality(img_path: str):
    image = Image.open(img_path).convert("RGB")
    arr = np.asarray(image).astype("float32")
    gray = arr.mean(axis=2)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    edge_strength = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())

    quality_score = 55.0
    quality_score += min(20.0, contrast / 3.5)
    quality_score += min(20.0, edge_strength / 2.4)

    if 70 <= brightness <= 200:
        quality_score += 8.0
    elif brightness < 50 or brightness > 225:
        quality_score -= 12.0
    else:
        quality_score -= 2.0

    quality_score = int(max(0, min(100, round(quality_score))))

    tips = []
    badges = []

    if brightness < 70:
        tips.append("Image looks dark. Capture the leaf in daylight or near soft natural light.")
        badges.append("Low light")
    elif brightness > 210:
        tips.append("Image looks overexposed. Avoid direct harsh sunlight on the leaf.")
        badges.append("Too bright")
    else:
        badges.append("Lighting OK")

    if edge_strength < 12:
        tips.append("Image may be blurry. Hold the camera steady and move closer to a single leaf.")
        badges.append("Low sharpness")
    else:
        badges.append("Sharp enough")

    if contrast < 20:
        tips.append("Leaf details are faint. Use a plain background to improve visibility.")
        badges.append("Low contrast")
    else:
        badges.append("Good contrast")

    if not tips:
        tips.append("Good photo quality for prediction. Keep using one clear leaf against a plain background.")

    return {
        "score": quality_score,
        "tips": tips,
        "badges": badges,
    }


def analyze_plant_likelihood(img_path: str):
    image = Image.open(img_path).convert("RGB")
    arr = np.asarray(image).astype("float32")
    normalized = arr / 255.0
    red = normalized[:, :, 0]
    green = normalized[:, :, 1]
    blue = normalized[:, :, 2]

    max_channel = np.max(normalized, axis=2)
    min_channel = np.min(normalized, axis=2)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    nonzero = delta > 1e-6
    red_mask = nonzero & (max_channel == red)
    green_mask = nonzero & (max_channel == green)
    blue_mask = nonzero & (max_channel == blue)
    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4
    hue = hue * 60

    saturation = np.where(max_channel > 1e-6, delta / max_channel, 0.0)
    value = max_channel

    green_mask = (hue >= 35) & (hue <= 140) & (saturation >= 0.18) & (value >= 0.16)
    yellow_brown_mask = (hue >= 10) & (hue <= 50) & (saturation >= 0.12) & (value >= 0.12)
    foliage_mask = green_mask | yellow_brown_mask

    height, width = foliage_mask.shape
    y0, y1 = int(height * 0.2), int(height * 0.8)
    x0, x1 = int(width * 0.2), int(width * 0.8)
    center_mask = foliage_mask[y0:y1, x0:x1]

    foliage_ratio = float(foliage_mask.mean())
    center_foliage_ratio = float(center_mask.mean()) if center_mask.size else foliage_ratio
    green_strength = float(np.clip(((green - np.maximum(red, blue)).mean() + 0.12) * 4.0, 0.0, 1.0))
    plant_score = int(round(max(0.0, min(1.0, foliage_ratio * 1.8 + center_foliage_ratio * 2.2 + green_strength * 0.35)) * 100))

    return {
        "score": plant_score,
        "foliage_ratio": round(foliage_ratio, 3),
        "center_foliage_ratio": round(center_foliage_ratio, 3),
        "green_strength": round(green_strength, 3),
        "looks_like_plant": plant_score >= 22 and (foliage_ratio >= 0.06 or center_foliage_ratio >= 0.10),
    }


def should_reject_prediction(photo_quality: dict, plant_check: dict, prediction_payload: dict):
    confidence = float(prediction_payload.get("confidence", 0.0) or 0.0)
    top_predictions = prediction_payload.get("top_predictions") or []
    runner_up = float(top_predictions[1].get("confidence", 0.0)) if len(top_predictions) > 1 else 0.0
    confidence_gap = confidence - runner_up

    reasons = []
    if not plant_check.get("looks_like_plant"):
        reasons.append("Image does not look like a plant leaf.")
    if photo_quality.get("score", 0) < 35:
        reasons.append("Image quality is too low for a trusted diagnosis.")
    if confidence < 55:
        reasons.append("Model confidence is too low.")
    if confidence_gap < 8:
        reasons.append("Prediction is too close to other classes.")

    return reasons, {
        "confidence": confidence,
        "runner_up": runner_up,
        "confidence_gap": round(confidence_gap, 2),
    }
def to_static_web_path(file_path: Path) -> str:
    return f"static/{file_path.relative_to(STATIC_DIR).as_posix()}"


def decode_data_url(data: str) -> bytes:
    payload = (data or "").strip()
    if not payload:
        raise ValueError("Empty image payload received from the model API.")
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def save_base64_asset(data: str, output_path: Path) -> str | None:
    if not data:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(decode_data_url(data))).convert("RGB")
    image.save(output_path, quality=92)
    return to_static_web_path(output_path)


def build_remote_gradcam_context(payload: dict, original_image_path: Path):
    gradcam = payload.get("gradcam") or {}
    stem = original_image_path.stem
    overlay_path = save_base64_asset(gradcam.get("overlay_base64"), GRADCAM_DIR / f"{stem}_gradcam.jpg")
    heatmap_path = save_base64_asset(gradcam.get("heatmap_base64"), GRADCAM_DIR / f"{stem}_heatmap.jpg")
    return {
        "gradcam_path": overlay_path,
        "gradcam_overlay_path": overlay_path,
        "gradcam_heatmap_path": heatmap_path,
        "gradcam_layer": gradcam.get("target_layer"),
    }


def call_model_api(image_path: Path) -> dict:
    print(f"[model-api] sending image to {HF_MODEL_API_URL}: {image_path.name}")
    with image_path.open("rb") as image_file:
        response = requests.post(
            HF_MODEL_API_URL,
            files={"file": (image_path.name, image_file, "image/jpeg")},
            timeout=MODEL_API_TIMEOUT,
        )

    response.raise_for_status()
    payload = response.json()
    if not payload.get("label"):
        raise RuntimeError(f"Model API returned an unexpected payload: {payload}")

    print(
        "[model-api] received response:",
        {
            "label": payload.get("label"),
            "confidence": payload.get("confidence"),
            "target_layer": (payload.get("gradcam") or {}).get("target_layer"),
        },
    )
    return payload


def extract_crop_and_health(label: str):
    """
    Very lightweight parser to derive crop name and health flag
    from a class label string. This works best when labels contain
    the crop name and 'healthy' for healthy samples.
    """
    if not label:
        return "Unknown", 0

    if "___" in label:
        crop_part, disease_part = label.split("___", 1)
    elif "__" in label:
        crop_part, disease_part = label.split("__", 1)
    else:
        parts = label.split("_", 1)
        crop_part = parts[0]
        disease_part = parts[1] if len(parts) > 1 else ""

    crop = re.sub(r"[(),]", " ", crop_part).replace("_", " ").strip()
    crop = re.sub(r"\s+", " ", crop) or "Unknown"

    is_healthy = 1 if "healthy" in (disease_part or label).lower() else 0
    return crop, is_healthy


def pretty_label(label: str) -> str:
    if not label:
        return ""
    text = label.replace("___", " - ").replace("__", " - ").replace("_", " ")
    text = re.sub(r"[()]", "", text)
    return re.sub(r"\s+", " ", text).strip()


app.jinja_env.filters["pretty_label"] = pretty_label


CROP_NAME_TRANSLATIONS = {
    "hi": {
        "Apple": "सेब",
        "Blueberry": "ब्लूबेरी",
        "Cherry including sour": "चेरी",
        "Corn maize": "मक्का",
        "Grape": "अंगूर",
        "Orange": "संतरा",
        "Peach": "आड़ू",
        "Pepper bell": "शिमला मिर्च",
        "Potato": "आलू",
        "Raspberry": "रास्पबेरी",
        "Soybean": "सोयाबीन",
        "Squash": "स्क्वैश",
        "Strawberry": "स्ट्रॉबेरी",
        "Tomato": "टमाटर",
        "Unknown": "अज्ञात",
    },
    "te": {
        "Apple": "యాపిల్",
        "Blueberry": "బ్లూబెర్రీ",
        "Cherry including sour": "చెర్రీ",
        "Corn maize": "మొక్కజొన్న",
        "Grape": "ద్రాక్ష",
        "Orange": "నారింజ",
        "Peach": "పీచ్",
        "Pepper bell": "బెల్ పెప్పర్",
        "Potato": "బంగాళాదుంప",
        "Raspberry": "రాస్ప్‌బెర్రీ",
        "Soybean": "సోయాబీన్",
        "Squash": "స్క్వాష్",
        "Strawberry": "స్ట్రాబెర్రీ",
        "Tomato": "టమాటా",
        "Unknown": "తెలియదు",
    },
    "kn": {
        "Apple": "ಸೇಬು",
        "Blueberry": "ಬ್ಲೂಬೆರಿ",
        "Cherry including sour": "ಚೆರಿ",
        "Corn maize": "ಮಕ್ಕೆಜೋಳ",
        "Grape": "ದ್ರಾಕ್ಷಿ",
        "Orange": "ಕಿತ್ತಳೆ",
        "Peach": "ಪೀಚ್",
        "Pepper bell": "ಬೆಲ್ ಪೆಪ್ಪರ್",
        "Potato": "ಆಲೂಗಡ್ಡೆ",
        "Raspberry": "ರಾಸ್ಪ್‌ಬೆರಿ",
        "Soybean": "ಸೋಯಾಬೀನ್",
        "Squash": "ಸ್ಕ್ವಾಶ್",
        "Strawberry": "ಸ್ಟ್ರಾಬೆರಿ",
        "Tomato": "ಟೊಮೇಟೊ",
        "Unknown": "ಅಪರಿಚಿತ",
    },
}

CONDITION_TRANSLATIONS = {
    "hi": {
        "Apple scab": "एप्पल स्कैब",
        "Black rot": "ब्लैक रॉट",
        "Cedar apple rust": "सीडर एप्पल रस्ट",
        "Healthy": "स्वस्थ",
        "Powdery mildew": "पाउडरी मिल्ड्यू",
        "Cercospora leaf spot gray leaf spot": "सर्कोस्पोरा पत्ती धब्बा / ग्रे लीफ स्पॉट",
        "Common rust": "कॉमन रस्ट",
        "Northern leaf blight": "नॉर्दर्न लीफ ब्लाइट",
        "Esca black measles": "एस्का / ब्लैक मीज़ल्स",
        "Leaf blight Isariopsis leaf spot": "लीफ ब्लाइट / इसारियोप्सिस लीफ स्पॉट",
        "Haunglongbing Citrus greening": "सिट्रस ग्रीनिंग",
        "Bacterial spot": "बैक्टीरियल स्पॉट",
        "Early blight": "अर्ली ब्लाइट",
        "Late blight": "लेट ब्लाइट",
        "Leaf mold": "लीफ मोल्ड",
        "Septoria leaf spot": "सेप्टोरिया लीफ स्पॉट",
        "Spider mites Two spotted spider mite": "स्पाइडर माइट्स",
        "Target spot": "टारगेट स्पॉट",
        "Tomato yellow leaf curl virus": "टमाटर येलो लीफ कर्ल वायरस",
        "Tomato mosaic virus": "टमाटर मोज़ेक वायरस",
        "Leaf scorch": "लीफ स्कॉर्च",
    },
    "te": {
        "Apple scab": "ఆపిల్ స్కాబ్",
        "Black rot": "బ్లాక్ రాట్",
        "Cedar apple rust": "సీడర్ ఆపిల్ రస్ట్",
        "Healthy": "ఆరోగ్యంగా ఉంది",
        "Powdery mildew": "పౌడరీ మిల్డ్యూ",
        "Cercospora leaf spot gray leaf spot": "సెర్కోస్పోరా ఆకుమచ్చ / గ్రే లీఫ్ స్పాట్",
        "Common rust": "కామన్ రస్ట్",
        "Northern leaf blight": "ఉత్తర లీఫ్ బ్లైట్",
        "Esca black measles": "ఎస్కా / బ్లాక్ మీజిల్స్",
        "Leaf blight Isariopsis leaf spot": "లీఫ్ బ్లైట్ / ఇసారియాప్సిస్ లీఫ్ స్పాట్",
        "Haunglongbing Citrus greening": "సిట్రస్ గ్రీనింగ్",
        "Bacterial spot": "బాక్టీరియా మచ్చ",
        "Early blight": "ఎర్లీ బ్లైట్",
        "Late blight": "లేట్ బ్లైట్",
        "Leaf mold": "ఆకు మొల్డ్",
        "Septoria leaf spot": "సెప్టోరియా ఆకుమచ్చ",
        "Spider mites Two spotted spider mite": "స్పైడర్ మైట్స్",
        "Target spot": "టార్గెట్ స్పాట్",
        "Tomato yellow leaf curl virus": "టమాటా యెల్లో లీఫ్ కర్ల్ వైరస్",
        "Tomato mosaic virus": "టమాటా మోసాయిక్ వైరస్",
        "Leaf scorch": "ఆకు కాలిన మచ్చ",
    },
    "kn": {
        "Apple scab": "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್",
        "Black rot": "ಬ್ಲಾಕ್ ರಾಟ್",
        "Cedar apple rust": "ಸೀಡರ್ ಆಪಲ್ ರಸ್ಟ್",
        "Healthy": "ಆರೋಗ್ಯಕರ",
        "Powdery mildew": "ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ",
        "Cercospora leaf spot gray leaf spot": "ಸರ್ಕೋಸ್ಪೋರಾ ಎಲೆ ಮಚ್ಚೆ / ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್",
        "Common rust": "ಕಾಮನ್ ರಸ್ಟ್",
        "Northern leaf blight": "ನಾರ್ದರ್ನ್ ಲೀಫ್ ಬ್ಲೈಟ್",
        "Esca black measles": "ಎಸ್ಕಾ / ಬ್ಲಾಕ್ ಮೀಸಲ್ಸ್",
        "Leaf blight Isariopsis leaf spot": "ಲೀಫ್ ಬ್ಲೈಟ್ / ಇಸಾರಿಯೋಪ್ಸಿಸ್ ಲೀಫ್ ಸ್ಪಾಟ್",
        "Haunglongbing Citrus greening": "ಸಿಟ್ರಸ್ ಗ್ರೀನಿಂಗ್",
        "Bacterial spot": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್",
        "Early blight": "ಅರ್ಲಿ ಬ್ಲೈಟ್",
        "Late blight": "ಲೇಟ್ ಬ್ಲೈಟ್",
        "Leaf mold": "ಲೀಫ್ ಮೋಲ್ಡ್",
        "Septoria leaf spot": "ಸೆಪ್ಟೋರಿಯಾ ಲೀಫ್ ಸ್ಪಾಟ್",
        "Spider mites Two spotted spider mite": "ಸ್ಪೈಡರ್ ಮೈಟ್ಸ್",
        "Target spot": "ಟಾರ್ಗೆಟ್ ಸ್ಪಾಟ್",
        "Tomato yellow leaf curl virus": "ಟೊಮೇಟೊ ಯೆಲ್ಲೋ ಲೀಫ್ ಕರ್‌ಲ್ ವೈರಸ್",
        "Tomato mosaic virus": "ಟೊಮೇಟೊ ಮೋಸಾಯಿಕ್ ವೈರಸ್",
        "Leaf scorch": "ಲೀಫ್ ಸ್ಕಾರ್ಚ್",
    },
}


def normalize_label_phrase(text: str) -> str:
    value = pretty_label(text).replace(" - ", " ")
    return re.sub(r"\s+", " ", value).strip()


def localize_crop_name(crop: str, lang_code: str) -> str:
    normalized = normalize_label_phrase(crop)
    translations = CROP_NAME_TRANSLATIONS.get(lang_code, {})
    lowered = {str(k).lower(): v for k, v in translations.items()}
    return translations.get(normalized, lowered.get(normalized.lower(), normalized))


def localize_condition_name(condition: str, lang_code: str) -> str:
    normalized = normalize_label_phrase(condition)
    translations = CONDITION_TRANSLATIONS.get(lang_code, {})
    lowered = {str(k).lower(): v for k, v in translations.items()}
    return translations.get(normalized, lowered.get(normalized.lower(), normalized))


def localized_prediction_label(label: str, lang_code: str) -> str:
    if not label or lang_code == "en":
        return pretty_label(label)

    if "___" in label:
        crop_part, condition_part = label.split("___", 1)
    elif "__" in label:
        crop_part, condition_part = label.split("__", 1)
    else:
        parts = label.split("_", 1)
        crop_part = parts[0]
        condition_part = parts[1] if len(parts) > 1 else ""

    localized_crop = localize_crop_name(crop_part, lang_code)
    localized_condition = localize_condition_name(condition_part or label, lang_code)
    return f"{localized_crop} - {localized_condition}".strip(" -")


def get_weather_for_city(city: str):
    """
    Fetch basic weather info and rain probability for the next hours
    using OpenWeatherMap geocoding + 5-day forecast.
    """
    if not OPENWEATHER_API_KEY:
        return {"ok": False, "error": "missing_api_key"}

    try:
        geo_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_resp = requests.get(
            geo_url,
            params={"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=5,
        )
        if geo_resp.status_code == 401:
            return {"ok": False, "error": "invalid_api_key"}
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return {"ok": False, "error": "city_not_found"}

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        resolved_name = geo_data[0].get("name") or city

        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        resp = requests.get(
            forecast_url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            },
            timeout=5,
        )
        if resp.status_code == 401:
            return {"ok": False, "error": "invalid_api_key"}
        resp.raise_for_status()
        data = resp.json()

        forecast_list = data.get("list", [])[:8]
        if not forecast_list:
            return {"ok": False, "error": "forecast_unavailable"}

        pops = [item.get("pop", 0) for item in forecast_list]
        rain_probability = round(sum(pops) / len(pops) * 100, 1)

        first = forecast_list[0]
        main = first.get("main", {})
        weather_block = (first.get("weather") or [{}])[0]
        temp = main.get("temp")
        humidity = main.get("humidity")

        return {
            "ok": True,
            "rain_probability": rain_probability,
            "temp": temp,
            "humidity": humidity,
            "city": resolved_name,
            "weather_main": weather_block.get("main", ""),
            "weather_description": weather_block.get("description", ""),
        }
    except requests.RequestException:
        return {"ok": False, "error": "network_error"}
    except Exception:
        return {"ok": False, "error": "unknown_error"}


@app.route("/")
def home():
    lang_code, ui = get_lang()
    return render_template("home.html", ui=ui, lang_code=lang_code)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename:
            filename = secure_filename(file.filename)
            if not filename:
                filename = f"leaf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = UPLOAD_DIR / filename
            file.save(filepath)
            print(f"[upload] saved file: {filepath}")
            plant_check = analyze_plant_likelihood(str(filepath))
            photo_quality = analyze_image_quality(str(filepath))
            if not plant_check.get("looks_like_plant"):
                print("[upload] rejected before model call", {"file": filepath.name, "plant_check": plant_check})
                lang_code, ui = get_lang()
                return render_template(
                    "upload.html",
                    ui=ui,
                    lang_code=lang_code,
                    error_message=(
                        "Please upload only a plant leaf image. "
                        "Photos of people, objects, or unclear scenes are not supported."
                    ),
                ), 400

            try:
                prediction_payload = call_model_api(filepath)
            except requests.RequestException as exc:
                print(f"[upload] model API request failed: {exc}")
                lang_code, ui = get_lang()
                return render_template(
                    "upload.html",
                    ui=ui,
                    lang_code=lang_code,
                    error_message="Prediction service is temporarily unavailable. Please try again in a moment.",
                ), 502
            except Exception as exc:
                print(f"[upload] model API processing failed: {exc}")
                lang_code, ui = get_lang()
                return render_template(
                    "upload.html",
                    ui=ui,
                    lang_code=lang_code,
                    error_message=str(exc),
                ), 500

            confidence = float(round(float(prediction_payload.get("confidence", 0.0)), 2))
            predicted_class = prediction_payload.get("label", "Unknown")
            gradcam_context = build_remote_gradcam_context(prediction_payload, filepath)
            rejection_reasons, trust_metrics = should_reject_prediction(photo_quality, plant_check, prediction_payload)
            if rejection_reasons:
                print(
                    "[upload] prediction rejected",
                    {
                        "file": filepath.name,
                        "plant_check": plant_check,
                        "photo_quality": photo_quality.get("score"),
                        "trust_metrics": trust_metrics,
                        "reasons": rejection_reasons,
                    },
                )
                lang_code, ui = get_lang()
                return render_template(
                    "upload.html",
                    ui=ui,
                    lang_code=lang_code,
                    error_message=(
                        "Please upload a clear plant leaf photo only. "
                        "This image does not look trustworthy enough for diagnosis yet."
                    ),
                ), 400

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
            if weather and weather.get("ok"):
                rain_probability = float(weather["rain_probability"])
                temp = weather.get("temp")
                humidity = weather.get("humidity")
                weather_main = (weather.get("weather_main") or "").lower()

                if rain_probability >= 70:
                    weather_advice = "High rain chance in the next hours. Avoid spraying now and protect soil from runoff."
                elif rain_probability >= 40:
                    weather_advice = "Moderate rain risk. Prefer early morning or evening applications in a dry gap."
                else:
                    weather_advice = "Low rain risk. This looks like a better treatment window."

                if temp is not None and temp >= 34:
                    weather_advice += " Temperature is high, so avoid mid-day spraying."
                elif temp is not None and temp <= 18:
                    weather_advice += " Cooler conditions may slow leaf drying, so watch for lingering moisture."

                if humidity is not None and humidity >= 85:
                    weather_advice += " Humidity is high, which can support fungal spread."

                if weather_main in {"rain", "thunderstorm", "drizzle"}:
                    weather_advice += " Rain-related conditions are present in the forecast."
            else:
                rain_probability = 0.0
                temp = None
                humidity = None
                error_code = (weather or {}).get("error")
                if error_code == "invalid_api_key":
                    weather_advice = "Weather service could not be used because the OpenWeather API key is invalid. Update OPENWEATHER_API_KEY in .env."
                elif error_code == "city_not_found":
                    weather_advice = "Weather service could not find that city or village. Try a nearby larger town name."
                elif error_code == "missing_api_key":
                    weather_advice = "Weather advice is unavailable because no OpenWeather API key is configured."
                else:
                    weather_advice = "Weather data is temporarily unavailable. Use your local forecast as a reference."

            lang_code, ui = get_lang()
            localized_disease = localized_prediction_label(predicted_class, lang_code)
            localized_crop = localize_crop_name(crop, lang_code)
            recommended_products = localize_products(recommended_products, lang_code)
            return render_template("result.html",
                                   image_path=to_static_web_path(filepath),
                                   gradcam_path=gradcam_context["gradcam_path"],
                                   gradcam_overlay_path=gradcam_context["gradcam_overlay_path"],
                                   gradcam_heatmap_path=gradcam_context["gradcam_heatmap_path"],
                                   gradcam_layer=gradcam_context["gradcam_layer"],
                                   disease=predicted_class,
                                   localized_disease=localized_disease,
                                   crop=crop,
                                   localized_crop=localized_crop,
                                   city=city,
                                   confidence=confidence,
                                   description=info["description"],
                                   cause=info["cause"],
                                   treatment=info["treatment"],
                                   prevention=info["prevention"],
                                   severity_level=severity_level,
                                   should_contact_department=(severity_level == "high"),
                                   is_uncertain=(confidence < 60),
                                   care_today=care_today,
                                   care_next_days=care_next_days,
                                   care_next_weeks=care_next_weeks,
                                   recommended_products=recommended_products,
                                   fertilizer_days=fertilizer_days,
                                   next_date=next_date,
                                   rain_probability=rain_probability,
                                   photo_quality=photo_quality,
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

    lang_code_from_cookie, _ = get_lang()
    requested_lang = (data.get("lang_code") or "").strip()
    lang_code = requested_lang if requested_lang in ["en", "kn", "hi", "te"] else lang_code_from_cookie

    context = {
        "disease": data.get("disease", ""),
        "confidence": data.get("confidence", ""),
        "rain_probability": data.get("rain_probability", ""),
        "city": data.get("city", ""),
        "crop": data.get("crop", ""),
        "care_today": data.get("care_today", ""),
        "care_next_days": data.get("care_next_days", ""),
        "care_next_weeks": data.get("care_next_weeks", ""),
        "page_title": data.get("page_title", ""),
        "page_path": data.get("page_path", ""),
        "page_headings": data.get("page_headings", []),
        "page_summary": data.get("page_summary", ""),
    }
    return jsonify({"answer": offline_chat_answer(question, lang_code, context)})


@app.route("/schemes")
def schemes():
    lang_code, ui = get_lang()
    schemes = [localize_scheme(scheme, lang_code) for scheme in GOVERNMENT_SCHEMES]
    return render_template("schemes.html", schemes=schemes, ui=ui, lang_code=lang_code)


@app.route("/about")
def about():
    lang_code, ui = get_lang()
    return render_template("about.html", ui=ui, lang_code=lang_code)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    lang_code, ui = get_lang()
    submission = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "farmer_name": (request.form.get("farmer_name") or "").strip(),
            "phone": (request.form.get("phone") or "").strip(),
            "email": (request.form.get("email") or "").strip(),
            "state": (request.form.get("state") or "").strip(),
            "district": (request.form.get("district") or "").strip(),
            "village": (request.form.get("village") or "").strip(),
            "crop": (request.form.get("crop") or "").strip(),
            "issue_type": (request.form.get("issue_type") or "").strip(),
            "severity": (request.form.get("severity") or "").strip(),
            "affected_area": (request.form.get("affected_area") or "").strip(),
            "message": (request.form.get("message") or "").strip(),
        }
        reference_id = create_support_reference()
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO support_requests (
                    created_at, reference_id, farmer_name, phone, email, state, district,
                    village, crop, issue_type, severity, affected_area, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    reference_id,
                    form_data["farmer_name"],
                    form_data["phone"],
                    form_data["email"],
                    form_data["state"],
                    form_data["district"],
                    form_data["village"],
                    form_data["crop"],
                    form_data["issue_type"],
                    form_data["severity"],
                    form_data["affected_area"],
                    form_data["message"],
                ),
            )
            conn.commit()
            submission = {"reference_id": reference_id}
            form_data = {}
        finally:
            conn.close()

    return render_template(
        "contact.html",
        ui=ui,
        lang_code=lang_code,
        states=INDIAN_STATES,
        support_issues=SUPPORT_ISSUES,
        severity_options=SEVERITY_OPTIONS,
        submission=submission,
        form_data=form_data,
    )


@app.route("/store")
def store():
    """Very simple e‑commerce-style catalog page."""
    product_id = request.args.get("product_id", type=int)
    if product_id:
        return redirect(url_for("product_detail", product_id=product_id))

    label = request.args.get("label", "").strip()
    crop_filter = request.args.get("crop", "").strip()
    type_filter = request.args.get("type", "").strip()
    q = request.args.get("q", "").strip()

    items = recommended_products_for_label(label) if label else PRODUCT_CATALOG
    if crop_filter:
        items = [p for p in items if p["crop"].lower() == crop_filter.lower() or p["crop"] == "All"]
    if type_filter:
        items = [p for p in items if p["type"].lower() == type_filter.lower()]
    if q:
        ql = q.lower()
        items = [p for p in items if ql in p.get("name", "").lower() or ql in p.get("usage", "").lower()]

    lang_code, ui = get_lang()
    items = localize_products(items, lang_code)
    return render_template(
        "store.html",
        products=items,
        crop_filter=crop_filter,
        type_filter=type_filter,
        q=q,
        label=label,
        ui=ui,
        lang_code=lang_code,
    )


@app.route("/product/<int:product_id>")
def product_detail(product_id: int):
    product = get_product_by_id(product_id)
    if not product:
        abort(404)
    lang_code, ui = get_lang()
    product = localize_product(product, lang_code)
    return render_template("product.html", product=product, ui=ui, lang_code=lang_code)


@app.route("/simulate-payment", methods=["POST"])
def simulate_payment():
    data = request.get_json(silent=True) or {}
    raw_items = data.get("items") or []
    items = []

    if raw_items:
        for raw_id in raw_items:
            if str(raw_id).isdigit():
                product = get_product_by_id(int(raw_id))
                if product:
                    items.append(product)

    product_id = data.get("product_id")
    if not items and str(product_id).isdigit():
        product = get_product_by_id(int(product_id))
        if product:
            items.append(product)

    if not items:
        return jsonify({"error": "invalid_product"}), 400

    method = (data.get("payment_method") or "upi").strip().lower() or "upi"
    txn_id = "ORD-" + datetime.now().strftime("%Y%m%d%H%M%S") + f"-{random.randint(100, 999)}"
    return jsonify(
        {
            "ok": True,
            "order_id": txn_id,
            "method": method,
            "product_names": [item["name"] for item in items],
            "item_count": len(items),
        }
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
