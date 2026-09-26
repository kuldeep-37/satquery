"""Pre-seed persistent translation cache with rich domain vocabulary across 10 Indian languages."""

import json
import os

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "backend", "translation_cache.json")

SEED_ENTRIES = [
    # -------------------------------------------------------------------------
    # 1. Preset Chips & Common Questions
    # -------------------------------------------------------------------------
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "hi",
        "translated": "इस उपग्रह छवि में दिखाए गए भूमि आवरण का वर्णन करें।"
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "ta",
        "translated": "இந்த செயற்கைக்கோள் படத்தில் காட்டப்பட்டுள்ள நிலப்பரப்பை விவரிக்கவும்."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "te",
        "translated": "ఈ ఉపగ్రహ చిత్రంలో చూపబడిన భూ కవరేజీని వివరించండి."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "bn",
        "translated": "এই উপগ্রহ চিত্রে প্রদর্শিত ভূমি আবরণ বর্ণনা করুন।"
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "mr",
        "translated": "या उपग्रह प्रतिमेमध्ये दर्शविलेल्या जमिनीच्या आच्छादनाचे वर्णन करा."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "kn",
        "translated": "ಈ ಉಪಗ್ರಹ ಚಿತ್ರದಲ್ಲಿ ತೋರಿಸಲಾದ ಭೂ ಆವರಣವನ್ನು ವಿವರಿಸಿ."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "ml",
        "translated": "ഈ ഉപഗ്രഹ ചിത്രത്തിൽ കാണിച്ചിരിക്കുന്ന ഭൂവിസ്തൃതി വിവരിക്കുക."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "gu",
        "translated": "આ ઉપગ્રહ ચિત્રમાં દર્શાવેલ જમીન આવરણનું વર્ણન કરો."
    },
    {
        "text": "Describe the land cover shown in this satellite image.",
        "lang": "pa",
        "translated": "ਇਸ ਉਪਗ੍ਰਹਿ ਤਸਵੀਰ ਵਿੱਚ ਦਿਖਾਈ ਗਈ ਜ਼ਮੀਨ ਦੇ ਢੱਕਣ ਦਾ ਵਰਣਨ ਕਰੋ।"
    },

    # Assess urban building density and road network connectivity
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "hi",
        "translated": "शहरी इमारत घनत्व और सड़क नेटवर्क कनेक्टिविटी का आकलन करें।"
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "ta",
        "translated": "நகர்ப்புற கட்டிட அடர்த்தி மற்றும் சாலை இணைப்பு ஆகியவற்றை மதிப்பிடுக."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "te",
        "translated": "పట్టణ భవనాల సాంద్రత మరియు రహదారి కనెక్టివిటీని అంచనా వేయండి."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "bn",
        "translated": "শহুরে ভবনের ঘনত্ব এবং রাস্তার নেটওয়ার্ক সংযোগ মূল্যায়ন করুন।"
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "mr",
        "translated": "शहरी इमारतींची घनता आणि रस्ते नेटवर्क कनेक्टिव्हिटीचे मूल्यांकन करा."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "kn",
        "translated": "ನಗರ ಕಟ್ಟಡಗಳ ಸಾಂದ್ರತೆ ಮತ್ತು ರಸ್ತೆ ಜಾಲದ ಸಂಪರ್ಕವನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡಿ."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "ml",
        "translated": "നഗര കെട്ടിട സാന്ദ്രതയും റോഡ് ശൃംഖലയുടെ കണക്റ്റിവിറ്റിയും വിലയിരുത്തുക."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "gu",
        "translated": "શહેરી મકાનોની ઘનતા અને રોડ નેટવર્ક જોડાણનું મૂલ્યાંકન કરો."
    },
    {
        "text": "Assess urban building density and road network connectivity.",
        "lang": "pa",
        "translated": "ਸ਼ਹਿਰੀ ਇਮਾਰਤਾਂ ਦੀ ਘਣਤਾ ਅਤੇ ਸੜਕ ਨੈੱਟਵਰਕ ਕਨੈਕਟੀਵਿਟੀ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ।"
    },

    # Evaluate wildfire hazard, drought aridity, and fuel moisture status
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "hi",
        "translated": "दावानल का खतरा, सूखे की शुष्कता और ईंधन नमी की स्थिति का मूल्यांकन करें।"
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "ta",
        "translated": "காட்டுத்தீ ஆபத்து, வறட்சி மற்றும் ஈரப்பத நிலையை மதிப்பிடுக."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "te",
        "translated": "దావానల ప్రమాదం, కరువు తీవ్రత మరియు తేమ స్థితిని అంచనా వేయండి."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "bn",
        "translated": "দাবানলের ঝুঁকি, খরার শুষ্কতা এবং জ্বালানী আর্দ্রতার অবস্থা মূল্যায়ন করুন।"
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "mr",
        "translated": "वणव्याचा धोका, दुष्काळी कोरडेपणा आणि ओलाव्याची स्थिती तपासा."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "kn",
        "translated": "ಕಾಡ್ಗಿಚ್ಚಿನ ಅಪಾಯ, ಬರಗಾಲದ ತೀವ್ರತೆ ಮತ್ತು ತೇವಾಂಶದ ಸ್ಥಿತಿಯನ್ನು ನಿರ್ಣಯಿಸಿ."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "ml",
        "translated": "കാട്ടുതീ സാധ്യത, വരൾച്ച, ഈർപ്പത്തിന്റെ അളവ് എന്നിവ വിലയിരുത്തുക."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "gu",
        "translated": "દાવાનળનું જોખમ, દુષ્કાળ અને ભેજની સ્થિતિનું મૂલ્યાંકન કરો."
    },
    {
        "text": "Evaluate wildfire hazard, drought aridity, and fuel moisture status.",
        "lang": "pa",
        "translated": "ਜੰਗਲ ਦੀ ਅੱਗ ਦਾ ਖਤਰਾ, ਸੋਕਾ ਅਤੇ ਨਮੀ ਦੀ ਸਥਿਤੀ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ।"
    },

    # Is there any evidence of deforestation, logging, or tree canopy loss?
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "hi",
        "translated": "क्या वनों की कटाई, लकड़ी काटने या पेड़ की छतरी के नुकसान का कोई सबूत है?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "ta",
        "translated": "காடழிப்பு அல்லது மரங்கள் வெட்டப்பட்டதற்கான ஏதேனும் சான்றுகள் உள்ளதா?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "te",
        "translated": "అటవీ నిర్మూలన లేదా చెట్లు నరికివేతకు సంబంధించిన ఆధారాలు ఏమైనా ఉన్నాయా?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "bn",
        "translated": "বন উজাড় বা গাছ কাটার কোনো প্রমাণ আছে কি?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "mr",
        "translated": "जंगलतोड किंवा झाडे तोडल्याचा काही पुरावा आहे का?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "kn",
        "translated": "ಅರಣ್ಯನಾಶ ಅಥವಾ ಮರಗಳನ್ನು ಕಡಿದಿರುವ ಬಗ್ಗೆ ಯಾವುದೇ ಪುರಾವೆಗಳಿವೆಯೇ?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "ml",
        "translated": "വനനശീകരണത്തിന്റെയോ മരംമുറിയുടെയോ എന്തെങ്കിലും തെളിവുകളുണ്ടോ?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "gu",
        "translated": "શું જંગલ કાપવા અથવા વૃક્ષોના નુકસાનના કોઈ પુરાવા છે?"
    },
    {
        "text": "Is there any evidence of deforestation, logging, or tree canopy loss?",
        "lang": "pa",
        "translated": "ਕੀ ਜੰਗਲਾਂ ਦੀ ਕਟਾਈ ਜਾਂ ਰੁੱਖਾਂ ਦੇ ਨੁਕਸਾਨ ਦਾ ਕੋਈ ਸਬੂਤ ਹੈ?"
    },

    # Check water quality, sediment load, and flood inundation risk
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "hi",
        "translated": "जल गुणवत्ता, तलछट भार और बाढ़ के खतरे की जांच करें।"
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "ta",
        "translated": "நீரின் தரம், படிவு சுமை மற்றும் வெள்ள அபாயத்தை சரிபார்க்கவும்."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "te",
        "translated": "నీటి నాణ్యత, అవక్షేప భారం మరియు వరద ముంపు ప్రమాదాన్ని తనిఖీ చేయండి."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "bn",
        "translated": "পানির গুণমান, পলির ভার এবং বন্যার ঝুঁকি পরীক্ষা করুন।"
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "mr",
        "translated": "पाण्याची गुणवत्ता, गाळाचे प्रमाण आणि पुराचा धोका तपासा."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "kn",
        "translated": "ನೀರಿನ ಗುಣಮಟ್ಟ, ಕಲ್ಮಷದ ಹೊರೆ ಮತ್ತು ಪ್ರವಾಹದ ಅಪಾಯವನ್ನು ಪರೀಕ್ಷಿಸಿ."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "ml",
        "translated": "ജലഗുണനിലവാരം, എക്കൽ ഭാരം, വെള്ളപ്പൊക്ക സാധ്യത എന്നിവ പരിശോധിക്കുക."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "gu",
        "translated": "પાણીની ગુણવત્તા, કાંપનો ભાર અને પૂરના જોખમની તપાસ કરો."
    },
    {
        "text": "Check water quality, sediment load, and flood inundation risk.",
        "lang": "pa",
        "translated": "ਪਾਣੀ ਦੀ ਗੁਣਵੱਤਾ, ਤਲਛਟ ਲੋਡ ਅਤੇ ਹੜ੍ਹ ਦੇ ਖਤਰੇ ਦੀ ਜਾਂਚ ਕਰੋ।"
    },

    # Analyze agricultural crop vigor, parcel patterns, and irrigation status
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "hi",
        "translated": "कृषि फसल जीवन शक्ति, पार्सल पैटर्न और सिंचाई स्थिति का विश्लेषण करें।"
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "ta",
        "translated": "விவசாய பயிர் வளர்ச்சி, நில அமைப்புகள் மற்றும் பாசன நிலையை பகுப்பாய்வு செய்க."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "te",
        "translated": "వ్యవసాయ పంట ఆరోగ్యం, భూ నమూనాలు మరియు నీటిపారుదల స్థితిని విశ్లేషించండి."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "bn",
        "translated": "কৃষি ফসলের সতেজতা, জমির ধরণ এবং সেচ ব্যবস্থা বিশ্লেষণ করুন।"
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "mr",
        "translated": "शेतीतील पिकांचे आरोग्य, जमिनीचे नमुने आणि सिंचनाची स्थिती तपासा."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "kn",
        "translated": "ಕೃಷಿ ಬೆಳೆಗಳ ಹುರುಪು, ಕೃಷಿ ಭೂಮಿಯ ಮಾದರಿಗಳು ಮತ್ತು ನೀರಾವರಿ ಸ್ಥಿತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "ml",
        "translated": "കാർഷിക വിളകളുടെ ആരോഗ്യം, ഭൂമിയുടെ ഘടന, ജലസേചന സ്ഥിതി എന്നിവ വിശകലനം ചെയ്യുക."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "gu",
        "translated": "ખેતીના પાકની જોમ, જમીનની પેટર્ન અને સિંચાઈની સ્થિતિનું વિશ્લેષણ કરો."
    },
    {
        "text": "Analyze agricultural crop vigor, parcel patterns, and irrigation status.",
        "lang": "pa",
        "translated": "ਖੇਤੀਬਾੜੀ ਫਸਲਾਂ ਦੀ ਸਿਹਤ, ਪੈਟਰਨ ਅਤੇ ਸਿੰਚਾਈ ਸਥਿਤੀ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ।"
    },

    # -------------------------------------------------------------------------
    # 2. Hindi Questions to English Mapping (for Input Preprocessing)
    # -------------------------------------------------------------------------
    {
        "text": "क्या इस वन क्षेत्र में आग लगने या सूखे का कोई खतरा है?",
        "lang": "en",
        "translated": "Is there any risk of wildfire or drought in this forest area?"
    },
    {
        "text": "क्या यहाँ बाढ़ का कोई खतरा है?",
        "lang": "en",
        "translated": "Is there any risk of flooding here?"
    },
    {
        "text": "क्या यहाँ जलभराव का खतरा है?",
        "lang": "en",
        "translated": "Is there any danger of waterlogging here?"
    },
    {
        "text": "क्या यह पानी पीने योग्य है?",
        "lang": "en",
        "translated": "Is this water potable or of good quality?"
    },
    {
        "text": "क्या यहाँ सड़क या परिवहन नेटवर्क है?",
        "lang": "en",
        "translated": "Is there a road or transportation network here?"
    },
    {
        "text": "क्या यहाँ फसलें अच्छी हैं?",
        "lang": "en",
        "translated": "Are the crops healthy and productive here?"
    },
    {
        "text": "क्या यह हेलीकॉप्टर लैंडिंग के लिए सुरक्षित है?",
        "lang": "en",
        "translated": "Is this safe as a helicopter landing zone?"
    },
    {
        "text": "यहाँ इमारतों का घनत्व कितना है?",
        "lang": "en",
        "translated": "What is the building density and zoning here?"
    },

    # -------------------------------------------------------------------------
    # 3. Standard Model Output Captions
    # -------------------------------------------------------------------------
    {
        "text": "this satellite image shows a river",
        "lang": "hi",
        "translated": "यह उपग्रह छवि एक नदी को दर्शाती है"
    },
    {
        "text": "this satellite image shows a dense forest",
        "lang": "hi",
        "translated": "यह उपग्रह छवि एक घने जंगल को दर्शाती है"
    },
    {
        "text": "this satellite image shows residential buildings",
        "lang": "hi",
        "translated": "यह उपग्रह छवि आवासीय भवनों को दर्शाती है"
    },
    {
        "text": "this satellite image shows agricultural fields",
        "lang": "hi",
        "translated": "यह उपग्रह छवि कृषि क्षेत्रों को दर्शाती है"
    },
    {
        "text": "this satellite image shows a natural landscape",
        "lang": "hi",
        "translated": "यह उपग्रह छवि एक प्राकृतिक परिदृश्य दिखाती है"
    },
    {
        "text": "this satellite image shows a road and trees",
        "lang": "hi",
        "translated": "यह उपग्रह छवि एक सड़क और पेड़ों को दर्शाती है"
    },
]


def seed_cache():
    existing = []
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    seen = {(e.get("text", "").strip(), e.get("lang", "").strip()) for e in existing}

    added = 0
    for item in SEED_ENTRIES:
        k = (item["text"].strip(), item["lang"].strip())
        if k not in seen:
            existing.append(item)
            seen.add(k)
            added += 1

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"Pre-seeded {added} new translations into cache. Total entries: {len(existing)}")


if __name__ == "__main__":
    seed_cache()
