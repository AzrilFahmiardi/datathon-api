# Influencer Recommendation API

API Flask untuk sistem rekomendasi influencer berdasarkan notebook Jupyter yang telah dibuat. API ini mengimplementasikan semua logika dan algoritma yang sama persis dengan notebook asli.

## 🚀 Features

- **Budget Optimization**: Optimasi konten mix berdasarkan budget dan rate card influencer
- **Persona Semantic Matching**: Pencocokan persona menggunakan NLP dan sentence transformers
- **Audience Matching**: Analisis kecocokan audience berdasarkan data Instagram Insights
- **Performance Prediction**: Prediksi performa campaign menggunakan pendekatan heuristik
- **Adaptive Weight Calculator**: Kalkulasi bobot dinamis berdasarkan karakteristik brief
- **Content Mix Optimization**: Optimasi kombinasi konten (Stories, Feeds, Reels)
- **Multi-Objective Ranking**: Ranking influencer dengan multiple kriteria

## 📋 Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

## 🔧 Installation

1. **Clone atau download project ini**
   ```bash
   git clone <repository-url>
   cd datathon-api
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run aplikasi**
   ```bash
   python app.py
   ```

API akan berjalan di `http://localhost:5000`

## 🛠️ API Endpoints

### 1. Health Check
```http
GET /
```
Mengecek status API

**Response:**
```json
{
  "status": "healthy",
  "message": "Influencer Recommendation API is running",
  "version": "1.0.0",
  "timestamp": "2025-01-24T10:00:00.000000"
}
```

### 2. Recommend Influencers
```http
POST /api/recommend-influencers
```
Endpoint utama untuk mendapatkan rekomendasi influencer.

**Query Parameters:**
- `adaptive_weights` (boolean, default: true): Gunakan adaptive weight calculator
- `include_insights` (boolean, default: true): Include detailed insights untuk setiap influencer
- `include_raw` (boolean, default: false): Include raw data influencer

**Request Body:**
```json
{
  "brief_id": "BRIEF_001",
  "brand_name": "Avoskin",
  "industry": "Skincare & Beauty",
  "product_name": "GlowSkin Vitamin C Serum",
  "overview": "Premium vitamin C serum untuk mencerahkan dan melindungi kulit dari radikal bebas",
  "usp": "Formula 20% Vitamin C dengan teknologi nano-encapsulation untuk penetrasi optimal",
  "marketing_objective": ["Cognitive", "Affective"],
  "target_goals": ["Awareness", "Brand Perception", "Product Education"],
  "timing_campaign": "2025-02-15",
  "audience_preference": {
    "top_locations": {
      "countries": ["Indonesia", "Malaysia", "Singapore"],
      "cities": ["Jakarta", "Surabaya", "Bandung"]
    },
    "age_range": ["18-24", "25-34"],
    "gender": ["Female"]
  },
  "influencer_persona": "Beauty enthusiast, skincare expert, authentic product reviewer",
  "total_influencer": 3,
  "niche": ["Beauty", "Lifestyle"],
  "location_prior": ["Indonesia", "Malaysia"],
  "esg_allignment": ["Cruelty-free", "sustainable packaging"],
  "budget": 50000000.0,
  "output": {
    "content_types": ["Reels", "Feeds"],
    "deliverables": 6
  },
  "risk_tolerance": "Medium"
}
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-01-24T10:00:00.000000",
  "brief": {
    "brief_id": "BRIEF_001",
    "summary": "Brief summary with target audience info...",
    "total_requested": 3,
    "total_found": 3
  },
  "recommendations": [
    {
      "rank": 1,
      "username": "influencer_username",
      "tier": "Micro",
      "expertise": "Beauty",
      "scores": {
        "final_score": 0.8524,
        "persona_fit": 0.7823,
        "audience_fit": 0.9156,
        "performance_pred": 0.8234,
        "budget_efficiency": 0.7891
      },
      "performance_metrics": {
        "engagement_rate": 0.0324,
        "authenticity_score": 0.8456,
        "reach_potential": 0.7234,
        "brand_fit": 0.8123
      },
      "optimal_content_mix": {
        "story_count": 0,
        "feeds_count": 3,
        "reels_count": 3,
        "total_cost": 45000000,
        "total_impact": 78.5,
        "remaining_budget": 5000000
      },
      "insights": "Detailed text insights about the influencer..."
    }
  ],
  "metadata": {
    "use_adaptive_weights": true,
    "adaptive_weights_info": {
      "applied_adjustments": ["Cognitive focus: +persona_fit, +audience_fit"],
      "total_adjustments": 1,
      "final_weights": {
        "persona_fit": 0.3,
        "audience_fit": 0.3,
        "performance_pred": 0.25,
        "budget_efficiency": 0.15
      }
    },
    "scoring_strategy": {...},
    "include_insights": true
  }
}
```

### 3. Get Influencer Insight
```http
GET /api/influencer-insight/{username}
```
Mendapatkan insight detail untuk influencer tertentu.

**Response:**
```json
{
  "status": "success",
  "username": "influencer_username",
  "insight": "Detailed text insights about conversion potential, caption behavior, comment quality, etc...",
  "timestamp": "2025-01-24T10:00:00.000000"
}
```

### 4. Calculate Adaptive Weights
```http
POST /api/adaptive-weights
```
Menghitung adaptive weights berdasarkan karakteristik brief.

**Request Body:**
```json
{
  "marketing_objective": ["Cognitive", "Affective"],
  "target_goals": ["Awareness", "Brand Perception"],
  "timing_campaign": "2025-02-15",
  "esg_allignment": ["Cruelty-free"],
  "risk_tolerance": "Medium",
  "niche": ["Beauty", "Lifestyle"],
  "location_prior": ["Indonesia"]
}
```

**Response:**
```json
{
  "status": "success",
  "adaptive_weights": {
    "persona_fit": 0.35,
    "audience_fit": 0.35,
    "performance_pred": 0.2,
    "budget_efficiency": 0.1
  },
  "applied_adjustments": [
    "Cognitive focus: +persona_fit, +audience_fit",
    "Persona niche: +persona_fit"
  ],
  "total_adjustments": 2,
  "timestamp": "2025-01-24T10:00:00.000000"
}
```

### 5. Validate Brief
```http
POST /api/validate-brief
```
Memvalidasi struktur JSON brief.

**Response:**
```json
{
  "status": "success",
  "is_valid": true,
  "message": "Valid input",
  "timestamp": "2025-01-24T10:00:00.000000"
}
```

### 6. Data Status
```http
GET /api/data-status
```
Mengecek status loading dataset.

**Response:**
```json
{
  "status": "success",
  "all_data_loaded": true,
  "individual_status": {
    "instagram_influencers": true,
    "labeled_caption": true,
    "labeled_comment": true,
    "bio": true,
    "captions": true
  },
  "data_shapes": {
    "instagram_influencers": [1000, 25],
    "labeled_caption": [5000, 4],
    "labeled_comment": [10000, 4],
    "bio": [1000, 2],
    "captions": [8000, 3]
  },
  "timestamp": "2025-01-24T10:00:00.000000"
}
```

### 7. Reload Data
```http
POST /api/reload-data
```
Reload ulang semua dataset.

## 🔍 Core Components

API ini mengimplementasikan semua komponen dari notebook asli:

### 1. BudgetOptimizer
- Filter influencer berdasarkan budget
- Optimasi content mix menggunakan Linear Programming
- Support untuk requirement-based optimization

### 2. PersonaSemanticMatcher  
- Semantic matching menggunakan sentence transformers
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Cosine similarity untuk scoring

### 3. AudienceMatchingGNN
- Real Instagram Insights data matching
- Flexible weighting strategies
- Location, age, dan gender scoring

### 4. SocialMediaPerformancePredictor
- Heuristic-based performance prediction
- Engagement, authenticity, reach scoring
- Tier-based benchmarking

### 5. SimpleAdaptiveWeightCalculator
- Dynamic weight calculation
- Based on marketing objectives, goals, niche, etc.
- Multiple adjustment strategies

## 📊 Scoring System

API menggunakan sistem scoring multi-komponen:

1. **Persona Fit** (0-1): Kecocokan persona influencer dengan brief
2. **Audience Fit** (0-1): Kecocokan audience demographics 
3. **Performance Prediction** (0-1): Prediksi performa campaign
4. **Budget Efficiency**: Impact points per budget unit

Final score = weighted sum dari keempat komponen ini.

## 🎛️ Adaptive Weighting

Sistem adaptive weight menyesuaikan bobot scoring berdasarkan:

- **Marketing Objective**: Cognitive, Conative, Affective
- **Target Goals**: Awareness, Conversion, Engagement, dll
- **Risk Tolerance**: High, Medium, Low  
- **Niche**: Beauty, Tech, Lifestyle, dll
- **ESG Alignment**: Sustainability focus
- **Location Priority**: Geographic targeting
- **Campaign Timing**: Special events/seasons

## 🚨 Error Handling

API mengembalikan error responses dengan format:

```json
{
  "error": "Error message description",
  "status": "error"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid JSON, missing fields)
- `404`: Not Found (no recommendations, influencer not found)
- `500`: Internal Server Error

## 🔧 Configuration

API secara otomatis load dataset dari GitHub repository yang sama dengan notebook asli:
- Instagram Influencers Final
- Labeled Caption  
- Labeled Comment
- Bio
- Captions

## 📝 Example Usage

### Python Client Example
```python
import requests
import json

# Prepare brief data
brief_data = {
    "brief_id": "BRIEF_001",
    "brand_name": "Beauty Brand",
    "industry": "Beauty",
    "product_name": "Skincare Product",
    "overview": "Premium skincare product",
    "influencer_persona": "Beauty enthusiast, authentic reviewer",
    "total_influencer": 3,
    "budget": 50000000.0,
    "audience_preference": {
        "top_locations": {
            "countries": ["Indonesia"],
            "cities": ["Jakarta"]
        },
        "age_range": ["18-24", "25-34"],
        "gender": ["Female"]
    }
}

# Make request
response = requests.post(
    "http://localhost:5000/api/recommend-influencers",
    json=brief_data,
    params={"adaptive_weights": "true", "include_insights": "true"}
)

# Process response
if response.status_code == 200:
    result = response.json()
    recommendations = result["recommendations"]
    for rec in recommendations:
        print(f"Rank {rec['rank']}: @{rec['username']} - Score: {rec['scores']['final_score']:.2%}")
else:
    print(f"Error: {response.json()}")
```

## 🤝 Contributing

API ini dibuat berdasarkan notebook Jupyter yang telah dikembangkan. Untuk kontribusi:

1. Pastikan logika tetap konsisten dengan notebook asli
2. Test semua endpoint sebelum commit
3. Update dokumentasi jika ada perubahan

## 📄 License

Project ini menggunakan lisensi yang sama dengan notebook asli.

---

**Note**: API ini adalah implementasi langsung dari notebook Jupyter tanpa mengubah logika algoritma sama sekali. Semua komponen, scoring system, dan metodologi tetap identik dengan versi notebook asli.
