# Influencer Recommendation API

API Flask untuk sistem rekomendasi influencer berbasis machine learning yang dikembangkan untuk Datathon Ristek 2025. Sistem ini menggunakan berbagai algoritma advanced seperti semantic matching, budget optimization, dan multi-objective ranking untuk memberikan rekomendasi influencer yang optimal.


## 🛠 Cara Install & Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/AzrilFahmiardi/datathon-api.git
cd datathon-api
```

### 2. Setup Python Environment

**Menggunakan Virtual Environment (Recommended):**

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

**Atau menggunakan Conda:**

```bash
# Buat conda environment
conda create -n influencer-api python=3.11
conda activate influencer-api
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables (Opsional)

Buat file `.env` di root directory:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
LOG_LEVEL=INFO

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Data URLs (default sudah tersedia)
INFLUENCERS_DATA_URL=https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/instagram_influencers_final.csv
CAPTION_DATA_URL=https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/labeled_caption.csv
COMMENT_DATA_URL=https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/labeled_comment.csv
BIO_DATA_URL=https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/bio.csv
CAPTIONS_DATA_URL=https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/captions.csv
```

### 5. Jalankan Aplikasi

**Development Mode:**

```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

## 📡 API Endpoints

### Health Check
- **GET** `/` - Health check dengan dokumentasi API lengkap
- **GET** `/health` - Simple health check untuk monitoring

### Main Endpoints
- **POST** `/api/recommend-influencers` - Endpoint utama untuk rekomendasi influencer
- **GET** `/api/influencer-insight/{username}` - Detail insight influencer
- **POST** `/api/adaptive-weights` - Hitung adaptive weights
- **POST** `/api/validate-brief` - Validasi struktur JSON brief

### Utility Endpoints
- **GET** `/api/data-status` - Status loading dataset
- **POST** `/api/reload-data` - Reload dataset

### Query Parameters
- `adaptive_weights=true` - Gunakan adaptive weight calculation
- `include_insights=true` - Include detailed insights
- `include_raw=false` - Exclude raw data dari response

## 🎯 Cara Penggunaan

### 1. Health Check

```bash
curl http://localhost:5000/
```

### 2. Cek Status Data

```bash
curl http://localhost:5000/api/data-status
```

### 3. Request Rekomendasi Influencer

**Basic Request:**

```bash
curl -X POST http://localhost:5000/api/recommend-influencers \
  -H "Content-Type: application/json" \
  -d '{
    "brief_id": "BRIEF_001",
    "brand_name": "Beauty Brand",
    "industry": "Beauty",
    "influencer_persona": "Beauty enthusiast, authentic reviewer",
    "total_influencer": 3,
    "budget": 50000000.0,
    "audience_preference": {
      "age_range": ["18-24", "25-34"],
      "gender": ["Female"]
    }
  }'
```

**Advanced Request dengan Parameters:**

```bash
curl -X POST "http://localhost:5000/api/recommend-influencers?adaptive_weights=true&include_insights=true" \
  -H "Content-Type: application/json" \
  -d '{
    "brief_id": "BRIEF_002",
    "brand_name": "Avoskin",
    "industry": "Skincare & Beauty",
    "product_name": "GlowSkin Vitamin C Serum",
    "overview": "Premium vitamin C serum untuk kulit glowing",
    "usp": "Formula 20% Vitamin C dengan teknologi advanced",
    "marketing_objective": ["Cognitive", "Affective"],
    "target_goals": ["Awareness", "Brand Perception"],
    "timing_campaign": "2025-02-15",
    "audience_preference": {
      "top_locations": {
        "countries": ["Indonesia", "Malaysia"],
        "cities": ["Jakarta", "Surabaya"]
      },
      "age_range": ["18-24", "25-34"],
      "gender": ["Female"]
    },
    "influencer_persona": "Beauty enthusiast, skincare expert, authentic content creator",
    "total_influencer": 5,
    "niche": ["Beauty", "Lifestyle"],
    "location_prior": ["Indonesia", "Malaysia"],
    "esg_allignment": ["Cruelty-free", "sustainable packaging"],
    "budget": 100000000.0,
    "output": {
      "content_types": ["Reels", "Feeds"],
      "deliverables": 10
    }
  }'
```

### 4. Get Influencer Insight

```bash
curl http://localhost:5000/api/influencer-insight/username_influencer
```


**Dibuat untuk Datathon Ristek 2025 oleh Tim Yujiem Rookie** 🏆

**Tech Stack:** Flask, pandas, scikit-learn, PyTorch, sentence-transformers, Gunicorn
