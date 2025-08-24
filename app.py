from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
import traceback
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import semua class dan fungsi dari sistem rekomendasi
from influencer_recommendation_system import (
    BudgetOptimizer, 
    PersonaSemanticMatcher, 
    AudienceMatchingGNN,
    SocialMediaPerformancePredictor,
    MultiObjectiveRanker,
    SOTAInfluencerMatcher,
    SimpleAdaptiveWeightCalculator,
    get_top_influencers_for_brief,
    generate_brief_summary,
    generate_influencer_insight,
    process_json_brief_input,
    convert_json_to_brief_dataframe,
    validate_json_brief_input
)

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=os.getenv('CORS_ORIGINS', '*').split(','))

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Configuration from environment variables
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')
app.config['FLASK_DEBUG'] = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Data URLs from environment variables
INFLUENCERS_DATA_URL = os.getenv('INFLUENCERS_DATA_URL', 'https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/instagram_influencers_final.csv')
CAPTION_DATA_URL = os.getenv('CAPTION_DATA_URL', 'https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/labeled_caption.csv')
COMMENT_DATA_URL = os.getenv('COMMENT_DATA_URL', 'https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/labeled_comment.csv')
BIO_DATA_URL = os.getenv('BIO_DATA_URL', 'https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/bio.csv')
CAPTIONS_DATA_URL = os.getenv('CAPTIONS_DATA_URL', 'https://raw.githubusercontent.com/AzrilFahmiardi/yujiem-rookie-datathon-2025/refs/heads/master/input/captions.csv')

# Global variables untuk data
df_instagram_influencers = None
df_labeled_caption = None
df_labeled_comment = None
df_bio = None
df_captions = None

def load_data():
    """Load all required datasets"""
    global df_instagram_influencers, df_labeled_caption, df_labeled_comment, df_bio, df_captions
    
    try:
        logger.info("Loading datasets...")
        df_instagram_influencers = pd.read_csv(INFLUENCERS_DATA_URL)
        df_labeled_caption = pd.read_csv(CAPTION_DATA_URL)
        df_labeled_comment = pd.read_csv(COMMENT_DATA_URL)
        df_bio = pd.read_csv(BIO_DATA_URL)
        df_captions = pd.read_csv(CAPTIONS_DATA_URL)
        logger.info("All datasets loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return False

# Load data saat aplikasi start (hanya sekali)
logger.info("Loading datasets on startup...")
if not load_data():
    logger.error("Failed to load data. Some features may not work.")

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint with API documentation"""
    return jsonify({
        "status": "healthy",
        "message": "Influencer Recommendation API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "api_info": {
            "description": "API Flask untuk sistem rekomendasi influencer berdasarkan notebook Jupyter",
            "base_url": "http://localhost:5000",
            "endpoints": {
                "health": {
                    "method": "GET",
                    "path": "/",
                    "description": "Health check dan informasi API"
                },
                "recommend": {
                    "method": "POST", 
                    "path": "/api/recommend-influencers",
                    "description": "Endpoint utama untuk mendapatkan rekomendasi influencer",
                    "params": "?adaptive_weights=true&include_insights=true&include_raw=false"
                },
                "influencer_insight": {
                    "method": "GET",
                    "path": "/api/influencer-insight/{username}",
                    "description": "Mendapatkan insight detail untuk influencer tertentu"
                },
                "adaptive_weights": {
                    "method": "POST",
                    "path": "/api/adaptive-weights", 
                    "description": "Menghitung adaptive weights berdasarkan brief"
                },
                "validate_brief": {
                    "method": "POST",
                    "path": "/api/validate-brief",
                    "description": "Memvalidasi struktur JSON brief"
                },
                "data_status": {
                    "method": "GET",
                    "path": "/api/data-status",
                    "description": "Mengecek status loading dataset"
                },
                "reload_data": {
                    "method": "POST", 
                    "path": "/api/reload-data",
                    "description": "Reload ulang semua dataset"
                }
            },
            "features": [
                "Budget Optimization",
                "Persona Semantic Matching", 
                "Audience Matching",
                "Performance Prediction",
                "Adaptive Weight Calculator",
                "Content Mix Optimization",
                "Multi-Objective Ranking"
            ],
            "testing": {
                "postman_collection": "Import endpoints ke Postman untuk testing",
                "test_script": "Jalankan: python test_api.py",
                "sample_request": {
                    "url": "POST /api/recommend-influencers",
                    "body": {
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
                    }
                }
            }
        }
    })

@app.route('/health', methods=['GET'])
def simple_health():
    """Simple health check for Render"""
    return jsonify({"status": "ok"})

@app.route('/api/recommend-influencers', methods=['POST'])
def recommend_influencers():
    """
    Main endpoint untuk rekomendasi influencer
    
    Expected JSON payload:
    {
        "brief_id": "BRIEF_001",
        "brand_name": "Avoskin",
        "industry": "Skincare & Beauty",
        "product_name": "GlowSkin Vitamin C Serum",
        "overview": "Premium vitamin C serum...",
        "usp": "Formula 20% Vitamin C...",
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
        "influencer_persona": "Beauty enthusiast, skincare expert...",
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
    """
    try:
        # Get request data
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "status": "error"
            }), 400

        json_brief_input = request.get_json()
        
        # Validate required fields
        is_valid, validation_message = validate_json_brief_input(json_brief_input)
        if not is_valid:
            return jsonify({
                "error": validation_message,
                "status": "error"
            }), 400

        # Check if data is loaded
        if df_instagram_influencers is None:
            return jsonify({
                "error": "Data not loaded. Please try again later.",
                "status": "error"
            }), 500

        # Extract parameters dari request
        use_adaptive_weights = request.args.get('adaptive_weights', 'true').lower() == 'true'
        include_insights = request.args.get('include_insights', 'true').lower() == 'true'
        
        # Custom priorities jika tidak menggunakan adaptive weights
        custom_priorities = None
        if not use_adaptive_weights and 'brief_priorities' in json_brief_input:
            custom_priorities = json_brief_input['brief_priorities']

        # Process menggunakan sistem rekomendasi
        brief_df = convert_json_to_brief_dataframe(json_brief_input)
        brief_id = json_brief_input.get("brief_id")
        top_n = json_brief_input.get("total_influencer", 3)

        # Calculate adaptive weights jika diminta
        adaptive_weights = None
        adaptive_info = None
        
        if use_adaptive_weights:
            adaptive_calculator = SimpleAdaptiveWeightCalculator()
            
            adaptive_result = adaptive_calculator.calculate_adaptive_weights(
                marketing_objective=json_brief_input.get('marketing_objective'),
                target_goals=json_brief_input.get('target_goals'),
                timing_campaign=json_brief_input.get('timing_campaign'),
                esg_alignment=', '.join(json_brief_input.get('esg_allignment', [])) if json_brief_input.get('esg_allignment') else None,
                risk_tolerance=json_brief_input.get('risk_tolerance'),
                niche=json_brief_input.get('niche'),
                location_prior=', '.join(json_brief_input.get('location_prior', [])) if json_brief_input.get('location_prior') else None
            )
            
            adaptive_weights = adaptive_result['weights']
            adaptive_info = {
                "applied_adjustments": adaptive_result['applied_adjustments'],
                "total_adjustments": adaptive_result['total_adjustments'],
                "final_weights": adaptive_weights
            }

        # Get recommendations
        recommendations = get_top_influencers_for_brief(
            brief_id=brief_id,
            briefs_df=brief_df,
            influencers_df=df_instagram_influencers,
            bio_df=df_bio,
            caption_df=df_labeled_caption,
            top_n=top_n,
            brief_priorities=custom_priorities,
            adaptive_weights=adaptive_weights
        )

        if not recommendations:
            return jsonify({
                "error": "No recommendations found for the given brief",
                "status": "error"
            }), 404

        # Format response
        formatted_recommendations = []
        for i, rec in enumerate(recommendations, 1):
            
            # Generate insights jika diminta
            influencer_insight = None
            if include_insights:
                insight_text = generate_influencer_insight(
                    username=rec['influencer'],
                    caption_df=df_labeled_caption,
                    comment_df=df_labeled_comment,
                    show_plot=False  # API tidak support plotting
                )
                influencer_insight = insight_text

            formatted_rec = {
                "rank": i,
                "username": rec['influencer'],
                "tier": rec['tier'],
                "expertise": rec['expertise'],
                "scores": {
                    "final_score": round(rec['final_score'], 4),
                    "persona_fit": round(rec['persona_fit'], 4),
                    "audience_fit": round(rec['audience_fit'], 4),
                    "performance_pred": round(rec['performance_pred'], 4),
                    "budget_efficiency": round(rec['budget_efficiency'], 4)
                },
                "performance_metrics": {
                    "engagement_rate": round(rec.get('engagement_rate', 0), 4),
                    "authenticity_score": round(rec.get('authenticity_score', 0), 4),
                    "reach_potential": round(rec.get('reach_potential', 0), 4),
                    "brand_fit": round(rec.get('brand_fit', 0), 4)
                },
                "optimal_content_mix": rec.get('optimal_content_mix', {}),
                "insights": influencer_insight if include_insights else None,
                "raw_data": rec.get('raw_influencer_data', {}) if request.args.get('include_raw', 'false').lower() == 'true' else None
            }
            formatted_recommendations.append(formatted_rec)

        # Brief summary
        brief_summary = generate_brief_summary(brief_df, brief_id)

        # Build response
        response_data = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "brief": {
                "brief_id": brief_id,
                "summary": brief_summary,
                "total_requested": top_n,
                "total_found": len(recommendations)
            },
            "recommendations": formatted_recommendations,
            "metadata": {
                "use_adaptive_weights": use_adaptive_weights,
                "adaptive_weights_info": adaptive_info,
                "scoring_strategy": adaptive_weights or custom_priorities,
                "include_insights": include_insights
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in recommend_influencers: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/influencer-insight/<username>', methods=['GET'])
def get_influencer_insight(username):
    """
    Get detailed insight untuk influencer tertentu
    
    Query parameters:
    - include_plots: boolean (default: false) - untuk future implementation
    """
    try:
        if df_labeled_caption is None or df_labeled_comment is None:
            return jsonify({
                "error": "Data not loaded. Please try again later.",
                "status": "error"
            }), 500

        # Generate insight
        insight_text = generate_influencer_insight(
            username=username,
            caption_df=df_labeled_caption,
            comment_df=df_labeled_comment,
            show_plot=False
        )

        if not insight_text or "Tidak ada" in insight_text:
            return jsonify({
                "error": f"No insight data available for @{username}",
                "status": "error"
            }), 404

        return jsonify({
            "status": "success",
            "username": username,
            "insight": insight_text,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in get_influencer_insight: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/adaptive-weights', methods=['POST'])
def calculate_adaptive_weights():
    """
    Calculate adaptive weights berdasarkan brief characteristics
    
    Expected JSON payload (subset of brief):
    {
        "marketing_objective": ["Cognitive", "Affective"],
        "target_goals": ["Awareness", "Brand Perception"],
        "timing_campaign": "2025-02-15",
        "esg_allignment": ["Cruelty-free"],
        "risk_tolerance": "Medium",
        "niche": ["Beauty", "Lifestyle"],
        "location_prior": ["Indonesia"]
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "status": "error"
            }), 400

        brief_data = request.get_json()
        
        adaptive_calculator = SimpleAdaptiveWeightCalculator()
        
        result = adaptive_calculator.calculate_adaptive_weights(
            marketing_objective=brief_data.get('marketing_objective'),
            target_goals=brief_data.get('target_goals'),
            timing_campaign=brief_data.get('timing_campaign'),
            esg_alignment=', '.join(brief_data.get('esg_allignment', [])) if brief_data.get('esg_allignment') else None,
            risk_tolerance=brief_data.get('risk_tolerance'),
            niche=brief_data.get('niche'),
            location_prior=', '.join(brief_data.get('location_prior', [])) if brief_data.get('location_prior') else None
        )

        return jsonify({
            "status": "success",
            "adaptive_weights": result['weights'],
            "applied_adjustments": result['applied_adjustments'],
            "total_adjustments": result['total_adjustments'],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in calculate_adaptive_weights: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/validate-brief', methods=['POST'])
def validate_brief():
    """
    Validate brief JSON structure
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "status": "error"
            }), 400

        brief_data = request.get_json()
        
        is_valid, message = validate_json_brief_input(brief_data)
        
        return jsonify({
            "status": "success" if is_valid else "error",
            "is_valid": is_valid,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in validate_brief: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/data-status', methods=['GET'])
def data_status():
    """
    Check data loading status
    """
    global df_instagram_influencers, df_labeled_caption, df_labeled_comment, df_bio, df_captions
    
    status = {
        "instagram_influencers": df_instagram_influencers is not None,
        "labeled_caption": df_labeled_caption is not None,
        "labeled_comment": df_labeled_comment is not None,
        "bio": df_bio is not None,
        "captions": df_captions is not None
    }
    
    all_loaded = all(status.values())
    
    # Get data shapes if loaded
    shapes = {}
    if all_loaded:
        shapes = {
            "instagram_influencers": df_instagram_influencers.shape,
            "labeled_caption": df_labeled_caption.shape,
            "labeled_comment": df_labeled_comment.shape,
            "bio": df_bio.shape,
            "captions": df_captions.shape
        }
    
    return jsonify({
        "status": "success",
        "all_data_loaded": all_loaded,
        "individual_status": status,
        "data_shapes": shapes,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/reload-data', methods=['POST'])
def reload_data():
    """
    Reload all datasets
    """
    try:
        success = load_data()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "All datasets reloaded successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload datasets",
                "timestamp": datetime.now().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"Error in reload_data: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500

if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', os.getenv('API_PORT', '5000')))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask application on {host}:{port}")
    app.run(debug=debug, host=host, port=port)
