import requests
import json

# Test script untuk Influencer Recommendation API

API_BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    print("-" * 50)

def test_data_status():
    """Test data status endpoint"""
    print("🔍 Testing Data Status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/data-status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Data status check passed")
            print(f"All data loaded: {data['all_data_loaded']}")
            if data['data_shapes']:
                print("📊 Data shapes:")
                for dataset, shape in data['data_shapes'].items():
                    print(f"   {dataset}: {shape}")
        else:
            print(f"❌ Data status check failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Data status error: {e}")
    print("-" * 50)

def test_adaptive_weights():
    """Test adaptive weights calculation"""
    print("🔍 Testing Adaptive Weights...")
    
    sample_brief = {
        "marketing_objective": ["Cognitive", "Affective"],
        "target_goals": ["Awareness", "Brand Perception"],
        "timing_campaign": "2025-02-15",
        "esg_allignment": ["Cruelty-free"],
        "risk_tolerance": "Medium",
        "niche": ["Beauty", "Lifestyle"],
        "location_prior": ["Indonesia"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/adaptive-weights",
            json=sample_brief
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Adaptive weights calculation passed")
            print("🎛️ Adaptive weights:")
            for component, weight in data['adaptive_weights'].items():
                print(f"   {component}: {weight:.2%}")
            print(f"📝 Applied {data['total_adjustments']} adjustments")
        else:
            print(f"❌ Adaptive weights failed with status {response.status_code}")
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Adaptive weights error: {e}")
    print("-" * 50)

def test_brief_validation():
    """Test brief validation"""
    print("🔍 Testing Brief Validation...")
    
    # Test valid brief
    valid_brief = {
        "brief_id": "TEST_001",
        "brand_name": "Test Brand",
        "industry": "Beauty",
        "product_name": "Test Product",
        "overview": "Test overview",
        "influencer_persona": "Test persona",
        "total_influencer": 3,
        "budget": 50000000.0
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/validate-brief",
            json=valid_brief
        )
        if response.status_code == 200:
            data = response.json()
            if data['is_valid']:
                print("✅ Valid brief validation passed")
            else:
                print(f"❌ Brief validation failed: {data['message']}")
        else:
            print(f"❌ Brief validation failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Brief validation error: {e}")
    
    # Test invalid brief
    invalid_brief = {
        "brief_id": "TEST_002"
        # Missing required fields
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/validate-brief",
            json=invalid_brief
        )
        if response.status_code == 200:
            data = response.json()
            if not data['is_valid']:
                print("✅ Invalid brief validation passed")
                print(f"Expected error: {data['message']}")
            else:
                print("❌ Should have detected invalid brief")
    except Exception as e:
        print(f"❌ Invalid brief validation error: {e}")
    
    print("-" * 50)

def test_recommend_influencers():
    """Test main recommendation endpoint"""
    print("🔍 Testing Influencer Recommendations...")
    
    sample_brief = {
        "brief_id": "BRIEF_TEST_001",
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
        "influencer_persona": "Beauty enthusiast, skincare expert, authentic product reviewer dengan focus pada natural skincare dan anti-aging",
        "total_influencer": 2,
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
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/recommend-influencers",
            json=sample_brief,
            params={
                "adaptive_weights": "true",
                "include_insights": "true",
                "include_raw": "false"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Influencer recommendations passed")
            print(f"📊 Brief: {data['brief']['brief_id']}")
            print(f"🎯 Found {data['brief']['total_found']} recommendations")
            
            print("\n🏆 Top Recommendations:")
            for rec in data['recommendations'][:2]:  # Show first 2
                print(f"   Rank {rec['rank']}: @{rec['username']} ({rec['tier']})")
                print(f"   Final Score: {rec['scores']['final_score']:.2%}")
                print(f"   Content Mix: Stories={rec['optimal_content_mix']['story_count']}, Feeds={rec['optimal_content_mix']['feeds_count']}, Reels={rec['optimal_content_mix']['reels_count']}")
                print()
            
            print("🧠 Adaptive Weights Info:")
            if data['metadata']['adaptive_weights_info']:
                weights_info = data['metadata']['adaptive_weights_info']
                print(f"   Applied {weights_info['total_adjustments']} adjustments")
                for component, weight in weights_info['final_weights'].items():
                    print(f"   {component}: {weight:.2%}")
        else:
            print(f"❌ Recommendations failed with status {response.status_code}")
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"❌ Recommendations error: {e}")
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("🧪 INFLUENCER RECOMMENDATION API TESTING")
    print("=" * 60)
    
    test_health_check()
    test_data_status()
    test_brief_validation()
    test_adaptive_weights()
    test_recommend_influencers()
    
    print("✨ Testing completed!")
    print("\n💡 Tips:")
    print("- Make sure the API is running on http://localhost:5000")
    print("- Wait for data to be loaded before running recommendations")
    print("- Check the console output of the Flask app for detailed logs")

if __name__ == "__main__":
    main()
