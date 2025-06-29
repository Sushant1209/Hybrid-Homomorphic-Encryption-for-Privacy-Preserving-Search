import random
import json
import os
from datetime import datetime

# Configuration
KEYWORDS = ["java", "python", "react", "angular", "sql", "aws", "azure", "spring", "docker", "cpp"]
NUM_RESUMES = 1000

def prepare_execution_folder():
    """Always use execution01 folder - clean it if it exists"""
    folder_name = "execution01"
    
    if os.path.exists(folder_name):
        print(f"🧹 Cleaning existing {folder_name} folder...")
        # Remove all files in the folder
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"   Deleted: {filename}")
            except Exception as e:
                print(f"   Error deleting {filename}: {e}")
        print(f"✅ Cleaned {folder_name} folder")
    else:
        os.makedirs(folder_name, exist_ok=True)
        print(f"📁 Created {folder_name} folder")
    
    return folder_name

def generate_realistic_metadata():
    """Generate realistic resume metadata"""
    skill_profiles = [
        {"profile": "Full Stack Developer", "skills": ["java", "python", "react", "sql", "spring", "docker"], "weight": 0.25},
        {"profile": "Frontend Developer", "skills": ["react", "angular"], "weight": 0.20},
        {"profile": "Backend Developer", "skills": ["java", "python", "sql", "spring", "docker"], "weight": 0.20},
        {"profile": "Cloud Engineer", "skills": ["aws", "azure", "docker", "python"], "weight": 0.15},
        {"profile": "Data Engineer", "skills": ["python", "sql", "aws", "azure"], "weight": 0.10},
        {"profile": "C++ Developer", "skills": ["cpp", "java", "sql"], "weight": 0.10}
    ]
    
    metadata_scie = []
    metadata_hybrid = []
    metadata_profiles = []
    
    print(f"🚀 Generating {NUM_RESUMES} resumes...")
    
    for i in range(NUM_RESUMES):
        profile_choice = random.choices(skill_profiles, weights=[p["weight"] for p in skill_profiles])[0]
        resume_skills = set(profile_choice["skills"])
        
        # Add random skills
        for kw in KEYWORDS:
            if kw not in resume_skills and random.random() < 0.15:
                resume_skills.add(kw)
        
        # Binary vector for SCIE-Enc
        binary_vector = [1 if kw in resume_skills else 0 for kw in KEYWORDS]
        metadata_scie.append(binary_vector)
        
        # Weighted dict for Hybrid
        weighted_keywords = {}
        for kw in resume_skills:
            if kw in profile_choice["skills"]:
                weighted_keywords[kw] = random.randint(3, 5)
            else:
                weighted_keywords[kw] = random.randint(1, 2)
        
        metadata_hybrid.append(weighted_keywords)
        metadata_profiles.append({
            "resume_id": f"Resume_{i+1}",
            "profile": profile_choice["profile"],
            "skills": list(resume_skills),
            "skill_weights": weighted_keywords
        })
    
    return metadata_scie, metadata_hybrid, metadata_profiles

def save_metadata(metadata_scie, metadata_hybrid, metadata_profiles, folder_name):
    """Save metadata to execution01 folder"""
    
    # SCIE data
    with open(f"{folder_name}/metadata_scie.json", "w") as f:
        json.dump({"keywords": KEYWORDS, "metadata": metadata_scie}, f, indent=2)
    
    # Hybrid data  
    with open(f"{folder_name}/metadata_hybrid.json", "w") as f:
        json.dump({"keywords": KEYWORDS, "metadata": metadata_hybrid}, f, indent=2)
    
    # Profiles
    with open(f"{folder_name}/metadata_profiles.json", "w") as f:
        json.dump(metadata_profiles, f, indent=2)
    
    print(f"✅ Saved metadata to {folder_name}/")

if __name__ == "__main__":
    print("🔍 METADATA GENERATION")
    print("=" * 40)
    
    random.seed(42)
    
    # Always use execution01 folder (clean if exists)
    folder_name = prepare_execution_folder()
    
    # Generate new metadata
    metadata_scie, metadata_hybrid, metadata_profiles = generate_realistic_metadata()
    
    # Save to execution01 folder
    save_metadata(metadata_scie, metadata_hybrid, metadata_profiles, folder_name)
    
    print(f"\n✅ Done! All files saved to {folder_name}/")
    print(f"📁 Ready for: python ultra_optimized_hybrid.py")
    print(f"📊 Then run: python create_graphs.py")