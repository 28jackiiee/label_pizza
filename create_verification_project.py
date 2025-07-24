#!/usr/bin/env python3
"""
Script to create a Label Pizza verification project from a directory of videos.

This script will:
1. Scan a directory for video files
2. Upload videos to Hugging Face dataset (jackieyayqli/vqascore/true_positive/folder_name/)
3. Create a verification project with the folder name as project name
4. Pre-label all videos as "Yes" by a user named "labeler"
5. Put the project in a project group called "verification"
6. Use Hugging Face URLs (no local server needed!)

Prerequisites:
    - Label Pizza must be installed with all dependencies
    - Database must be initialized
    - Database URL must be configured in environment variables
    - Hugging Face authentication set up

Usage:
    python create_verification_project.py --video-dir /path/to/videos --database-url-name DBURL [--hf-token TOKEN]

Example:
    python create_verification_project.py --video-dir /Users/jackieli/Downloads/prof_code/data/true_positive/dolly_zoom --database-url-name DBURL
    # This will upload videos to: jackieyayqli/vqascore/true_positive/dolly_zoom/

Setup HF authentication:
    huggingface-cli login
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add the label_pizza directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "label_pizza"))

try:
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("❌ Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

try:
    from label_pizza.db import init_database, SessionLocal
    from label_pizza.sync_utils import (
        sync_videos,
        sync_users,
        sync_question_groups,
        sync_schemas,
        sync_projects,
        sync_project_groups,
        sync_users_to_projects,
        sync_annotations
    )
except ImportError as e:
    print(f"❌ Error importing Label Pizza modules: {e}", file=sys.stderr)
    print("\nPlease ensure that:", file=sys.stderr)
    print("1. You are in the Label Pizza project directory", file=sys.stderr)
    print("2. All dependencies are installed (pip install -e .)", file=sys.stderr)
    print("3. The database is properly configured", file=sys.stderr)
    sys.exit(1)

# Common video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}

def upload_videos_to_huggingface(
    video_dir: str,
    dataset_repo: str = "jackieyayqli/vqascore",
    subfolder: str = "true_positive",
    token: Optional[str] = None
) -> Dict[str, str]:
    """
    Upload video files from a local directory to Hugging Face dataset using upload_folder.
    
    Args:
        video_dir: Local directory containing video files
        dataset_repo: Hugging Face dataset repository (format: username/dataset-name)
        subfolder: Subfolder within the dataset to upload to
        token: Hugging Face API token (if None, will try to use cached token)
        
    Returns:
        Dictionary mapping local filenames to Hugging Face URLs
    """
    if token:
        login(token=token)
    
    api = HfApi()
    video_dir_path = Path(video_dir)
    
    if not video_dir_path.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Find all video files
    video_files = []
    for file_path in video_dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(file_path)
    
    if not video_files:
        print(f"⚠️  No video files found in {video_dir}")
        return {}
    
    print(f"🎬 Found {len(video_files)} video files to upload to {dataset_repo}")
    
    # Try to create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"✅ Dataset repository {dataset_repo} is ready")
    except Exception as e:
        print(f"⚠️  Could not create/verify repository: {e}")
    
    print(f"📁 Uploading to subfolder: {subfolder}/")
    
    try:
        # Upload the entire folder to the specified subfolder
        print("🚀 Uploading folder to Hugging Face (this may take a while for large files)...")
        
        api.upload_folder(
            folder_path=str(video_dir_path),
            repo_id=dataset_repo,
            repo_type="dataset",
            path_in_repo=subfolder,
            allow_patterns=["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm", "*.m4v", "*.3gp", "*.ogv"]
        )
        
        print(f"✅ Successfully uploaded folder to {dataset_repo}/{subfolder}")
        
    except Exception as e:
        print(f"❌ Failed to upload folder: {e}")
        raise
    
    # Generate URL mapping for all video files
    file_url_mapping = {}
    for video_file in video_files:
        path_in_repo = f"{subfolder}/{video_file.name}"
        hf_url = f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{path_in_repo}"
        file_url_mapping[video_file.name] = hf_url
    
    print(f"🎉 Successfully uploaded {len(file_url_mapping)} videos to {dataset_repo}/{subfolder}")
    return file_url_mapping

def get_video_files(directory: Path) -> list:
    """
    Scan directory for video files and return list of video file paths.
    
    Args:
        directory: Path to directory containing videos
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(file_path)
    
    if not video_files:
        raise ValueError(f"No video files found in directory: {directory}")
    
    return sorted(video_files)

def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name to be database-friendly.
    
    Args:
        name: Raw project name
        
    Returns:
        Sanitized project name
    """
    import re
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def get_existing_verification_projects():
    """Get list of existing verification projects."""
    from label_pizza.db import SessionLocal
    if SessionLocal is None:
        return []
    
    with SessionLocal() as session:
        from label_pizza.services import ProjectGroupService
        try:
            # Try to get the verification project group
            group = ProjectGroupService.get_project_group_by_name("verification", session)
            if group:
                projects = ProjectGroupService.get_projects_in_group(group.id, session)
                return [p["name"] for p in projects]
        except:
            pass
    return []

def create_verification_project(video_dir: str, database_url_name: str, hf_token: Optional[str] = None):
    """
    Create a verification project from a directory of videos using Hugging Face URLs.
    
    Args:
        video_dir: Path to directory containing videos
        database_url_name: Name of database URL in environment variables
        hf_token: Optional Hugging Face API token
    """
    video_dir_path = Path(video_dir)
    raw_project_name = video_dir_path.name
    project_name = sanitize_project_name(raw_project_name)
    
    if raw_project_name != project_name:
        print(f"⚠️  Project name sanitized: '{raw_project_name}' → '{project_name}'")
    
    print(f"Creating verification project '{project_name}' from directory: {video_dir}")
    
    # Initialize database connection
    print("Initializing database connection...")
    try:
        init_database(database_url_name)
    except Exception as e:
        raise ValueError(f"Failed to initialize database: {e}")
    
    # Get video files
    print("Scanning for video files...")
    video_files = get_video_files(video_dir_path)
    print(f"Found {len(video_files)} video files")
    
    # Upload videos to Hugging Face
    print(f"\n🚀 Uploading videos to Hugging Face...")
    try:
        video_url_mapping = upload_videos_to_huggingface(
            video_dir=video_dir,
            subfolder=f"true_positive/{raw_project_name}",
            token=hf_token
        )
        
        if not video_url_mapping:
            raise ValueError("No videos were successfully uploaded")
            
    except Exception as e:
        print(f"❌ Error during Hugging Face upload: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Check for existing verification projects
    print("Checking for existing verification projects...")
    try:
        existing_projects = get_existing_verification_projects()
    except Exception as e:
        print(f"❌ Error getting existing projects: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 1. Create videos data with Hugging Face URLs
    print("Creating videos data with Hugging Face URLs...")
    videos_data = []
    for video_file in video_files:
        # Sanitize video UID (filename) as well
        video_uid = sanitize_project_name(video_file.name)
        if video_uid != video_file.name:
            print(f"⚠️  Video UID sanitized: '{video_file.name}' → '{video_uid}'")
        
        # Get Hugging Face URL
        if video_file.name in video_url_mapping:
            video_url = video_url_mapping[video_file.name]
        else:
            print(f"⚠️  Warning: No HF URL found for {video_file.name}, skipping...")
            continue
        
        videos_data.append({
            "video_uid": video_uid,
            "url": video_url,
            "is_active": True,
            "metadata": {
                "original_path": str(video_file),
                "original_filename": video_file.name,
                "file_size": video_file.stat().st_size,
                "verification_project": True,
                "huggingface_url": video_url,
                "uploaded_to_hf": True,
                "served_from": "huggingface"
            }
        })
    
    if not videos_data:
        raise ValueError("No videos available for project creation")
    
    # 2. Create labeler user (if it doesn't exist, sync_users will handle it gracefully)
    print("Creating/updating labeler user...")
    users_data = [{
        "user_id": "labeler",
        "email": "labeler@verification.local",
        "password": "labeler123",
        "user_type": "human",
        "is_active": True
    }]
    
    # 3. Create verification question group
    print("Creating verification question group...")
    question_groups_data = [{
        "title": f"Verification_{project_name}",
        "display_title": f"Verify {project_name}",
        "description": f"Verify that videos are correctly classified as {project_name}",
        "is_reusable": False,
        "is_auto_submit": False,
        "verification_function": None,
        "questions": [{
            "text": f"Is this video correctly classified as {project_name}?",
            "display_text": f"Is this video correctly classified as {project_name}?",
            "qtype": "single",
            "options": ["Yes", "No"],
            "display_values": ["Yes", "No"],
            "option_weights": [1.0, 0.0],
            "default_option": "Yes"
        }]
    }]
    
    # 4. Create verification schema
    print("Creating verification schema...")
    schemas_data = [{
        "schema_name": f"Verification_Schema_{project_name}",
        "question_group_names": [f"Verification_{project_name}"],
        "instructions_url": None,
        "has_custom_display": False,
        "is_active": True
    }]
    
    # 5. Create verification project
    print("Creating verification project...")
    projects_data = [{
        "project_name": project_name,
        "description": f"Verification project for {project_name} videos",
        "schema_name": f"Verification_Schema_{project_name}",
        "videos": [v["video_uid"] for v in videos_data],
        "is_active": True
    }]
    
    # 6. Create/update project group
    print("Creating/updating project group...")
    project_groups_data = [{
        "project_group_name": "verification",
        "description": "Projects for verifying video classifications",
        "projects": existing_projects + [project_name]
    }]
    
    # 7. Create assignment
    print("Creating project assignment...")
    assignments_data = [{
        "user_name": "labeler",
        "project_name": project_name,
        "role": "annotator",
        "user_weight": 1.0,
        "is_active": True
    }]
    
    # 8. Create annotations (pre-label as "Yes")
    print("Creating pre-filled annotations...")
    annotations_data = []
    for video_data in videos_data:
        annotations_data.append({
            "project_name": project_name,
            "video_uid": video_data["video_uid"],
            "user_name": "labeler",
            "question_group_title": f"Verification_{project_name}",
            "answers": {
                f"Is this video correctly classified as {project_name}?": "Yes"
            },
            "is_ground_truth": False
        })
    
    # Sync everything to database
    print("\n📊 Syncing data to Label Pizza database...")
    
    try:
        sync_videos(videos_data=videos_data)
        sync_users(users_data=users_data)
        sync_question_groups(question_groups_data=question_groups_data)
        sync_schemas(schemas_data=schemas_data)
        sync_projects(projects_data=projects_data)
        sync_project_groups(project_groups_data=project_groups_data)
        sync_users_to_projects(assignments_data=assignments_data)
        sync_annotations(annotations_data=annotations_data)
        
        print(f"\n🎉 Verification project '{project_name}' created successfully!")
        print(f"📊 Created project with {len(videos_data)} videos")
        print(f"✅ All videos pre-labeled as 'Yes' by labeler")
        print(f"🗂️  Added to 'verification' project group")
        print(f"🌐 Videos uploaded to: jackieyayqli/vqascore/true_positive/{raw_project_name}/")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Launch Label Pizza: streamlit run label_pizza/label_pizza_app.py -- --database-url-name {database_url_name}")
        print(f"2. Navigate to Review → verification → {project_name}")
        print(f"3. Review the pre-labeled videos")
        
    except Exception as e:
        print(f"❌ Error syncing data to database: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Create Label Pizza verification project with Hugging Face upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_verification_project.py --video-dir /path/to/videos --database-url-name DBURL
  python create_verification_project.py --video-dir /Users/jackieli/Downloads/prof_code/data/true_positive/dolly_zoom --database-url-name DBURL --hf-token YOUR_TOKEN

What happens:
  - Videos from /path/to/videos/ get uploaded to jackieyayqli/vqascore/true_positive/videos/
  - Videos from /path/to/dolly_zoom/ get uploaded to jackieyayqli/vqascore/true_positive/dolly_zoom/

Setup:
  1. Install dependencies: pip install -e .
  2. Set up HF auth: huggingface-cli login
  3. Initialize database: python label_pizza/init_or_reset_db.py --mode init --database-url-name DBURL --email admin@example.com --password admin123 --user-id "Admin"

Benefits over old version:
  - No local server needed (uses Hugging Face)
  - Videos persist even when machine is off
  - Better sharing and collaboration
  - Automatic cloud backup
  - Organized by folder name in HF dataset

Video formats supported: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv
        """
    )
    
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Path to directory containing video files"
    )
    
    parser.add_argument(
        "--database-url-name",
        required=True,
        help="Name of the database URL environment variable (e.g., DBURL)"
    )
    
    parser.add_argument(
        "--hf-token",
        help="Hugging Face API token (optional if using cached token from huggingface-cli login)"
    )
    
    args = parser.parse_args()
    
    try:
        create_verification_project(args.video_dir, args.database_url_name, args.hf_token)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 