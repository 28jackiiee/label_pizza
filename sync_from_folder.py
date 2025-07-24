import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from huggingface_hub import HfApi, login, create_repo

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
    
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    
    # Find all video files
    video_files = []
    for file_path in video_dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
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

def create_or_update_videos_json(
    videos_json_path: str,
    video_url_mapping: Dict[str, str],
    video_dir: Optional[str] = None,
    overwrite: bool = False
) -> None:
    """
    Create or update videos.json with Hugging Face URLs.
    
    Args:
        videos_json_path: Path to videos.json file
        video_url_mapping: Dictionary mapping filenames to HF URLs
        video_dir: Optional local video directory for metadata
        overwrite: Whether to overwrite existing videos.json
    """
    videos_json_path = Path(videos_json_path)
    
    # Load existing videos.json if it exists and we're not overwriting
    existing_videos = []
    if videos_json_path.exists() and not overwrite:
        try:
            with open(videos_json_path, 'r') as f:
                existing_videos = json.load(f)
            print(f"📄 Loaded existing videos.json with {len(existing_videos)} entries")
        except Exception as e:
            print(f"⚠️  Could not load existing videos.json: {e}")
            existing_videos = []
    
    # Create new video entries
    new_videos = []
    for filename, hf_url in video_url_mapping.items():
        video_uid = filename  # Use filename as UID
        
        # Check if this video already exists in the list
        existing_video = next((v for v in existing_videos if v.get("video_uid") == video_uid), None)
        
        if existing_video:
            # Update the URL of existing video
            existing_video["url"] = hf_url
            existing_video["is_active"] = True
            
            # Update metadata to include HF info
            if "metadata" not in existing_video:
                existing_video["metadata"] = {}
            existing_video["metadata"]["huggingface_url"] = hf_url
            existing_video["metadata"]["uploaded_to_hf"] = True
            
            print(f"🔄 Updated existing video: {video_uid}")
        else:
            # Create new video entry
            video_entry = {
                "video_uid": video_uid,
                "url": hf_url,
                "is_active": True,
                "metadata": {
                    "huggingface_url": hf_url,
                    "uploaded_to_hf": True,
                    "original_filename": filename
                }
            }
            
            # Add file size if video_dir is provided
            if video_dir:
                video_file_path = Path(video_dir) / filename
                if video_file_path.exists():
                    video_entry["metadata"]["file_size"] = video_file_path.stat().st_size
            
            new_videos.append(video_entry)
            print(f"➕ Added new video: {video_uid}")
    
    # Combine existing and new videos
    all_videos = existing_videos + new_videos
    
    # Write updated videos.json
    videos_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(videos_json_path, 'w') as f:
        json.dump(all_videos, f, indent=2)
    
    print(f"💾 Saved videos.json with {len(all_videos)} total videos")

def run_label_pizza_setup(database_url_name, folder_path, video_dir=None, hf_token=None, skip_upload=False):
    """
    Run the complete label pizza setup process with optional Hugging Face upload.
    Only processes files/folders that exist.
    
    Args:
        database_url_name (str): Database URL name
        folder_path (str): Base folder path containing all data files
        video_dir (str, optional): Directory containing local video files to upload
        hf_token (str, optional): Hugging Face API token
        skip_upload (bool): Whether to skip the HF upload step
    """
    # Handle video upload first if specified
    if video_dir and not skip_upload:
        print(f"\n🚀 Starting Hugging Face upload process...")
        try:
            video_url_mapping = upload_videos_to_huggingface(
                video_dir=video_dir,
                token=hf_token
            )
            
            if video_url_mapping:
                # Update or create videos.json
                videos_json_path = os.path.join(folder_path, "videos.json")
                create_or_update_videos_json(
                    videos_json_path=videos_json_path,
                    video_url_mapping=video_url_mapping,
                    video_dir=video_dir
                )
                print(f"✅ Updated videos.json with Hugging Face URLs")
            
        except Exception as e:
            print(f"❌ Error during Hugging Face upload: {e}")
            print("Continuing with existing videos.json...")
    
    # Initialize database (Important to do this before importing utils which uses the database session)
    from label_pizza.db import init_database
    init_database(database_url_name) # This will initialize the database; importantly to do this before importing utils which uses the database session

    # from label_pizza.upload_utils import upload_videos, upload_users, upload_question_groups, upload_schemas, create_projects, bulk_assign_users, batch_upload_annotations, batch_upload_reviews, apply_simple_video_configs

    from label_pizza.sync_utils import sync_videos
    sync_videos(videos_path=os.path.join(folder_path, "videos.json"))
    
    from label_pizza.sync_utils import sync_users
    sync_users(users_path=os.path.join(folder_path, "users.json"))
    
    from label_pizza.sync_utils import sync_question_groups
    sync_question_groups(question_groups_folder=os.path.join(folder_path, "question_groups"))
    
    from label_pizza.sync_utils import sync_schemas
    sync_schemas(schemas_path=os.path.join(folder_path, "schemas.json"))

    from label_pizza.sync_utils import sync_projects
    sync_projects(projects_path=os.path.join(folder_path, "projects.json"))
    
    from label_pizza.sync_utils import sync_project_groups
    sync_project_groups(project_groups_path=os.path.join(folder_path, "project_groups.json"))
    
    from label_pizza.sync_utils import sync_users_to_projects
    sync_users_to_projects(assignment_path=os.path.join(folder_path, "assignments.json"))
    
    from label_pizza.sync_utils import sync_annotations
    sync_annotations(annotations_folder=os.path.join(folder_path, "annotations"), max_workers=8)

    from label_pizza.sync_utils import sync_ground_truths
    sync_ground_truths(ground_truths_folder=os.path.join(folder_path, "ground_truths"), max_workers=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync Label Pizza data from folder, with optional Hugging Face upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sync without upload
  python sync_from_folder.py --folder-path ./workspace

  # Sync with Hugging Face upload
  python sync_from_folder.py --folder-path ./workspace --video-dir /path/to/videos --hf-token your_hf_token

  # Sync with upload using cached HF token
  python sync_from_folder.py --folder-path ./workspace --video-dir /path/to/videos

  # Skip upload but use existing videos.json
  python sync_from_folder.py --folder-path ./workspace --skip-upload

Note: If you don't provide --hf-token, the script will try to use a cached Hugging Face token.
You can set up a token by running: huggingface-cli login
        """
    )
    
    parser.add_argument("--database-url-name", default="DBURL")
    parser.add_argument("--folder-path", default="./workspace", help="Folder path containing data files")
    parser.add_argument("--video-dir", help="Directory containing local video files to upload to Hugging Face")
    parser.add_argument("--hf-token", help="Hugging Face API token (optional if using cached token)")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Hugging Face upload step")
    
    args = parser.parse_args()
    
    run_label_pizza_setup(
        database_url_name=args.database_url_name, 
        folder_path=args.folder_path,
        video_dir=args.video_dir,
        hf_token=args.hf_token,
        skip_upload=args.skip_upload
    )
