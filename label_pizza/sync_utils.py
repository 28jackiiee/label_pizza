import json
from sqlalchemy.orm import Session
from functools import lru_cache
from tqdm import tqdm
from label_pizza.services import (
    VideoService, 
    ProjectService, 
    SchemaService, 
    QuestionGroupService, 
    QuestionService,
    AuthService,
    AnnotatorService,
    GroundTruthService,
    CustomDisplayService,
    ProjectGroupService
)
import label_pizza.db
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
import pandas as pd
import os
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import glob
from copy import deepcopy

# --------------------------------------------------------------------------- #
# Core operations                                                             #
# --------------------------------------------------------------------------- #

def _process_video_add(video_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Process and verify a single video addition in a thread-safe manner.
    
    Args:
        video_data: Dictionary containing video_uid, url, and optional metadata
        
    Returns:
        Tuple of (video_uid, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            VideoService.verify_add_video(
                video_uid=video_data["video_uid"],
                url=video_data["url"],
                metadata=video_data.get("metadata"),
                session=sess,
            )
            return video_data["video_uid"], True, None
        except ValueError as err:
            if "already exists" in str(err):
                return video_data["video_uid"], False, "already exists"
            else:
                return video_data["video_uid"], False, str(err)

def _add_single_video(video_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Add a single video in a thread-safe manner.
    
    Args:
        video_data: Dictionary containing video_uid, url, and optional metadata
        
    Returns:
        Tuple of (video_uid, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            VideoService.add_video(
                video_uid=video_data["video_uid"],
                url=video_data["url"],
                metadata=video_data.get("metadata"),
                session=sess,
            )
            return video_data["video_uid"], True, None
        except Exception as e:
            return video_data["video_uid"], False, str(e)

def add_videos(videos_data: List[Dict], max_workers: int = 10) -> None:
    """Insert videos that are not yet in database with parallel verification.
    
    Args:
        videos_data: List of video dictionaries with video_uid, url, metadata
        max_workers: Number of parallel worker threads (default: 10)
        
    Raises:
        TypeError: If videos_data is not a list of dictionaries
        ValueError: If videos already exist or verification fails
    """
    if not isinstance(videos_data, list):
        raise TypeError("videos_data must be a list[dict]")

    # Verify all videos with ThreadPoolExecutor
    duplicates = []
    errors = []
    
    with tqdm(total=len(videos_data), desc="Verifying videos for addition", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_video_add, v): v for v in videos_data}
            
            for future in concurrent.futures.as_completed(futures):
                video_uid, success, error_msg = future.result()
                if not success:
                    if error_msg == "already exists":
                        duplicates.append(video_uid)
                    else:
                        errors.append(f"{video_uid}: {error_msg}")
                pbar.update(1)

    if duplicates:
        raise ValueError("Add aborted – already in DB: " + ", ".join(duplicates))
    
    if errors:
        raise ValueError("Add aborted – verification errors: " + "; ".join(errors))

    # Add videos with ThreadPoolExecutor
    with tqdm(total=len(videos_data), desc="Adding videos", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_add_single_video, v): v for v in videos_data}
            
            for future in concurrent.futures.as_completed(futures):
                video_uid, success, error_msg = future.result()
                if not success:
                    raise ValueError(f"Failed to add video {video_uid}: {error_msg}")
                pbar.set_postfix(uid=video_uid[:20] + "..." if len(video_uid) > 20 else video_uid)
                pbar.update(1)
                
    print(f"✔ Added {len(videos_data)} new video(s)")


def _process_video_update(video_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Process and verify a single video update in a thread-safe manner.
    
    Args:
        video_data: Dictionary containing video_uid, url, and optional metadata
        
    Returns:
        Tuple of (video_uid, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            VideoService.verify_update_video(
                video_uid=video_data["video_uid"],
                new_url=video_data["url"],
                new_metadata=video_data.get("metadata"),
                session=sess,
            )
            return video_data["video_uid"], True, None
        except ValueError as err:
            if "not found" in str(err):
                return video_data["video_uid"], False, "not found"
            else:
                return video_data["video_uid"], False, str(err)

def _update_single_video(video_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Update a single video in a thread-safe manner with change detection.
    
    Args:
        video_data: Dictionary containing video_uid, url, metadata, optional is_archived
        
    Returns:
        Tuple of (video_uid, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            # Get existing video info
            existing_video = VideoService.get_video_by_uid(video_data["video_uid"], sess)
            if not existing_video:
                return video_data["video_uid"], False, "Video not found"
            
            # Check if any information has changed
            needs_update = False
            
            # Check URL
            if video_data["url"] != existing_video.url:
                needs_update = True
            
            # Check metadata
            new_metadata = video_data.get("metadata", {})
            existing_metadata = existing_video.video_metadata or {}
            if new_metadata != existing_metadata:
                needs_update = True
            
            # Check archive status
            if "is_archived" in video_data:
                if video_data["is_archived"] != existing_video.is_archived:
                    needs_update = True
            
            # If no changes needed, skip update
            if not needs_update:
                return video_data["video_uid"], True, "No changes needed"
            
            # Perform the update
            VideoService.update_video(
                video_uid=video_data["video_uid"],
                new_url=video_data["url"],
                new_metadata=video_data.get("metadata"),
                session=sess,
            )
            
            rec = VideoService.get_video_by_uid(video_data["video_uid"], sess)
            if rec and video_data["is_archived"] != rec.is_archived:
                if video_data["is_archived"]:
                    VideoService.archive_video(rec.id, sess)
                else:
                    VideoService.unarchive_video(rec.id, sess)
            
            return video_data["video_uid"], True, None
        except Exception as e:
            return video_data["video_uid"], False, str(e)


def update_videos(videos_data: List[Dict], max_workers: int = 10) -> None:
    """Update videos that must exist in database with parallel verification.
    
    Args:
        videos_data: List of video dictionaries with video_uid, url, metadata
        max_workers: Number of parallel worker threads (default: 10)
        
    Raises:
        TypeError: If videos_data is not a list of dictionaries
        ValueError: If videos not found or verification fails
    """
    if not isinstance(videos_data, list):
        raise TypeError("videos_data must be a list[dict]")

    # Verify all videos with ThreadPoolExecutor
    missing = []
    errors = []
    
    with tqdm(total=len(videos_data), desc="Verifying videos for update", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_video_update, v): v for v in videos_data}
            
            for future in concurrent.futures.as_completed(futures):
                video_uid, success, error_msg = future.result()
                if not success:
                    if error_msg == "not found":
                        missing.append(video_uid)
                    else:
                        errors.append(f"{video_uid}: {error_msg}")
                pbar.update(1)

    if missing:
        raise ValueError("Update aborted – not found in DB: " + ", ".join(missing))
    
    if errors:
        raise ValueError("Update aborted – verification errors: " + "; ".join(errors))

    # Update videos with ThreadPoolExecutor
    updated_count = 0
    skipped_count = 0
    
    with tqdm(total=len(videos_data), desc="Updating videos", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_update_single_video, v): v for v in videos_data}
            
            for future in concurrent.futures.as_completed(futures):
                video_uid, success, error_msg = future.result()
                if not success:
                    raise ValueError(f"Failed to update video {video_uid}: {error_msg}")
                
                if error_msg == "No changes needed":
                    skipped_count += 1
                else:
                    updated_count += 1
                
                pbar.set_postfix(uid=video_uid[:20] + "..." if len(video_uid) > 20 else video_uid)
                pbar.update(1)

    print(f"✔ Updated {updated_count} video(s), skipped {skipped_count} video(s) (no changes)")

# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def sync_videos(
    *, videos_path: str | Path | None = None, videos_data: List[Dict] | None = None
) -> None:
    """Load, validate, and route videos to add/update pipelines automatically.
    
    Args:
        videos_path: Path to JSON file containing video list
        videos_data: Pre-loaded list of video dictionaries
        
    Raises:
        ValueError: If neither or both parameters provided, or validation fails
        TypeError: If videos_data is not a list of dictionaries
        
    Note:
        Exactly one of videos_path or videos_data must be provided.
        Each video dict requires: url, video_uid, metadata, is_active.
    """

    if videos_path is None and videos_data is None:
        raise ValueError("Provide either videos_path or videos_data")

    if videos_path and videos_data:
        raise ValueError("Provide either videos_path or videos_data, not both")

    # Load JSON if a path is provided
    if videos_path:
        print(f"📂 Loading videos from {videos_path}")
        with open(videos_path, "r") as f:
            videos_data = json.load(f)

    if not isinstance(videos_data, list):
        raise TypeError("videos_data must be a list[dict]")

    # Deep copy videos_data to avoid modifying the original list
    videos_data = deepcopy(videos_data)
    
    print(f"\n🚀 Starting video sync pipeline with {len(videos_data)} videos...")
    
    # Validate & enrich each record with progress bar
    processed: List[Dict] = []
    with tqdm(total=len(videos_data), desc="Validating video data", unit="video") as pbar:
        for idx, item in enumerate(videos_data, 1):
            required = {"url", "video_uid", "metadata", "is_active"}
            item_keys = set(item.keys())

            if item_keys != required:
                missing = required - item_keys
                extra = item_keys - required
                
                error_parts = []
                if missing:
                    error_parts.append(f"missing: {', '.join(missing)}")
                if extra:
                    error_parts.append(f"extra: {', '.join(extra)}")
                
                raise ValueError(f"Entry #{idx} invalid fields: {', '.join(error_parts)}")

            item["is_archived"] = not item.pop("is_active")

            processed.append(item)
            pbar.update(1)
    
    # Check for duplicate video_uid values first
    print("\n🔍 Checking for duplicate video_uid and urls values...")
    urls = []
    video_uids = []
    uid_duplicates = []
    url_duplicates = []
    
    for idx, item in enumerate(processed, 1):
        video_uid = item["video_uid"]
        url = item["url"]
        if video_uid in video_uids:
            uid_duplicates.append((video_uid, idx))
        else:
            video_uids.append(video_uid)
        if url in urls:
            url_duplicates.append((url, idx))
        else:
            urls.append(url)
    
    if uid_duplicates:
        duplicate_info = [f"video_uid '{uid}' at entry #{idx}" for uid, idx in uid_duplicates]
        raise ValueError(f"Duplicate video_uid values found: {', '.join(duplicate_info)}")
    if url_duplicates:
        duplicate_info = [f"url '{url}' at entry #{idx}" for url, idx in url_duplicates]
        raise ValueError(f"Duplicate url values found: {', '.join(duplicate_info)}")
    
    print(f"✅ No duplicates found - all {len(video_uids)} video_uid values and {len(urls)} url values are unique")

    # Decide add vs update with a single read-only look‑up
    print("\n📊 Categorizing videos...")
    
    def _check_video_exists(video_data: Dict) -> Tuple[str, bool]:
        """Check if a video exists in a thread-safe manner."""
        with label_pizza.db.SessionLocal() as sess:
            try:
                existing = VideoService.get_video_by_uid(video_data["video_uid"], sess)
                return video_data["video_uid"], existing is not None
            except Exception as e:
                # If there's an error checking, assume it doesn't exist
                return video_data["video_uid"], False
    
    to_add, to_update = [], []
    with tqdm(total=len(processed), desc="Checking existing videos", unit="video") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_check_video_exists, v): v for v in processed}
            
            for future in concurrent.futures.as_completed(futures):
                video_uid, exists = future.result()
                video_data = futures[future]
                if exists:
                    to_update.append(video_data)
                else:
                    to_add.append(video_data)
                pbar.update(1)
    
    print(f"\n📈 Summary: {len(to_add)} videos to add, {len(to_update)} videos to update")
    
    if to_add:
        print(f"\n➕ Adding {len(to_add)} new videos...")
        add_videos(to_add)
        
    if to_update:
        print(f"\n🔄 Updating {len(to_update)} existing videos...")
        update_videos(to_update)
        
    print("\n🎉 Video pipeline complete!")


# --------------------------------------------------------------------------- #
# Core operations                                                             #
# --------------------------------------------------------------------------- #

def add_users(users_data: List[Dict]) -> None:
    """Insert users that are not yet in database with verification.
    
    Args:
        users_data: List of user dictionaries with user_id, email, password, user_type
        
    Raises:
        TypeError: If users_data is not a list of dictionaries
        ValueError: If users already exist or verification fails
    """
    if not isinstance(users_data, list):
        raise TypeError("users_data must be a list[dict]")

    with label_pizza.db.SessionLocal() as sess:
        duplicates = []
        for u in users_data:
            try:
                AuthService.verify_create_user(
                    user_id=u.get("user_id"),
                    email=u.get("email"),
                    password_hash=u.get("password"),
                    user_type=u.get("user_type", "human"),
                    session=sess,
                )
            except ValueError as err:
                if "already exists" in str(err):
                    duplicates.append(u.get("user_id") or u.get("email"))
                else:
                    raise

        if duplicates:
            raise ValueError("Add aborted – already in DB: " + ", ".join(duplicates))

        for u in users_data:
            AuthService.create_user(
                user_id=u.get("user_id"),
                email=u.get("email"),
                password_hash=u.get("password"),
                user_type=u.get("user_type", "human"),
                is_archived=u.get("is_archived", False),
                session=sess,
            )
        sess.commit()
        print(f"✔ Added {len(users_data)} new user(s)")


def update_users(users_data: List[Dict]) -> None:
    """Update users that must exist in database with change detection.
    
    Args:
        users_data: List of user dictionaries with user_id/email and optional updates
        
    Raises:
        TypeError: If users_data is not a list of dictionaries
        ValueError: If users not found
        RuntimeError: If update operation fails
    """
    if not isinstance(users_data, list):
        raise TypeError("users_data must be a list[dict]")

    # Process users in single session to avoid connection exhaustion
    validated_entries = []
    skipped_entries = []
    
    print("🔍 Validating and updating users...")
    with label_pizza.db.SessionLocal() as session:
        try:
            # Validation phase with progress bar
            for idx, user in enumerate(tqdm(users_data, desc="Validating", unit="users"), 1):
                try:
                    # Get existing user
                    user_rec = None
                    if user.get("user_id"):
                        try:
                            user_rec = AuthService.get_user_by_id(user["user_id"], session)
                        except ValueError:
                            pass
                    
                    if not user_rec and user.get("email"):
                        try:
                            user_rec = AuthService.get_user_by_email(user["email"], session)
                        except ValueError:
                            pass
                    
                    if not user_rec:
                        raise ValueError(f"User not found: {user.get('user_id') or user.get('email')}")
                    
                    # Check if any information has changed
                    needs_update = False
                    changes = []
                    
                    # Check email
                    if user["email"] != user_rec.email:
                        AuthService.verify_update_user_email(user_rec.id, user["email"], session)
                        needs_update = True
                        changes.append("email")
                    
                    # Check password (we can't compare hashes, so we'll update if provided)
                    if user["password"] != user_rec.password_hash:
                        
                        AuthService.verify_update_user_password(user_rec.id, user["password"], session)
                        
                        needs_update = True
                        changes.append("password")
                    
                    # Check user_type
                    if user["user_type"] != user_rec.user_type:
                        
                        AuthService.verify_update_user_role(user_rec.id, user["user_type"], session)
                        
                        needs_update = True
                        changes.append("user_type")
                    
                    # Check user_id
                    if user["user_id"] != user_rec.user_id_str:
                        
                        AuthService.verify_update_user_id(user_rec.id, user["user_id"], session)
                        
                        needs_update = True
                        changes.append("user_id")
                    
                    # Check archive status
                    if user["is_archived"] != user_rec.is_archived:
                        needs_update = True
                        changes.append("archive_status")
                    
                    if not needs_update:
                        skipped_entries.append({
                            "user_id": user.get("user_id"),
                            "email": user.get("email")
                        })
                    else:
                        validated_entries.append({
                            "user_rec": user_rec,
                            "user_data": user,
                            "changes": changes
                        })
                        
                except Exception as e:
                    raise ValueError(f"[Row {idx}] {user.get('user_id') or user.get('email')}: {e}")
            
            print(f"✅ Validation passed: {len(validated_entries)} to update, {len(skipped_entries)} skipped")
            
            # Update validated entries in same session with progress bar
            if validated_entries:
                print("📤 Updating users...")
                for entry in tqdm(validated_entries, desc="Updating", unit="users"):
                    user_rec = entry["user_rec"]
                    user_data = entry["user_data"]
                    changes = entry["changes"]
                    
                    # Apply only the changes that are needed
                    if "email" in changes:
                        AuthService.update_user_email(user_rec.id, user_data["email"], session)
                    
                    if "password" in changes:
                        AuthService.update_user_password(user_rec.id, user_data["password"], session)
                    
                    if "user_type" in changes:
                        AuthService.update_user_role(user_rec.id, user_data["user_type"], session)
                    
                    if "user_id" in changes:
                        AuthService.update_user_id(user_rec.id, user_data["user_id"], session)
                    
                    if "archive_status" in changes:
                        if user_data["is_archived"] != user_rec.is_archived:
                            AuthService.toggle_user_archived(user_rec.id, session)
                
                session.commit()
                print(f"🎉 Successfully updated {len(validated_entries)} users!")
                
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Update failed: {e}")

# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def sync_users(
    *, users_path: str | Path | None = None, users_data: List[Dict] | None = None
) -> None:
    """Load, validate, and route users to add/update pipelines automatically.
    
    Args:
        users_path: Path to JSON file containing user list
        users_data: Pre-loaded list of user dictionaries
        
    Raises:
        ValueError: If neither or both parameters provided, or validation fails
        TypeError: If users_data is not a list of dictionaries
        
    Note:
        Exactly one of users_path or users_data must be provided.
        Each user dict requires: user_id, email, password, user_type, is_active.
    """

    if users_path is None and users_data is None:
        raise ValueError("Provide either users_path or users_data")

    if users_path and users_data:
        raise ValueError("Provide either users_path or users_data, not both")

    if users_path:
        with open(users_path, "r") as f:
            users_data = json.load(f)

    if not isinstance(users_data, list):
        raise TypeError("users_data must be a list[dict]")

    # Deep copy users_data to avoid modifying the original list
    users_data = deepcopy(users_data)

    print(f"🚀 Starting user sync with {len(users_data)} users...")

    # Validate, clean, and check duplicates
    user_ids, emails = set(), set()
    duplicates = []
    
    for idx, user in enumerate(users_data, 1):
        # Check required fields
        required = {"user_id", "email", "password", "user_type", "is_active"}
        user_keys = set(user.keys())

        if user_keys != required:
            missing = required - user_keys
            extra = user_keys - required
            
            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")
            
            raise ValueError(f"Entry #{idx} {', '.join(error_parts)}")
        
        # Validate user_id
        user_id = str(user["user_id"]).strip() if user["user_id"] else ""
        if not user_id:
            raise ValueError(f"Entry #{idx}: user_id cannot be empty")
        user["user_id"] = user_id
        
        # Check user_id duplicates
        if user_id in user_ids:
            duplicates.append(f"user_id '{user_id}' at #{idx}")
        user_ids.add(user_id)
        
        # Validate user_type
        if user["user_type"] not in {"admin", "human", "model"}:
            raise ValueError(f"Entry #{idx}: invalid user_type: {user['user_type']}")
        
        is_model = user["user_type"] == "model"
        
        # Process email
        email = user["email"]
        if is_model:
            if email is not None:
                raise ValueError(f"Entry #{idx}: email must be None for model user")
        else:
            if email is None:
                raise ValueError(f"Entry #{idx}: email cannot be None for {user['user_type']} user")
            email = str(email).strip().lower()
            if not email or "@" not in email:
                raise ValueError(f"Entry #{idx}: invalid email")
        user["email"] = email
        
        # Check email duplicates (only non-None)
        if email is not None:
            if email in emails:
                duplicates.append(f"email '{email}' at #{idx}")
            emails.add(email)
        
        # Process password
        password = user["password"]
        if is_model:
            # Model users: empty string or None gets converted to placeholder
            password = "" if password is None or str(password).strip() == "" else str(password).strip()
        else:
            if password is None or not str(password).strip():
                raise ValueError(f"Entry #{idx}: password cannot be empty for {user['user_type']} user")
            password = str(password).strip()
        user["password"] = password
        
        # Convert is_active → is_archived
        user["is_archived"] = not user.pop("is_active")
    
    if duplicates:
        raise ValueError(f"Duplicates found: {'; '.join(duplicates)}")
    
    print(f"✅ Validated {len(user_ids)} users, {len(emails)} unique emails")

    # Categorize add vs update
    to_add, to_update = [], []
    with label_pizza.db.SessionLocal() as sess:
        for user in users_data:
            user_exists = False
            
            # Check if user exists by user_id first
            if user.get("user_id"):
                try:
                    existing_user = AuthService.get_user_by_id(user["user_id"], sess)
                    if existing_user:
                        user_exists = True
                except (ValueError, Exception) as e:
                    # If error contains "not found", user doesn't exist
                    if "not found" in str(e).lower():
                        user_exists = False
                    else:
                        # Re-raise unexpected errors
                        raise
            
            # If not found by user_id, check by email
            if not user_exists and user.get("email"):
                try:
                    existing_user = AuthService.get_user_by_email(user["email"], sess)
                    if existing_user:
                        user_exists = True
                except (ValueError, Exception) as e:
                    # If error contains "not found", user doesn't exist
                    if "not found" in str(e).lower():
                        user_exists = False
                    else:
                        # Re-raise unexpected errors
                        raise
            
            (to_update if user_exists else to_add).append(user)

    print(f"📊 {len(to_add)} to add, {len(to_update)} to update")
    
    # Execute
    if to_add:
        add_users(to_add)
    if to_update:
        update_users(to_update)
    
    print("🎉 Complete")


# --------------------------------------------------------------------------- #
# Core operations                                                             #
# --------------------------------------------------------------------------- #

def add_question_groups(groups: List[Tuple[str, Dict]], question_text_to_id: Dict[str, int]) -> List[Dict]:
    """Create new question groups with full verification and atomic transaction.
    
    Args:
        groups: List of (filename, group_dict) tuples with question group data
        question_text_to_id: Mapping from question text to question ID (all questions should already exist)
        
    Returns:
        List of created group information
        
    Raises:
        TypeError: If groups is not a list of tuples
        ValueError: If groups already exist or verification fails
        
    Note:
        All questions must already exist in the database. This function only handles group creation.
    """
    if not isinstance(groups, list):
        raise TypeError("groups must be list[(filename, dict)]")

    if not isinstance(question_text_to_id, dict):
        raise TypeError("question_text_to_id must be a dictionary")

    created: List[Dict] = []

    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: duplicate title check (cheap, read‑only) ───────────────
        dup_titles = []
        for _, g in groups:
            try:
                QuestionGroupService.get_group_by_name(g["title"], sess)
                dup_titles.append(g["title"])
            except ValueError as err:
                # Only ignore "not found" errors, re-raise others
                if "not found" not in str(err).lower():
                    raise
                # Group doesn't exist, which is what we want for adding
        
        if dup_titles:
            raise ValueError("Add aborted – already in DB: " + ", ".join(dup_titles))

        # ── Phase 1: Process each group ──────────────────────────────────────
        for _, group_data in groups:
            # Build question IDs list for this group
            question_ids = []
            for question_data in group_data.get("questions", []):
                question_text = question_data["text"]
                if question_text not in question_text_to_id:
                    raise ValueError(f"Question '{question_text}' not found in question mapping - questions must be processed first")
                question_ids.append(question_text_to_id[question_text])
            
            # Check for auto-submit validation
            is_auto_submit = group_data.get("is_auto_submit", False)
            if is_auto_submit:
                for question_data in group_data.get("questions", []):
                    if question_data.get("default_option") is None:
                        raise ValueError(
                            f"Cannot set default_option to None for question '{question_data['text']}' "
                            f"in auto-submit group '{group_data['title']}'. Auto-submit groups require non-None default values."
                        )
            
            # Verify group
            QuestionGroupService.verify_create_group(
                title=group_data["title"],
                display_title=group_data.get("display_title", group_data["title"]),
                description=group_data["description"],
                is_reusable=group_data.get("is_reusable", True),
                question_ids=question_ids,
                verification_function=group_data.get("verification_function"),
                is_auto_submit=group_data.get("is_auto_submit", False),
                session=sess,
            )
            
            # Create group
            grp = QuestionGroupService.create_group(
                title=group_data["title"],
                display_title=group_data.get("display_title", group_data["title"]),
                description=group_data["description"],
                is_reusable=group_data.get("is_reusable", True),
                question_ids=question_ids,
                verification_function=group_data.get("verification_function"),
                is_auto_submit=group_data.get("is_auto_submit", False),
                session=sess,
            )
            
            created.append({"title": group_data["title"], "id": grp.id})

        sess.commit()
    
    return created


def update_question_groups(groups: List[Tuple[str, Dict]], question_text_to_id: Dict[str, int]) -> List[Dict]:
    """Update existing question groups with full verification and atomic transaction.
    
    Args:
        groups: List of (filename, group_dict) tuples with question group data
        question_text_to_id: Mapping from question text to question ID (all questions should already exist)
        
    Returns:
        List of updated group information with changes made
        
    Raises:
        TypeError: If groups is not a list of tuples
        ValueError: If groups not found or verification fails
        
    Note:
        All questions must already exist in the database. This function only handles group updates.
        The question set for each group must remain the same (same questions, possibly different order).
    """
    if not isinstance(groups, list):
        raise TypeError("groups must be list[(filename, dict)]")

    if not isinstance(question_text_to_id, dict):
        raise TypeError("question_text_to_id must be a dictionary")

    updated: List[Dict] = []
    skipped: List[Dict] = []
    
    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: existence check (cheap, read‑only) ────────────────────
        missing = []
        for _, g in groups:
            try:
                QuestionGroupService.get_group_by_name(g["title"], sess)
            except ValueError as err:
                # Only treat "not found" as missing, re-raise other errors
                if "not found" not in str(err).lower():
                    raise
                # Group doesn't exist
                missing.append(g["title"])
        
        if missing:
            raise ValueError("Update aborted – not found in DB: " + ", ".join(missing))

        # ── Phase 1: Process each group ──────────────────────────────────────
        for _, group_data in groups:
            grp = QuestionGroupService.get_group_by_name(group_data["title"], sess)
            
            # Build question IDs list for this group
            question_ids = []
            for question_data in group_data.get("questions", []):
                question_text = question_data["text"]
                if question_text not in question_text_to_id:
                    raise ValueError(f"Question '{question_text}' not found in question mapping - questions must be processed first")
                question_ids.append(question_text_to_id[question_text])
            
            # Check for duplicates in question list
            if len(question_ids) != len(set(question_ids)):
                question_texts = [q["text"] for q in group_data.get("questions", [])]
                from collections import Counter
                question_counter = Counter(question_texts)
                duplicates = [text for text, count in question_counter.items() if count > 1]
                raise ValueError(f"Group '{group_data['title']}': Duplicate questions found: {', '.join(duplicates)}")
            
            # Check if question set has changed
            current_question_ids = set(QuestionGroupService.get_question_order(grp.id, sess))
            new_question_ids = set(question_ids)
            
            if current_question_ids != new_question_ids:
                missing_questions_in_set = current_question_ids - new_question_ids
                extra_questions_in_set = new_question_ids - current_question_ids
                raise ValueError(
                    f"Update aborted - Group '{group_data['title']}': Question set must remain the same. "
                    f"Missing questions: {missing_questions_in_set}. "
                    f"Extra questions: {extra_questions_in_set}"
                )
            
            # Check what needs to be updated
            needs_update = False
            changes = []
            
            # Check display title
            new_display_title = group_data.get("display_title", group_data["title"])
            if new_display_title != grp.display_title:
                needs_update = True
                changes.append("display_title")
            
            # Check description
            if group_data["description"] != grp.description:
                needs_update = True
                changes.append("description")
            
            # Check is_reusable
            new_is_reusable = group_data.get("is_reusable", True)
            if new_is_reusable != grp.is_reusable:
                needs_update = True
                changes.append("is_reusable")
            
            # Check verification_function
            new_verification_function = group_data.get("verification_function")
            if new_verification_function != grp.verification_function:
                needs_update = True
                changes.append("verification_function")
            
            # Check is_auto_submit
            new_is_auto_submit = group_data.get("is_auto_submit", False)
            if new_is_auto_submit != grp.is_auto_submit:
                needs_update = True
                changes.append("is_auto_submit")
            
            # Check question order
            current_order = QuestionGroupService.get_question_order(grp.id, sess)
            if current_order != question_ids:
                needs_update = True
                changes.append("question_order")
            
            if not needs_update:
                skipped.append({"title": group_data["title"], "id": grp.id})
                continue
            
            # Check for auto-submit validation
            is_auto_submit = group_data.get("is_auto_submit", False)
            if is_auto_submit:
                for question_data in group_data.get("questions", []):
                    if question_data.get("default_option") is None:
                        raise ValueError(
                            f"Cannot set default_option to None for question '{question_data['text']}' "
                            f"in auto-submit group '{group_data['title']}'. Auto-submit groups require non-None default values."
                        )
            
            # Verify group update
            QuestionGroupService.verify_edit_group(
                group_id=grp.id,
                new_display_title=group_data.get("display_title", group_data["title"]),
                new_description=group_data["description"],
                is_reusable=group_data.get("is_reusable", True),
                verification_function=group_data.get("verification_function"),
                is_auto_submit=group_data.get("is_auto_submit", False),
                session=sess,
            )
            
            # Apply group updates
            QuestionGroupService.edit_group(
                group_id=grp.id,
                new_display_title=group_data.get("display_title", group_data["title"]),
                new_description=group_data["description"],
                is_reusable=group_data.get("is_reusable", True),
                verification_function=group_data.get("verification_function"),
                is_auto_submit=group_data.get("is_auto_submit", False),
                session=sess,
            )
            
            # Handle question order updates
            if "question_order" in changes:
                QuestionGroupService.update_question_order(grp.id, question_ids, sess)
            
            updated.append({"title": group_data["title"], "id": grp.id, "changes": changes})

        sess.commit()

    
    return updated

# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def sync_question_groups(
    question_groups_folder: str = None, 
    question_groups_data: List[Dict] = None) -> None:
    """Load, validate, and route question groups to add/update pipelines.
    
    Args:
        question_groups_folder: Path to folder containing JSON group files
        question_groups_data: Pre-loaded list of question group dictionaries
        
    Raises:
        ValueError: If neither or both parameters provided, validation fails, or duplicates found
        TypeError: If question_groups_data is not a list of dictionaries
        
    Note:
        Exactly one parameter must be provided.
        Each group dict requires: title, display_title, description, is_reusable, is_auto_submit, verification_function, questions.
        All title values must be unique within the dataset.
    """

    if question_groups_folder is None and question_groups_data is None:
        raise ValueError("Provide either question_groups_folder or question_groups_data")
    
    if question_groups_folder and question_groups_data:
        raise ValueError("Provide either question_groups_folder or question_groups_data, not both")

    # Load data from folder if provided
    if question_groups_folder:
        folder = Path(question_groups_folder)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid folder: {question_groups_folder}")
        
        question_groups_data = []
        for json_path in folder.glob("*.json"):
            with open(json_path, "r") as f:
                try:
                    # Custom hook to detect duplicate keys
                    def detect_duplicate_keys(pairs):
                        seen = set()
                        for key, value in pairs:
                            if key in seen:
                                raise ValueError(f"Duplicate key found: '{key}'")
                            seen.add(key)
                        return dict(pairs)
                    
                    data = json.load(f, object_pairs_hook=detect_duplicate_keys)
                    if not isinstance(data, dict):
                        raise ValueError(f"{json_path.name}: file must contain a JSON object")
                    question_groups_data.append(data)
                except (json.JSONDecodeError, ValueError) as e:
                    raise ValueError(f"{json_path.name}: {str(e)}")

    if not isinstance(question_groups_data, list):
        raise TypeError("question_groups_data must be a list of dictionaries")
    
    question_groups_data = deepcopy(question_groups_data)
    print(f"\n🚀 Starting sync pipeline with {len(question_groups_data)} groups...")

    # Check for duplicate title values
    print("\n🔍 Checking for duplicate title values...")
    
    titles = []
    title_duplicates = []
    
    for idx, data in enumerate(question_groups_data, 1):
        # Validate required fields
        required = {"title", "display_title", "description", "is_reusable", "is_auto_submit", "verification_function", "questions"}
        data_keys = set(data.keys())

        if data_keys != required:
            missing = required - data_keys
            extra = data_keys - required
            
            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")
            
            raise ValueError(f"Entry #{idx} {', '.join(error_parts)}")
        
        title = data["title"]
        if title in titles:
            title_duplicates.append((title, idx))
        else:
            titles.append(title)
    
    if title_duplicates:
        duplicate_info = [f"title '{title}' at entry #{idx}" for title, idx in title_duplicates]
        raise ValueError(f"Duplicate title values found: {', '.join(duplicate_info)}")
    
    print(f"✅ No duplicates found - all {len(titles)} title values are unique")

    # Validate and normalize data
    processed: List[Dict] = []
    for idx, data in enumerate(question_groups_data, 1):
        
        # Set defaults and normalize
        data.setdefault("display_title", data["title"])
        
        if not isinstance(data["questions"], list):
            raise ValueError(f"Entry #{idx}: 'questions' must be a list")
        
        processed.append(data)

    print(f"✅ Validation passed for {len(processed)} groups")

    # Collect all unique questions from all groups
    all_questions_by_text = {}  # text -> question_data
    for group_data in processed:
        for question in group_data["questions"]:
            question_text = question["text"]
            if question_text not in all_questions_by_text:
                all_questions_by_text[question_text] = question

    # Classify add vs update with one read-only session
    to_add, to_update = [], []
    with label_pizza.db.SessionLocal() as sess:
        # Categorize questions: new, existing that need updates, existing unchanged
        unique_questions_to_add = {}      # text -> question_data (for new questions)
        unique_questions_to_update = {}   # text -> question_data (for existing questions that need updates)
        question_text_to_id = {}          # text -> id (for all questions, existing + new)
        
        for question_text, question_data in all_questions_by_text.items():
            try:
                # Try to find existing question
                q_rec = QuestionService.get_question_by_text(question_text, sess)
                question_text_to_id[question_text] = q_rec["id"]
                
                # Check if this existing question needs updates
                needs_update = False
                
                # Check display_text
                new_display_text = question_data.get("display_text", question_text)
                if new_display_text != q_rec["display_text"]:
                    needs_update = True
                
                # Check default_option
                if "default_option" in question_data:
                    new_default = question_data["default_option"]
                    current_default = q_rec.get("default_option")
                    if new_default != current_default:
                        needs_update = True
                
                # For single-choice questions, check other properties
                if q_rec["type"] == "single":
                    new_display_values = question_data.get("display_values")
                    if new_display_values is not None and new_display_values != q_rec.get("display_values"):
                        needs_update = True
                    
                    new_option_weights = question_data.get("option_weights")
                    if new_option_weights is not None and new_option_weights != q_rec.get("option_weights"):
                        needs_update = True
                
                if needs_update:
                    unique_questions_to_update[question_text] = question_data
                    
            except ValueError as err:
                if "not found" not in str(err).lower():
                    raise
                # Question doesn't exist, needs to be created
                unique_questions_to_add[question_text] = question_data
        
        print(f"  📝 {len(unique_questions_to_add)} questions to create")
        print(f"  🔄 {len(unique_questions_to_update)} questions to update")
        print(f"  ✅ {len(question_text_to_id) - len(unique_questions_to_update)} questions unchanged")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: PROCESS ALL QUESTIONS (ADD AND UPDATE)
        # ═══════════════════════════════════════════════════════════════════
        
        print("🔨 Step 2: Processing all questions...")
        
        # Verify all new questions
        for question_text, q_data in unique_questions_to_add.items():
            QuestionService.verify_add_question(
                text=q_data["text"],
                qtype=q_data["qtype"],
                options=q_data.get("options"),
                default=q_data.get("default_option"),
                display_values=q_data.get("display_values"),
                display_text=q_data.get("display_text"),
                option_weights=q_data.get("option_weights"),
                session=sess
            )
        
        # Verify all question updates
        for question_text, q_data in unique_questions_to_update.items():
            q_id = question_text_to_id[question_text]
            QuestionService.verify_edit_question(
                question_id=q_id,
                new_display_text=q_data.get("display_text", q_data["text"]),
                new_opts=q_data.get("options"),
                new_default=q_data.get("default_option"),
                session=sess,
                new_display_values=q_data.get("display_values"),
                new_option_weights=q_data.get("option_weights")
            )
        
        # Create all new questions
        questions_created = []
        for question_text, q_data in unique_questions_to_add.items():
            q_rec = QuestionService.add_question(
                text=q_data["text"],
                qtype=q_data["qtype"],
                options=q_data.get("options"),
                default=q_data.get("default_option"),
                display_values=q_data.get("display_values"),
                display_text=q_data.get("display_text"),
                option_weights=q_data.get("option_weights"),
                session=sess,
            )
            question_text_to_id[question_text] = q_rec.id
            questions_created.append(question_text)
        
        # Update all existing questions
        for question_text, q_data in unique_questions_to_update.items():
            q_id = question_text_to_id[question_text]
            QuestionService.edit_question(
                question_id=q_id,
                new_display_text=q_data.get("display_text", q_data["text"]),
                new_opts=q_data.get("options"),
                new_default=q_data.get("default_option"),
                session=sess,
                new_display_values=q_data.get("display_values"),
                new_option_weights=q_data.get("option_weights")
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: PROCESS ALL QUESTION GROUPS (ADD AND UPDATE)
        # ═══════════════════════════════════════════════════════════════════
        
        print("🏗️  Step 3: Processing all question groups...")
        
        # Classify groups: add vs update
        groups_to_add = []
        groups_to_update = []
        
        for group_data in question_groups_data:
            try:
                QuestionGroupService.get_group_by_name(group_data["title"], sess)
                groups_to_update.append(group_data)
            except ValueError as err:
                if "not found" not in str(err).lower():
                    raise
                groups_to_add.append(group_data)
        
        print(f"  ➕ {len(groups_to_add)} groups to create")
        print(f"  🔄 {len(groups_to_update)} groups to update")
        
        # Process groups to add
        created_groups = []
        if groups_to_add:
            add_data = [(f"item_{i}", group) for i, group in enumerate(groups_to_add)]
            created_groups = add_question_groups(add_data, question_text_to_id)
        
        # Process groups to update  
        updated_groups = []
        if groups_to_update:
            update_data = [(f"item_{i}", group) for i, group in enumerate(groups_to_update)]
            updated_groups = update_question_groups(update_data, question_text_to_id)

        sess.commit()

    # Final summary
    print("🎉 Sync pipeline complete!")
    print(f"   • Questions created: {len(questions_created)}")
    print(f"   • Questions updated: {len(unique_questions_to_update)}")
    print(f"   • Groups created: {len(created_groups)}")
    print(f"   • Groups updated: {len(updated_groups)}")
    
    if created_groups:
        for group in created_groups:
            print(f"     ➕ Created: {group['title']}")
    
    if updated_groups:
        for group in updated_groups:
            print(f"     🔄 Updated: {group['title']} ({', '.join(group['changes'])})")

# --------------------------------------------------------------------------- #
# Core operations                                                             #
# --------------------------------------------------------------------------- #


def add_schemas(schemas: List[Dict]) -> List[Dict]:
    """Create new schemas with full verification and atomic transaction.
    
    Args:
        schemas: List of schema dictionaries with schema_name, question_group_names
        
    Returns:
        List of created schema information
        
    Raises:
        TypeError: If schemas is not a list of dictionaries
        ValueError: If schemas already exist or verification fails
    """
    if not isinstance(schemas, list):
        raise TypeError("schemas must be list[dict]")

    created: List[Dict] = []

    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: duplicate name check (cheap, read‑only) ───────────────
        dup_names = []
        for s in schemas:
            try:
                SchemaService.get_schema_by_name(s["schema_name"], sess)
                dup_names.append(s["schema_name"])
            except ValueError as err:
                # Only ignore "not found" errors, re-raise others
                if "not found" not in str(err).lower():
                    raise
                # Schema doesn't exist, which is what we want for adding
        
        if dup_names:
            raise ValueError("Add aborted – already in DB: " + ", ".join(dup_names))

        # ── Phase 1: resolve group names → ids & run verify_create_schema ────
        prepared: List[Tuple[Dict, List[int]]] = []
        for s in schemas:
            group_ids: List[int] = []
            for gname in s["question_group_names"]:
                group = QuestionGroupService.get_group_by_name(gname, sess)
                group_ids.append(group.id)

            SchemaService.verify_create_schema(
                name=s["schema_name"],
                question_group_ids=group_ids,
                instructions_url=s.get("instructions_url"),
                has_custom_display=s.get("has_custom_display", False),
                session=sess,
            )
            prepared.append((s, group_ids))

        # ── Phase 2: create after all verifications passed ──────────────────
        for s, group_ids in prepared:
            sch = SchemaService.create_schema(
                name=s["schema_name"],
                question_group_ids=group_ids,
                instructions_url=s.get("instructions_url"),
                has_custom_display=s.get("has_custom_display", False),
                session=sess,
            )
            if s.get("is_archived", False):
                SchemaService.archive_schema(sch.id, sess)
            created.append({"name": s["schema_name"], "id": sch.id})

        sess.commit()
    return created


def update_schemas(schemas: List[Dict]) -> List[Dict]:
    """Update existing schemas with full verification and atomic transaction.
    
    Args:
        schemas: List of schema dictionaries with schema_name and updates
        
    Returns:
        List of updated schema information with changes made
        
    Raises:
        TypeError: If schemas is not a list of dictionaries
        ValueError: If schemas not found or verification fails
    """
    if not isinstance(schemas, list):
        raise TypeError("schemas must be list[dict]")

    updated: List[Dict] = []
    skipped: List[Dict] = []
    
    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: existence check ───────────────────────────────────────
        missing = []
        for s in schemas:
            try:
                SchemaService.get_schema_by_name(s["schema_name"], sess)
            except ValueError as err:
                # Only treat "not found" as missing, re-raise other errors
                if "not found" not in str(err).lower():
                    raise
                # Schema doesn't exist
                missing.append(s["schema_name"])
        
        if missing:
            raise ValueError("Update aborted – not found in DB: " + ", ".join(missing))

        # ── Phase 1: prepare each schema and validate question group sets ───
        prepared: List[Tuple[Dict, List[int], object]] = []  # (schema_data, group_ids, schema_record)
        missing_groups = []
        question_group_set_errors = []
        
        for s in schemas:
            sch = SchemaService.get_schema_by_name(s["schema_name"], sess)
            group_ids: List[int] = []
            
        # Get question group IDs from the schema data using question_group_names
        if "question_group_names" in s and s["question_group_names"]:
            for gname in s["question_group_names"]:
                try:
                    group_rec = QuestionGroupService.get_group_by_name(gname, sess)
                    group_ids.append(group_rec.id)
                except ValueError as err:
                    # Only treat "not found" as missing, re-raise other errors
                    if "not found" not in str(err).lower():
                        raise
                    # Question group doesn't exist
                    missing_groups.append(gname)
            
            # Check if question group set has changed (before any database modifications)
            current_group_ids = set(SchemaService.get_question_group_order(sch.id, sess))
            new_group_ids = set(group_ids)
            
            if current_group_ids != new_group_ids:
                missing_groups_in_set = current_group_ids - new_group_ids
                extra_groups_in_set = new_group_ids - current_group_ids
                question_group_set_errors.append(
                    f"Schema '{s['schema_name']}': Question group set must remain the same. "
                    f"Missing groups: {missing_groups_in_set}. "
                    f"Extra groups: {extra_groups_in_set}"
                )
            
            prepared.append((s, group_ids, sch))
        
        # Check for any missing question groups and abort if found
        if missing_groups:
            raise ValueError("Update aborted – question groups not found in DB: " + ", ".join(missing_groups))
        
        # Check for question group set changes and abort if found
        if question_group_set_errors:
            raise ValueError("Update aborted – question group sets changed: " + "; ".join(question_group_set_errors))

        # ── Phase 2: check for differences and skip if no changes ───────────
        to_update = []
        for s, group_ids, sch in prepared:
            needs_update = False
            changes = []
            
            # Check name
            if s.get("schema_name") != sch.name:
                needs_update = True
                changes.append("name")
            
            # Check instructions_url
            new_instructions_url = s.get("instructions_url")
            if new_instructions_url != sch.instructions_url:
                needs_update = True
                changes.append("instructions_url")
            
            # Check has_custom_display
            new_has_custom_display = s.get("has_custom_display", False)
            if new_has_custom_display != sch.has_custom_display:
                needs_update = True
                changes.append("has_custom_display")
            
            # Check is_archived
            if "is_archived" in s and s["is_archived"] != sch.is_archived:
                needs_update = True
                changes.append("archive_status")
            
            # Check question group order
            current_order = SchemaService.get_question_group_order(sch.id, sess)
            if current_order != group_ids:
                needs_update = True
                changes.append("question_group_order")
            
            if needs_update:
                to_update.append((s, group_ids, sch, changes))
            else:
                skipped.append({"name": s["schema_name"], "id": sch.id})

        # ── Phase 3: verify ALL edits first ─────────────────────────────────
        for s, group_ids, sch, changes in to_update:
            SchemaService.verify_edit_schema(
                schema_id=sch.id,
                name=s.get("schema_name"),
                instructions_url=s.get("instructions_url"),
                has_custom_display=s.get("has_custom_display"),
                is_archived=s.get("is_archived"),
                session=sess,
            )

        # ── Phase 4: apply edits after all verifications passed ─────────────
        for s, group_ids, sch, changes in to_update:
            SchemaService.edit_schema(
                schema_id=sch.id,
                name=s.get("schema_name"),
                instructions_url=s.get("instructions_url"),
                has_custom_display=s.get("has_custom_display"),
                is_archived=s.get("is_archived"),
                session=sess,
            )
            
            # Handle question group order updates
            if "question_group_order" in changes:
                SchemaService.update_question_group_order(sch.id, group_ids, sess)
            
            # Handle archiving/unarchiving
            if "archive_status" in changes:
                if s["is_archived"]:
                    SchemaService.archive_schema(sch.id, sess)
                else:
                    SchemaService.unarchive_schema(sch.id, sess)
            
            updated.append({"name": s["schema_name"], "id": sch.id, "changes": changes})

        sess.commit()
    
    # Print summary
    if skipped:
        print(f"⏭️  Skipped {len(skipped)} schema(s) (no changes needed)")
    
    if updated:
        print(f"✅ Updated {len(updated)} schema(s)")
        for item in updated:
            print(f"   • {item['name']}: {', '.join(item['changes'])}")
    
    return updated

# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def sync_schemas(*, schemas_path: str | Path | None = None, schemas_data: List[Dict] | None = None) -> None:
    """Load, validate, and route schemas to add/update pipelines automatically.
    
    Args:
        schemas_path: Path to JSON file containing schema list
        schemas_data: Pre-loaded list of schema dictionaries
        
    Raises:
        ValueError: If neither or both parameters provided, or validation fails
        TypeError: If schemas_data is not a list of dictionaries
        
    Note:
        Exactly one parameter must be provided.
        Each schema dict requires: schema_name, question_group_names, instructions_url, has_custom_display, is_active.
    """

    if schemas_path and schemas_data:
        raise ValueError("Only one of schemas_path or schemas_data can be provided")

    if schemas_path is None and schemas_data is None:
        raise ValueError("Provide either schemas_path or schemas_data")

    # Load JSON if path provided
    if schemas_path:
        with open(schemas_path, "r") as f:
            schemas_data = json.load(f)

    if not isinstance(schemas_data, list):
        raise TypeError("schemas_data must be list[dict]")
    
    # Deep copy schemas_data to avoid modifying the original list
    schemas_data = deepcopy(schemas_data)

    print(f"\n🚀 Starting schema sync pipeline with {len(schemas_data)} schemas...")

    # Check for duplicate schema_name values
    print("\n🔍 Checking for duplicate schema_name values...")
    
    schema_names = []
    schema_name_duplicates = []
    
    processed: List[Dict] = []
    for idx, s in enumerate(schemas_data, 1):
        required = {"schema_name", "question_group_names", "instructions_url", "has_custom_display", "is_active"}
        schema_keys = set(s.keys())

        if schema_keys != required:
            missing = required - schema_keys
            extra = schema_keys - required
            
            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")
            
            raise ValueError(f"Entry #{idx} {', '.join(error_parts)}")
        if not isinstance(s["question_group_names"], list):
            raise ValueError(f"Entry #{idx}: 'question_group_names' must be list")
        s["is_archived"] = not s.pop("is_active")
        processed.append(s)
    
    for idx, schema in enumerate(processed, 1):
        schema_name = schema["schema_name"]
        if schema_name in schema_names:
            schema_name_duplicates.append((schema_name, idx))
        else:
            schema_names.append(schema_name)
    
    if schema_name_duplicates:
        duplicate_info = [f"schema_name '{name}' at entry #{idx}" for name, idx in schema_name_duplicates]
        raise ValueError(f"Duplicate schema_name values found: {', '.join(duplicate_info)}")
    
    print(f"✅ No duplicates found - all {len(schema_names)} schema_name values are unique")

    # Decide add vs update ---------------------------------------------------
    to_add, to_update = [], []
    with label_pizza.db.SessionLocal() as sess:
        for s in processed:
            schema_exists = False
            try:
                SchemaService.get_schema_by_name(s["schema_name"], sess)
                schema_exists = True
            except ValueError as err:
                # Only treat "not found" as non-existence, re-raise other errors
                if "not found" not in str(err).lower():
                    raise
                # Schema doesn't exist
                schema_exists = False
            
            if schema_exists:
                to_update.append(s)
            else:
                to_add.append(s)

    print(f"📊 {len(to_add)} to add · {len(to_update)} to update")

    created, updated = [], []
    if to_add:
        created.extend(add_schemas(to_add))
    if to_update:
        updated.extend(update_schemas(to_update))

    print("🎉 Schema pipeline complete")
    print(f"   • Schemas created: {len(created)}")
    print(f"   • Schemas updated: {len(updated)}")



# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #

def _normalize_video_data(videos: list[Any]) -> Dict[str, List[Dict]]:
    """Convert video list formats into normalized dictionary structure.
    
    Args:
        videos: List of video UIDs (strings) or video dictionaries with questions
        
    Returns:
        Dictionary mapping video_uid to list of question configurations
        
    Raises:
        TypeError: If videos is not a list
        ValueError: If video entries have invalid format
    """
    if not isinstance(videos, list):
        raise TypeError("'videos' must be a list")
    out: Dict[str, List[Dict]] = {}
    for item in videos:
        if isinstance(item, str):
            out[item] = []
        elif isinstance(item, dict) and "video_uid" in item:
            q_cfgs: List[Dict] = []
            for q in item.get("questions", []):
                with label_pizza.db.SessionLocal() as sess:
                    try:
                        question = QuestionService.get_question_by_text(q.get("question_text"), sess)
                    except ValueError as err:
                        raise ValueError(f"Question '{q.get('question_text')}' not found in database")
                if question["type"] == "single":
                    if q.get("custom_question") is None:
                        q["custom_question"] = question["display_text"]
                    if q.get("custom_option") is None:
                        options = question["options"]
                        display_values = question["display_values"]
                        q["custom_option"] = {options[i]: display_values[i] for i in range(len(options))}
                elif question["type"] == "description":
                    if q.get("custom_question") is None:
                        q["custom_question"] = question["display_text"]

                # Check whether question options are valid
                if q.get("custom_option") is not None:
                    for opt, value in q.get("custom_option").items():
                        if value is None:
                            raise ValueError(f"Question '{q.get('question_text')}' has a custom option that is None")
                    opts = set(q.get("custom_option").keys())
                    db_opts = set(question["options"])
                    if opts != db_opts:
                        raise ValueError(f"Question '{q.get('question_text')}' has custom options that do not match the database options")
                q_cfgs.append(
                    {
                        "question_text": q.get("question_text"),
                        "display_text": q.get("custom_question"),
                        "option_map": q.get("custom_option"),
                    }
                )
            out[item["video_uid"]] = q_cfgs
        else:
            raise ValueError(f"Invalid video entry: {item}")
    return out

# --------------------------------------------------------------------------- #
# Custom‑display synchroniser                                                  #
# --------------------------------------------------------------------------- #

@staticmethod
def _sync_custom_displays(project_id: int, videos: list[Any], sess) -> Dict[str, int]:
    """Synchronize custom displays for project videos with verification.
    
    Args:
        project_id: ID of the project
        videos: List of video configurations with custom display settings
        sess: Database session
        
    Returns:
        Dictionary with operation counts (created, updated, removed, skipped)
        
    Raises:
        ValueError: If verification fails for any custom display operation
    """
    stats = {"created": 0, "updated": 0, "removed": 0, "skipped": 0}

    # Get project info including schema
    project = ProjectService.get_project_by_id(project_id, sess)
    schema_id = project.schema_id
    
    # Early exit if schema doesn't support custom displays
    schema = SchemaService.get_schema_by_id(schema_id, sess)
    if not schema.has_custom_display:
        # Count all potential operations as skipped for reporting
        proj_q = ProjectService.get_project_questions(project_id, sess)
        proj_v = VideoService.get_project_videos(project_id, sess)
        stats["skipped"] = len(proj_q) * len(proj_v)
        return stats

    cfg = _normalize_video_data(videos)
    
    # Get project questions and videos using service methods
    proj_q = {q["id"]: q["text"] for q in ProjectService.get_project_questions(project_id, sess)}
    proj_v = {v["id"]: v["uid"] for v in VideoService.get_project_videos(project_id, sess)}

    # ── Phase 1: Plan all operations and verify them ──────────────────────
    operations = []  # List of (operation_type, params) tuples
    verification_errors = []
    
    print("🔍 Planning and verifying custom display operations...")
    
    with tqdm(total=len(proj_v) * len(proj_q), desc="Verifying operations", unit="operation") as pbar:
        for vid_id, uid in proj_v.items():
            json_q_cfg = {qc["question_text"]: qc for qc in cfg.get(uid, [])}

            # First, validate that all question_text in JSON exist in database
            for json_question_text in json_q_cfg.keys():
                if json_question_text not in [q_text for q_text in proj_q.values()]:
                    verification_errors.append(f"Video '{uid}': question_text '{json_question_text}' not found in database")

            for q_id, q_text in proj_q.items():
                # Get existing custom display
                db_rec = CustomDisplayService.get_custom_display(q_id, project_id, vid_id, sess)
                json_cfg = json_q_cfg.get(q_text)

                if db_rec and not json_cfg:
                    # Plan removal operation
                    operations.append(("remove", {
                        "project_id": project_id,
                        "video_id": vid_id, 
                        "question_id": q_id,
                        "video_uid": uid,
                        "question_text": q_text
                    }))
                elif json_cfg:
                    # Check if we need to update or create
                    same_text = db_rec and db_rec.get("display_text") == json_cfg["display_text"]
                    same_map = db_rec and db_rec.get("display_values") == json_cfg["option_map"]
                    
                    if db_rec and same_text and same_map:
                        # Plan skip operation
                        operations.append(("skip", {
                            "video_uid": uid,
                            "question_text": q_text
                        }))
                    else:
                        # Plan create/update operation and verify it
                        operation_type = "update" if db_rec else "create"
                        operation_params = {
                            "project_id": project_id,
                            "video_id": vid_id,
                            "question_id": q_id,
                            "custom_display_text": json_cfg["display_text"],
                            "custom_option_display_map": json_cfg["option_map"],
                            "video_uid": uid,
                            "question_text": q_text
                        }
                        
                        # Verify this operation
                        try:
                            CustomDisplayService.verify_set_custom_display(
                                project_id=project_id,
                                video_id=vid_id,
                                question_id=q_id,
                                custom_display_text=json_cfg["display_text"],
                                custom_option_display_map=json_cfg["option_map"],
                                session=sess
                            )
                            operations.append((operation_type, operation_params))
                        except ValueError as e:
                            verification_errors.append(f"Question '{q_text}' on video '{uid}': {e}")
                else:
                    # No operation needed - neither in DB nor JSON
                    operations.append(("skip", {
                        "video_uid": uid,
                        "question_text": q_text
                    }))
                
                pbar.update(1)
    
    # Check if any verifications failed
    if verification_errors:
        error_summary = f"Custom display verification failed for {len(verification_errors)} operations:\n"
        # Show first 5 errors, then summarize if more
        shown_errors = verification_errors[:5]
        error_summary += "\n".join(f"  • {err}" for err in shown_errors)
        if len(verification_errors) > 5:
            error_summary += f"\n  ... and {len(verification_errors) - 5} more errors"
        raise ValueError(error_summary)

    # ── Phase 2: Execute all operations after verification passed ──────────
    print(f"✅ All verifications passed. Executing {len(operations)} operations...")
    
    with tqdm(total=len(operations), desc="Executing operations", unit="operation") as pbar:
        for operation_type, params in operations:
            if operation_type == "remove":
                CustomDisplayService.remove_custom_display(
                    params["project_id"], 
                    params["video_id"], 
                    params["question_id"], 
                    sess
                )
                stats["removed"] += 1
                
            elif operation_type in ["create", "update"]:
                CustomDisplayService.set_custom_display(
                    project_id=params["project_id"],
                    video_id=params["video_id"],
                    question_id=params["question_id"],
                    custom_display_text=params["custom_display_text"],
                    custom_option_display_map=params["custom_option_display_map"],
                    session=sess
                )
                stats[operation_type + "d"] += 1
                
            elif operation_type == "skip":
                stats["skipped"] += 1
            
            pbar.update(1)
            pbar.set_postfix(created=stats["created"], updated=stats["updated"], 
                            removed=stats["removed"], skipped=stats["skipped"])
                        
    return stats

# --------------------------------------------------------------------------- #
# Creation logic                                                               #
# --------------------------------------------------------------------------- #

def _process_project_validation(project_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Validate single project creation in a thread-safe manner.
    
    Args:
        project_data: Dictionary containing project_name, schema_name, videos
        
    Returns:
        Tuple of (project_name, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            project_name = project_data["project_name"]
            
            # Get schema ID
            schema_id = SchemaService.get_schema_id_by_name(project_data["schema_name"], sess)
            
            # Get video IDs
            video_uids = list(_normalize_video_data(project_data["videos"]).keys())
            video_ids = ProjectService.get_video_ids_by_uids(video_uids, sess)
            # Verify all videos were found
            if len(video_ids) != len(video_uids):
                print('--------------------------------')
                print('video_ids', len(video_ids))
                print('video_uids', len(video_uids))
                print('--------------------------------')
                # Find which ones are missing by checking each one
                missing_uids = []
                for uid in video_uids:
                    video = VideoService.get_video_by_uid(uid, sess)
                    if not video:
                        missing_uids.append(uid)
                
                if missing_uids:
                    raise ValueError(f"Videos not found: {', '.join(missing_uids)}")
            description = project_data.get("description", "")
            
            # Verify creation parameters
            ProjectService.verify_create_project(project_name, description, schema_id, video_ids, sess)
            
            return project_name, True, None
        except ValueError as err:
            if "already exists" in str(err):
                return project_data["project_name"], False, "already exists"
            else:
                return project_data["project_name"], False, str(err)
        except Exception as e:
            return project_data["project_name"], False, str(e)

def _create_single_project(project_data: Dict) -> Tuple[str, bool, Optional[str], Dict]:
    """Create single project in a thread-safe manner with custom displays.
    
    Args:
        project_data: Dictionary containing project creation parameters
        
    Returns:
        Tuple of (project_name, success, error_message, result_info)
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            project_name = project_data["project_name"]
            
            # Get schema ID
            schema_id = SchemaService.get_schema_id_by_name(project_data["schema_name"], sess)
            
            # Get video IDs
            video_uids = list(_normalize_video_data(project_data["videos"]).keys())
            video_ids = ProjectService.get_video_ids_by_uids(video_uids, sess)
            description = project_data.get("description", "")
            
            # Create the project
            ProjectService.create_project(
                name=project_name, 
                description=description, 
                schema_id=schema_id, 
                video_ids=video_ids, 
                session=sess
            )
            
            # Get the created project by name to get its ID
            proj = ProjectService.get_project_by_name(project_name, sess)
            
            # Handle archive status
            if project_data.get("is_archived", False):
                ProjectService.verify_archive_project(proj.id, sess)
                ProjectService.archive_project(proj.id, sess)
            
            if project_data.get("is_active") == True:
                ProjectService.verify_unarchive_project(proj.id, sess)
                ProjectService.unarchive_project(proj.id, sess)
            
            # Sync custom displays
            stats = _sync_custom_displays(proj.id, project_data["videos"], sess)
            
            result = {
                "name": proj.name, 
                "id": proj.id, 
                **stats
            }
            
            return project_name, True, None, result
            
        except Exception as e:
            return project_data["project_name"], False, str(e), {}

def add_projects_parallel(projects: List[Dict], max_workers: int = 20) -> List[Dict]:
    """Create projects using parallel processing with full verification.
    
    Args:
        projects: List of project dictionaries with project_name, schema_name, videos
        max_workers: Number of parallel worker threads (default: 20)
        
    Returns:
        List of created project information with custom display stats
        
    Raises:
        TypeError: If projects is not a list of dictionaries
        ValueError: If projects already exist or verification fails
    """
    if not isinstance(projects, list):
        raise TypeError("projects must be list[dict]")

    # Phase 1: Verify all projects
    duplicates = []
    errors = []
    
    print("🔍 Verifying project creation parameters...")
    with tqdm(total=len(projects), desc="Verifying projects", unit="project") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_project_validation, p): p for p in projects}
            
            for future in concurrent.futures.as_completed(futures):
                project_name, success, error_msg = future.result()
                if not success:
                    if error_msg == "already exists":
                        duplicates.append(project_name)
                    else:
                        errors.append(f"{project_name}: {error_msg}")
                pbar.update(1)

    if duplicates:
        raise ValueError("Add aborted – already in DB: " + ", ".join(duplicates))
    
    if errors:
        raise ValueError("Add aborted – verification errors: " + "; ".join(errors))

    # Phase 2: Create all projects
    output = []
    print("📤 Creating projects...")
    with tqdm(total=len(projects), desc="Creating projects", unit="project") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_create_single_project, p): p for p in projects}
            
            for future in concurrent.futures.as_completed(futures):
                project_name, success, error_msg, result = future.result()
                if not success:
                    raise ValueError(f"Failed to create project {project_name}: {error_msg}")
                
                output.append(result)
                pbar.set_postfix(name=project_name[:20] + "..." if len(project_name) > 20 else project_name)
                pbar.update(1)
                
    print(f"✔ Added {len(projects)} new project(s)")
    return output

def _process_project_update_validation(project_data: Dict) -> Tuple[str, bool, Optional[str]]:
    """Validate single project update in a thread-safe manner.
    
    Args:
        project_data: Dictionary containing project update parameters
        
    Returns:
        Tuple of (project_name, success, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            proj = ProjectService.get_project_by_name(project_data["project_name"], sess)
            
            # Handle archive flag
            desired_archived = None
            if "is_archived" in project_data:
                desired_archived = project_data["is_archived"]
            
            # Check if project is currently archived and if we want to unarchive it
            current_is_archived = proj.is_archived
            
            # Verify archive/unarchive operations
            if desired_archived is not None and desired_archived != proj.is_archived:
                if desired_archived:
                    ProjectService.verify_archive_project(proj.id, sess)
                else:
                    ProjectService.verify_unarchive_project(proj.id, sess)

            if "videos" in project_data:
                # Get current videos in project
                current_videos = VideoService.get_project_videos(proj.id, sess)
                current_video_uids = set(v["uid"] for v in current_videos)
                
                # Get video UIDs from input
                input_video_uids = list(_normalize_video_data(project_data["videos"]).keys())
                input_video_uids_set = set(input_video_uids)
                
                # Check if all videos exist in database
                video_ids = ProjectService.get_video_ids_by_uids(input_video_uids, sess)
                
                if len(video_ids) != len(input_video_uids):
                    # Find which ones are missing
                    missing_uids = []
                    for uid in input_video_uids:
                        video = VideoService.get_video_by_uid(uid, sess)
                        if not video:
                            missing_uids.append(uid)
                    
                    if missing_uids:
                        return project_data["project_name"], False, f"Videos not found in database: {', '.join(missing_uids)}"
                
                # Check if video list matches current project videos
                if current_video_uids != input_video_uids_set:
                    videos_to_remove = current_video_uids - input_video_uids_set
                    videos_to_add = input_video_uids_set - current_video_uids
                    
                    error_parts = []
                    if videos_to_remove:
                        error_parts.append(f"would remove: {', '.join(sorted(videos_to_remove))}")
                    if videos_to_add:
                        error_parts.append(f"would add: {', '.join(sorted(videos_to_add))}")
                    
                    return project_data["project_name"], False, f"Video list mismatch - {'; '.join(error_parts)}. Video updates are not supported."

            # Only verify other updates if project is not archived OR is being unarchived
            # If project is currently archived and not being unarchived, skip other validations
            if not current_is_archived or (desired_archived is False):
                # Verify description updates if provided
                if "description" in project_data:
                    ProjectService.verify_update_project_description(project_id =proj.id, description = project_data["description"], session = sess)
            elif current_is_archived and desired_archived != False:
                # Project is archived and not being unarchived, but trying to update other fields
                if "description" in project_data or "videos" in project_data:
                    return project_data["project_name"], False, "Cannot update archived project. Set is_active=True to unarchive first."
            
            return project_data["project_name"], True, None
        
        except ValueError as err:
            if "not found" in str(err).lower():
                return project_data["project_name"], False, "not found"
            else:
                return project_data["project_name"], False, str(err)
        except Exception as e:
            return project_data["project_name"], False, str(e)

def _update_single_project(project_data: Dict) -> Tuple[str, bool, Optional[str], Dict]:
    """Update single project in a thread-safe manner with change detection."""
    with label_pizza.db.SessionLocal() as sess:
        try:
            project_name = project_data["project_name"]
            proj = ProjectService.get_project_by_name(project_name, sess)
            
            # Check if any information has changed
            needs_update = False
            changes = []
            
            # Check archive status
            desired_archived = None
            if "is_active" in project_data:
                desired_archived = not project_data["is_active"]
                
            if desired_archived is not None and desired_archived != proj.is_archived:
                needs_update = True
                changes.append("archive_status")
            
            # Check description
            if "description" in project_data and project_data["description"] != proj.description:
                needs_update = True
                changes.append("description")
            
            # Check if custom displays need updating (only if schema supports it)
            custom_displays_changed = False
            custom_display_stats = {"created": 0, "updated": 0, "removed": 0, "skipped": 0}
            
            # Only check custom displays if schema supports it AND videos data is provided
            if "videos" in project_data:
                schema = SchemaService.get_schema_by_id(proj.schema_id, sess)
                if schema.has_custom_display:
                    # Get current custom displays for comparison
                    current_custom_displays = CustomDisplayService.get_all_custom_displays_for_project(proj.id, sess)
                    
                    # Normalize the new video data for comparison
                    cfg = _normalize_video_data(project_data["videos"])
                    proj_q = {q["id"]: q["text"] for q in ProjectService.get_project_questions(proj.id, sess)}
                    proj_v = {v["id"]: v["uid"] for v in VideoService.get_project_videos(proj.id, sess)}
                    
                    # Check if any custom displays have actually changed
                    for vid_id, uid in proj_v.items():
                        json_q_cfg = {qc["question_text"]: qc for qc in cfg.get(uid, [])}
                        
                        for q_id, q_text in proj_q.items():
                            db_rec = CustomDisplayService.get_custom_display(q_id, proj.id, vid_id, sess)
                            json_cfg = json_q_cfg.get(q_text)
                            
                            if db_rec and not json_cfg:
                                # Custom display exists in DB but not in JSON - will be removed
                                custom_displays_changed = True
                                break
                            elif json_cfg:
                                # Check if content has changed
                                same_text = db_rec and db_rec.get("display_text") == json_cfg["display_text"]
                                same_map = db_rec and db_rec.get("display_values") == json_cfg["option_map"]
                                
                                if not (db_rec and same_text and same_map):
                                    custom_displays_changed = True
                                    break
                            elif not db_rec and json_cfg:
                                # New custom display will be created
                                custom_displays_changed = True
                                break
                        
                        if custom_displays_changed:
                            break
            
            # Only proceed with updates if there are actual changes
            if needs_update or custom_displays_changed:
                # Apply changes
                if "archive_status" in changes:
                    if desired_archived:
                        ProjectService.archive_project(proj.id, sess)
                    else:
                        ProjectService.unarchive_project(proj.id, sess)
                
                if "description" in changes:
                    ProjectService.update_project_description(proj.id, project_data["description"], sess)
                
                # Sync custom displays only if there are changes
                if custom_displays_changed:
                    custom_display_stats = _sync_custom_displays(proj.id, project_data["videos"], sess)
                    changes.append("custom_displays")
                
                result = {
                    "name": proj.name, 
                    "id": proj.id, 
                    "changes": changes,
                    **custom_display_stats
                }
                
                return project_name, True, None, result
            else:
                # No changes needed - this is the skip case
                result = {
                    "name": proj.name, 
                    "id": proj.id, 
                    "changes": [],
                    "created": 0,
                    "updated": 0,
                    "removed": 0,
                    "skipped": 0
                }
                
                return project_name, True, "No changes needed", result
            
        except Exception as e:
            return project_data["project_name"], False, str(e), {}

def update_projects_parallel(projects: List[Dict], max_workers: int = 20) -> List[Dict]:
    """Update projects using parallel processing with full verification.
    
    Args:
        projects: List of project dictionaries with updates
        max_workers: Number of parallel worker threads (default: 20)
        
    Returns:
        List of updated project information with changes and custom display stats
        
    Raises:
        TypeError: If projects is not a list of dictionaries
        ValueError: If projects not found or verification fails
    """
    if not isinstance(projects, list):
        raise TypeError("projects must be list[dict]")

    # Phase 1: Verify all project updates
    missing = []
    errors = []
    
    print("🔍 Verifying project update parameters...")
    with tqdm(total=len(projects), desc="Verifying project updates", unit="project") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_project_update_validation, p): p for p in projects}
            
            for future in concurrent.futures.as_completed(futures):
                project_name, success, error_msg = future.result()
                if not success:
                    if error_msg == "not found":
                        missing.append(project_name)
                    else:
                        errors.append(f"{project_name}: {error_msg}")
                pbar.update(1)

    if missing:
        raise ValueError("Update aborted – not found in DB: " + ", ".join(missing))
    
    if errors:
        raise ValueError("Update aborted – verification errors: " + "; ".join(errors))

    # Phase 2: Update all projects
    output = []
    updated_count = 0
    skipped_count = 0
    
    print("📤 Updating projects...")
    with tqdm(total=len(projects), desc="Updating projects", unit="project") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_update_single_project, p): p for p in projects}
            
            for future in concurrent.futures.as_completed(futures):
                project_name, success, error_msg, result = future.result()
                if not success:
                    raise ValueError(f"Failed to update project {project_name}: {error_msg}")
                
                if error_msg == "No changes needed":
                    skipped_count += 1
                else:
                    updated_count += 1
                
                output.append(result)
                pbar.set_postfix(name=project_name[:20] + "..." if len(project_name) > 20 else project_name)
                pbar.update(1)

    print(f"✔ Updated {updated_count} project(s), skipped {skipped_count} project(s) (no changes)")
    return output

def sync_projects(*, projects_path: str | Path | None = None, projects_data: List[Dict] | None = None, max_workers: int = 10) -> None:
    """Load, validate, and route projects to add/update pipelines with parallel processing.
    
    Args:
        projects_path: Path to JSON file containing project list
        projects_data: Pre-loaded list of project dictionaries
        max_workers: Number of parallel worker threads (default: 10)
        
    Raises:
        ValueError: If neither or both parameters provided, or validation fails
        TypeError: If projects_data is not a list of dictionaries
        
    Note:
        Exactly one path parameter must be provided.
        Each project dict requires: project_name, schema_name, is_active, videos.
    """
    if projects_path is None and projects_data is None:
        raise ValueError("Provide either projects_path or projects_data")
    
    if projects_path and projects_data:
        raise ValueError("Provide either projects_path or projects_data, not both")
    
    if projects_path:
        with open(projects_path, "r") as f:
            projects_data = json.load(f)
            
    if not isinstance(projects_data, list):
        raise TypeError("projects_data must be list[dict]")
    
    # Deep copy projects_data to avoid modifying the original list
    projects_data = deepcopy(projects_data)
    
    print(f"\n🚀 Starting project upload pipeline with {len(projects_data)} projects...")
    
    # Validate and normalize project data
    processed: List[Dict] = []
    with tqdm(total=len(projects_data), desc="Validating project data", unit="project") as pbar:
        for idx, cfg in enumerate(projects_data, 1):
            # Validate required fields
            required = {"project_name", "description", "schema_name", "is_active", "videos"}
            config_keys = set(cfg.keys())

            if config_keys != required:
                missing = required - config_keys
                extra = config_keys - required
                
                error_parts = []
                if missing:
                    error_parts.append(f"missing: {', '.join(missing)}")
                if extra:
                    error_parts.append(f"extra: {', '.join(extra)}")
                
                raise ValueError(f"Entry #{idx} {', '.join(error_parts)}")
            
            video_uids = []
            video_duplicates = []
            
            for video_idx, video in enumerate(cfg["videos"]):
                # Extract video UID based on format (string or dict)
                if isinstance(video, str):
                    video_uid = video
                elif isinstance(video, dict):
                    required_video = {'video_uid', 'questions'}
                    video_keys = set(video.keys())
                    if video_keys != required_video:
                        missing = required_video - video_keys
                        extra = video_keys - required_video
                        if missing:
                            raise ValueError(f"Entry #{idx}, video #{video_idx + 1}: Missing required fields: {', '.join(missing)}")
                        if extra:
                            raise ValueError(f"Entry #{idx}, video #{video_idx + 1}: Extra fields: {', '.join(extra)}")
                        
                    video_uid = video["video_uid"]
                    # Get question types from database
                    question_types = {}
                    try:
                        with label_pizza.db.SessionLocal() as sess:
                            schema_id = SchemaService.get_schema_id_by_name(cfg["schema_name"], sess)
                            questions_df = SchemaService.get_schema_questions(schema_id, sess)
                            if not questions_df.empty:
                                question_types = dict(zip(questions_df["Text"], questions_df["Type"]))
                    except:
                        # Schema doesn't exist, skip validation
                        pass
                    
                    for question_idx, q in enumerate(video["questions"]):
                        if not isinstance(q, dict):
                            raise ValueError(f"Entry #{idx}, video '{video_uid}', question #{question_idx + 1}: Invalid format")
                        question_type = None
                        with label_pizza.db.SessionLocal() as sess:
                            try:
                                question_type = QuestionService.get_question_by_text(q["question_text"], sess)["type"]
                            except:
                                raise ValueError(f"Entry #{idx}, video '{video_uid}', question #{question_idx + 1}: Question not found in database")

                        if question_type not in ["single", "description"]:
                            raise ValueError(f"Entry #{idx}, video '{video_uid}', question #{question_idx + 1}: Question type must be 'single' or 'description'")
                        if question_type == "single":
                            required = {"question_text", "custom_question", "custom_option"}
                            question_keys = set(q.keys())
                            
                            if question_keys != required:
                                missing = required - question_keys
                                extra = question_keys - required
                                
                                error_parts = []
                                if missing:
                                    error_parts.append(f"missing: {', '.join(missing)}")
                                if extra:
                                    error_parts.append(f"extra: {', '.join(extra)}")
                                
                                raise ValueError(f"Entry #{idx}, video '{video_uid}', question #{question_idx + 1}: {'; '.join(error_parts)}")

                        elif question_type == "description":
                            required = {"question_text", "custom_question"}
                            question_keys = set(q.keys())
                            
                            if question_keys != required:
                                missing = required - question_keys
                                extra = question_keys - required
                                
                                error_parts = []
                                if missing:
                                    error_parts.append(f"missing: {', '.join(missing)}")
                                if extra:
                                    error_parts.append(f"extra: {', '.join(extra)}")
                                
                                raise ValueError(f"Entry #{idx}, video '{video_uid}', question #{question_idx + 1}: {'; '.join(error_parts)}")
                        question_text = q["question_text"]
                        
                        # Check if question exists in database
                        if question_text not in question_types:
                            raise ValueError(f"Entry #{idx}, video '{video_uid}', question '{question_text}': Question not found in schema")
                else:
                    raise ValueError(f"Entry #{idx}, video #{video_idx + 1}: Invalid video format. Must be string or dict with 'video_uid'")
                
                if video_uid in video_uids:
                    video_duplicates.append(video_uid)
                else:
                    video_uids.append(video_uid)
            
            if video_duplicates:
                duplicate_info = [f"video_uid '{uid}'" for uid in video_duplicates]
                raise ValueError(f"Entry #{idx} (project '{cfg['project_name']}'): Duplicate videos found: {', '.join(duplicate_info)}")
            
            # Normalize is_active to is_archived
            cfg["is_archived"] = not cfg.pop("is_active")
                
            processed.append(cfg)
            pbar.update(1)
            
    # Check for duplicate project_name values
    print("\n🔍 Checking for duplicate project_name values...")
    
    project_names = []
    project_name_duplicates = []
    
    for idx, project in enumerate(projects_data, 1):
        project_name = project["project_name"]
        if project_name in project_names:
            project_name_duplicates.append((project_name, idx))
        else:
            project_names.append(project_name)
    
    if project_name_duplicates:
        duplicate_info = [f"project_name '{name}' at entry #{idx}" for name, idx in project_name_duplicates]
        raise ValueError(f"Duplicate project_name values found: {', '.join(duplicate_info)}")
    
    print(f"✅ No duplicates found - all {len(project_names)} project_name values are unique")

    # Separate projects to add vs sync
    to_add, to_sync = [], []
    
    def _check_project_exists(project_data: Dict) -> Tuple[str, bool]:
        """Check if project exists in a thread-safe manner."""
        with label_pizza.db.SessionLocal() as sess:
            try:
                ProjectService.get_project_by_name(project_data["project_name"], sess)
                return project_data["project_name"], True
            except ValueError:
                return project_data["project_name"], False
    
    print("\n📊 Categorizing projects...")
    with tqdm(total=len(processed), desc="Checking existing projects", unit="project") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_check_project_exists, p): p for p in processed}
            
            for future in concurrent.futures.as_completed(futures):
                project_name, exists = future.result()
                project_data = futures[future]
                if exists:
                    to_sync.append(project_data)  # exists → sync
                else:
                    to_add.append(project_data)  # not found → add
                pbar.update(1)

    print(f"\n📈 Summary: {len(to_add)} projects to add, {len(to_sync)} projects to sync")

    # Process projects
    add_results = []
    sync_results = []
    
    if to_add:
        print(f"\n➕ Adding {len(to_add)} new projects...")
        add_results = add_projects_parallel(to_add, max_workers)
            
    if to_sync:
        print(f"\n🔄 Syncing {len(to_sync)} existing projects...")
        sync_results = update_projects_parallel(to_sync, max_workers)

    # Final summary
    print("\n🎉 Project pipeline complete!")
    print(f"✅ Added: {len(add_results)} projects")
    print(f"🔄 Synced: {len(sync_results)} projects")
    
    total_custom_displays = 0
    for result in add_results + sync_results:
        total_custom_displays += result['created'] + result['updated']
    
    if total_custom_displays > 0:
        print(f"🎨 Total custom displays processed: {total_custom_displays}")


def add_project_groups(groups: List[Tuple[str, Dict]]) -> List[Dict]:
    """Create new project groups with full verification and atomic transaction.
    
    Args:
        groups: List of (filename, group_dict) tuples with project group data
        
    Returns:
        List of created project group information
        
    Raises:
        TypeError: If groups is not a list of tuples
        ValueError: If groups already exist or projects not found
    """
    if not isinstance(groups, list):
        raise TypeError("groups must be list[(filename, dict)]")

    created: List[Dict] = []

    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: duplicate name check (cheap, read‑only) ───────────────
        dup_names = []
        for _, g in groups:
            try:
                ProjectGroupService.get_project_group_by_name(g["project_group_name"], sess)
                dup_names.append(g["project_group_name"])
            except ValueError as err:
                # Only ignore "not found" errors, re-raise others
                if "not found" not in str(err).lower():
                    raise
                # Group doesn't exist, which is what we want for adding
        
        if dup_names:
            raise ValueError("Add aborted – already in DB: " + ", ".join(dup_names))

        # ── Phase 1: prepare each group (get project IDs) ──────────
        prepared: List[Tuple[Dict, List[int]]] = []  # (group_data, project_ids)
        missing_projects = []
        
        for _, g in groups:
            project_ids: List[int] = []
            for project_name in g.get("projects", []):
                try:
                    project = ProjectService.get_project_by_name(project_name, sess)
                    project_ids.append(project.id)
                except ValueError as err:
                    # Only treat "not found" as missing, re-raise other errors
                    if "not found" not in str(err).lower():
                        raise
                    # Project doesn't exist - collect for error reporting
                    missing_projects.append(project_name)
            
            prepared.append((g, project_ids))
        
        # Check for any missing projects and abort if found
        if missing_projects:
            raise ValueError("Add aborted – projects not found in DB: " + ", ".join(missing_projects))

        # ── Phase 2: verify ALL groups before any create_group ──────────────
        for g, project_ids in prepared:
            ProjectGroupService.verify_create_project_group(
                name=g["project_group_name"],
                description=g.get("description", ""),
                project_ids=project_ids if project_ids else None,
                session=sess,
            )

        # ── Phase 3: all verifications passed – perform creations ───────────
        for g, project_ids in prepared:
            grp = ProjectGroupService.create_project_group(
                name=g["project_group_name"],
                description=g.get("description", ""),
                project_ids=project_ids if project_ids else None,
                session=sess,
            )
            created.append({"name": g["project_group_name"], "id": grp.id})

        sess.commit()
    return created


def update_project_groups(groups: List[Tuple[str, Dict]]) -> List[Dict]:
    """Update existing project groups with full verification and atomic transaction.
    
    Args:
        groups: List of (filename, group_dict) tuples with project group data
        
    Returns:
        List of updated project group information with changes made
        
    Raises:
        TypeError: If groups is not a list of tuples
        ValueError: If groups not found or projects not found
    """
    if not isinstance(groups, list):
        raise TypeError("groups must be list[(filename, dict)]")

    updated: List[Dict] = []
    skipped: List[Dict] = []
    
    with label_pizza.db.SessionLocal() as sess:
        # ── Phase 0: existence check (cheap, read‑only) ────────────────────
        missing = []
        for _, g in groups:
            try:
                ProjectGroupService.get_project_group_by_name(g["project_group_name"], sess)
            except ValueError as err:
                # Only treat "not found" as missing, re-raise other errors
                if "not found" not in str(err).lower():
                    raise
                # Group doesn't exist
                missing.append(g["project_group_name"])
        
        if missing:
            raise ValueError("Update aborted – not found in DB: " + ", ".join(missing))

        # ── Phase 1: prepare each group and check for changes ──────────────────
        prepared: List[Tuple[Dict, List[int], object]] = []  # (group_data, project_ids, group_record)
        missing_projects = []
        
        for _, g in groups:
            grp = ProjectGroupService.get_project_group_by_name(g["project_group_name"], sess)
            project_ids: List[int] = []
            
            # Get project IDs from the group data - all projects must exist
            for project_name in g.get("projects", []):
                try:
                    project = ProjectService.get_project_by_name(project_name, sess)
                    project_ids.append(project.id)
                except ValueError as err:
                    # Only treat "not found" as missing, re-raise other errors
                    if "not found" not in str(err).lower():
                        raise
                    # Project doesn't exist - collect for error reporting
                    missing_projects.append(project_name)
            
            prepared.append((g, project_ids, grp))
        
        # Check for any missing projects and abort if found
        if missing_projects:
            raise ValueError("Update aborted – projects not found in DB: " + ", ".join(missing_projects))

        # ── Phase 2: check for changes and skip if no changes ────────────────
        to_update = []
        for g, project_ids, grp in prepared:
            # Get current project IDs using ProjectGroupService instead of direct SQL
            group_info = ProjectGroupService.get_project_group_by_id(grp.id, sess)
            current_project_ids = set(p["id"] for p in group_info["projects"])
            current_description = group_info.get("description", "")
            
            new_project_ids = set(project_ids)
            new_description = g.get("description", "")
            
            # Check if any changes are needed
            needs_update = False
            changes = []
            
            # Check description
            if new_description != current_description:
                needs_update = True
                changes.append("description")
            
            # Check project list
            if new_project_ids != current_project_ids:
                needs_update = True
                changes.append("projects")
            
            if needs_update:
                to_update.append((g, project_ids, grp, changes))
            else:
                # No changes needed
                skipped.append({
                    "name": g["project_group_name"], 
                    "id": grp.id,
                    "changes": []
                })

        # ── Phase 3: verify ALL edits first ─────────────────────────────────
        for g, project_ids, grp, changes in to_update:
            # Get current project IDs using ProjectGroupService instead of direct SQL
            group_info = ProjectGroupService.get_project_group_by_id(grp.id, sess)
            current_project_ids = set(p["id"] for p in group_info["projects"])
            
            new_project_ids = set(project_ids)
            
            # Calculate what to add and remove
            add_project_ids = list(new_project_ids - current_project_ids)
            remove_project_ids = list(current_project_ids - new_project_ids)
            
            ProjectGroupService.verify_edit_project_group(
                group_id=grp.id,
                name=None,  # No name change in this implementation
                description=g.get("description", ""),
                add_project_ids=add_project_ids if add_project_ids else None,
                remove_project_ids=remove_project_ids if remove_project_ids else None,
                session=sess,
            )

        # ── Phase 4: apply edits after all verifications passed ─────────────
        for g, project_ids, grp, changes in to_update:
            # Get current project IDs using ProjectGroupService instead of direct SQL
            group_info = ProjectGroupService.get_project_group_by_id(grp.id, sess)
            current_project_ids = set(p["id"] for p in group_info["projects"])
            
            new_project_ids = set(project_ids)
            
            # Calculate what to add and remove
            add_project_ids = list(new_project_ids - current_project_ids)
            remove_project_ids = list(current_project_ids - new_project_ids)
            
            ProjectGroupService.edit_project_group(
                group_id=grp.id,
                name=None,  # No name change in this implementation
                description=g.get("description", ""),
                add_project_ids=add_project_ids if add_project_ids else None,
                remove_project_ids=remove_project_ids if remove_project_ids else None,
                session=sess,
            )
            
            updated.append({
                "name": g["project_group_name"], 
                "id": grp.id,
                "changes": changes
            })

        sess.commit()
    
    # Print summary
    if skipped:
        print(f"⏭️  Skipped {len(skipped)} groups with no changes")
    if updated:
        print(f"🔄 Updated {len(updated)} groups")
    
    return updated + skipped


def sync_project_groups(
    *, project_groups_path: str | Path | None = None, 
    project_groups_data: List[Dict] | None = None) -> None:
    """Load, validate, and route project groups to add/update pipelines.
    
    Args:
        project_groups_path: Path to JSON file containing project group list
        project_groups_data: Pre-loaded list of project group dictionaries
        
    Raises:
        ValueError: If neither or both parameters provided, or validation fails
        TypeError: If project_groups_data is not a list of dictionaries
        
    Note:
        Exactly one parameter must be provided.
        Each group dict requires: project_group_name, projects.
    """

    if project_groups_path is None and project_groups_data is None:
        raise ValueError("Provide either project_groups_path or project_groups_data")

    if project_groups_path and project_groups_data:
        raise ValueError("Provide either project_groups_path or project_groups_data, not both")

    # Load JSON if path provided
    if project_groups_path:
        with open(project_groups_path, "r") as f:
            project_groups_data = json.load(f)

    if not isinstance(project_groups_data, list):
        raise TypeError("project_groups_data must be list[dict]")


    # Deep copy project_groups_data to avoid modifying the original list
    project_groups_data = deepcopy(project_groups_data)
    
    print(f"\n🚀 Starting project groups sync pipeline with {len(project_groups_data)} groups...")

    # Validate and normalize project groups data
    processed: List[Dict] = []
    for idx, g in enumerate(project_groups_data, 1):
        # Validate required fields
        required = {"project_group_name", "projects", "description"}
        group_keys = set(g.keys())

        if group_keys != required:
            missing = required - group_keys
            extra = group_keys - required
            
            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")
            
            raise ValueError(f"Entry #{idx} {', '.join(error_parts)}")
        
        # Set defaults and normalize
        g.setdefault("description", "")
        
        if not isinstance(g["projects"], list):
            raise ValueError(f"Entry #{idx}: 'projects' must be a list")
        
        processed.append(g)

    print(f"✅ JSON validation passed for {len(processed)} items")

    # Check for duplicate project_group_name values
    print("\n🔍 Checking for duplicate project_group_name values...")
    
    project_group_names = []
    project_group_name_duplicates = []
    
    for idx, group in enumerate(project_groups_data, 1):
        project_group_name = group["project_group_name"]
        if project_group_name in project_group_names:
            project_group_name_duplicates.append((project_group_name, idx))
        else:
            project_group_names.append(project_group_name)
    
    if project_group_name_duplicates:
        duplicate_info = [f"project_group_name '{name}' at entry #{idx}" for name, idx in project_group_name_duplicates]
        raise ValueError(f"Duplicate project_group_name values found: {', '.join(duplicate_info)}")
    
    print(f"✅ No duplicates found - all {len(project_group_names)} project_group_name values are unique")

    # Classify add vs update with one read-only session
    to_add, to_update = [], []
    with label_pizza.db.SessionLocal() as sess:
        for g in processed:
            group_exists = False
            try:
                ProjectGroupService.get_project_group_by_name(g["project_group_name"], sess)
                group_exists = True
            except ValueError as err:
                # Only treat "not found" as non-existence, re-raise other errors
                if "not found" not in str(err).lower():
                    raise
                # Group doesn't exist
                group_exists = False
            
            if group_exists:
                to_update.append(g)
            else:
                to_add.append(g)

    print(f"📊 {len(to_add)} to add · {len(to_update)} to update")

    # Execute operations
    created = []
    updated = []
    
    if to_add:
        # Convert to the format expected by add_project_groups
        add_data = [(f"item_{i}", g) for i, g in enumerate(to_add)]
        created.extend(add_project_groups(add_data))
    
    if to_update:
        # Convert to the format expected by update_project_groups
        update_data = [(f"item_{i}", g) for i, g in enumerate(to_update)]
        updated.extend(update_project_groups(update_data))

    print("🎉 Project-group pipeline complete")
    print(f"   • Groups created: {len(created)}")
    print(f"   • Groups updated: {len(updated)}")


def _process_assignment_validation(assignment_data: Dict) -> Tuple[int, Dict, Optional[str]]:
    """Process and validate a single assignment in a thread-safe manner.
    
    Args:
        assignment_data: Dictionary containing assignment fields (user_name/user_email, project_name, role)
        
    Returns:
        Tuple of (index, processed_data, error_message). Error message is None on success.
        
    Raises:
        ValueError: If entity lookup fails with unhandled error
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            required = {"user_name", "project_name", "role", "user_weight", "is_active", "_index"}
            assignment_keys = set(assignment_data.keys())
            if assignment_keys != required:
                missing = required - assignment_keys
                extra = assignment_keys - required
                print("extra", extra)
                print("missing", missing)
                print('--------------------------------')
                error_parts = []
                if missing:
                    error_parts.append(f"missing: {', '.join(missing)}")
                if extra:
                    error_parts.append(f"extra: {', '.join(extra)}")
                
                return assignment_data.get('_index', 0), {}, f"Field validation failed: {', '.join(error_parts)}"
            
            # Validate role
            valid_roles = {'annotator', 'reviewer', 'admin', 'model'}
            if assignment_data['role'] == 'admin':
                raise ValueError("Admin role is not allowed")
            if assignment_data['role'] not in valid_roles:
                return assignment_data.get('_index', 0), {}, f"Invalid role '{assignment_data['role']}'"
            
            # Validate entities exist and aren't archived
            user = AuthService.get_user_by_name(assignment_data['user_name'], sess)
            project = ProjectService.get_project_by_name(assignment_data['project_name'], sess)
            
            if user.is_archived:
                return assignment_data.get('_index', 0), {}, f"User '{assignment_data['user_name']}' is archived"
            if project.is_archived:
                return assignment_data.get('_index', 0), {}, f"Project '{assignment_data['project_name']}' is archived"
                
            processed = {
                **assignment_data,
                'is_active': assignment_data.get('is_active', True),
                'user_id': user.id,
                'project_id': project.id
            }
            
            return assignment_data.get('_index', 0), processed, None
            
        except ValueError as e:
            if "not found" in str(e).lower():
                return assignment_data.get('_index', 0), {}, str(e)
            raise


def _apply_single_assignment(assignment_data: Dict) -> Tuple[str, str, bool, Optional[str]]:
    """Apply a single assignment operation in a thread-safe manner.
    
    Args:
        assignment_data: Validated assignment dictionary with user_id, project_id, role, is_active
        
    Returns:
        Tuple of (assignment_name, operation, success, error_message). 
        Operation is one of: "created", "updated", "removed", "skipped", "error".
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            # Check existing assignment using service method
            if assignment_data['role'] == 'model':
                existing = False
            else:
                user_projects = AuthService.get_user_projects_by_role(assignment_data['user_id'], sess)
                existing = any(
                    assignment_data['project_id'] in [p['id'] for p in projects] 
                    for projects in user_projects.values()
                )
            
            if assignment_data['is_active']:
                ProjectService.add_user_to_project(
                    project_id=assignment_data['project_id'],
                    user_id=assignment_data['user_id'],
                    role=assignment_data['role'],
                    session=sess,
                    user_weight=assignment_data.get('user_weight')
                )
                operation = "updated" if existing else "created"
                return f"{assignment_data['user_name']} -> {assignment_data['project_name']}", operation, True, None
            elif existing:
                # Use remove_user_from_project instead of archive_user_from_project
                AuthService.remove_user_from_project(
                    assignment_data['user_id'], 
                    assignment_data['project_id'], 
                    assignment_data['role'], 
                    sess
                )
                return f"{assignment_data['user_name']} -> {assignment_data['project_name']}", "removed", True, None
            else:
                return f"{assignment_data['user_name']} -> {assignment_data['project_name']}", "skipped", True, None
                
        except Exception as e:
            return f"{assignment_data['user_name']} -> {assignment_data['project_name']}", "error", False, str(e)

def sync_users_to_projects(assignment_path: str = None, assignments_data: list[dict] = None, max_workers: int = 10) -> None:
    """Bulk assign users to projects with parallel validation and application.
    
    Args:
        assignment_path: Path to JSON file containing assignment list
        assignments_data: Pre-loaded list of assignment dictionaries
        max_workers: Number of parallel worker threads (default: 10)
        
    Raises:
        ValueError: If validation fails or input parameters invalid
        TypeError: If assignments_data is not a list of dictionaries  
        RuntimeError: If assignment application fails
        
    Note:
        Exactly one of assignment_path or assignments_data must be provided.
        Each assignment dict requires: user_name/user_email, project_name, role.
        Optional: is_active (default: True), user_weight.
    """
    
    # Load and validate input
    if assignment_path is None and assignments_data is None:
        raise ValueError("Either assignment_path or assignments_data must be provided")
    
    if assignment_path:
        with open(assignment_path, 'r') as f:
            assignments_data = json.load(f)
    
    if not isinstance(assignments_data, list):
        raise TypeError("assignments_data must be a list of dictionaries")
    
    # Deep copy assignments_data to avoid modifying the original list
    assignments_data = deepcopy(assignments_data)

    if not assignments_data:
        print("ℹ️  No assignments to process")
        return

    # Add index for tracking
    for idx, assignment in enumerate(assignments_data):
        assignment['_index'] = idx + 1

    # Process and validate assignments with ThreadPoolExecutor
    processed = []
    seen_pairs = set()
    validation_errors = []
    
    print("🔍 Validating assignments...")
    with tqdm(total=len(assignments_data), desc="Validating assignments", unit="assignment") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_assignment_validation, a): a for a in assignments_data}
            
            for future in concurrent.futures.as_completed(futures):
                assignment = futures[future]
                idx, processed_data, error_msg = future.result()
                
                if error_msg:
                    validation_errors.append(f"#{idx}: {error_msg}")
                else:
                    # Check for duplicates
                    pair = (processed_data['user_name'], processed_data['project_name'])
                    if pair in seen_pairs:
                        validation_errors.append(f"#{idx}: Duplicate assignment {pair[0]} -> {pair[1]}")
                    else:
                        seen_pairs.add(pair)
                        processed.append(processed_data)
                
                pbar.update(1)
                pbar.set_postfix(valid=len(processed), errors=len(validation_errors))

    if validation_errors:
        error_summary = f"Validation failed for {len(validation_errors)} assignments:\n"
        # Show first 5 errors, then summarize if more
        shown_errors = validation_errors[:5]
        error_summary += "\n".join(f"  • {err}" for err in shown_errors)
        if len(validation_errors) > 5:
            error_summary += f"\n  ... and {len(validation_errors) - 5} more errors"
        raise ValueError(error_summary)

    print(f"✅ Validation passed for {len(processed)} assignments")

    # Verify all operations before applying them
    print("🔍 Verifying all operations...")
    verification_errors = []
    
    with tqdm(total=len(processed), desc="Verifying operations", unit="operation") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_verify_single_assignment, a): a for a in processed}
            
            for future in concurrent.futures.as_completed(futures):
                assignment = futures[future]
                assignment_name, error_msg = future.result()
                
                if error_msg:
                    verification_errors.append(f"{assignment_name}: {error_msg}")
                
                pbar.update(1)
                pbar.set_postfix(errors=len(verification_errors))

    if verification_errors:
        error_summary = f"Verification failed for {len(verification_errors)} operations:\n"
        # Show first 5 errors, then summarize if more
        shown_errors = verification_errors[:5]
        error_summary += "\n".join(f"  • {err}" for err in shown_errors)
        if len(verification_errors) > 5:
            error_summary += f"\n  ... and {len(verification_errors) - 5} more errors"
        raise ValueError(error_summary)

    print("✅ All operations verified")

    # Apply assignments with ThreadPoolExecutor
    created = updated = removed = skipped = 0
    application_errors = []
    
    print("📤 Applying assignments...")
    with tqdm(total=len(processed), desc="Applying assignments", unit="assignment") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_apply_single_assignment, a): a for a in processed}
            
            for future in concurrent.futures.as_completed(futures):
                assignment = futures[future]
                assignment_name, operation, success, error_msg = future.result()
                
                if success:
                    if operation == "created":
                        created += 1
                    elif operation == "updated":
                        updated += 1
                    elif operation == "removed":
                        removed += 1
                    elif operation == "skipped":
                        skipped += 1
                else:
                    application_errors.append(f"{assignment_name}: {error_msg}")
                
                pbar.update(1)
                pbar.set_postfix(created=created, updated=updated, removed=removed, skipped=skipped, errors=len(application_errors))

    if application_errors:
        error_summary = f"Application failed for {len(application_errors)} assignments:\n"
        # Show first 5 errors, then summarize if more
        shown_errors = application_errors[:5]
        error_summary += "\n".join(f"  • {err}" for err in shown_errors)
        if len(application_errors) > 5:
            error_summary += f"\n  ... and {len(application_errors) - 5} more errors"
        raise RuntimeError(error_summary)

    print(f"✅ Completed: {created} created, {updated} updated, {removed} removed, {skipped} skipped")

def _verify_single_assignment(assignment_data: Dict) -> Tuple[str, Optional[str]]:
    """Verify a single assignment operation in a thread-safe manner.
    
    Args:
        assignment_data: Assignment dictionary with user_id, project_id, role, is_active
        
    Returns:
        Tuple of (assignment_name, error_message). Error message is None on success.
    """
    with label_pizza.db.SessionLocal() as sess:
        try:
            assignment_name = f"{assignment_data['user_name']} -> {assignment_data['project_name']}"
            
            if assignment_data['is_active']:
                # Verify adding user to project
                ProjectService.verify_add_user_to_project(
                    project_id=assignment_data['project_id'],
                    user_id=assignment_data['user_id'],
                    role=assignment_data['role'],
                    session=sess,
                )
            else:
                # Verify removing user from project
                AuthService.verify_remove_user_from_project(
                    assignment_data['user_id'],
                    assignment_data['project_id'],
                    assignment_data['role'],
                    sess
                )
            
            return assignment_name, None
            
        except Exception as e:
            assignment_name = f"{assignment_data['user_name']} -> {assignment_data['project_name']}"
            return assignment_name, str(e)


def load_and_flatten_json_files(folder_path: str) -> list[dict]:
    """Load all JSON files from folder and flatten into single list.
    
    Args:
        folder_path: Path to folder containing JSON files
        
    Returns:
        Flattened list of dictionaries from all JSON files
        
    Note:
        Handles both single objects and arrays in JSON files.
        Prints success/failure for each file loaded.
    """
    json_files = glob.glob(f"{folder_path}/*.json")
    flattened_data = []
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle both single items and lists
            if isinstance(data, list):
                flattened_data.extend(data)
            else:
                flattened_data.append(data)
            
            print(f"✓ Loaded {filepath}")
        except Exception as e:
            print(f"✗ Failed to load {filepath}: {e}")
    
    return flattened_data


def check_for_duplicates(data: list[dict], data_type: str) -> None:
    """Check for duplicate entries with different logic for annotations vs ground truths.
    
    For annotations: Check duplicates based on (video_uid, user_name, question_text, project_name)
    - Same user cannot answer the same question for the same video in the same project twice
    
    For ground truths: Check duplicates based on (video_uid, question_text, project_name)
    - There can only be one ground truth answer per question per video per project
    - User doesn't matter for ground truth uniqueness
    
    Args:
        data: List of dictionaries to check for duplicates
        data_type: Type description for error messages (e.g., "annotation", "ground truth")
        
    Raises:
        ValueError: If duplicates are found (includes detailed duplicate list)
    """
    seen = set()
    duplicates = []
    
    # Determine checking mode based on data_type parameter
    is_ground_truth_mode = "ground truth" in data_type.lower()
    
    for idx, item in enumerate(data):
        video_uid = item.get("video_uid", "").split("/")[-1]
        user_name = item.get("user_name", "")
        project_name = item.get("project_name", "")
        answers = item.get("answers", {})
        
        # Check each question in the answers dict
        for question_text, answer_value in answers.items():
            if is_ground_truth_mode:
                # For ground truths: only check (video_uid, question_text, project_name)
                # User doesn't matter - there should be only one ground truth per question per video per project
                key = (
                    video_uid,
                    question_text,
                    project_name
                )
            else:
                # For regular annotations: check (video_uid, user_name, question_text, project_name)
                # Same user cannot answer the same question twice
                key = (
                    video_uid,
                    user_name,
                    question_text,
                    project_name
                )
            
            if key in seen:
                duplicate_info = {
                    "index": idx + 1,  # 1-based indexing for user-friendly error messages
                    "video_uid": item.get("video_uid"),
                    "user_name": user_name,
                    "question_text": question_text,
                    "project_name": project_name,
                    "answer": answer_value
                }
                duplicates.append(duplicate_info)
            else:
                seen.add(key)
    
    if duplicates:
        if is_ground_truth_mode:
            error_msg = f"Found {len(duplicates)} duplicate {data_type} entries (multiple ground truths for same question/video/project):\n"
            for dup in duplicates:
                error_msg += f"  - Entry #{dup['index']}: {dup['video_uid']} | '{dup['question_text']}' | {dup['project_name']}\n"
        else:
            error_msg = f"Found {len(duplicates)} duplicate {data_type} question entries:\n"
            for dup in duplicates:
                error_msg += f"  - Entry #{dup['index']}: {dup['video_uid']} | {dup['user_name']} | '{dup['question_text']}' | {dup['project_name']}\n"
        
        raise ValueError(error_msg.rstrip())


def sync_annotations(annotations_folder: str = None, 
                           annotations_data: list[dict] = None, 
                           max_workers: int = 15) -> None:
    """Batch upload annotations with parallel validation and submission.
    
    Args:
        annotations_folder: Path to folder containing JSON annotation files
        annotations_data: Pre-loaded list of annotation dictionaries
        max_workers: Number of parallel validation/submission threads (default: 15)
        
    Raises:
        ValueError: If validation fails, duplicates found, or invalid data structure
        TypeError: If annotations_data is not a list of dictionaries
        RuntimeError: If batch processing fails
        
    Note:
        Exactly one of annotations_folder or annotations_data must be provided.
        All annotations validated in parallel before any database operations.
        Submissions are also processed in parallel for better performance.
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if annotations_folder and annotations_data:
        raise ValueError("Only one of annotations_folder or annotations_data can be provided")
    
    # Load and flatten data
    if annotations_folder:
        annotations_data = load_and_flatten_json_files(annotations_folder)
    
    if not annotations_data:
        print("No annotation data to process")
        return
    
    # Validate data structure
    if not isinstance(annotations_data, list):
        raise TypeError("annotations_data must be a list of dictionaries")
    
    # Deep copy annotations_data to avoid modifying the original list
    annotations_data = deepcopy(annotations_data)
    
    # Check for duplicates
    check_for_duplicates(annotations_data, "annotation")
    
    # Validate all data BEFORE any database operations using ThreadPool
    print("🔍 Validating all annotations...")
    
    def validate_single_annotation(annotation_with_idx):
        idx, annotation = annotation_with_idx
        try:
            required = {"question_group_title", "project_name", "user_name", "video_uid", "answers", "is_ground_truth"}
            optional = {"confidence_scores"}  # Allowed optional fields
            annotation_keys = set(annotation.keys())

            # Check for missing required fields
            missing = required - annotation_keys
            # Check for extra fields, but exclude allowed optional ones
            extra = annotation_keys - required - optional

            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")

            if error_parts:
                raise ValueError(f"Field validation failed: {', '.join(error_parts)}")
            
            # Validate ground truth flag
            if annotation.get("is_ground_truth", False):
                raise ValueError(f"is_ground_truth must be False for annotations")
            
            with label_pizza.db.SessionLocal() as session:
                # Resolve IDs
                video_uid = annotation.get("video_uid", "").split("/")[-1]
                video = VideoService.get_video_by_uid(video_uid, session)
                project = ProjectService.get_project_by_name(annotation["project_name"], session)
                user = AuthService.get_user_by_name(annotation["user_name"], session)
                group = QuestionGroupService.get_group_by_name(annotation["question_group_title"], session)
                
                # Verify submission format
                AnnotatorService.verify_submit_answer_to_question_group(
                    video_id=video.id,
                    project_id=project.id,
                    user_id=user.id,
                    question_group_id=group.id,
                    answers=annotation["answers"],
                    session=session,
                    confidence_scores=annotation.get("confidence_scores"),
                    notes=annotation.get("notes")
                )
                
                # Return validated entry
                return {
                    "success": True,
                    "annotation": annotation,
                    "video_id": video.id,
                    "project_id": project.id,
                    "user_id": user.id,
                    "group_id": group.id,
                    "video_uid": video_uid
                }
                
        except Exception as e:
            return {
                "success": False,
                "idx": idx,
                "annotation": annotation,
                "error": f"[Row {idx}] {annotation.get('video_uid')} | "
                        f"{annotation.get('user_name')} | "
                        f"{annotation.get('question_group_title')}: {e}"
            }
    
    # Parallel validation
    validation_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        enumerated_data = [(idx + 1, annotation) for idx, annotation in enumerate(annotations_data)]
        results = list(tqdm(
            executor.map(validate_single_annotation, enumerated_data),
            total=len(enumerated_data),
            desc="Validating annotations"
        ))
        validation_results.extend(results)
    
    # Check for validation errors - ALL must pass or NONE are submitted
    failed_validations = [r for r in validation_results if not r["success"]]
    if failed_validations:
        print(f"❌ {len(failed_validations)} validation errors found:")
        for failure in failed_validations[:10]:  # Show first 10 errors
            print(f"  {failure['error']}")
        if len(failed_validations) > 10:
            print(f"  ... and {len(failed_validations) - 10} more errors")
        print(f"\n🚫 ABORTING: All {len(validation_results)} annotations must pass validation before any submissions occur.")
        raise ValueError(f"Validation failed for {len(failed_validations)} annotations. No data was submitted.")
    
    print(f"✅ All {len(validation_results)} annotations validated successfully")
    
    # All validations passed - safe to proceed with submissions
    successful_validations = validation_results  # All are successful at this point
    
    # Parallel submission function
    def submit_single_annotation(validation_result):
        """Submit a single annotation entry to the database."""
        try:
            annotation = validation_result["annotation"]
            
            with label_pizza.db.SessionLocal() as session:
                # Check if answers already exist
                existing = AnnotatorService.get_user_answers_for_question_group(
                    video_id=validation_result["video_id"],
                    project_id=validation_result["project_id"],
                    user_id=validation_result["user_id"],
                    question_group_id=validation_result["group_id"],
                    session=session
                )
                
                # Determine if update needed - check if any answer differs
                needs_update = False
                for q_text, answer in annotation["answers"].items():
                    if q_text not in existing or existing[q_text] != answer:
                        needs_update = True
                        break
                    elif existing[q_text] == answer and "confidence_scores" in annotation:
                        # Check if confidence score differs
                        new_confidence = annotation["confidence_scores"].get(q_text)
                        if new_confidence is not None:
                            # Get all answers for this video and filter by user
                            all_answers_df = AnnotatorService.get_answers(
                                video_id=validation_result["video_id"],
                                project_id=validation_result["project_id"],
                                session=session
                            )
                            
                            # Filter for this specific user
                            user_answers = all_answers_df[all_answers_df["User ID"] == validation_result["user_id"]]
                            
                            # Find the specific question and get its confidence score
                            for _, row in user_answers.iterrows():
                                question_info = QuestionService.get_question_by_id(row["Question ID"], session)
                                if question_info and question_info["text"] == q_text:
                                    existing_confidence = row["Confidence Score"]
                                    if existing_confidence != new_confidence:
                                        needs_update = True
                                        break
                            if needs_update:
                                break
                        
                
                if not needs_update:
                    return {
                        "success": True,
                        "status": "skipped",
                        "video_uid": validation_result["video_uid"],
                        "user_name": annotation["user_name"],
                        "group": annotation["question_group_title"],
                        "reason": "No changes needed"
                    }
                
                # Submit the annotation
                AnnotatorService.submit_answer_to_question_group(
                    video_id=validation_result["video_id"],
                    project_id=validation_result["project_id"],
                    user_id=validation_result["user_id"],
                    question_group_id=validation_result["group_id"],
                    answers=annotation["answers"],
                    session=session,
                    confidence_scores=annotation.get("confidence_scores"),
                    notes=annotation.get("notes")
                )
                
                return {
                    "success": True,
                    "status": "uploaded",
                    "video_uid": validation_result["video_uid"],
                    "user_name": annotation["user_name"],
                    "group": annotation["question_group_title"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "video_uid": validation_result["video_uid"],
                "user_name": annotation["user_name"],
                "group": annotation["question_group_title"],
                "error": str(e)
            }
    
    # Parallel submission
    print("📤 Submitting annotations to database...")
    submission_results = []
    failed_submissions = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit futures and track progress
        future_to_validation = {
            executor.submit(submit_single_annotation, validation_result): validation_result
            for validation_result in successful_validations
        }
        
        with tqdm(total=len(successful_validations), desc="Submitting annotations") as pbar:
            for future in as_completed(future_to_validation):
                result = future.result()
                submission_results.append(result)
                
                if not result["success"]:
                    failed_submissions.append(result)
                    print(f"❌ Failed submission: {result['video_uid']} | {result['user_name']} | {result['group']}: {result['error']}")
                
                pbar.update(1)
    
    # Categorize results
    successful_submissions = [r for r in submission_results if r["success"]]
    uploaded = [r for r in successful_submissions if r["status"] == "uploaded"]
    skipped = [r for r in successful_submissions if r["status"] == "skipped"]
    
    # Report results
    if failed_submissions:
        print(f"❌ {len(failed_submissions)} submission errors occurred:")
        for failure in failed_submissions[:10]:  # Show first 10 errors
            print(f"  {failure['video_uid']} | {failure['user_name']} | {failure['group']}: {failure['error']}")
        if len(failed_submissions) > 10:
            print(f"  ... and {len(failed_submissions) - 10} more errors")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  ✅ Uploaded: {len(uploaded)}")
    print(f"  ⏭️  Skipped: {len(skipped)}")
    if failed_submissions:
        print(f"  ❌ Failed: {len(failed_submissions)}")
    
    if uploaded:
        print(f"🎉 Successfully uploaded {len(uploaded)} annotations!")
    
    if failed_submissions and not uploaded:
        raise RuntimeError(f"All {len(failed_submissions)} annotation submissions failed")


def sync_ground_truths(ground_truths_folder: str = None, 
                            ground_truths_data: list[dict] = None, 
                            max_workers: int = 15) -> None:
    """Batch upload ground truths with parallel validation and submission.
    
    Args:
        ground_truths_folder: Path to folder containing JSON ground truth files
        ground_truths_data: Pre-loaded list of ground truth dictionaries  
        max_workers: Number of parallel validation/submission threads (default: 15)
        
    Raises:
        ValueError: If validation fails, duplicates found, or invalid data structure
        TypeError: If ground_truths_data is not a list of dictionaries
        RuntimeError: If batch processing fails (all changes rolled back)
        
    Note:
        Exactly one of ground_truths_folder or ground_truths_data must be provided.
        All ground truths validated in parallel before any database operations.
        Submissions are also processed in parallel for better performance.
        ALL validations must pass before ANY submissions occur (all-or-nothing).
        """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if ground_truths_folder and ground_truths_data:
        raise ValueError("Only one of ground_truths_folder or ground_truths_data can be provided")
    
    # Load and flatten data
    if ground_truths_folder:
        ground_truths_data = load_and_flatten_json_files(ground_truths_folder)
    
    if not ground_truths_data:
        print("No ground truth data to process")
        return
    
    # Validate data structure
    if not isinstance(ground_truths_data, list):
        raise TypeError("ground_truths_data must be a list of dictionaries")
    
    # Deep copy ground_truths_data to avoid modifying the original list
    ground_truths_data = deepcopy(ground_truths_data)
    
    # Check for duplicates
    check_for_duplicates(ground_truths_data, "ground truth")
    
    # Validate all data BEFORE any database operations using ThreadPool
    print("🔍 Validating all ground truths...")
    
    def validate_single_ground_truth(ground_truth_with_idx):
        idx, ground_truth = ground_truth_with_idx
        try:
            required = {"question_group_title", "project_name", "user_name", "video_uid", "answers", "is_ground_truth"}
            optional = {"confidence_scores"}  # Allowed optional fields
            ground_truth_keys = set(ground_truth.keys())

            # Check for missing required fields
            missing = required - ground_truth_keys
            # Check for extra fields, but exclude allowed optional ones
            extra = ground_truth_keys - required - optional

            error_parts = []
            if missing:
                error_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                error_parts.append(f"extra: {', '.join(extra)}")

            if error_parts:
                raise ValueError(f"Field validation failed: {', '.join(error_parts)}")
            # Validate ground truth flag
            if not ground_truth.get("is_ground_truth", False):
                raise ValueError(f"is_ground_truth must be True for ground truths")
            
            with label_pizza.db.SessionLocal() as session:
                # Resolve IDs
                video_uid = ground_truth.get("video_uid", "").split("/")[-1]
                video = VideoService.get_video_by_uid(video_uid, session)
                project = ProjectService.get_project_by_name(ground_truth["project_name"], session)
                reviewer = AuthService.get_user_by_name(ground_truth["user_name"], session)
                group = QuestionGroupService.get_group_by_name(ground_truth["question_group_title"], session)
                
                # Verify submission format
                GroundTruthService.verify_submit_ground_truth_to_question_group(
                    video_id=video.id,
                    project_id=project.id,
                    reviewer_id=reviewer.id,
                    question_group_id=group.id,
                    answers=ground_truth["answers"],
                    session=session,
                    confidence_scores=ground_truth.get("confidence_scores"),
                    notes=ground_truth.get("notes")
                )
                
                # Get questions for admin modification check
                group, questions = GroundTruthService._get_question_group_with_questions(question_group_id=group.id, session=session)

                # Check if any existing ground truth was set by admin
                for question in questions:
                    if GroundTruthService.check_question_modified_by_admin(
                        video_id=video.id, 
                        project_id=project.id, 
                        question_id=question.id, 
                        session=session
                    ):
                        # Get admin modification details for better error message
                        admin_details = GroundTruthService.get_admin_modification_details(
                            video_id=video.id,
                            project_id=project.id,
                            question_id=question.id,
                            session=session
                        )
                        
                        if admin_details:
                            raise ValueError(
                                f"Cannot submit ground truth for question '{question.text}'. "
                                f"This question's ground truth was previously set by admin '{admin_details['admin_name']}' "
                                f"on {admin_details['modified_at'].strftime('%Y-%m-%d %H:%M:%S')}. "
                                f"Only admins can modify admin-set ground truth."
                            )
                        else:
                            raise ValueError(
                                f"Cannot submit ground truth for question '{question.text}'. "
                                f"This question's ground truth was previously modified by an admin. "
                                f"Only admins can modify admin-set ground truth."
                            )
                
                # Return validated entry
                return {
                    "success": True,
                    "ground_truth": ground_truth,
                    "video_id": video.id,
                    "project_id": project.id,
                    "reviewer_id": reviewer.id,
                    "group_id": group.id,
                    "video_uid": video_uid
                }
                
        except Exception as e:
            return {
                "success": False,
                "idx": idx,
                "ground_truth": ground_truth,
                "error": f"[Row {idx}] {ground_truth.get('video_uid')} | "
                        f"reviewer:{ground_truth.get('user_name')}: {e}"
            }
    
    # Parallel validation
    validation_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        enumerated_data = list(enumerate(ground_truths_data))
        results = list(tqdm(
            executor.map(validate_single_ground_truth, enumerated_data),
            total=len(enumerated_data),
            desc="Validating ground truths"
        ))
        validation_results.extend(results)
    
    # Check for validation errors - ALL must pass or NONE are submitted
    failed_validations = [r for r in validation_results if not r["success"]]
    if failed_validations:
        print(f"❌ {len(failed_validations)} validation errors found:")
        for failure in failed_validations[:10]:  # Show first 10 errors
            print(f"  {failure['error']}")
        if len(failed_validations) > 10:
            print(f"  ... and {len(failed_validations) - 10} more errors")
        print(f"\n🚫 ABORTING: All {len(validation_results)} ground truths must pass validation before any submissions occur.")
        raise ValueError(f"Validation failed for {len(failed_validations)} ground truths. No data was submitted.")
    
    print(f"✅ All {len(validation_results)} ground truths validated successfully")
    
    # All validations passed - safe to proceed with submissions
    successful_validations = validation_results  # All are successful at this point
    
    # Parallel submission function
    def submit_single_ground_truth(validation_result):
        """Submit a single ground truth entry to the database."""
        try:
            ground_truth = validation_result["ground_truth"]
            
            with label_pizza.db.SessionLocal() as session:
                # Check if ground truth already exists
                existing = GroundTruthService.get_ground_truth_dict_for_question_group(
                    video_id=validation_result["video_id"],
                    project_id=validation_result["project_id"],
                    question_group_id=validation_result["group_id"],
                    session=session
                )
                
                # Determine if update needed - check if any answer differs
                needs_update = False
                for q_text, answer in ground_truth["answers"].items():
                    if q_text not in existing or existing[q_text] != answer:
                        needs_update = True
                        break
                    elif existing[q_text] == answer and "confidence_scores" in ground_truth:
                        # Check if confidence score differs
                        new_confidence = ground_truth["confidence_scores"].get(q_text)
                        if new_confidence is not None:
                            # Get existing confidence score using service method
                            all_answers_df = GroundTruthService.get_ground_truth_for_question_group(
                                video_id=validation_result["video_id"],
                                project_id=validation_result["project_id"],
                                question_group_id=validation_result["group_id"],
                                session=session
                            )
                            
                            # Find the specific question and get its confidence score
                            for _, row in all_answers_df.iterrows():
                                question_info = QuestionService.get_question_by_id(row["Question ID"], session)
                                if question_info and question_info["text"] == q_text:
                                    existing_confidence = row["Confidence Score"]
                                    if existing_confidence != new_confidence:
                                        needs_update = True
                                        break
                            if needs_update:
                                break
                
                if not needs_update:
                    return {
                        "success": True,
                        "status": "skipped",
                        "video_uid": validation_result["video_uid"],
                        "user_name": ground_truth["user_name"],
                        "reason": "No changes needed"
                    }
                
                # Submit the ground truth
                GroundTruthService.submit_ground_truth_to_question_group(
                    video_id=validation_result["video_id"],
                    project_id=validation_result["project_id"], 
                    reviewer_id=validation_result["reviewer_id"],
                    question_group_id=validation_result["group_id"],
                    answers=ground_truth["answers"],
                    session=session,
                    confidence_scores=ground_truth.get("confidence_scores"),
                    notes=ground_truth.get("notes")
                )
                
                return {
                    "success": True,
                    "status": "uploaded",
                    "video_uid": validation_result["video_uid"],
                    "user_name": ground_truth["user_name"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "video_uid": validation_result["video_uid"],
                "user_name": ground_truth["user_name"],
                "error": str(e)
            }
    
    # Parallel submission
    print("📤 Submitting ground truths to database...")
    submission_results = []
    failed_submissions = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit futures and track progress
        future_to_validation = {
            executor.submit(submit_single_ground_truth, validation_result): validation_result
            for validation_result in successful_validations
        }
        
        with tqdm(total=len(successful_validations), desc="Submitting ground truths") as pbar:
            for future in as_completed(future_to_validation):
                result = future.result()
                submission_results.append(result)
                
                if not result["success"]:
                    failed_submissions.append(result)
                    print(f"❌ Failed submission: {result['video_uid']} | {result['user_name']}: {result['error']}")
                
                pbar.update(1)
    
    # Report results
    successful_submissions = [r for r in submission_results if r["success"]]
    uploaded = [r for r in successful_submissions if r["status"] == "uploaded"]
    skipped = [r for r in successful_submissions if r["status"] == "skipped"]

    # Report results
    if failed_submissions:
        print(f"❌ {len(failed_submissions)} submission errors occurred:")
        for failure in failed_submissions[:10]:  # Show first 10 errors
            print(f"  {failure['video_uid']} | {failure['user_name']}: {failure['error']}")
        if len(failed_submissions) > 10:
            print(f"  ... and {len(failed_submissions) - 10} more errors")

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  ✅ Uploaded: {len(uploaded)}")
    print(f"  ⏭️  Skipped: {len(skipped)}")
    if failed_submissions:
        print(f"  ❌ Failed: {len(failed_submissions)}")

    if uploaded:
        print(f"🎉 Successfully uploaded {len(uploaded)} ground truths!")

    if failed_submissions and not uploaded:
        raise RuntimeError(f"All {len(failed_submissions)} ground truth submissions failed")
                    
                    
