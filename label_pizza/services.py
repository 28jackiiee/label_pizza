from sqlalchemy import select, insert, update, func, delete, exists, join, distinct, and_, or_, case, text
from sqlalchemy.orm import Session, selectinload, joinedload, contains_eager  
from sqlalchemy.sql import literal_column
from typing import List, Optional, Dict, Any, Tuple
from label_pizza.models import (
    Video, Project, ProjectVideo, Schema, QuestionGroup,
    Question, ProjectUserRole, AnnotatorAnswer, ReviewerGroundTruth, User, AnswerReview,
    QuestionGroupQuestion, SchemaQuestionGroup, ProjectGroup, ProjectGroupProject,
    ProjectVideoQuestionDisplay
)
import pandas as pd
from datetime import datetime, timezone
import hashlib
import os
from dotenv import load_dotenv
import importlib.util
import sys
from pathlib import Path

load_dotenv()

# Import verify module
# verify_path = Path(__file__).parent / "verify.py"
# spec = importlib.util.spec_from_file_location("verify", verify_path)
# verify = importlib.util.module_from_spec(spec)
# sys.modules["verify"] = verify
# spec.loader.exec_module(verify)
from label_pizza import verify


class VideoService:
    @staticmethod
    def get_video_by_uid(video_uid: str, session: Session) -> Optional[Video]:
        """Get a video by its UID.
        
        Args:
            video_uid: The UID of the video
            session: Database session
            
        Returns:
            Video object if found, None otherwise
        """
        return session.scalar(select(Video).where(Video.video_uid == video_uid))
    
    @staticmethod
    def get_video_by_url(url: str, session: Session) -> Optional[Video]:
        """Get a video by its URL.
        
        Args:
            url: The URL of the video
            session: Database session

        Returns:
            Video object if found, None otherwise
        """
        return session.scalar(select(Video).where(Video.url == url))
    
    @staticmethod
    def get_video_info_by_uid(video_uid: str, session: Session) -> Dict[str, Any]:
        """Get video info by UID.
        
        Args:
            video_uid: The UID of the video
            session: Database session   
            
        Returns:
            Video object if found, None otherwise
        """
        video = VideoService.get_video_by_uid(video_uid=video_uid, session=session)
        if not video:
            raise ValueError(f"Video with UID '{video_uid}' not found")
        return {
            "id": video.id,
            "uid": video.video_uid,
            "url": video.url,
            "metadata": video.video_metadata or {},
            "created_at": video.created_at,
            "is_archived": video.is_archived
        }

    @staticmethod
    def get_video_url(video_id: int, session: Session) -> str:
        """Get a video's URL by its ID.
        
        Args:
            video_id: The ID of the video
            session: Database session
            
        Returns:
            The video's URL
            
        Raises:
            ValueError: If video not found
        """
        video = session.get(Video, video_id)
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        return video.url

    @staticmethod
    def get_video_metadata(video_id: int, session: Session) -> dict:
        """Get a video's metadata by its ID.
        
        Args:
            video_id: The ID of the video
            session: Database session
            
        Returns:
            The video's metadata dictionary
            
        Raises:
            ValueError: If video not found
        """
        video = session.get(Video, video_id)
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        return video.video_metadata

    @staticmethod
    def archive_video(video_id: int, session: Session) -> None:
        """Archive a video by its ID.
        
        Args:
            video_id: The ID of the video to archive
            session: Database session
            
        Raises:
            ValueError: If video not found
        """
        video = session.get(Video, video_id)
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        video.is_archived = True
        session.commit()
    
    @staticmethod
    def unarchive_video(video_id: int, session: Session) -> None:
        """Unarchive a video by its ID.
        
        Args:
            video_id: The ID of the video to unarchive
            session: Database session
            
        Raises:
            ValueError: If video not found
        """
        video = session.get(Video, video_id)
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        video.is_archived = False
        session.commit()
    
    @staticmethod
    def get_all_videos(session: Session) -> pd.DataFrame:
        """Get all videos.
        
        Args:
            session: Database session
            
        Returns:
            DataFrame containing videos with columns:
            - Video UID: Unique identifier for the video
            - URL: Video URL
            - Archived: Whether the video is archived
        """
        videos = session.scalars(select(Video)).all()
        
        return pd.DataFrame([
            {
                "ID": v.id,
                "Video UID": v.video_uid,
                "URL": v.url,
                "Created At": v.created_at,
                "Updated At": v.updated_at,
                "Archived": v.is_archived
            }
            for v in videos
        ])

    @staticmethod
    def get_videos_with_project_status(session: Session) -> pd.DataFrame:
        """Get all videos with their project assignments and ground truth status.
        
        Args:
            session: Database session
            
        Returns:
            DataFrame containing videos with columns:
            - Video UID: Unique identifier for the video
            - URL: Video URL
            - Projects: Comma-separated list of project names and their ground truth status
        """
        rows = []
        for v in session.scalars(select(Video).where(Video.is_archived == False)).all():
            # Get all non-archived projects this video belongs to
            projects = session.scalars(
                select(Project)
                .join(ProjectVideo, Project.id == ProjectVideo.project_id)
                .where(
                    ProjectVideo.video_id == v.id,
                    Project.is_archived == False
                )
            ).all()
            
            # Skip videos that only belong to archived projects
            if not projects:
                continue
            
            # For each project, check if video has complete ground truth
            project_status = []
            for p in projects:
                # Get total questions in schema through question groups
                total_questions = session.scalar(
                    select(func.count())
                    .select_from(Question)
                    .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                    .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                    .where(SchemaQuestionGroup.schema_id == p.schema_id)
                )
                
                # Get ground truth answers for this video in this project
                gt_answers = session.scalar(
                    select(func.count())
                    .select_from(ReviewerGroundTruth)
                    .where(
                        ReviewerGroundTruth.video_id == v.id,
                        ReviewerGroundTruth.project_id == p.id
                    )
                )
                
                # Get any annotator answers for this video in this project
                any_answers = session.scalar(
                    select(func.count())
                    .select_from(AnnotatorAnswer)
                    .where(
                        AnnotatorAnswer.video_id == v.id,
                        AnnotatorAnswer.project_id == p.id
                    )
                )
                
                # Determine status based on answers
                if total_questions == 0:
                    status = "No questions"
                elif gt_answers == total_questions:
                    status = "✓"
                else:
                    status = "✗"
                project_status.append(f"{p.name}: {status}")
            
            rows.append({
                "Video UID": v.video_uid,
                "URL": v.url,
                "Projects": ", ".join(project_status) if project_status else "No projects",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def verify_add_video(video_uid: str=None, url: str=None, session: Session=None, metadata: dict = None) -> None:
        """Verify parameters for adding a new video.

        Args:
            video_uid: The UID of the video
            url: The URL of the video
            session: Database session
            metadata: Optional dictionary containing video metadata

        Raises:
            ValueError: If URL is invalid, metadata is invalid, or video already exists
        """
        if video_uid is None or url is None:
            raise ValueError("video_uid and url must be provided")

        # Validate URL format
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        if len(video_uid) > 255:
            raise ValueError("Video UID is too long")

        # Validate metadata type - must be None or a dictionary
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")

            # Validate metadata value types if metadata is provided
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool, list, dict)):
                    raise ValueError(f"Invalid metadata value type for key '{key}': {type(value)}")
                if isinstance(value, list):
                    # Validate list elements
                    for item in value:
                        if not isinstance(item, (str, int, float, bool, dict)):
                            raise ValueError(f"Invalid list element type in metadata key '{key}': {type(item)}")
                elif isinstance(value, dict):
                    # Validate nested dictionary values
                    for k, v in value.items():
                        if not isinstance(v, (str, int, float, bool, list, dict)):
                            raise ValueError(f"Invalid nested metadata value type for key '{key}.{k}': {type(v)}")

        # Check if video already exists (case-sensitive check)
        existing = VideoService.get_video_by_uid(video_uid, session)
        if existing:
            raise ValueError(f"Video with UID '{video_uid}' already exists")
        
        # Check if video url already exists
        existing = VideoService.get_video_by_url(url, session)
        if existing:
            raise ValueError(f"Video with URL '{url}' already exists")

    @staticmethod
    def add_video(video_uid: str=None, url: str=None, session: Session=None, metadata: dict = None) -> None:
        """Add a new video to the database.

        Args:
            video_uid: The UID of the video
            url: The URL of the video
            session: Database session
            metadata: Optional dictionary containing video metadata

        Raises:
            ValueError: If URL is invalid, video already exists, or metadata is invalid
        """
        # Verify input parameters and get the filename
        VideoService.verify_add_video(video_uid, url, session, metadata)

        # Create video
        video = Video(
            video_uid=video_uid,
            url=url,
            video_metadata=metadata or {}
        )
        session.add(video)
        session.commit()

    @staticmethod
    def get_project_videos(project_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all non-archived videos in a project.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            List of video dictionaries containing: id, uid, url, metadata
            
        Raises:
            ValueError: If project not found
        """
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        videos = session.scalars(
            select(Video)
            .join(ProjectVideo, Video.id == ProjectVideo.video_id)
            .where(
                ProjectVideo.project_id == project_id,
                Video.is_archived == False
            )
        ).all()
        
        return [{
            "id": v.id,
            "uid": v.video_uid,
            "url": v.url,
            "metadata": v.video_metadata
        } for v in videos]

    @staticmethod
    def verify_update_video(video_uid: str=None, new_url: str=None, new_metadata: dict=None, session: Session=None) -> None:
        """Verify parameters for updating a video.

        Args:
            video_uid: Video UID to update
            new_url: New video URL
            new_metadata: New metadata dictionary
            session: Database session

        Raises:
            ValueError: If video not found or validation fails
        """
        # Check if video exists
        video = VideoService.get_video_by_uid(video_uid=video_uid, session=session)
        if not video:
            raise ValueError(f"Video with UID '{video_uid}' not found")

        # Validate new URL if provided
        if new_url and new_url != video.url:
            if not new_url.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            
            # Check if video url already exists
            existing = VideoService.get_video_by_url(new_url, session)
            if existing:
                raise ValueError(f"Video with URL '{new_url}' already exists")

        # Validate new metadata if provided
        if new_metadata is not None:
            if not isinstance(new_metadata, dict):
                raise ValueError("Metadata must be a dictionary")

            # Validate metadata value types
            for key, value in new_metadata.items():
                if not isinstance(value, (str, int, float, bool, list, dict)):
                    raise ValueError(f"Invalid metadata value type for key '{key}': {type(value)}")
                if isinstance(value, list):
                    # Validate list elements
                    for item in value:
                        if not isinstance(item, (str, int, float, bool, dict)):
                            raise ValueError(f"Invalid list element type in metadata key '{key}': {type(item)}")
                elif isinstance(value, dict):
                    # Validate nested dictionary values
                    for k, v in value.items():
                        if not isinstance(v, (str, int, float, bool, list, dict)):
                            raise ValueError(f"Invalid nested metadata value type for key '{key}.{k}': {type(v)}")

    @staticmethod
    def update_video(video_uid: str, new_url: str, new_metadata: dict, session: Session) -> None:
        """Update video URL and metadata.

        Args:
            video_uid: Video UID to update
            new_url: New video URL
            new_metadata: New metadata dictionary
            session: Database session

        Raises:
            ValueError: If video not found or validation fails
        """
        # Verify parameters (raises ValueError if invalid)
        VideoService.verify_update_video(video_uid, new_url, new_metadata, session)

        # Get the video object for updating
        video = VideoService.get_video_by_uid(video_uid=video_uid, session=session)

        # Update URL if provided and different
        if new_url and new_url != video.url:
            video.url = new_url

        # Update metadata if provided
        if new_metadata is not None:
            video.video_metadata = new_metadata

        video.updated_at = datetime.now(timezone.utc)
        session.commit()

    # Add these new methods to VideoService class
    @staticmethod
    def get_video_counts(session: Session) -> Dict[str, int]:
        """Get video counts without loading actual data."""
        total_videos = session.scalar(select(func.count(Video.id)))
        active_videos = session.scalar(select(func.count(Video.id)).where(Video.is_archived == False))
        archived_videos = total_videos - active_videos
        
        # Get unassigned count efficiently
        assigned_video_ids = session.scalars(
            select(ProjectVideo.video_id).distinct()
        ).all()
        
        unassigned_count = session.scalar(
            select(func.count(Video.id)).where(
                Video.is_archived == False,
                ~Video.id.in_(assigned_video_ids) if assigned_video_ids else True
            )
        )
        
        return {
            "total": total_videos,
            "active": active_videos, 
            "archived": archived_videos,
            "unassigned": unassigned_count
        }

    @staticmethod
    def search_videos(search_term: str = "", show_archived: bool = False, 
                    show_only_unassigned: bool = False, page: int = 0, 
                    page_size: int = 50, session: Session = None) -> Dict[str, Any]:
        """Search videos with pagination and filters."""
        # Build base query
        query = select(Video)
        
        # Apply filters
        conditions = []
        if not show_archived:
            conditions.append(Video.is_archived == False)
        
        if search_term:
            search_filter = or_(
                Video.video_uid.ilike(f"%{search_term}%"),
                Video.url.ilike(f"%{search_term}%")
            )
            conditions.append(search_filter)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Handle unassigned filter
        if show_only_unassigned:
            assigned_video_ids = session.scalars(
                select(ProjectVideo.video_id).distinct()
            ).all()
            if assigned_video_ids:
                query = query.where(~Video.id.in_(assigned_video_ids))
        
        # Get total count for pagination
        count_query = select(func.count()).select_from(query.subquery())
        total_count = session.scalar(count_query)
        
        # Apply pagination
        query = query.offset(page * page_size).limit(page_size)
        videos = session.scalars(query).all()
        
        # Convert to dataframe format
        video_data = []
        for v in videos:
            video_data.append({
                "ID": v.id,
                "Video UID": v.video_uid,
                "URL": v.url,
                "Created At": v.created_at,
                "Updated At": v.updated_at,
                "Archived": v.is_archived
            })
        
        return {
            "videos": pd.DataFrame(video_data),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count - 1) // page_size + 1 if total_count > 0 else 1
        }

    @staticmethod
    def search_videos_for_selection(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search videos for selection dropdowns - returns limited results."""
        query = select(Video).where(
            Video.is_archived == False,
            Video.video_uid.ilike(f"%{search_term}%")
        ).limit(limit)
        
        videos = session.scalars(query).all()
        return [{"id": v.id, "uid": v.video_uid, "url": v.url} for v in videos]


class ProjectService:

    @staticmethod
    def get_bulk_project_completion_data(project_ids: List[int], session: Session) -> Dict[int, Dict[str, Any]]:
        """Get completion data for multiple projects in bulk - much faster than individual calls."""
        
        if not project_ids:
            return {}
        
        # Single query to get all project basic info
        projects_query = select(
            Project.id,
            Project.name,
            Project.description,
            Project.schema_id,
            Project.is_archived,
            Project.created_at
        ).where(Project.id.in_(project_ids))
        
        projects_result = session.execute(projects_query).all()
        
        # Single query to get video counts for all projects
        video_counts_query = select(
            ProjectVideo.project_id,
            func.count(ProjectVideo.video_id).label('video_count')
        ).select_from(
            ProjectVideo
        ).join(
            Video, ProjectVideo.video_id == Video.id
        ).where(
            ProjectVideo.project_id.in_(project_ids),
            Video.is_archived == False
        ).group_by(ProjectVideo.project_id)
        
        video_counts_result = session.execute(video_counts_query).all()
        video_counts_map = {row.project_id: row.video_count for row in video_counts_result}
        
        # Single query to get question counts for all schemas
        schema_ids = [p.schema_id for p in projects_result]
        question_counts_query = select(
            SchemaQuestionGroup.schema_id,
            func.count(Question.id).label('question_count')
        ).select_from(
            Question
        ).join(
            QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id
        ).join(
            SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id
        ).where(
            SchemaQuestionGroup.schema_id.in_(schema_ids),
            Question.is_archived == False
        ).group_by(SchemaQuestionGroup.schema_id)
        
        question_counts_result = session.execute(question_counts_query).all()
        question_counts_map = {row.schema_id: row.question_count for row in question_counts_result}
        
        # Single query to get ground truth counts for all projects
        gt_counts_query = select(
            ReviewerGroundTruth.project_id,
            func.count().label('gt_count')
        ).select_from(
            ReviewerGroundTruth
        ).join(
            Question, ReviewerGroundTruth.question_id == Question.id
        ).where(
            ReviewerGroundTruth.project_id.in_(project_ids),
            Question.is_archived == False
        ).group_by(ReviewerGroundTruth.project_id)
        
        gt_counts_result = session.execute(gt_counts_query).all()
        gt_counts_map = {row.project_id: row.gt_count for row in gt_counts_result}
        
        # Build result
        result = {}
        for project in projects_result:
            video_count = video_counts_map.get(project.id, 0)
            question_count = question_counts_map.get(project.schema_id, 0)
            gt_count = gt_counts_map.get(project.id, 0)
            
            total_possible = video_count * question_count
            completion_percentage = (gt_count / total_possible * 100) if total_possible > 0 else 0.0
            
            result[project.id] = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "schema_id": project.schema_id,
                "is_archived": project.is_archived,
                "created_at": project.created_at,
                "video_count": video_count,
                "question_count": question_count,
                "gt_count": gt_count,
                "completion_percentage": completion_percentage,
                "has_full_ground_truth": completion_percentage >= 100.0
            }
        
        return result

    @staticmethod
    def get_project_by_name(name: str, session: Session) -> Optional[Project]:
        """Get a project by its name.
        
        Args:
            name: The name of the project
            session: Database session
            
        Returns:
            Project object if found, raises ValueError otherwise
        """
        project = session.scalar(select(Project).where(Project.name == name))
        if not project:
            raise ValueError(f"Project with name '{name}' not found")
        return project
    
    @staticmethod
    def get_all_projects(session: Session) -> pd.DataFrame:
        """Get all non-archived projects with their video counts and ground truth percentages."""
        
        # Single optimized query using subqueries and joins
        query = select(
            Project.id,
            Project.name,
            Project.schema_id,
            Project.is_archived,
            Project.created_at,
            Project.updated_at,
            # Count videos in project
            func.coalesce(
                select(func.count(ProjectVideo.video_id))
                .select_from(ProjectVideo)
                .where(ProjectVideo.project_id == Project.id)
                .scalar_subquery(), 0
            ).label('video_count'),
            # Count total questions in schema
            func.coalesce(
                select(func.count(Question.id))
                .select_from(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .where(SchemaQuestionGroup.schema_id == Project.schema_id)
                .scalar_subquery(), 0
            ).label('total_questions'),
            # Count ground truth answers
            func.coalesce(
                select(func.count(ReviewerGroundTruth.project_id))
                .select_from(ReviewerGroundTruth)
                .where(ReviewerGroundTruth.project_id == Project.id)
                .scalar_subquery(), 0
            ).label('gt_answers')
        ).select_from(
            Project
        ).join(
            Schema, Project.schema_id == Schema.id
        ).where(
            and_(Project.is_archived == False, Schema.is_archived == False)
        )
        
        result = session.execute(query).all()
        
        rows = []
        for row in result:
            # Calculate GT percentage
            total_possible = row.video_count * row.total_questions
            gt_percentage = (row.gt_answers / total_possible * 100) if total_possible > 0 else 0.0
            
            rows.append({
                "ID": row.id,
                "Name": row.name,
                "Archived": row.is_archived,
                "Videos": row.video_count,
                "Schema ID": row.schema_id,
                "Created At": row.created_at,
                "Updated At": row.updated_at,
                "GT %": gt_percentage
            })
        
        return pd.DataFrame(rows)

    @staticmethod
    def get_all_projects_including_archived(session: Session) -> pd.DataFrame:
        """Get ALL projects (including archived) with their video counts and ground truth percentages - for admin use."""
        
        # Single optimized query using subqueries and joins - NO archive filter
        query = select(
            Project.id,
            Project.name,
            Project.schema_id,
            Project.is_archived,
            Project.created_at,
            Project.updated_at,
            # Count videos in project
            func.coalesce(
                select(func.count(ProjectVideo.video_id))
                .select_from(ProjectVideo)
                .where(ProjectVideo.project_id == Project.id)
                .scalar_subquery(), 0
            ).label('video_count'),
            # Count total questions in schema
            func.coalesce(
                select(func.count(Question.id))
                .select_from(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .where(SchemaQuestionGroup.schema_id == Project.schema_id)
                .scalar_subquery(), 0
            ).label('total_questions'),
            # Count ground truth answers
            func.coalesce(
                select(func.count(ReviewerGroundTruth.project_id))
                .select_from(ReviewerGroundTruth)
                .where(ReviewerGroundTruth.project_id == Project.id)
                .scalar_subquery(), 0
            ).label('gt_answers')
        ).select_from(
            Project
        ).join(
            Schema, Project.schema_id == Schema.id
        )
        # NOTE: No archive filter here - we want ALL projects for admin view
        
        result = session.execute(query).all()
        
        rows = []
        for row in result:
            # Calculate GT percentage
            total_possible = row.video_count * row.total_questions
            gt_percentage = (row.gt_answers / total_possible * 100) if total_possible > 0 else 0.0
            
            rows.append({
                "ID": row.id,
                "Name": row.name,
                "Archived": row.is_archived,
                "Videos": row.video_count,
                "Schema ID": row.schema_id,
                "Created At": row.created_at,
                "Updated At": row.updated_at,
                "GT %": gt_percentage
            })
        
        return pd.DataFrame(rows)

    @staticmethod
    def verify_create_project(
        name: str=None,
        description: str=None,
        schema_id: int=None,
        video_ids: List[int]=None,
        session: Session=None) -> None:
        """Verify parameters for creating a new project.
        
        Args:
            name: Project name
            description: Project description
            schema_id: ID of the schema to use
            video_ids: List of video IDs to include in the project
            session: Database session
            
        Raises:
            ValueError: If schema or any video is archived, or project name already exists
        """
        # Check if schema exists and is not archived
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        if schema.is_archived:
            raise ValueError(f"Schema with ID {schema_id} is archived")
        
        if name == None:
            raise ValueError("Project name is required")
        
        # Check if project name already exists
        existing_project = session.scalar(select(Project).where(Project.name == name))
        if existing_project:
            raise ValueError(f"Project with name '{name}' already exists")
        
        if len(video_ids) == 0:
            raise ValueError("At least one video is required")
        
        # Check if all videos exist and are not archived
        for vid in video_ids:
            video = session.get(Video, vid)
            if not video:
                raise ValueError(f"Video with ID {vid} not found")
            if video.is_archived:
                raise ValueError(f"Video with ID {vid} is archived")
        
        # Assert that all videos are unique
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("All videos must be unique")

    @staticmethod
    def create_project(
        name: str=None,
        description: str=None,
        schema_id: int=None,
        video_ids: List[int]=None,
        session: Session=None) -> None:
        """Create a new project and assign all admin users to it.
        
        Args:
            name: Project name
            description: Project description
            schema_id: ID of the schema to use
            video_ids: List of video IDs to include in the project
            session: Database session
            
        Raises:
            ValueError: If schema or any video is archived
        """
        # Verify input parameters
        ProjectService.verify_create_project(name=name, description=description, schema_id=schema_id, video_ids=video_ids, session=session)
        
        # Create project
        project = Project(name=name, description=description, schema_id=schema_id)
        session.add(project)
        session.flush()  # Get the project ID
        
        # Add videos to project
        for vid in video_ids:
            session.add(ProjectVideo(project_id=project.id, video_id=vid))
        
        # Assign all admin users to the project using KEYWORD ARGUMENTS
        admin_users = session.scalars(
            select(User).where(User.user_type == "admin", User.is_archived == False)
        ).all()
        
        for admin in admin_users:
            ProjectService.add_user_to_project(
                project_id=project.id, 
                user_id=admin.id, 
                role="admin", 
                session=session
            )
        
        session.commit()

    @staticmethod
    def get_video_ids_by_uids(video_uids: List[str], session: Session) -> List[int]:
        """Get video IDs from their UIDs.
        
        Args:
            video_uids: List of video UIDs
            session: Database session
            
        Returns:
            List of video IDs
        """
        return session.scalars(select(Video.id).where(Video.video_uid.in_(video_uids))).all()

    @staticmethod
    def verify_archive_project(project_id: int, session: Session) -> None:
        """Verify that a project exists and is not already archived.
        
        Args:
            project_id: The ID of the project to verify
            session: Database session

        Raises:
            ValueError: If project not found or already archived
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        if project.is_archived:
            raise ValueError(f"Project with ID {project_id} is already archived")

    @staticmethod
    def archive_project(project_id: int, session: Session) -> None:
        """Archive a project and block new answers.
        
        Args:
            project_id: The ID of the project to archive
            session: Database session
            
        Raises:
            ValueError: If project not found
        """
        ProjectService.verify_archive_project(project_id=project_id, session=session)
        
        project = session.get(Project, project_id)
        project.is_archived = True
        session.commit()
    
    @staticmethod
    def verify_unarchive_project(project_id: int, session: Session) -> None:
        """Verify that a project exists and is archived.
        
        Args:
            project_id: The ID of the project to verify
            session: Database session
            
        Raises:
            ValueError: If project not found or not archived
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        if not project.is_archived:
            raise ValueError(f"Project with ID {project_id} is not archived")
    
    @staticmethod
    def unarchive_project(project_id: int, session: Session) -> None:
        """Unarchive a project and allow new answers.
        
        Args:
            project_id: The ID of the project to unarchive
            session: Database session

        Raises:
            ValueError: If project not found
        """
        ProjectService.verify_unarchive_project(project_id=project_id, session=session)
        
        project = session.get(Project, project_id)
        
        project.is_archived = False
        session.commit()
    
    @staticmethod
    def verify_update_project_description(project_id: int, description: str, session: Session) -> None:
        """Verify that a project exists and is not archived.
        
        Args:
            project_id: The ID of the project to verify
            description: The new description for the project
            session: Database session

        Raises:
            ValueError: If project not found or not archived
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
    
    @staticmethod
    def update_project_description(project_id: int, description: str, session: Session) -> None:
        """Update the description of a project.
        
        Args:
            project_id: The ID of the project to update
            description: The new description for the project
            session: Database session

        Raises:
            ValueError: If project not found or not archived
        """
        ProjectService.verify_update_project_description(project_id=project_id, description=description, session=session)
        
        project = session.get(Project, project_id)
        project.description = description
        session.commit()

    @staticmethod
    def get_project_reviewers(project_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all reviewers who have submitted ground truth in a project.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            List of reviewer dictionaries containing: id, name, email
            
        Raises:
            ValueError: If project not found
        """
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Get all users who have submitted ground truth in this project
        reviewer_ids = session.scalars(
            select(ReviewerGroundTruth.reviewer_id)
            .where(ReviewerGroundTruth.project_id == project_id)
            .distinct()
        ).all()
        
        reviewers = []
        for reviewer_id in reviewer_ids:
            user = session.get(User, reviewer_id)
            if user and not user.is_archived:
                reviewers.append({
                    'id': reviewer_id,
                    'name': user.user_id_str,
                    'email': user.email
                })
        
        return reviewers

    
    @staticmethod
    def get_project_questions(project_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all questions in a project's schema.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            List of question dictionaries containing: id, text, type, display_text, options, display_values, option_weights, default_option
            
        Raises:
            ValueError: If project not found
        """
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Get all questions in the project's schema
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(
                SchemaQuestionGroup.schema_id == project.schema_id,
                Question.is_archived == False
            )
        ).all()
        
        return [{
            'id': q.id,
            'text': q.text,
            'type': q.type,
            'display_text': q.display_text,
            'options': q.options,
            'display_values': q.display_values,
            'option_weights': q.option_weights,
            'default_option': q.default_option
        } for q in questions]
    

    @staticmethod
    def get_project_questions_with_custom_display(project_id: int, video_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all questions in a project's schema with custom display applied for a specific video.
        
        Args:
            project_id: The ID of the project
            video_id: The ID of the video (for custom display context)
            session: Database session
            
        Returns:
            List of question dictionaries with custom display applied
            
        Raises:
            ValueError: If project not found
        """
        # Get original questions
        original_questions = ProjectService.get_project_questions(project_id=project_id, session=session)
        
        # Apply custom display to each question
        result = []
        for q in original_questions:
            # Get full question data with custom display
            question_with_custom = CustomDisplayService.get_custom_display(
                question_id=q['id'],
                project_id=project_id,
                video_id=video_id,
                session=session
            )
            
            # Return same format as original but with custom display fields
            result.append({
                'id': question_with_custom['id'],
                'text': question_with_custom['text'],
                'display_text': question_with_custom['display_text'],  # Custom display applied
                'type': question_with_custom['type'],
                'options': question_with_custom['options'],
                'display_values': question_with_custom['display_values'],  # Custom display applied
                'option_weights': question_with_custom['option_weights'],
                'default_option': question_with_custom['default_option']
            })
        
        return result


    @staticmethod
    def progress(project_id: int, session: Session) -> dict:
        """Get project progress statistics.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            Dictionary containing:
            - total_videos: Number of videos in project
            - total_questions: Number of questions in schema
            - total_answers: Total number of annotator answers submitted
            - ground_truth_answers: Number of ground truth answers
            - completion_percentage: Percentage of questions with ground truth answers
            
        Raises:
            ValueError: If project not found
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Get total videos in project
        total_videos = session.scalar(
            select(func.count())
            .select_from(ProjectVideo)
            .where(ProjectVideo.project_id == project_id)
        )
        
        # Get total questions in schema through question groups
        total_questions = session.scalar(
            select(func.count())
            .select_from(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == project.schema_id)
        )
        
        # Get total annotator answers
        total_answers = session.scalar(
            select(func.count())
            .select_from(AnnotatorAnswer)
            .where(AnnotatorAnswer.project_id == project_id)
        )
        
        # Get ground truth answers
        ground_truth_answers = session.scalar(
            select(func.count())
            .select_from(ReviewerGroundTruth)
            .where(ReviewerGroundTruth.project_id == project_id)
        )
        
        # Calculate completion percentage
        total_possible_answers = total_videos * total_questions
        completion_percentage = round(
            (ground_truth_answers / total_possible_answers * 100) if total_possible_answers > 0 else 0,
            2
        )
        
        return {
            "total_videos": total_videos,
            "total_questions": total_questions,
            "total_answers": total_answers,
            "ground_truth_answers": ground_truth_answers,
            "completion_percentage": completion_percentage
        }

    @staticmethod
    def get_project_by_id(project_id: int, session: Session) -> Optional[Project]:
        """Get a project by its ID.
        
        Args:
            project_id: The ID of the project
            session: Database session
        
        Returns:
            Project object if found, raises ValueError otherwise
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        return project
    
    @staticmethod
    def get_project_dict_by_id(project_id: int, session: Session) -> Dict[str, Any]:
        """Get project details as a dictionary by ID.
        
        Returns:
            Dictionary with project details: id, name, description, schema_id, etc.
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "schema_id": project.schema_id,
            "created_at": project.created_at,
            "is_archived": project.is_archived,
        }

    @staticmethod
    def verify_add_user_to_project(project_id: int, user_id: int, role: str, session: Session) -> None:
        """Verify parameters for adding a user to a project.

        Args:
            project_id: The ID of the project
            user_id: The ID of the user
            role: The role to assign ('annotator', 'reviewer', 'admin', or 'model')
            session: Database session

        Raises:
            ValueError: If project or user not found, archived, or role assignment is invalid
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        if project.is_archived:
            raise ValueError(f"Project '{project.name}' is archived")  # Use name
        
        # Validate user exists and is not archived
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        if user.is_archived:
            raise ValueError(f"User '{user.user_id_str}' is archived")  # Use name
            
        # Validate role is valid
        if role not in ["annotator", "reviewer", "admin", "model"]:
            raise ValueError(f"Invalid role: {role}")
            
        # Validate role assignment rules
        if role == "admin" and user.user_type != "admin":
            raise ValueError(f"User '{user.user_id_str}' must be a global admin to be assigned admin role")
        
        if user.user_type == "admin" and role != "admin":
            raise ValueError(f"Admin user '{user.user_id_str}' cannot be assigned non-admin roles")
        
        if role == "model" and user.user_type != "model":
            raise ValueError(f"User '{user.user_id_str}' must be a model to be assigned model role")
        
        if user.user_type == "model" and role != "model":
            raise ValueError(f"Model user '{user.user_id_str}' cannot be assigned non-model roles")

    @staticmethod
    def add_user_to_project(project_id: int, user_id: int, role: str, session: Session, user_weight: Optional[float] = None) -> None:
        """Add a user to a project with the specified role.
        
        Args:
            project_id: The ID of the project
            user_id: The ID of the user
            role: The role to assign ('annotator', 'reviewer', 'admin', or 'model')
            session: Database session
            user_weight: Optional weight for the user's answers (defaults to 1.0)
            
        Raises:
            ValueError: If project or user not found, or if role is invalid
        """
        # Verify input parameters
        ProjectService.verify_add_user_to_project(project_id, user_id, role, session)
        
        # Archive any existing roles for this user in this project
        session.execute(
            update(ProjectUserRole)
            .where(
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.user_id == user_id
            )
            .values(is_archived=True)
        )
        
        def ensure_role(role_type: str) -> None:
            """Helper function to ensure a role exists and is active."""
            existing = session.scalar(
                select(ProjectUserRole).where(
                    ProjectUserRole.project_id == project_id,
                    ProjectUserRole.user_id == user_id,
                    ProjectUserRole.role == role_type
                )
            )
            if existing:
                existing.is_archived = False
                existing.user_weight = user_weight if user_weight is not None else 1.0
            else:
                session.add(ProjectUserRole(
                    project_id=project_id,
                    user_id=user_id,
                    role=role_type,
                    user_weight=user_weight if user_weight is not None else 1.0
                ))
        
        # Add roles based on the requested role
        if role == "annotator":
            ensure_role("annotator")
        elif role == "reviewer":
            # Reviewers get both annotator and reviewer roles
            ensure_role("annotator")
            ensure_role("reviewer")
        elif role == "model":
            ensure_role("model")
        elif role == "admin":
            # Admins get all three roles
            ensure_role("annotator")
            ensure_role("reviewer")
            ensure_role("admin")
        else:
            raise ValueError(f"Invalid role: {role}")
            
        session.commit()
    
    @staticmethod
    def get_project_annotators(project_id: int, session: Session) -> Dict[str, Dict[str, Any]]:
        """Get all annotators who have submitted answers in a project.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            Dictionary mapping display names to annotator info.
            Each annotator info contains: id, name, email
            
        Raises:
            ValueError: If project not found
        """
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Get all users who have submitted answers in this project
        user_ids = session.scalars(
            select(AnnotatorAnswer.user_id)
            .where(AnnotatorAnswer.project_id == project_id)
            .distinct()
        ).all()
        
        annotators = {}
        for user_id in user_ids:
            user = session.get(User, user_id)
            if user and not user.is_archived:
                user_name = user.user_id_str
                user_email = user.email or f"user_{user.id}@example.com"
                
                # Use the centralized function
                display_name, initials = AuthService.get_user_display_name_with_initials(user_name)
                
                annotators[display_name] = {
                    'id': user_id,
                    'name': user_name,
                    'email': user_email
                }
        
        return annotators

    @staticmethod
    def check_project_has_full_ground_truth(project_id: int, session: Session) -> bool:
        """Check if project has complete ground truth for all questions and videos.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            True if project has complete ground truth, False otherwise
            
        Raises:
            ValueError: If project not found
        """
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        try:
            # Get total questions in schema
            total_questions = session.scalar(
                select(func.count())
                .select_from(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .where(
                    SchemaQuestionGroup.schema_id == project.schema_id,
                    Question.is_archived == False
                )
            )
            
            # Get total videos in project
            total_videos = session.scalar(
                select(func.count())
                .select_from(ProjectVideo)
                .join(Video, ProjectVideo.video_id == Video.id)
                .where(
                    ProjectVideo.project_id == project_id,
                    Video.is_archived == False
                )
            )
            
            if total_questions == 0 or total_videos == 0:
                return False
            
            # Get total ground truth answers
            gt_count = session.scalar(
                select(func.count())
                .select_from(ReviewerGroundTruth)
                .join(Question, ReviewerGroundTruth.question_id == Question.id)
                .where(
                    ReviewerGroundTruth.project_id == project_id,
                    Question.is_archived == False
                )
            )
            
            expected_answers = total_questions * total_videos
            return gt_count >= expected_answers
            
        except Exception:
            return False

        # Add these new methods to ProjectService class  
    
    @staticmethod
    def get_project_counts(session: Session) -> Dict[str, int]:
        """Get project counts without loading actual data."""
        total_projects = session.scalar(select(func.count(Project.id)))
        archived_projects = session.scalar(select(func.count(Project.id)).where(Project.is_archived == True))
        active_projects = total_projects - archived_projects
        
        # Get training mode count more simply - projects that have any ground truth
        training_mode = session.scalar(
            select(func.count(func.distinct(ReviewerGroundTruth.project_id)))
            .select_from(ReviewerGroundTruth)
            .join(Project, ReviewerGroundTruth.project_id == Project.id)
            .where(Project.is_archived == False)
        )
        
        annotation_mode = active_projects - (training_mode or 0)
        
        return {
            "total": total_projects,
            "active": active_projects,
            "archived": archived_projects, 
            "training_mode": training_mode or 0,
            "annotation_mode": annotation_mode
        }
    
    @staticmethod
    def search_projects(search_term: str = "", show_archived: bool = True, 
                    page: int = 0, page_size: int = 50, session: Session = None) -> Dict[str, Any]:
        """Search projects with pagination and filters."""
        # Build base query  
        query = select(Project)
        
        # Apply filters
        conditions = []
        if not show_archived:
            conditions.append(Project.is_archived == False)
        
        if search_term:
            conditions.append(Project.name.ilike(f"%{search_term}%"))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_count = session.scalar(count_query)
        
        # Apply pagination
        query = query.offset(page * page_size).limit(page_size)
        projects = session.scalars(query).all()
        
        # Process projects with enhanced data
        enhanced_projects = []
        for project in projects:
            schema_info = SchemaService.get_schema_name_by_id_with_archived(schema_id=project.schema_id, session=session)
            schema_name = f"{schema_info['name']} {'[Archived]' if schema_info['is_archived'] else ''}"
            
            try:
                progress = ProjectService.progress(project_id=project.id, session=session)
                has_full_gt = ProjectService.check_project_has_full_ground_truth(project_id=project.id, session=session)
                mode = "🎓 Training" if has_full_gt else "📝 Annotation"
                
                enhanced_projects.append({
                    "ID": project.id,
                    "Name": project.name,
                    "Archived": project.is_archived,
                    "Videos": progress['total_videos'],
                    "Schema Name": schema_name,
                    "Mode": mode,
                    "GT Progress": f"{progress['completion_percentage']:.1f}%",
                    "GT Answers": f"{progress['ground_truth_answers']}/{progress['total_videos'] * progress['total_questions']}"
                })
            except Exception as e:
                print(f"Error getting project counts for project {project.id}: {e}")
                enhanced_projects.append({
                    "ID": project.id,
                    "Name": project.name,
                    "Archived": project.is_archived,
                    "Videos": 0,
                    "Schema Name": schema_name,
                    "Mode": "Error",
                    "GT Progress": "Error", 
                    "GT Answers": "Error"
                })
        
        return {
            "projects": pd.DataFrame(enhanced_projects),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count - 1) // page_size + 1 if total_count > 0 else 1
        }

    @staticmethod
    def search_projects_for_selection(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search projects for selection dropdowns - returns limited results."""
        query = select(Project).where(
            Project.name.ilike(f"%{search_term}%")
        ).limit(limit)
        
        projects = session.scalars(query).all()
        return [{"id": p.id, "name": p.name} for p in projects]


class SchemaService:
    
    @staticmethod
    def get_all_schemas(session: Session) -> pd.DataFrame:
        """Get all schemas with their question groups.
        
        Args:
            session: Database session
            
        Returns:
            DataFrame containing schemas with columns:
            - ID: Schema ID
            - Name: Schema name
            - Rules: Schema rules JSON
            - Question Groups: List of question groups in schema
        """
        schemas = session.scalars(select(Schema)).all()
        rows = []
        for s in schemas:
            # Get question groups for this schema
            groups = session.scalars(
                select(QuestionGroup)
                .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
                .where(SchemaQuestionGroup.schema_id == s.id)
            ).all()
            
            rows.append({
                "ID": s.id,
                "Name": s.name,
                "Instructions URL": s.instructions_url,
                "Question Groups": ", ".join(g.title for g in groups) if groups else "No groups",
                "Has Custom Display": s.has_custom_display,
                "Archived": s.is_archived
            })
        return pd.DataFrame(rows)

    @staticmethod
    def get_schema_questions(schema_id: int, session: Session) -> pd.DataFrame:
        """Get all questions in a schema through its question groups.
        
        Args:
            schema_id: The ID of the schema
            session: Database session
            
        Returns:
            DataFrame containing questions with columns:
            - ID: Question ID
            - Text: Question text
            - Group: Question group name
            - Type: Question type
            - Options: Available options for single-choice questions
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        
        # Get questions through question groups
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == schema_id)
        ).all()
        
        return pd.DataFrame([
            {
                "ID": q.id,
                "Text": q.text,
                "Group": session.scalar(
                    select(QuestionGroup.title)
                    .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                    .where(QuestionGroupQuestion.question_id == q.id)
                ),
                "Type": q.type,
                "Options": ", ".join(q.options or []) if q.options else ""
            }
            for q in questions
        ])

    @staticmethod
    def get_schema_questions_with_custom_display(schema_id: int, project_id: int, video_id: int, session: Session) -> pd.DataFrame:
        """Get all questions in a schema with custom display applied for a specific project-video combination.
        
        Args:
            schema_id: The ID of the schema
            project_id: The ID of the project (for custom display context)
            video_id: The ID of the video (for custom display context)
            session: Database session
            
        Returns:
            DataFrame containing questions with custom display applied (same columns as original)
            
        Raises:
            ValueError: If schema not found
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")

        if not schema.has_custom_display:
            raise ValueError(f"Schema with ID {schema_id} does not have custom display enabled")
        
        # Get questions through question groups with custom display
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == schema_id)
        ).all()
        
        rows = []
        for q in questions:
            # Get custom display data
            question_with_custom = CustomDisplayService.get_custom_display(
                question_id=q.id,
                project_id=project_id,
                video_id=video_id,
                session=session
            )
            
            # Get group name
            group_name = session.scalar(
                select(QuestionGroup.title)
                .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                .where(QuestionGroupQuestion.question_id == q.id)
            )
            
            # Same format as original but with custom display
            rows.append({
                "ID": question_with_custom["id"],
                "Text": question_with_custom["text"],
                "Display Text": question_with_custom["display_text"],  # Custom display applied
                "Group": group_name,
                "Type": question_with_custom["type"],
                "Options": ", ".join(question_with_custom["options"] or []) if question_with_custom["options"] else "",
                "Display Values": ", ".join(question_with_custom["display_values"] or []) if question_with_custom["display_values"] else ""  # Custom display applied
            })
        
        return pd.DataFrame(rows)

    @staticmethod
    def get_schema_id_by_name(name: str, session: Session) -> int:
        """Get schema ID by name.
        
        Args:
            name: Schema name
            session: Database session
            
        Returns:
            Schema ID
            
        Raises:
            ValueError: If schema not found
        """
        schema = session.scalar(select(Schema).where(Schema.name == name))
        if not schema:
            raise ValueError(f"Schema '{name}' not found")
        return schema.id
    
    @staticmethod
    def get_schema_name_by_id(schema_id: int, session: Session) -> str:
        """Get schema name by ID.
        
        Args:
            schema_id: The ID of the schema
            session: Database session
            
        Returns:
            Schema name
            
        Raises:
            ValueError: If schema not found or if schema is archived
        """
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        if schema.is_archived:
            raise ValueError(f"Schema with ID {schema_id} is archived")
        return schema.name
    
    @staticmethod
    def get_schema_name_by_id_with_archived(schema_id: int, session: Session) -> str:
        """Get schema name by ID.
        
        Args:
            schema_id: The ID of the schema
            session: Database session
            
        Returns:
            Schema name
            
        Raises:
            ValueError: If schema not found
        """
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        return {
            "name": schema.name,
            "is_archived": schema.is_archived
        }

    @staticmethod
    def verify_create_schema(
            name: str,
            question_group_ids: List[int],
            instructions_url: Optional[str] = None,
            has_custom_display: bool = False,
            session: Session = None
    ) -> None:
        """Verify parameters for creating a new schema.

        This function performs all validation checks and raises ValueError
        if any validation fails. It does not return any objects.

        Args:
            name: Schema name
            question_group_ids: List of question group IDs in desired order
            instructions_url: URL for schema instructions
            has_custom_display: Whether schema has custom display
            session: Database session

        Raises:
            ValueError: If schema with same name exists or validation fails
        """
        # Validate name
        if not name or not name.strip():
            raise ValueError("Schema name is required")

        # Check if schema with same name exists
        existing = session.scalar(select(Schema).where(Schema.name == name))
        if existing:
            raise ValueError(f"Schema with name '{name}' already exists")

        # Validate question group IDs
        if not question_group_ids:
            raise ValueError("Schema must contain at least one question group")

        # Validate instructions URL if provided
        if instructions_url:
            instructions_url = instructions_url.strip()
            if not instructions_url.startswith(("http://", "https://")):
                raise ValueError("Instructions URL must start with http:// or https://")

        # Validate all question groups
        for group_id in question_group_ids:
            # Check if group exists
            group = QuestionGroupService.get_group_by_id(group_id, session)
            if group.is_archived:
                raise ValueError(f"Question group with ID {group_id} is archived")

            # Check if non-reusable group is already used in another schema
            if not group.is_reusable:
                existing_schema = session.scalar(
                    select(Schema)
                    .join(SchemaQuestionGroup, Schema.id == SchemaQuestionGroup.schema_id)
                    .where(SchemaQuestionGroup.question_group_id == group_id)
                )
                if existing_schema:
                    raise ValueError(
                        f"Question group {group.title} is not reusable and is already used in schema {existing_schema.name}")
        
        # Validate that all questions in schema are unique (no shared questions between groups)
        all_question_ids = []
        group_questions = {}  # group_id -> list of question_ids
        
        for group_id in question_group_ids:
            questions = QuestionService.get_questions_by_group_id(group_id=group_id, session=session)
            question_ids = [q["id"] for q in questions]
            group_questions[group_id] = question_ids
            all_question_ids.extend(question_ids)
        
        # Check if all questions are unique
        if len(all_question_ids) != len(set(all_question_ids)):
            # Find which groups share questions
            from collections import Counter
            question_counts = Counter(all_question_ids)
            shared_questions = [qid for qid, count in question_counts.items() if count > 1]
            
            # Find which groups share these questions
            sharing_groups = []
            for shared_qid in shared_questions:
                groups_with_question = []
                for group_id, question_ids in group_questions.items():
                    if shared_qid in question_ids:
                        group = session.get(QuestionGroup, group_id)
                        if group:
                            groups_with_question.append(group.title)
                sharing_groups.append(groups_with_question)
            
            group_names = list(set().union(*sharing_groups))
            raise ValueError(f"Question groups {group_names} share the same questions")

    @staticmethod
    def create_schema(
        name: str,
        question_group_ids: List[int],
        instructions_url: Optional[str] = None,
        has_custom_display: bool = False,
        session: Session = None
    ) -> Schema:
        """Create a new schema with its question groups.

        Args:
            name: Schema name
            question_group_ids: List of question group IDs in desired order
            instructions_url: URL for schema instructions
            has_custom_display: Whether schema has custom display
            session: Database session

        Returns:
            Created schema

        Raises:
            ValueError: If schema with same name exists or validation fails
        """
        # First, verify all parameters (will raise ValueError if validation fails)
        SchemaService.verify_create_schema(name, question_group_ids, instructions_url, has_custom_display, session)

        # Create schema object
        schema = Schema(name=name, instructions_url=instructions_url, has_custom_display=has_custom_display)
        session.add(schema)
        session.flush()  # Get schema ID

        # Add question groups to schema in specified order
        for i, group_id in enumerate(question_group_ids):
            sqg = SchemaQuestionGroup(
                schema_id=schema.id,
                question_group_id=group_id,
                display_order=i
            )
            session.add(sqg)

        session.commit()
        return schema
    
    @staticmethod
    def get_schema_details(schema_id: int, session: Session) -> Dict[str, Any]:
        """Get complete schema details including instructions URL.
        
        Args:
            schema_id: Schema ID
            session: Database session
            
        Returns:
            Dictionary containing schema details
            
        Raises:
            ValueError: If schema not found
        """
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        
        return {
            "id": schema.id,
            "name": schema.name,
            "instructions_url": schema.instructions_url,
            "created_at": schema.created_at,
            "updated_at": schema.updated_at,
            "has_custom_display": schema.has_custom_display,
            "is_archived": schema.is_archived
        }

    @staticmethod
    def verify_edit_schema(
        schema_id: int,
        name: Optional[str] = None,
        instructions_url: Optional[str] = None,
        has_custom_display: Optional[bool] = None,
        is_archived: Optional[bool] = None,
        session: Session = None
    ) -> None:
        """Verify parameters for editing a schema.

        This function performs all validation checks and raises ValueError
        if any validation fails. It does not return any objects.

        Args:
            schema_id: Schema ID
            name: New schema name (optional)
            instructions_url: New instructions URL (optional, use empty string to clear)
            has_custom_display: New custom display flag (optional)
            is_archived: New archive status (optional)
            session: Database session

        Raises:
            ValueError: If schema not found or validation fails
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        
        # Validate name if provided
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Schema name cannot be empty")
            
            # Check for unique name (excluding current schema)
            existing = session.scalar(
                select(Schema).where(Schema.name == name, Schema.id != schema_id)
            )
            if existing:
                raise ValueError(f"Schema with name '{name}' already exists")
        
        # Validate instructions URL if provided
        if instructions_url is not None:
            if instructions_url.strip():
                instructions_url = instructions_url.strip()
                if not instructions_url.startswith(("http://", "https://")):
                    raise ValueError("Instructions URL must start with http:// or https://")
        
        # Validate archive status change if provided
        if is_archived is not None:
            if is_archived and not schema.is_archived:
                # Trying to archive - check no non-archived projects are using this schema
                projects_using_schema = session.scalars(
                    select(Project)
                    .where(
                        Project.schema_id == schema_id,
                        Project.is_archived == False
                    )
                ).all()
                
                if projects_using_schema:
                    project_names = [p.name for p in projects_using_schema]
                    raise ValueError(
                        f"Cannot archive schema. The following non-archived projects are using it: "
                        f"{', '.join(project_names)}"
                    )
            
            elif not is_archived and schema.is_archived:
                # Trying to unarchive - check all question groups in schema are not archived
                archived_groups = session.scalars(
                    select(QuestionGroup)
                    .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
                    .where(
                        SchemaQuestionGroup.schema_id == schema_id,
                        QuestionGroup.is_archived == True
                    )
                ).all()
                
                if archived_groups:
                    group_names = [g.title for g in archived_groups]
                    raise ValueError(
                        f"Cannot unarchive schema. The following question groups in the schema are archived: "
                        f"{', '.join(group_names)}"
                    )

    @staticmethod
    def edit_schema(
        schema_id: int, 
        name: Optional[str] = None,
        instructions_url: Optional[str] = None,
        has_custom_display: Optional[bool] = None,
        is_archived: Optional[bool] = None,
        session: Session = None
    ) -> None:
        """Edit a schema's properties.
        
        Args:
            schema_id: Schema ID
            name: New schema name (optional)
            instructions_url: New instructions URL (optional, use empty string to clear)
            has_custom_display: New custom display flag (optional)
            is_archived: New archive status (optional)
            session: Database session
            
        Raises:
            ValueError: If schema not found or validation fails
        """
        # First, verify all parameters (will raise ValueError if validation fails)
        SchemaService.verify_edit_schema(
            schema_id, name, instructions_url, has_custom_display, is_archived, session
        )
        
        # Get schema object for updating
        schema = session.get(Schema, schema_id)
        
        # Update name if provided
        if name is not None:
            schema.name = name.strip()
        
        # Update instructions URL if provided
        if instructions_url is not None:
            if instructions_url.strip():
                schema.instructions_url = instructions_url.strip()
            else:
                # Empty string means clear the URL
                schema.instructions_url = None
        
        # Update custom display flag if provided
        if has_custom_display is not None:
            schema.has_custom_display = has_custom_display
        
        # Update archive status if provided
        if is_archived is not None:
            schema.is_archived = is_archived
        
        schema.updated_at = datetime.now(timezone.utc)
        session.commit()

    @staticmethod
    def archive_schema(schema_id: int, session: Session) -> None:
        """Archive a schema and prevent its use in new projects.
        
        Args:
            schema_id: The ID of the schema to archive
            session: Database session
            
        Raises:
            ValueError: If schema not found or has active projects
        """
        # Use the edit function with proper validation
        SchemaService.edit_schema(schema_id=schema_id, is_archived=True, session=session)

    @staticmethod
    def unarchive_schema(schema_id: int, session: Session) -> None:
        """Unarchive a schema to allow its use in new projects.
        
        Args:
            schema_id: The ID of the schema to unarchive
            session: Database session
            
        Raises:
            ValueError: If schema not found or has archived question groups
        """
        # Use the edit function with proper validation
        SchemaService.edit_schema(schema_id=schema_id, is_archived=False, session=session)

    @staticmethod
    def get_question_group_order(schema_id: int, session: Session) -> List[int]:
        """Get the ordered list of question group IDs in a schema.
        
        Args:
            schema_id: Schema ID
            session: Database session
            
        Returns:
            List of question group IDs in display order
            
        Raises:
            ValueError: If schema not found
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
            
        # Get all question groups in schema ordered by display_order
        assignments = session.scalars(
            select(SchemaQuestionGroup)
            .where(SchemaQuestionGroup.schema_id == schema_id)
            .order_by(SchemaQuestionGroup.display_order)
        ).all()
        
        return [a.question_group_id for a in assignments]

    @staticmethod
    def verify_update_question_group_order(schema_id: int, group_ids: List[int], session: Session) -> None:
        """Verify that a schema exists and that all question groups in the list exist in the schema.
        
        Args:
            schema_id: Schema ID
            group_ids: List of question group IDs in desired order
            session: Database session

        Raises:
            ValueError: If schema not found, or if any group not in schema
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        
        # Get all current assignments
        assignments = session.scalars(
            select(SchemaQuestionGroup)
            .where(SchemaQuestionGroup.schema_id == schema_id)
        ).all()
        
        # Create lookup for assignments
        assignment_map = {a.question_group_id: a for a in assignments}
        
        # Validate all groups exist in schema
        for group_id in group_ids:
            if group_id not in assignment_map:
                raise ValueError(f"Question group {group_id} not in schema {schema_id}")
        
        current_ids = set(assignment_map.keys())
        new_ids = set(group_ids)
        if current_ids != new_ids:
            missing = current_ids - new_ids
            extra = new_ids - current_ids
            error_msg = f"New group_ids must be a permutation of the current group IDs in schema {schema_id}"
            if missing:
                error_msg += f". Missing groups: {list(missing)}"
            if extra:
                error_msg += f". Extra groups: {list(extra)}"
            raise ValueError(error_msg)
            
    @staticmethod
    def update_question_group_order(schema_id: int, group_ids: List[int], session: Session) -> None:
        """Update the order of question groups in a schema.
        
        Args:
            schema_id: Schema ID
            group_ids: List of question group IDs in desired order
            session: Database session
            
        Raises:
            ValueError: If schema not found, or if any group not in schema
        """
        SchemaService.verify_update_question_group_order(schema_id=schema_id, group_ids=group_ids, session=session)


        assignments = session.scalars(
            select(SchemaQuestionGroup)
            .where(SchemaQuestionGroup.schema_id == schema_id)
        ).all()
        
        # Create lookup for assignments
        assignment_map = {a.question_group_id: a for a in assignments}
        
        # Update orders based on list position
        for i, group_id in enumerate(group_ids):
            assignment_map[group_id].display_order = i
            
        session.commit()

    @staticmethod
    def get_schema_by_name(name: str, session: Session) -> Schema:
        """Get a schema by its name.
        
        Args:
            name: Schema name
            session: Database session
            
        Returns:
            Schema object
            
        Raises:
            ValueError: If schema not found
        """
        schema = session.scalar(select(Schema).where(Schema.name == name))
        if not schema:
            raise ValueError("Schema not found")
        return schema

    @staticmethod
    def get_schema_by_id(schema_id: int, session: Session) -> Schema:
        """Get a schema by its ID.
        
        Args:
            schema_id: Schema ID
            session: Database session
            
        Returns:
            Schema object
            
        Raises:
            ValueError: If schema not found
        """
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        return schema

    
    @staticmethod
    def get_schema_question_groups(schema_id: int, session: Session) -> pd.DataFrame:
        """Get all question groups in a schema.
        
        Args:
            schema_id: The ID of the schema
            session: Database session
            
        Returns:
            DataFrame containing question groups with columns:
            - ID: Question group ID
            - Title: Question group title
            - Display Title: Question group display title
            - Description: Question group description
            - Reusable: Whether the group is reusable
            - Archived: Whether the group is archived
            - Display Order: Order in which the group appears in the schema
            - Question Count: Number of questions in the group
            
        Raises:
            ValueError: If schema not found
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
            
        # Get all question groups in schema ordered by display_order
        groups = session.scalars(
            select(QuestionGroup)
            .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == schema_id)
            .order_by(SchemaQuestionGroup.display_order)
        ).all()
        
        # Get question counts for each group
        rows = []
        for group in groups:
            # Count questions in this group
            question_count = session.scalar(
                select(func.count())
                .select_from(QuestionGroupQuestion)
                .where(QuestionGroupQuestion.question_group_id == group.id)
            )
            
            # Get display order
            display_order = session.scalar(
                select(SchemaQuestionGroup.display_order)
                .where(
                    SchemaQuestionGroup.schema_id == schema_id,
                    SchemaQuestionGroup.question_group_id == group.id
                )
            )
            
            rows.append({
                "ID": group.id,
                "Title": group.title,
                "Display Title": group.display_title,
                "Description": group.description,
                "Reusable": group.is_reusable,
                "Archived": group.is_archived,
                "Display Order": display_order,
                "Question Count": question_count
            })
            
        return pd.DataFrame(rows)
    
    @staticmethod
    def get_schema_question_groups_list(schema_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get question groups in a schema as a list of dictionaries.
        
        Args:
            schema_id: The ID of the schema
            session: Database session
            
        Returns:
            List of question group dictionaries containing: ID, Title, Description
            
        Raises:
            ValueError: If schema not found
        """
        # Check if schema exists
        schema = session.get(Schema, schema_id)
        if not schema:
            raise ValueError(f"Schema with ID {schema_id} not found")
        
        groups = session.scalars(
            select(QuestionGroup)
            .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == schema_id)
            .order_by(SchemaQuestionGroup.display_order)
        ).all()
        
        return [{
            "ID": g.id,
            "Title": g.title,
            "Display Title": g.display_title,
            "Description": g.description
        } for g in groups]

    # Add these new methods to SchemaService class in services.py
    @staticmethod
    def get_schema_counts(session: Session) -> Dict[str, int]:
        """Get schema counts without loading actual data."""
        total_schemas = session.scalar(select(func.count(Schema.id)))
        archived_schemas = session.scalar(select(func.count(Schema.id)).where(Schema.is_archived == True))
        active_schemas = total_schemas - archived_schemas
        schemas_with_custom_display = session.scalar(
            select(func.count(Schema.id)).where(
                Schema.has_custom_display == True,
                Schema.is_archived == False
            )
        )
        
        return {
            "total": total_schemas,
            "active": active_schemas,
            "archived": archived_schemas,
            "custom_display": schemas_with_custom_display
        }

    @staticmethod
    def search_schemas(search_term: str = "", show_archived: bool = False,
                    page: int = 0, page_size: int = 20, session: Session = None) -> Dict[str, Any]:
        """Search schemas with pagination and filters."""
        query = select(Schema)
        
        # Apply filters
        conditions = []
        if not show_archived:
            conditions.append(Schema.is_archived == False)
        
        if search_term:
            # We'll need to search in question groups - simplified for now
            conditions.append(Schema.name.ilike(f"%{search_term}%"))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_count = session.scalar(count_query)
        
        # Apply pagination
        query = query.offset(page * page_size).limit(page_size)
        schemas = session.scalars(query).all()
        
        # Process schemas
        schema_data = []
        for schema in schemas:
            # Get question groups for this schema
            groups = session.scalars(
                select(QuestionGroup)
                .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
                .where(SchemaQuestionGroup.schema_id == schema.id)
            ).all()
            
            schema_data.append({
                "ID": schema.id,
                "Name": schema.name,
                "Instructions URL": schema.instructions_url,
                "Question Groups": ", ".join(g.title for g in groups) if groups else "No groups",
                "Has Custom Display": schema.has_custom_display,
                "Archived": schema.is_archived
            })
        
        return {
            "schemas": pd.DataFrame(schema_data),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count - 1) // page_size + 1 if total_count > 0 else 1
        }
    
    @staticmethod
    def search_schemas_for_selection(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search schemas for selection dropdowns - returns limited results including archived."""
        query = select(Schema).where(
            Schema.name.ilike(f"%{search_term}%")
        ).limit(limit)
        
        schemas = session.scalars(query).all()
        return [{"id": s.id, "name": s.name, "archived": s.is_archived} for s in schemas]



class QuestionService:
    
    @staticmethod
    def get_all_questions(session: Session) -> pd.DataFrame:
        """Get all questions with their group information.
        
        Args:
            session: Database session
        
        Returns:
            DataFrame containing questions with columns:
            - ID: Question ID
            - Text: Question text (immutable, unique)
            - Display Text: UI display text
            - Type: Question type
            - Group: Question group name
            - Options: Available options for single-choice questions
            - Default: Default option for single-choice questions
            - Archived: Whether the question is archived
        """
        qs = session.scalars(select(Question)).all()
        return pd.DataFrame([
            {
                "ID": q.id, 
                "Text": q.text, 
                "Display Text": q.display_text,
                "Type": q.type,
                "Group": session.scalar(
                    select(QuestionGroup.title)
                    .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                    .where(QuestionGroupQuestion.question_id == q.id)
                ),
                "Options": ", ".join(q.options or []) if q.options else "",
                "Default": q.default_option or "",
                "Archived": q.is_archived
            } for q in qs
        ])
    
    @staticmethod
    def get_all_questions_with_custom_display(project_id: int, video_id: int, session: Session) -> pd.DataFrame:
        """Get all questions in a project's schema with custom display applied for a specific video.
        
        Args:
            project_id: Project ID 
            video_id: Video ID (for custom display context)
            session: Database session
            
        Returns:
            DataFrame containing all questions with custom display applied
            
        Raises:
            ValueError: If project not found
        """
        # Get project to find schema
        project = ProjectService.get_project_by_id(project_id=project_id, session=session)
        
        # Get all questions in project's schema
        qs = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(SchemaQuestionGroup.schema_id == project.schema_id)
        ).all()
        
        rows = []
        for q in qs:
            # Get custom display data
            question_with_custom = CustomDisplayService.get_custom_display(
                question_id=q.id,
                project_id=project_id,
                video_id=video_id,
                session=session
            )
            
            # Get group name
            group_name = session.scalar(
                select(QuestionGroup.title)
                .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                .where(QuestionGroupQuestion.question_id == q.id)
            )
            
            # Same format as original get_all_questions but with custom display
            rows.append({
                "ID": question_with_custom["id"], 
                "Text": question_with_custom["text"], 
                "Display Text": question_with_custom["display_text"],  # Custom display applied
                "Type": question_with_custom["type"],
                "Group": group_name,
                "Options": ", ".join(question_with_custom["options"] or []) if question_with_custom["options"] else "",
                "Display Values": ", ".join(question_with_custom["display_values"] or []) if question_with_custom["display_values"] else "",  # Custom display applied  
                "Default": question_with_custom["default_option"] or "",
                "Archived": question_with_custom["archived"]
            })
    
    @staticmethod
    def get_question_object_by_id(question_id: int, session: Session) -> Question:
        """Get a question object by its ID.
        
        Args:
            question_id: Question ID
            session: Database session
            
        Returns:
            Question object if found
            
        Raises:
            ValueError: If question not found
        """
        question = session.get(Question, question_id)
        if not question:
            raise ValueError(f"Question with ID {question_id} not found")
        return question

    @staticmethod
    def verify_add_question(text: str, qtype: str, options: Optional[List[str]], default: Optional[str], 
                    session: Session, display_values: Optional[List[str]] = None, display_text: Optional[str] = None,
                    option_weights: Optional[List[float]] = None) -> None:
        """Verify parameters for adding a new question.

        Args:
            text: Question text (immutable, unique)
            qtype: Question type ('single' or 'description')
            options: List of options for single-choice questions
            default: Default option/answer for single-choice/description questions
            session: Database session
            display_values: Optional list of display text for options. For single-type questions, if not provided, uses options as display values.
            display_text: Optional display text for UI. If not provided, uses text.
            option_weights: Optional list of weights for each option. If not provided, defaults to 1.0 for each option.

        Raises:
            ValueError: If validation fails
        """
        # Check if question text already exists
        existing = session.scalar(select(Question).where(Question.text == text))
        if existing:
            raise ValueError(f"Question with text '{text}' already exists")
        
        # Validate default option for single-choice questions
        if qtype == "single":
            if options is None:
                raise ValueError("Single-choice questions must have options")
            if default is not None and default not in options:
                raise ValueError(f"Default option '{default}' must be one of the available options: {', '.join(options)}")
            if len(options) != len(set(options)):
                raise ValueError("Options must be unique")
            
            # For single-type questions, display_values must be provided or default to options
            if display_values is not None:
                if len(display_values) != len(options):
                    raise ValueError("Number of display values must match number of options")
                if len(display_values) != len(set(display_values)):
                    raise ValueError("Display values must be unique")
                
            # Handle option weights
            if option_weights is not None:
                if len(option_weights) != len(options):
                    raise ValueError("Number of option weights must match number of options")
        else:
            if options is not None or display_values is not None or option_weights is not None:
                raise ValueError("Options, display values, and option weights are not allowed for description questions")

    @staticmethod
    def add_question(text: str, qtype: str, options: Optional[List[str]], default: Optional[str], 
                    session: Session, display_values: Optional[List[str]] = None, display_text: Optional[str] = None,
                    option_weights: Optional[List[float]] = None) -> Question:
        """Add a new question.
        
        Args:
            text: Question text (immutable, unique)
            qtype: Question type ('single' or 'description')
            options: List of options for single-choice questions
            default: Default option/answer for single-choice/description questions
            session: Database session
            display_values: Optional list of display text for options. For single-type questions, if not provided, uses options as display values.
            display_text: Optional display text for UI. If not provided, uses text.
            option_weights: Optional list of weights for each option. If not provided, defaults to 1.0 for each option.
        
        Returns:
            Created question
        
        Raises:
            ValueError: If question text already exists or validation fails
        """
        QuestionService.verify_add_question(text, qtype, options, default, session, display_values, display_text, option_weights)
        # Validate default option for single-choice questions
        if qtype == "single":
            # For single-type questions, display_values must be provided or default to options
            if display_values is None:
                display_values = options  # Use options as display values if not provided
                
            # Handle option weights
            if option_weights is None:
                option_weights = [1.0] * len(options)  # Default to 1.0 for each option
        else:
            # For description-type questions, display_values and option_weights should be None
            display_values = None
            option_weights = None
            options = None
        
        # Set display_text
        if display_text is None:
            display_text = text
        
        # Create question
        q = Question(
            text=text, 
            display_text=display_text,
            type=qtype, 
            options=options, 
            display_values=display_values,
            option_weights=option_weights,
            default_option=default
        )
        session.add(q)
        session.commit()
        return q

    @staticmethod
    def verify_edit_question(
            question_id: int,
            new_display_text: str,
            new_opts: Optional[List[str]],
            new_default: Optional[str],
            session: Session,
            new_display_values: Optional[List[str]] = None,
            new_option_weights: Optional[List[float]] = None
    ) -> None:
        """Verify parameters for editing an existing question.

        Args:
            question_id: Current question ID
            new_display_text: New display text for UI
            new_opts: New options for single-choice questions. Must include all existing options.
            new_default: New default option for single-choice questions
            session: Database session
            new_display_values: Optional new display values for options. For single-type questions, if not provided, maintains existing display values or uses options.
            new_option_weights: Optional new weights for options. For single-type questions, if not provided, maintains existing weights.

        Raises:
            ValueError: If question not found or validation fails
        """
        # Get question
        q = QuestionService.get_question_object_by_id(question_id=question_id, session=session)

        # Check if question is archived
        if q.is_archived:
            raise ValueError(f"Question with ID {question_id} is archived")

        # For single-choice questions, validate options and display values
        if q.type == "single":
            if not new_opts:
                raise ValueError("Cannot change question type")
            if new_default and new_default not in new_opts:
                raise ValueError(
                    f"Default option '{new_default}' must be one of the available options: {', '.join(new_opts)}")
            if len(new_opts) != len(set(new_opts)):
                raise ValueError("Options must be unique")

            # Validate that all existing options are included in new options
            missing_opts = set(q.options) - set(new_opts)
            if missing_opts:
                raise ValueError(f"Cannot remove existing options: {', '.join(missing_opts)}")

            # Validate display values
            if new_display_values:
                if len(new_display_values) != len(new_opts):
                    raise ValueError("Number of display values must match number of options")
                if len(new_display_values) != len(set(new_display_values)):
                    raise ValueError("Display values must be unique")

            # Validate option weights
            if new_option_weights:
                if len(new_option_weights) != len(new_opts):
                    raise ValueError("Number of option weights must match number of options")
        else:  # description type
            if new_opts is not None or new_display_values is not None or new_option_weights is not None:
                raise ValueError("Cannot change question type")

    @staticmethod
    def edit_question(question_id: int, new_display_text: str, new_opts: Optional[List[str]],
                      new_default: Optional[str],
                      session: Session, new_display_values: Optional[List[str]] = None,
                      new_option_weights: Optional[List[float]] = None) -> None:
        """Edit an existing question (only display_text and options, not text).

        Args:
            question_id: Current question ID
            new_display_text: New display text for UI.
            new_opts: New options for single-choice questions. Must include all existing options.
            new_default: New default option for single-choice questions
            session: Database session
            new_display_values: Optional new display values for options. For single-type questions, if not provided, maintains existing display values or uses options.
            new_option_weights: Optional new weights for options. For single-type questions, if not provided, maintains existing weights or defaults to 1.0.

        Raises:
            ValueError: If question not found or validation fails
        """
        # Verify parameters (raises ValueError if invalid)
        QuestionService.verify_edit_question(
            question_id, new_display_text, new_opts, new_default, session, new_display_values, new_option_weights
        )

        # Get question again after validation
        q = QuestionService.get_question_object_by_id(question_id=question_id, session=session)

        # Process defaults after validation passes
        if q.type == "single":
            # For single-type questions, ensure we have display values
            if new_display_values is None:
                # If no new display values provided, maintain existing mapping for unchanged options
                new_display_values = []
                for opt in new_opts:
                    if opt in q.options:
                        idx = q.options.index(opt)
                        new_display_values.append(q.display_values[idx])
                    else:
                        new_display_values.append(opt)

            # Handle option weights
            if new_option_weights is None:
                # If no new weights provided, maintain existing weights for unchanged options
                new_option_weights = []
                for opt in new_opts:
                    if opt in q.options:
                        idx = q.options.index(opt)
                        new_option_weights.append(q.option_weights[idx])
                    else:
                        new_option_weights.append(1.0)  # Default to 1.0 for new options
        # Update only display_text, options, display_values, option_weights, default_option
        q.display_text = new_display_text
        q.options = new_opts
        q.display_values = new_display_values
        q.option_weights = new_option_weights
        q.default_option = new_default
        session.commit()

    @staticmethod
    def archive_question(question_id: int, session: Session) -> None:
        """Archive a question.
        
        Args:
            question_id: Question ID
            session: Database session
            
        Raises:
            ValueError: If question not found
        """
        q = session.get(Question, question_id)
        if not q:
            raise ValueError(f"Question with ID {question_id} not found")
        q.is_archived = True
        session.commit()

    @staticmethod
    def unarchive_question(question_id: int, session: Session) -> None:
        """Unarchive a question.
        
        Args:
            question_id: Question ID
            session: Database session
            
        Raises:
            ValueError: If question not found
        """
        q = session.get(Question, question_id)
        if not q:
            raise ValueError(f"Question with ID {question_id} not found")
        q.is_archived = False
        session.commit()

    @staticmethod
    def get_question_by_text(text: str, session: Session) -> Dict[str, Any]:
        """Get a question by its text.
        
        Args:
            text: Question text
            session: Database session
            
        Returns:
            Dictionary with question data if found
            
        Raises:
            ValueError: If question not found
        """
        question = session.scalar(select(Question).where(Question.text == text))
        if not question:
            raise ValueError(f"Question with text '{text}' not found")
        
        return {
            "id": question.id,
            "text": question.text,
            "display_text": question.display_text,
            "type": question.type,
            "options": question.options,
            "display_values": question.display_values,
            "default_option": question.default_option,
            "option_weights": question.option_weights,
            "created_at": question.created_at,
            "archived": question.is_archived
        }

    @staticmethod
    def get_question_by_text_with_custom_display(text: str, project_id: int, video_id: int, session: Session) -> Dict[str, Any]:
        """Get a question by its text with custom display applied for a specific project-video combination.
        
        Args:
            text: Question text (immutable text field)
            project_id: Project ID (for custom display context)
            video_id: Video ID (for custom display context)
            session: Database session
            
        Returns:
            Dictionary with question data including custom display (same format as original)
            
        Raises:
            ValueError: If question not found
        """
        # First get the question by text to find its ID
        original_question = QuestionService.get_question_by_text(text=text, session=session)
        
        # Then get it with custom display applied
        return CustomDisplayService.get_custom_display(
            question_id=original_question["id"],
            project_id=project_id,
            video_id=video_id,
            session=session
        )

    @staticmethod
    def get_question_by_id(question_id: int, session: Session) -> Dict[str, Any]:
        """Get a question by its ID.
        
        Args:
            question_id: Question ID
            session: Database session
            
        Returns:
            Dictionary with question data if found
            
        Raises:
            ValueError: If question not found
        """
        question = session.get(Question, question_id)
        if not question:
            raise ValueError(f"Question with ID {question_id} not found")
        
        return {
            "id": question.id,
            "text": question.text,
            "display_text": question.display_text,
            "type": question.type,
            "options": question.options,
            "display_values": question.display_values,
            "default_option": question.default_option,
            "option_weights": question.option_weights,
            "created_at": question.created_at,
            "archived": question.is_archived
        }
    
    @staticmethod 
    def get_question_by_id_with_custom_display(question_id: int, project_id: int, video_id: int, session: Session) -> Dict[str, Any]:
        """Get a question by its ID with custom display applied for a specific project-video combination.
        
        Args:
            question_id: Question ID
            project_id: Project ID (for custom display context)
            video_id: Video ID (for custom display context)
            session: Database session
            
        Returns:
            Dictionary with question data including custom display (same format as original)
            
        Raises:
            ValueError: If question not found
        """
        # This directly uses our custom display service which already returns the right format
        return CustomDisplayService.get_custom_display(
            question_id=question_id,
            project_id=project_id,
            video_id=video_id,
            session=session
        )
    
    @staticmethod
    def get_questions_by_group_id(group_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all questions in a group as dictionaries.
        
        Args:
            group_id: The ID of the question group
            session: Database session
        
        Returns:
            List of question dictionaries containing: id, text, display_text, type, options, display_values, default_option
        
        Raises:
            ValueError: If group not found
        """
        # Check if group exists
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == group_id)
            .order_by(QuestionGroupQuestion.display_order)
        ).all()
        
        return [{
            "id": q.id,
            "text": q.text,
            "display_text": q.display_text,
            "type": q.type,
            "options": q.options,
            "option_weights": q.option_weights,
            "display_values": q.display_values,
            "default_option": q.default_option
        } for q in questions]
    
    @staticmethod
    def get_questions_by_group_id_with_custom_display(group_id: int, project_id: int, video_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all questions in a group with custom display applied for a specific project-video combination.
        
        Args:
            group_id: The ID of the question group
            project_id: The ID of the project (for custom display context)
            video_id: The ID of the video (for custom display context)
            session: Database session
        
        Returns:
            List of question dictionaries with custom display applied (same format as original)
        
        Raises:
            ValueError: If group not found
        """
        # Get original questions
        original_questions = QuestionService.get_questions_by_group_id(group_id=group_id, session=session)
        
        # Apply custom display to each question
        result = []
        for q in original_questions:
            # Get question with custom display applied
            question_with_custom = CustomDisplayService.get_custom_display(
                question_id=q['id'],
                project_id=project_id,
                video_id=video_id,
                session=session
            )
            
            # Return same format as original but with custom display
            result.append({
                "id": question_with_custom["id"],
                "text": question_with_custom["text"],
                "display_text": question_with_custom["display_text"],  # Custom display applied
                "type": question_with_custom["type"],
                "options": question_with_custom["options"],
                "option_weights": question_with_custom["option_weights"],
                "display_values": question_with_custom["display_values"],  # Custom display applied
                "default_option": question_with_custom["default_option"]
            })
        
        return result

    # Add these new methods to QuestionService class
    @staticmethod
    def get_question_counts(session: Session) -> Dict[str, int]:
        """Get question counts without loading actual data."""
        total_questions = session.scalar(select(func.count(Question.id)))
        archived_questions = session.scalar(select(func.count(Question.id)).where(Question.is_archived == True))
        active_questions = total_questions - archived_questions
        single_choice = session.scalar(select(func.count(Question.id)).where(Question.type == "single", Question.is_archived == False))
        description_type = session.scalar(select(func.count(Question.id)).where(Question.type == "description", Question.is_archived == False))
        
        return {
            "total": total_questions,
            "active": active_questions,
            "archived": archived_questions,
            "single_choice": single_choice,
            "description": description_type
        }

    @staticmethod
    def search_questions(search_term: str = "", show_archived: bool = False,
                        page: int = 0, page_size: int = 20, session: Session = None) -> Dict[str, Any]:
        """Search questions with pagination and filters."""
        query = select(Question)
        
        # Apply filters
        conditions = []
        if not show_archived:
            conditions.append(Question.is_archived == False)
        
        if search_term:
            search_filter = or_(
                Question.text.ilike(f"%{search_term}%"),
                Question.display_text.ilike(f"%{search_term}%")
            )
            conditions.append(search_filter)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_count = session.scalar(count_query)
        
        # Apply pagination
        query = query.offset(page * page_size).limit(page_size)
        questions = session.scalars(query).all()
        
        # Process questions
        question_data = []
        for q in questions:
            # Get group for this question
            group_title = session.scalar(
                select(QuestionGroup.title)
                .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                .where(QuestionGroupQuestion.question_id == q.id)
            )
            
            question_data.append({
                "ID": q.id,
                "Text": q.text,
                "Display Text": q.display_text,
                "Type": q.type,
                "Group": group_title or "No group",
                "Options": ", ".join(q.options or []) if q.options else "",
                "Default": q.default_option or "",
                "Archived": q.is_archived
            })
        
        return {
            "questions": pd.DataFrame(question_data),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count - 1) // page_size + 1 if total_count > 0 else 1
        }

    @staticmethod
    def search_questions_for_selection(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search questions for selection dropdowns - returns limited results including archived."""
        query = select(Question).where(
            or_(
                Question.text.ilike(f"%{search_term}%"),
                Question.display_text.ilike(f"%{search_term}%")
            )
        ).limit(limit)
        
        questions = session.scalars(query).all()
        return [{"id": q.id, "text": q.text, "display_text": q.display_text, "type": q.type, "archived": q.is_archived} for q in questions]

class CustomDisplayService:
    @staticmethod
    def _validate_project_video_question_relationship(project_id: int, video_id: int, question_id: int, session: Session) -> None:
        """Validate that video and question are both in the specified project.
        
        Args:
            project_id: Project ID
            video_id: Video ID  
            question_id: Question ID
            session: Database session
            
        Raises:
            ValueError: If validation fails
        """
        # Check if video is in project
        project_video = session.scalar(
            select(ProjectVideo).where(
                ProjectVideo.project_id == project_id,
                ProjectVideo.video_id == video_id
            )
        )
        if not project_video:
            raise ValueError(f"Video {video_id} is not assigned to project {project_id}")
        
        # Check if question is in project's schema
        project = session.get(Project, project_id)
        question_in_schema = session.scalar(
            select(func.count())
            .select_from(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(
                SchemaQuestionGroup.schema_id == project.schema_id,
                Question.id == question_id
            )
        )
        if not question_in_schema:
            raise ValueError(f"Question {question_id} is not in project {project_id}'s schema")

    @staticmethod
    def verify_set_custom_display(
        project_id: int,
        video_id: int, 
        question_id: int,
        custom_display_text: Optional[str] = None,
        custom_option_display_map: Optional[Dict[str, str]] = None,
        session: Session = None
    ) -> None:
        """Verify parameters for setting custom display.
        
        Args:
            project_id: Project ID
            video_id: Video ID
            question_id: Question ID
            custom_display_text: Custom display text for question
            custom_option_display_map: Dict mapping option values to custom display text
            session: Database session
            
        Raises:
            ValueError: If validation fails
        """
        # Validate project exists and is not archived
        project = ProjectService.get_project_by_id(project_id=project_id, session=session)
        if project.is_archived:
            raise ValueError(f"Project {project_id} is archived")
        
        # Validate video exists and is not archived
        video = session.get(Video, video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found")
        if video.is_archived:
            raise ValueError(f"Video {video_id} is archived")
        
        # Validate question exists and is not archived
        question_dict = QuestionService.get_question_by_id(question_id=question_id, session=session)
        if question_dict["archived"]:
            raise ValueError(f"Question {question_id} is archived")
        
        reusable_group_count = session.scalar(
            select(func.count())
            .select_from(QuestionGroup)
            .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
            .where(
                QuestionGroupQuestion.question_id == question_id,
                QuestionGroup.is_reusable == True,
                QuestionGroup.is_archived == False
            )
        )

        if reusable_group_count > 0:
            # Get the reusable group names for a better error message
            reusable_groups = session.scalars(
                select(QuestionGroup.title)
                .join(QuestionGroupQuestion, QuestionGroup.id == QuestionGroupQuestion.question_group_id)
                .where(
                    QuestionGroupQuestion.question_id == question_id,
                    QuestionGroup.is_reusable == True,
                    QuestionGroup.is_archived == False
                )
            ).all()
            
            group_names = ", ".join(reusable_groups)
            raise ValueError(
                f"Cannot set custom display for question {question_id}. "
                f"Question belongs to reusable question group(s): {group_names}. "
                f"Reusable groups must maintain consistent display across all schemas."
            )
        
        # Validate project-video-question relationship
        CustomDisplayService._validate_project_video_question_relationship(
            project_id=project_id, video_id=video_id, question_id=question_id, session=session
        )
        
        # Check that schema has custom display enabled
        schema = session.get(Schema, project.schema_id)
        if not schema.has_custom_display:
            raise ValueError(f"Schema {schema.id} does not have custom display enabled")
        
        # For single-choice questions with custom option mapping
        if custom_option_display_map is not None:
            if question_dict["type"] != "single":
                raise ValueError(f"Custom option display mapping can only be set for single-choice questions, not {question_dict['type']}")
            
            if not question_dict["options"]:
                raise ValueError(f"Question {question_id} has no options defined")
            
            # Check that all keys in the mapping are valid option values
            valid_options = set(question_dict["options"])
            invalid_options = set(custom_option_display_map.keys()) - valid_options
            if invalid_options:
                raise ValueError(f"Invalid option values in custom mapping: {invalid_options}. Valid options: {valid_options}")

    @staticmethod
    def set_custom_display(
        project_id: int,
        video_id: int, 
        question_id: int,
        custom_display_text: Optional[str] = None,
        custom_option_display_map: Optional[Dict[str, str]] = None,
        session: Session = None
    ) -> None:
        """Set custom display for a specific project-video-question combination.
        
        Args:
            project_id: Project ID
            video_id: Video ID
            question_id: Question ID
            custom_display_text: Custom display text for question
            custom_option_display_map: Dict mapping option values to custom display text
            session: Database session
            
        Raises:
            ValueError: If validation fails
        """
        # Verify all parameters
        CustomDisplayService.verify_set_custom_display(
            project_id=project_id,
            video_id=video_id,
            question_id=question_id,
            custom_display_text=custom_display_text,
            custom_option_display_map=custom_option_display_map,
            session=session
        )
        
        # Get or create custom display entry
        custom_display = session.get(
            ProjectVideoQuestionDisplay,
            (project_id, video_id, question_id)
        )
        
        if custom_display:
            # Update existing entry
            if custom_display_text is not None:
                custom_display.custom_display_text = custom_display_text
            if custom_option_display_map is not None:
                custom_display.custom_option_display_map = custom_option_display_map
            custom_display.updated_at = datetime.now(timezone.utc)
        else:
            # Create new entry
            custom_display = ProjectVideoQuestionDisplay(
                project_id=project_id,
                video_id=video_id,
                question_id=question_id,
                custom_display_text=custom_display_text,
                custom_option_display_map=custom_option_display_map
            )
            session.add(custom_display)
        
        session.commit()

    @staticmethod
    def get_custom_display(
        question_id: int, 
        project_id: int, 
        video_id: int, 
        session: Session
    ) -> Dict[str, Any]:
        """Get question display data with custom display if applicable.
        
        Args:
            question_id: Question ID
            project_id: Project ID
            video_id: Video ID
            session: Database session
            
        Returns:
            Dictionary with question data including custom display if applicable
            
        Raises:
            ValueError: If question, project, or video not found
        """
        # Get the original question
        question_dict = QuestionService.get_question_by_id(question_id=question_id, session=session)
        
        # Get the project to check if custom display is enabled
        project = ProjectService.get_project_by_id(project_id=project_id, session=session)
        
        # Get the schema to check custom display flag
        schema = session.get(Schema, project.schema_id)
        if not schema or not schema.has_custom_display:
            # No custom display enabled - return original data
            return question_dict
        
        # Try to get custom display entries
        custom_display = session.get(
            ProjectVideoQuestionDisplay, 
            (project_id, video_id, question_id)
        )
        
        # Start with original question data
        result = question_dict.copy()
        
        if custom_display:
            # Override display text if available
            if custom_display.custom_display_text:
                result["display_text"] = custom_display.custom_display_text
            
            # Override option display values if available
            if (custom_display.custom_option_display_map and 
                question_dict["type"] == "single" and 
                question_dict["options"]):
                
                # Build new display_values array by mapping original options
                new_display_values = []
                for option_value in question_dict["options"]:
                    custom_display_text = custom_display.custom_option_display_map.get(option_value)
                    if custom_display_text:
                        new_display_values.append(custom_display_text)
                    else:
                        # Fall back to original display value
                        original_index = question_dict["options"].index(option_value)
                        original_display = (question_dict["display_values"][original_index] 
                                          if question_dict["display_values"] else option_value)
                        new_display_values.append(original_display)
                
                result["display_values"] = new_display_values
        
        return result

    @staticmethod
    def remove_custom_display(
        project_id: int,
        video_id: int,
        question_id: int,
        session: Session
    ) -> bool:
        """Remove custom display for a specific project-video-question combination.
        
        Args:
            project_id: Project ID
            video_id: Video ID
            question_id: Question ID
            session: Database session
            
        Returns:
            True if overrides were removed, False if none existed
        """
        custom_display = session.get(
            ProjectVideoQuestionDisplay,
            (project_id, video_id, question_id)
        )
        
        if custom_display:
            session.delete(custom_display)
            session.commit()
            return True
        
        return False

    @staticmethod
    def get_all_custom_displays_for_video(
        project_id: int,
        video_id: int,
        session: Session
    ) -> List[Dict[str, Any]]:
        """Get all questions that have custom display for a specific project-video combination.
        
        Args:
            project_id: Project ID
            video_id: Video ID
            session: Database session
            
        Returns:
            List of dictionaries with custom display information
        """
        overrides = session.scalars(
            select(ProjectVideoQuestionDisplay)
            .where(
                ProjectVideoQuestionDisplay.project_id == project_id,
                ProjectVideoQuestionDisplay.video_id == video_id
            )
        ).all()
        
        result = []
        for override in overrides:
            question_dict = QuestionService.get_question_by_id(question_id=override.question_id, session=session)
            result.append({
                "question_id": override.question_id,
                "question_text": question_dict["text"],
                "has_custom_text": override.custom_display_text is not None,
                "has_custom_options": override.custom_option_display_map is not None,
                "custom_display_text": override.custom_display_text,
                "custom_option_display_map": override.custom_option_display_map
            })
        
        return result

    @staticmethod
    def get_all_custom_displays_for_project(
        project_id: int,
        session: Session
    ) -> List[Dict[str, Any]]:
        """Get all custom display entries for a project.
        
        Args:
            project_id: Project ID
            session: Database session
            
        Returns:
            List of dictionaries with all custom display overrides
        """
        # Check if the schema has custom display enabled
        # get schema from project
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        schema = session.get(Schema, project.schema_id)
        if not schema or not schema.has_custom_display:
            return []
        
        overrides = session.scalars(
            select(ProjectVideoQuestionDisplay)
            .where(ProjectVideoQuestionDisplay.project_id == project_id)
        ).all()
        
        result = []
        for override in overrides:
            video = session.get(Video, override.video_id)
            question_dict = QuestionService.get_question_by_id(question_id=override.question_id, session=session)
            
            result.append({
                "project_id": override.project_id,
                "video_id": override.video_id,
                "video_uid": video.video_uid if video else None,
                "question_id": override.question_id,
                "question_text": question_dict["text"],
                "custom_display_text": override.custom_display_text,
                "custom_option_display_map": override.custom_option_display_map,
                "created_at": override.created_at,
                "updated_at": override.updated_at
            })
        
        return result


class AuthService:
    @staticmethod
    def get_user_display_name_with_initials(user_name: str) -> tuple[str, str]:
        """Get user display name with initials.
        
        Args:
            user_name: The user's name/ID string
            
        Returns:
            Tuple of (display_name_with_initials, initials_only)
            
        Example:
            ("alice_annotator (AL)", "AL")
        """
        name_parts = user_name.split()
        if len(name_parts) >= 2:
            initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
        else:
            initials = user_name[:2].upper()
        
        display_name = f"{user_name} ({initials})"
        return display_name, initials

    @staticmethod
    def get_user_info_by_id(user_id: int, session: Session) -> Dict[str, Any]:
        """Get user information by numeric ID as a dictionary.
        
        Args:
            user_id: The numeric ID of the user
            session: Database session
            
        Returns:
            Dictionary containing user info: id, user_id_str, email, user_type, is_archived
            
        Raises:
            ValueError: If user not found
        """
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        return {
            "id": user.id,
            "user_id_str": user.user_id_str,
            "email": user.email,
            "user_type": user.user_type,
            "is_archived": user.is_archived,
            "created_at": user.created_at
        }
    
    @staticmethod
    def get_user_by_id(user_id: str, session: Session) -> Optional[User]:
        """Get a user by their ID string.
        
        Args:
            user_id: The ID string of the user
            session: Database session
            
        Returns:
            User object if found, raises ValueError otherwise
        """
        user = session.scalar(select(User).where(User.user_id_str == user_id))
        if not user:
            raise ValueError(f"User with ID '{user_id}' not found")
        return user
    
    @staticmethod
    def get_user_by_name(user_name: str, session: Session) -> Optional[User]:
        """Get a user by their name.
        
        Args:
            user_name: The name of the user
            session: Database session
            
        Returns:
            User object if found, raises ValueError otherwise
        """
        user = session.scalar(select(User).where(User.user_id_str == user_name))
        if not user:
            raise ValueError(f"User with name '{user_name}' not found")
        return user

    @staticmethod
    def get_user_by_email(email: str, session: Session) -> Optional[User]:
        """Get a user by their email.
        
        Args:
            email: The email of the user
            session: Database session
            
        Returns:
            User object if found, raises ValueError otherwise
            
        Raises:
            ValueError: If user not found
        """
        if not email:
            raise ValueError("Email is required")
        user = session.scalar(select(User).where(User.email == email))
        if not user:
            raise ValueError(f"User with email '{email}' not found")
        return user

    @staticmethod
    def authenticate(email: str, pwd: str, role: str, session: Session) -> Optional[dict]:
        """Authenticate a user.
        
        Args:
            email: User's email
            pwd: User's password
            role: Required role
            session: Database session
            
        Returns:
            Dictionary containing user info if authenticated, None otherwise
        """
        u = session.scalar(select(User).where(
            User.email == email, 
            User.password_hash == pwd, 
            User.is_archived == False
        ))
        if not u:
            return None
        if role != "admin" and u.user_type != role:
            return None
        return {"id": u.id, "name": u.user_id_str, "role": u.user_type}

    @staticmethod
    def seed_admin(session: Session, email: str = "admin@example.com", password: str = "password123", user_id: str = "Admin") -> None:
        """Create admin user with specified credentials if not present.
        
        Args:
            session: Database session
            email: Admin email address
            password: Admin password  
            user_id: Admin user ID string
        """
        try:
            # Use create_user which handles all validation and duplicate checking
            AuthService.create_user(
                user_id=user_id,
                email=email,
                password_hash=password,
                user_type="admin",
                session=session
            )
        except ValueError:
            # User already exists, that's fine for seeding
            pass

    @staticmethod
    def get_all_users(session: Session) -> pd.DataFrame:
        """Get all users in a DataFrame format."""
        users = session.scalars(select(User)).all()
        return pd.DataFrame([
            {
                "ID": u.id,
                "User ID": u.user_id_str,
                "Email": u.email,
                "Password Hash": u.password_hash,
                "Role": u.user_type,
                "Archived": u.is_archived,
                "Created At": u.created_at
            } for u in users
        ])

    @staticmethod
    def get_users_by_type(user_type: str, session: Session) -> List[User]:
        """Get all users of a specific type.
        
        Args:
            user_type: The type of users to get ('human', 'model', or 'admin')
            session: Database session
            
        Returns:
            List of User objects of the specified type
            
        Raises:
            ValueError: If user_type is invalid
        """
        if user_type not in ["human", "model", "admin"]:
            raise ValueError(f"Invalid user type: {user_type}")
            
        return session.scalars(
            select(User).where(User.user_type == user_type)
        ).all()

    @staticmethod
    def verify_update_user_id(user_id: int, new_user_id: str, session: Session) -> None:
        """Verify parameters for updating a user's ID."""
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Check if new user ID already exists
        existing = session.scalar(
            select(User).where(User.user_id_str == new_user_id)
        )
        if existing and existing.id != user_id:
            raise ValueError(f"User ID '{new_user_id}' already exists")
        
        
    @staticmethod
    def update_user_id(user_id: int, new_user_id: str, session: Session) -> None:
        """Update a user's ID."""
        AuthService.verify_update_user_id(user_id, new_user_id, session)

        user = session.get(User, user_id)
        user.user_id_str = new_user_id
        session.commit()

    @staticmethod
    def verify_update_user_email(user_id: int, new_email: str, session: Session) -> None:
        """Verify parameters for updating a user's email."""
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Model users cannot have emails
        if user.user_type == "model":
            raise ValueError("Model users cannot have emails")
        
        # Email is required for human and admin users
        if not new_email:
            raise ValueError("Email is required for human and admin users")
        
        # Check if new email already exists
        existing = session.scalar(
            select(User).where(User.email == new_email)
        )
        if existing and existing.id != user_id:
            raise ValueError(f"Email '{new_email}' already exists")

    @staticmethod
    def update_user_email(user_id: int, new_email: str, session: Session) -> None:
        """Update a user's email."""
        AuthService.verify_update_user_email(user_id, new_email, session)
        user = session.get(User, user_id)    
        user.email = new_email
        session.commit()

    @staticmethod
    def verify_update_user_password(user_id: int, new_password: str, session: Session) -> None:
        """Verify parameters for updating a user's password."""
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

    @staticmethod
    def update_user_password(user_id: int, new_password: str, session: Session) -> None:
        """Update a user's password."""
        AuthService.verify_update_user_password(user_id, new_password, session)
        user = session.get(User, user_id)
        user.password_hash = new_password  # Note: In production, this should be hashed
        session.commit()

    @staticmethod
    def verify_update_user_role(user_id: int, new_role: str, session: Session) -> None:
        """Verify parameters for updating a user's role."""
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        if new_role not in ["human", "model", "admin"]:
            raise ValueError(f"Invalid role: {new_role}")

        # Cannot change from human/admin to model
        if user.user_type == "human" or user.user_type == "admin":
            if new_role == "model":
                raise ValueError("Cannot change from human/admin to model")

        # Cannot change from model to human/admin
        if user.user_type == "model":
            if new_role == "human" or new_role == "admin":
                raise ValueError("Cannot change from model to human/admin")

    @staticmethod
    def update_user_role(user_id: int, new_role: str, session: Session) -> None:
        """Update a user's role and handle admin project assignments."""
        AuthService.verify_update_user_role(user_id, new_role, session)

        user = session.get(User, user_id)
        # If changing from admin to human, remove all project roles
        if user.user_type == "admin" and new_role == "human":
            # Get all non-archived project assignments for this user
            assignments = session.scalars(
                select(ProjectUserRole)
                .where(
                    ProjectUserRole.user_id == user_id,
                    ProjectUserRole.is_archived == False
                )
            ).all()
            
            # Archive each assignment
            for assignment in assignments:
                assignment.is_archived = True
        
        # If changing to admin role, assign to all projects
        if new_role == "admin" and user.user_type != "admin":
            user.user_type = new_role
            session.commit()
            # Get all non-archived projects
            projects = session.scalars(
                select(Project).where(Project.is_archived == False)
            ).all()
            
            # Assign user as admin to each project using KEYWORD ARGUMENTS
            for project in projects:
                ProjectService.add_user_to_project(
                    project_id=project.id, 
                    user_id=user_id, 
                    role="admin", 
                    session=session
                )
        else:
            user.user_type = new_role
            session.commit()
    
    @staticmethod
    def get_user_weights_for_project(project_id: int, session: Session) -> Dict[int, float]:
        """Get user weights for a specific project.
        
        Args:
            project_id: Project ID
            session: Database session
            
        Returns:
            Dictionary mapping user_id to user_weight
            
        Raises:
            ValueError: If error getting user weights
        """
        try:
            assignments_df = AuthService.get_project_assignments(session=session)
            project_assignments = assignments_df[assignments_df["Project ID"] == project_id]
            
            user_weights = {}
            for _, assignment in project_assignments.iterrows():
                user_id = assignment["User ID"]
                weight = assignment.get("User Weight", 1.0)
                user_weights[user_id] = weight
            
            return user_weights
        except Exception as e:
            raise ValueError(f"Error getting user weights for project: {str(e)}")


    @staticmethod
    def toggle_user_archived(user_id: int, session: Session) -> None:
        """Toggle a user's archived status."""
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        user.is_archived = not user.is_archived
        session.commit()

    @staticmethod
    def get_project_assignments(session: Session) -> pd.DataFrame:
        """Get all project assignments in a DataFrame format."""
        
        # Single query with joins instead of individual gets
        query = select(
            ProjectUserRole.project_id,
            Project.name.label('project_name'),
            ProjectUserRole.user_id,
            User.user_id_str.label('user_name'),
            ProjectUserRole.role,
            ProjectUserRole.is_archived,
            ProjectUserRole.assigned_at,
            ProjectUserRole.completed_at,
            ProjectUserRole.user_weight
        ).select_from(
            ProjectUserRole
        ).join(
            Project, ProjectUserRole.project_id == Project.id
        ).join(
            User, ProjectUserRole.user_id == User.id
        ).where(
            ProjectUserRole.is_archived == False
        )
        
        result = session.execute(query).all()
        
        return pd.DataFrame([
            {
                "Project ID": row.project_id,
                "Project Name": row.project_name,
                "User ID": row.user_id,
                "User Name": row.user_name,
                "Role": row.role,
                "Archived": row.is_archived,
                "Assigned At": row.assigned_at,
                "Completed At": row.completed_at,
                "User Weight": row.user_weight
            }
            for row in result
        ])

    @staticmethod
    def verify_create_user(user_id: str, email: str, password_hash: str, user_type: str, session: Session, is_archived: bool = False) -> None:
        """Verify parameters for creating a new user.

        Args:
            user_id: The unique identifier for the user
            email: User's email address (required for human/admin, None for model)
            password_hash: Hashed password
            user_type: Type of user (human, model, admin)
            session: Database session
            is_archived: Whether the user should be archived (default: False)

        Raises:
            ValueError: If user_type is invalid, email validation fails, or user already exists
        """
        # Validate user type
        if user_type not in ["human", "model", "admin"]:
            raise ValueError("Invalid user type. Must be one of: human, model, admin")

        # Validate email requirements based on user type
        if user_type == "model":
            if email:
                raise ValueError("Model users cannot have emails")
        elif not email:
            raise ValueError("Email is required for human and admin users")

        # Check if user already exists - handle model users differently
        if user_type == "model":
            # For model users, only check user_id_str since all model users have email=None
            existing_user = session.scalar(
                select(User).where(User.user_id_str == user_id)
            )
        else:
            # For human and admin users, check both user_id_str and email
            existing_user = session.scalar(
                select(User).where(
                    (User.user_id_str == user_id) | 
                    (User.email == email)
                )
            )
        
        if existing_user:
            if user_type == "model":
                raise ValueError(f"Model user with ID '{user_id}' already exists")
            else:
                raise ValueError(f"User with ID '{user_id}' or email '{email}' already exists")

    @staticmethod
    def bulk_assign_user_to_projects(user_id: int, project_ids: List[int], role: str, session: Session, user_weight: float = 1.0) -> None:
        """Assign a single user to multiple projects with the specified role.
        
        Args:
            user_id: The ID of the user
            project_ids: List of project IDs to assign the user to
            role: The role to assign ('annotator', 'reviewer', 'admin', or 'model')
            session: Database session
            user_weight: Weight for the user's answers (defaults to 1.0)
            
        Raises:
            ValueError: If role is invalid
        """
        if role not in ["annotator", "reviewer", "admin", "model"]:
            raise ValueError(f"Invalid role: {role}")
        
        if not project_ids:
            return
        
        # Determine which roles to assign based on the requested role
        roles_to_assign = []
        if role == "annotator":
            roles_to_assign = ["annotator"]
        elif role == "reviewer":
            roles_to_assign = ["annotator", "reviewer"]
        elif role == "model":
            roles_to_assign = ["model"]
        elif role == "admin":
            roles_to_assign = ["annotator", "reviewer", "admin"]
        
        # Create all role assignments in bulk
        role_assignments = []
        for project_id in project_ids:
            for role_name in roles_to_assign:
                role_assignments.append(ProjectUserRole(
                    project_id=project_id,
                    user_id=user_id,
                    role=role_name,
                    user_weight=user_weight,
                    is_archived=False
                ))
        
        # Bulk insert all assignments
        session.add_all(role_assignments)

    @staticmethod
    def create_user(user_id: str, email: str, password_hash: str, user_type: str, session: Session, is_archived: bool = False) -> User:
        """Create a new user with validation."""
        # Verify input parameters
        AuthService.verify_create_user(user_id, email, password_hash, user_type, session, is_archived)
        
        # Create user
        user = User(
            user_id_str=user_id,
            email=email,
            password_hash=password_hash,
            user_type=user_type,
            is_archived=is_archived
        )
        session.add(user)
        session.flush()  # Get user ID
        
        # If user is admin, assign to all existing projects
        if user_type == "admin" and not is_archived:
            project_ids = session.scalars(
                select(Project.id).where(Project.is_archived == False)
            ).all()
            
            AuthService.bulk_assign_user_to_projects(
                user_id=user.id,
                project_ids=project_ids,
                role="admin",
                session=session
            )
        
        session.commit()
        return user

    # @staticmethod
    # def verify_assign_user_to_project(user_id: int, project_id: int, role: str, session: Session) -> None:
    #     """Verify that a user can be assigned to a project with the specified role.
        
    #     Args:
    #         user_id: The ID of the user
    #         project_id: The ID of the project
    #         role: The role to assign ('annotator', 'reviewer', 'admin', or 'model')
    #         session: Database session

    #     Raises:
    #         ValueError: If role is invalid
    #     """
    #     if role not in ["annotator", "reviewer", "admin", "model"]:
    #         raise ValueError("Invalid role. Must be one of: annotator, reviewer, admin, model")
        
    #     # Get user and project
    #     user = session.get(User, user_id)
    #     if not user:
    #         raise ValueError(f"User with ID {user_id} not found")
    #     if user.is_archived:
    #         raise ValueError(f"User '{user.user_id_str}' is archived")  # Use name instead of ID
        
    #     project = session.get(Project, project_id)
    #     if not project:
    #         raise ValueError(f"Project with ID {project_id} not found")
    #     if project.is_archived:
    #         raise ValueError(f"Project '{project.name}' is archived")  # Use name instead of ID
        
    #     if user.user_type == "admin" and role != "admin":
    #         raise ValueError(f"Admin user '{user.user_id_str}' cannot be assigned to projects with roles other than admin")
        
    
    # @staticmethod
    # def assign_user_to_project(user_id: int, project_id: int, role: str, session: Session, user_weight: Optional[float] = None) -> None:
    #     """Assign a user to a project with a specific role.
        
    #     Args:
    #         user_id: The ID of the user
    #         project_id: The ID of the project
    #         role: The role to assign ('annotator', 'reviewer', 'admin', or 'model')
    #         session: Database session
    #         user_weight: Optional weight for the user's answers (defaults to 1.0)
            
    #     Raises:
    #         ValueError: If user or project not found, or if role is invalid
    #     """
    #     AuthService.verify_assign_user_to_project(user_id, project_id, role, session)
        
    #     # Check if THIS SPECIFIC role assignment already exists
    #     existing = session.scalar(
    #         select(ProjectUserRole).where(
    #             ProjectUserRole.user_id == user_id,
    #             ProjectUserRole.project_id == project_id,
    #             ProjectUserRole.role == role  # Now checking for specific role
    #         )
    #     )
        
    #     if existing:
    #         existing.is_archived = False  # Unarchive if it was archived
    #         existing.user_weight = user_weight if user_weight is not None else 1.0
    #     else:
    #         assignment = ProjectUserRole(
    #             project_id=project_id,
    #             user_id=user_id,
    #             role=role,
    #             user_weight=user_weight if user_weight is not None else 1.0,
    #             is_archived=False
    #         )
    #         session.add(assignment)
        
    #     session.commit()

    @staticmethod
    def verify_remove_user_from_project(user_id: int, project_id: int, role: str, session: Session) -> None:
        """Verify that a user can be removed from a project for a specific role."""
        # Validate user exists and isn't archived
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        if user.is_archived:
            raise ValueError(f"User '{user.user_id_str}' is archived")
        
        # Admin users cannot be removed from any project role
        if user.user_type == "admin":
            raise ValueError(f"Cannot remove admin user '{user.user_id_str}' from any project role")
        
        # Validate project exists
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        # Validate role is valid
        if role not in ["annotator", "reviewer", "admin", "model"]:
            raise ValueError(f"Invalid role: {role}")
        
        # Determine which roles would be affected by removal
        roles_to_check = []
        if role == "annotator":
            roles_to_check = ["annotator", "reviewer", "admin"]
        elif role == "reviewer":
            roles_to_check = ["reviewer", "admin"]
        elif role == "admin":
            roles_to_check = ["admin"]
        elif role == "model":
            roles_to_check = ["model"]
        
        # Check if user has any active assignments at the requested level or above
        active_assignments = session.scalars(
            select(ProjectUserRole).where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.role.in_(roles_to_check),
                ProjectUserRole.is_archived == False
            )
        ).all()
        
        if not active_assignments:
            raise ValueError(f"No active assignments found for user '{user.user_id_str}' in project '{project.name}' at role level '{role}' or above")
        
    @staticmethod
    def remove_user_from_project(user_id: int, project_id: int, role: str, session: Session) -> None:
        """Remove a user's specific role assignment from a project.

        Hierarchy logic:
            - Remove "annotator" → removes annotator + reviewer + admin (all depend on annotator)
            - Remove "reviewer" → removes reviewer + admin (admin depends on reviewer)  
            - Remove "admin" → removes only admin
            - Remove "model" → removes only model
        
        Args:
            user_id: The ID of the user
            project_id: The ID of the project
            role: The specific role to remove ('annotator', 'reviewer', 'admin', or 'model')
            session: Database session
            
        Raises:
            ValueError: If no assignment found for the specific role
        """
        AuthService.verify_remove_user_from_project(user_id, project_id, role, session)

        if role == "annotator":
            roles_to_remove = ["annotator", "reviewer", "admin"]
        elif role == "reviewer":
            roles_to_remove = ["reviewer", "admin"]
        elif role == "admin":
            roles_to_remove = ["admin"]
        elif role == "model":
            roles_to_remove = ["model"]
        
        # Get and archive all affected role assignments
        assignments_to_remove = session.scalars(
            select(ProjectUserRole).where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.role.in_(roles_to_remove),
                ProjectUserRole.is_archived == False
            )
        ).all()
        
        if not assignments_to_remove:
            # Get project name for better error message
            project = session.get(Project, project_id)
            project_name = project.name if project else f"Project {project_id}"
            raise ValueError(f"No active assignments found for user '{user.user_id_str}' in project '{project_name}' at role level '{role}' or above")
        
        # Archive all affected role assignments
        for assignment in assignments_to_remove:
            assignment.is_archived = True
        
        session.commit()

    
    @staticmethod
    def remove_all_user_roles_from_project(user_id: int, project_id: int, session: Session) -> int:
        """Remove ALL role assignments for a user from a project.
        
        Args:
            user_id: The ID of the user
            project_id: The ID of the project
            session: Database session
            
        Returns:
            Number of role assignments removed
        """
        # Get the user to check if they are a global admin
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Admin users cannot be removed from any project role
        if user.user_type == "admin":
            raise ValueError(f"Cannot remove admin user {user_id} from any project role")
        
        # Get all active role assignments for this user in this project
        assignments = session.scalars(
            select(ProjectUserRole).where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.is_archived == False
            )
        ).all()
        
        if not assignments:
            return 0
        
        # Archive all role assignments
        for assignment in assignments:
            assignment.is_archived = True
        
        session.commit()
        return len(assignments)

    

    @staticmethod
    def verify_bulk_assign_users_to_projects(user_ids: List[int], project_ids: List[int], role: str, session: Session) -> None:
        """Verify that all users can be assigned to all projects with the specified role.
        
        Args:
            user_ids: List of user IDs
            project_ids: List of project IDs  
            role: The role to assign
            session: Database session
            
        Raises:
            ValueError: If any assignment would fail, with details about all failures
        """
        failures = []
        
        for user_id in user_ids:
            for project_id in project_ids:
                try:
                    ProjectService.verify_add_user_to_project(project_id, user_id, role, session)  # Updated call
                except ValueError as e:
                    # Get user and project names for better error messages
                    try:
                        user = session.get(User, user_id)
                        project = session.get(Project, project_id)
                        user_name = user.user_id_str if user else f"User {user_id}"
                        project_name = project.name if project else f"Project {project_id}"
                        failures.append(f"• {user_name} → {project_name}: {str(e)}")
                    except:
                        failures.append(f"• User {user_id} → Project {project_id}: {str(e)}")
        
        if failures:
            failure_summary = f"Cannot assign {len(user_ids)} users to {len(project_ids)} projects as {role}. Found {len(failures)} conflicts:\n\n" + "\n".join(failures)
            raise ValueError(failure_summary)

    @staticmethod  
    def bulk_assign_users_to_projects(user_ids: List[int], project_ids: List[int], role: str, session: Session, user_weight: Optional[float] = None) -> int:
        """Assign all users to all projects after verification.
        
        Args:
            user_ids: List of user IDs
            project_ids: List of project IDs
            role: The role to assign
            session: Database session
            user_weight: Optional weight for the user's answers
            
        Returns:
            Number of assignments created/updated
            
        Raises:
            ValueError: If verification fails
        """
        # First verify all assignments would succeed
        AuthService.verify_bulk_assign_users_to_projects(user_ids, project_ids, role, session)
        
        # If verification passes, do all assignments
        assignment_count = 0
        for user_id in user_ids:
            for project_id in project_ids:
                ProjectService.add_user_to_project(project_id, user_id, role, session, user_weight)  # Updated call
                assignment_count += 1
        
        return assignment_count

    @staticmethod
    def verify_bulk_remove_users_from_projects(user_ids: List[int], project_ids: List[int], role: str, session: Session) -> None:
        """Verify that all users can be removed from all projects for a specific role.
        
        Args:
            user_ids: List of user IDs
            project_ids: List of project IDs
            role: The specific role to remove
            session: Database session
            
        Raises:
            ValueError: If any removal would fail, with details about all failures
        """
        failures = []
    
        for user_id in user_ids:
            for project_id in project_ids:
                try:
                    AuthService.verify_remove_user_from_project(user_id, project_id, role, session)
                except ValueError as e:
                    # Get user and project names for better error messages
                    try:
                        user = session.get(User, user_id)
                        project = session.get(Project, project_id)
                        user_name = user.user_id_str if user else f"User {user_id}"  # Fixed: was user.user_id
                        project_name = project.name if project else f"Project {project_id}"
                        failures.append(f"• {user_name} → {project_name}: {str(e)}")
                    except:
                        failures.append(f"• User {user_id} → Project {project_id}: {str(e)}")
        
        if failures:
            failure_summary = f"Cannot remove {role} role for {len(user_ids)} users from {len(project_ids)} projects. Found {len(failures)} conflicts:\n\n" + "\n".join(failures)
            raise ValueError(failure_summary)

    @staticmethod
    def bulk_remove_users_from_projects(user_ids: List[int], project_ids: List[int], role: str, session: Session) -> int:
        """Remove specific role for all users from all projects after verification.
        
        Args:
            user_ids: List of user IDs  
            project_ids: List of project IDs
            role: The specific role to remove
            session: Database session
            
        Returns:
            Number of role assignments removed
            
        Raises:
            ValueError: If verification fails or any individual operation fails
        """
        # First verify all removals would succeed
        AuthService.verify_bulk_remove_users_from_projects(user_ids, project_ids, role, session)
        
        # If verification passes, do all removals WITHOUT try/catch
        # If any fail after verification, that's a real error that should bubble up
        removal_count = 0
        for user_id in user_ids:
            for project_id in project_ids:
                AuthService.remove_user_from_project(user_id, project_id, role, session)
                removal_count += 1
        
        return removal_count

    # @staticmethod
    # def remove_user_from_project(user_id: int, project_id: int, session: Session) -> None:
    #     """Archive a user's assignment from a project.
        
    #     Args:
    #         user_id: The ID of the user
    #         project_id: The ID of the project
    #         session: Database session
            
    #     Raises:
    #         ValueError: If no assignments found
    #     """
    #     # Get all role assignments for this user in this project
    #     assignments = session.scalars(
    #         select(ProjectUserRole).where(
    #             ProjectUserRole.user_id == user_id,
    #             ProjectUserRole.project_id == project_id,
    #             ProjectUserRole.is_archived == False
    #         )
    #     ).all()
        
    #     if not assignments:
    #         return
        
    #     # Archive all role assignments
    #     for assignment in assignments:
    #         assignment.is_archived = True
        
    #     session.commit()
    
    @staticmethod
    def get_annotator_user_ids_from_display_names(display_names: List[str], project_id: int, session: Session) -> List[int]:
        """Convert display names to user IDs for annotators in a project.
        
        Args:
            display_names: List of display names like "John Doe (JD)"
            project_id: The ID of the project
            session: Database session
            
        Returns:
            List of unique user IDs (no duplicates)
        """
        # Remove duplicates from input list first
        unique_display_names = list(dict.fromkeys(display_names))
        
        user_ids = []
        
        # Get all annotators in project
        annotators = session.scalars(
            select(User)
            .join(ProjectUserRole, User.id == ProjectUserRole.user_id)
            .where(
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.role.in_(["annotator", "reviewer", "model", "admin"]),
                ProjectUserRole.is_archived == False,
                User.is_archived == False
            )
            .distinct()  # Ensure no duplicate users
        ).all()
        
        # Create mapping from display names to user IDs
        for annotator in annotators:
            name_parts = annotator.user_id_str.split()
            if len(name_parts) >= 2:
                initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
            else:
                initials = annotator.user_id_str[:2].upper()
            
            display_name = f"{annotator.user_id_str} ({initials})"
            if display_name in unique_display_names:
                user_ids.append(annotator.id)
        
        # Return unique user IDs only
        return list(set(user_ids))
    
    @staticmethod
    def get_user_projects_by_role(user_id: int, session: Session) -> Dict[str, List[Dict[str, Any]]]:
        """Get projects assigned to user by role.
        
        Args:
            user_id: The ID of the user
            session: Database session
            
        Returns:
            Dictionary with role keys containing lists of project info dicts
        """
        assignments = session.scalars(
            select(ProjectUserRole)
            .join(Project, ProjectUserRole.project_id == Project.id)
            .where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.is_archived == False,
                Project.is_archived == False
            )
        ).all()
        
        projects = {"annotator": [], "reviewer": [], "admin": []}
        for assignment in assignments:
            project = session.get(Project, assignment.project_id)
            if project and not project.is_archived:
                projects[assignment.role].append({
                    "id": project.id,
                    "name": project.name,
                    "description": project.description
                })
        
        return projects

    @staticmethod
    def get_assignment_counts(session: Session) -> Dict[str, int]:
        """Get assignment counts without loading actual data."""
        # Count total assignment records (use a different field since ProjectUserRole has no id)
        total_assignments = session.scalar(
            select(func.count()).select_from(ProjectUserRole)
        )
        
        active_assignments = session.scalar(
            select(func.count()).select_from(ProjectUserRole)
            .where(ProjectUserRole.is_archived == False)
        )
        
        archived_assignments = total_assignments - active_assignments
        
        # Count unique users with assignments
        unique_users = session.scalar(
            select(func.count(func.distinct(ProjectUserRole.user_id)))
            .where(ProjectUserRole.is_archived == False)
        )
        
        # Count unique projects with assignments
        unique_projects = session.scalar(
            select(func.count(func.distinct(ProjectUserRole.project_id)))
            .where(ProjectUserRole.is_archived == False)
        )
        
        return {
            "total_assignments": total_assignments,
            "active_assignments": active_assignments,
            "archived_assignments": archived_assignments,
            "unique_users": unique_users,
            "unique_projects": unique_projects
        }
    
    @staticmethod
    def get_user_counts(session: Session) -> Dict[str, int]:
        """Get count statistics for users."""
        from sqlalchemy import func, select
        
        total_users = session.scalar(select(func.count()).select_from(User))
        archived_users = session.scalar(
            select(func.count()).select_from(User)
            .where(User.is_archived == True)
        )
        active_users = total_users - archived_users
        
        # Count by role
        admin_users = session.scalar(
            select(func.count()).select_from(User)
            .where(User.user_type == 'admin')
        )
        human_users = session.scalar(
            select(func.count()).select_from(User)
            .where(User.user_type == 'human')
        )
        model_users = session.scalar(
            select(func.count()).select_from(User)
            .where(User.user_type == 'model')
        )
        
        return {
            "total": total_users,
            "active": active_users,
            "archived": archived_users,
            "admin": admin_users,
            "human": human_users,
            "model": model_users
        }
    
    @staticmethod
    def search_assignments(search_term: str = "", status_filter: str = "All", 
                        user_role_filter: str = "All", project_role_filter: str = "All",
                        page: int = 0, page_size: int = 20, session: Session = None) -> Dict[str, Any]:
        """Search assignments with pagination and filters - paginated by users, not assignment records."""
        
        # Build base query to get all matching assignment records first
        query = select(
            ProjectUserRole.user_id,
            ProjectUserRole.project_id,
            ProjectUserRole.role,
            ProjectUserRole.is_archived,
            ProjectUserRole.assigned_at,
            ProjectUserRole.completed_at,
            ProjectUserRole.user_weight,
            User.user_id_str,
            User.email,
            User.user_type,
            Project.name.label('project_name')
        ).select_from(
            ProjectUserRole
        ).join(
            User, ProjectUserRole.user_id == User.id
        ).join(
            Project, ProjectUserRole.project_id == Project.id
        )
        
        # Apply filters
        conditions = []
        
        # Status filter
        if status_filter == "Active":
            conditions.append(ProjectUserRole.is_archived == False)
        elif status_filter == "Archived":
            conditions.append(ProjectUserRole.is_archived == True)
        
        # User role filter
        if user_role_filter != "All":
            conditions.append(User.user_type == user_role_filter)
        
        # Project role filter
        if project_role_filter != "All":
            conditions.append(ProjectUserRole.role == project_role_filter)
        
        # Search filter
        if search_term:
            search_filter = or_(
                User.user_id_str.ilike(f"%{search_term}%"),
                User.email.ilike(f"%{search_term}%"),
                Project.name.ilike(f"%{search_term}%")
            )
            conditions.append(search_filter)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get ALL matching assignments (no pagination yet)
        all_assignments = session.execute(query).all()
        
        # Group by USER first (this is the key change!)
        user_assignments = {}
        for assignment in all_assignments:
            user_id = assignment.user_id
            
            if user_id not in user_assignments:
                user_assignments[user_id] = {
                    "name": assignment.user_id_str,
                    "email": assignment.email,
                    "user_role": assignment.user_type,
                    "projects": {},
                    "is_archived": True  # Will be set to False if any assignment is active
                }
            
            # Check if user has any active assignments
            if not assignment.is_archived:
                user_assignments[user_id]["is_archived"] = False
            
            project_id = assignment.project_id
            if project_id not in user_assignments[user_id]["projects"]:
                user_assignments[user_id]["projects"][project_id] = {
                    "name": assignment.project_name,
                    "role_assignments": {}
                }
            
            role = assignment.role
            user_assignments[user_id]["projects"][project_id]["role_assignments"][role] = {
                "assigned_date": assignment.assigned_at.strftime("%Y-%m-%d") if assignment.assigned_at else None,
                "completed_date": assignment.completed_at.strftime("%Y-%m-%d") if assignment.completed_at else None,
                "archived": assignment.is_archived,
                "user_weight": assignment.user_weight
            }
        
        # NOW paginate by USERS, not assignment records
        user_list = list(user_assignments.items())
        total_users = len(user_list)
        total_pages = (total_users - 1) // page_size + 1 if total_users > 0 else 1
        
        # Get users for current page
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_users)
        page_users = user_list[start_idx:end_idx]
        
        # Convert back to dict format
        paginated_user_assignments = {user_id: user_data for user_id, user_data in page_users}
        
        return {
            "user_assignments": paginated_user_assignments,  # Back to user_assignments!
            "total_count": total_users,  # Now counts users, not assignment records
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    
    @staticmethod
    def search_users_for_assignment(search_term: str, user_role_filter: str = "All", 
                                limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search users for assignment - returns limited results."""
        query = select(User).where(User.is_archived == False)
        
        if user_role_filter != "All":
            query = query.where(User.user_type == user_role_filter)
        
        if search_term:
            query = query.where(
                or_(
                    User.user_id_str.ilike(f"%{search_term}%"),
                    User.email.ilike(f"%{search_term}%")
                )
            )
        
        query = query.limit(limit)
        users = session.scalars(query).all()
        
        return [{
            "id": u.id,
            "name": u.user_id_str,
            "email": u.email,
            "role": u.user_type
        } for u in users]

    @staticmethod
    def search_projects_for_assignment(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search projects for assignment - returns limited results including archived."""
        query = select(Project).where(
            Project.name.ilike(f"%{search_term}%")
        ).limit(limit)
        
        projects = session.scalars(query).all()
        return [{"id": p.id, "name": p.name, "archived": p.is_archived} for p in projects]

    @staticmethod
    def search_users_for_selection(search_term: str, user_role_filter: str = "All", limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search users for selection dropdowns - returns limited results including archived."""
        query = select(User)
        
        conditions = []
        if user_role_filter != "All":
            conditions.append(User.user_type == user_role_filter)
        
        if search_term:
            conditions.append(
                or_(
                    User.user_id_str.ilike(f"%{search_term}%"),
                    User.email.ilike(f"%{search_term}%")
                )
            )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.limit(limit)
        users = session.scalars(query).all()
        
        return [{
            "id": u.id,
            "name": u.user_id_str,
            "email": u.email,
            "role": u.user_type,
            "archived": u.is_archived
        } for u in users]    

class QuestionGroupService:
    @staticmethod
    def get_group_verification_function(group_id: int, session: Session) -> Optional[str]:
        """Get the verification function for a question group.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            Verification function name or None if not set
            
        Raises:
            ValueError: If group not found
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        return group.verification_function

    @staticmethod
    def get_group_details_with_verification(group_id: int, session: Session) -> dict:
        """Get complete details of a question group including verification function.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            Dictionary containing all group details including verification_function
            
        Raises:
            ValueError: If group not found
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        return {
            "title": group.title,
            "display_title": group.display_title,
            "description": group.description,
            "is_reusable": group.is_reusable,
            "is_auto_submit": group.is_auto_submit,
            "is_archived": group.is_archived,
            "verification_function": group.verification_function
        }
    
    
    @staticmethod
    def get_all_groups(session: Session) -> pd.DataFrame:
        """Get all question groups with their questions and schema usage.
        
        Args:
            session: Database session
            
        Returns:
            DataFrame containing groups with columns:
            - ID: Group ID
            - Name: Group name
            - Description: Group description
            - Questions: List of questions in the group
            - Reusable: Whether the group is reusable
            - Auto Submit: Whether the group is auto-submit
            - Archived: Whether the group is archived
            - Question Count: Number of questions
            - Archived Questions: Number of archived questions
            - Used in Schemas: List of schemas using this group
        """
        groups = session.scalars(select(QuestionGroup)).all()
        rows = []
        for g in groups:
            # Get all questions in this group
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .where(QuestionGroupQuestion.question_group_id == g.id)
            ).all()
            
            # Get all schemas using this group
            schemas = session.scalars(
                select(Schema)
                .join(SchemaQuestionGroup, Schema.id == SchemaQuestionGroup.schema_id)
                .where(SchemaQuestionGroup.question_group_id == g.id)
                .distinct()
            ).all()
            
            # Format questions as a list of strings
            question_list = []
            for q in questions:
                q_str = f"- {q.text} ({q.type})"
                if q.type == "single" and q.options:
                    q_str += f" [Options: {', '.join(q.options)}"
                    if q.default_option:
                        q_str += f", Default: {q.default_option}"
                    q_str += "]"
                if q.is_archived:
                    q_str += " [ARCHIVED]"
                question_list.append(q_str)
            
            rows.append({
                "ID": g.id,
                "Name": g.title,
                "Display Title": g.display_title,
                "Description": g.description,
                "Questions": "\n".join(question_list) if question_list else "No questions",
                "Reusable": g.is_reusable,
                "Auto Submit": g.is_auto_submit,
                "Archived": g.is_archived,
                "Question Count": len(questions),
                "Archived Questions": sum(1 for q in questions if q.is_archived),
                "Used in Schemas": ", ".join(s.name for s in schemas) if schemas else "None"
            })
        return pd.DataFrame(rows)

    
    @staticmethod
    def get_group_questions(group_id: int, session: Session) -> pd.DataFrame:
        """Get all questions in a group.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            DataFrame containing questions with columns:
            - ID: Question ID
            - Text: Question text
            - Type: Question type
            - Options: Available options for single-choice questions
            - Default: Default option for single-choice questions
            - Archived: Whether the question is archived
        """
        # Check if group exists
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == group_id)
        ).all()
        
        return pd.DataFrame([
            {
                "ID": q.id,
                "Text": q.text,
                "Display Text": q.display_text,
                "Type": q.type,
                "Options": ", ".join(q.options or []) if q.options else "",
                "Default": q.default_option or "",
                "Archived": q.is_archived
            }
            for q in questions
        ])

    @staticmethod
    def get_group_counts(session: Session) -> Dict[str, int]:
        """Get count statistics for question groups.
        
        Args:
            session: Database session
            
        Returns:
            Dictionary with count statistics
        """
        total_groups = session.scalar(select(func.count()).select_from(QuestionGroup))
        archived_groups = session.scalar(
            select(func.count()).select_from(QuestionGroup)
            .where(QuestionGroup.is_archived == True)
        )
        active_groups = total_groups - archived_groups
        reusable_groups = session.scalar(
            select(func.count()).select_from(QuestionGroup)
            .where(QuestionGroup.is_reusable == True, QuestionGroup.is_archived == False)
        )
        
        return {
            "total": total_groups,
            "active": active_groups,
            "archived": archived_groups,
            "reusable": reusable_groups
        }

    @staticmethod
    def get_group_questions_with_custom_display(group_id: int, project_id: int, video_id: int, session: Session) -> pd.DataFrame:
        """Get all questions in a group with custom display applied for a specific project-video combination.
        
        Args:
            group_id: Group ID
            project_id: Project ID (for custom display context)
            video_id: Video ID (for custom display context)
            session: Database session
            
        Returns:
            DataFrame containing questions with custom display applied (same columns as original)
            
        Raises:
            ValueError: If group not found
        """
        # Check if group exists
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == group_id)
        ).all()
        
        rows = []
        for q in questions:
            # Get custom display data
            question_with_custom = CustomDisplayService.get_custom_display(
                question_id=q.id,
                project_id=project_id,
                video_id=video_id,
                session=session
            )
            
            # Same format as original but with custom display
            rows.append({
                "ID": question_with_custom["id"],
                "Text": question_with_custom["text"],
                "Display Text": question_with_custom["display_text"],  # Custom display applied
                "Type": question_with_custom["type"],
                "Options": ", ".join(question_with_custom["options"] or []) if question_with_custom["options"] else "",
                "Display Values": ", ".join(question_with_custom["display_values"] or []) if question_with_custom["display_values"] else "",  # Custom display applied
                "Default": question_with_custom["default_option"] or "",
                "Archived": question_with_custom["archived"]
            })
            
        return pd.DataFrame(rows)
    
    @staticmethod
    def get_group_details(group_id: int, session: Session) -> dict:
        """Get details of a question group.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            Dictionary containing group details
            
        Raises:
            ValueError: If group not found
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        return {
            "title": group.title,
            "display_title": group.display_title,
            "description": group.description,
            "is_reusable": group.is_reusable,
            "is_auto_submit": group.is_auto_submit,
            "is_archived": group.is_archived,
            "verification_function": group.verification_function
        }

    @staticmethod
    def set_group_verification_function(group_id: int, verification_function: Optional[str], session: Session) -> None:
        """Set the verification function for a question group.
        
        Args:
            group_id: Group ID
            verification_function: Function name or None to remove
            session: Database session
            
        Raises:
            ValueError: If group not found or function invalid
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        # Validate verification function if provided
        if verification_function:
            if not hasattr(verify, verification_function):
                raise ValueError(f"Verification function '{verification_function}' not found in verify.py")
        
        group.verification_function = verification_function
        session.commit()

    @staticmethod
    def verify_create_group(
            title: str,
            display_title: str,
            description: str,
            is_reusable: bool,
            question_ids: List[int],
            verification_function: Optional[str],
            is_auto_submit: bool = False,
            session: Session = None
    ) -> None:
        """Verify parameters for creating a new question group.

        This function performs all validation checks and raises ValueError
        if any validation fails. It does not return any objects.

        Args:
            title: Group title
            description: Group description
            is_reusable: Whether group can be used in multiple schemas
            question_ids: List of question IDs in desired order
            verification_function: Optional name of verification function from verify.py
            is_auto_submit: If TRUE, answers are automatically submitted for annotation mode
            session: Database session

        Raises:
            ValueError: If title already exists or validation fails
        """
        # Validate title
        if not title or not title.strip():
            raise ValueError("Title is required")

        # Validate questions
        if not question_ids:
            raise ValueError("Question group must contain at least one question")

        # Check if title already exists
        existing = session.scalar(
            select(QuestionGroup).where(
                QuestionGroup.title == title
            )
        )
        if existing:
            raise ValueError(f"Question group with title '{title}' already exists")
        
        if is_reusable:
            questions_with_custom_displays = session.scalar(
                select(func.count())
                .select_from(ProjectVideoQuestionDisplay)
                .where(ProjectVideoQuestionDisplay.question_id.in_(question_ids))
            )
            
            if questions_with_custom_displays > 0:
                raise ValueError(
                    f"Cannot create reusable question group. {questions_with_custom_displays} questions "
                    f"already have custom displays. Reusable groups must maintain consistent display."
                )

        # Validate verification function if provided
        if verification_function:
            if not hasattr(verify, verification_function):
                raise ValueError(f"Verification function '{verification_function}' not found in verify.py")

        # Validate all questions exist and aren't archived
        for question_id in question_ids:
            question = session.scalar(select(Question).where(Question.id == question_id))
            if not question:
                raise ValueError(f"Question with ID {question_id} not found")
            if question.is_archived:
                raise ValueError(f"Question with ID {question_id} is archived")
        
        # Validate questions are unique
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Question IDs must be unique")

        # If auto submit is TRUE, check that all questions have a default option
        if is_auto_submit:
            for question_id in question_ids:
                question = session.scalar(select(Question).where(Question.id == question_id))
                if question.default_option is None:
                    raise ValueError(f"Question with ID {question_id} does not have a default option")

    @staticmethod
    def create_group(
            title: str,
            display_title: str,
            description: str,
            is_reusable: bool,
            question_ids: List[int],
            verification_function: Optional[str],
            is_auto_submit: bool = False,
            session: Session = None
    ) -> QuestionGroup:
        """Create a new question group.

        Args:
            title: Group title (unique, immutable)
            display_title: Group display title (can be changed)
            description: Group description
            is_reusable: Whether group can be used in multiple schemas
            question_ids: List of question IDs in desired order
            verification_function: Optional name of verification function from verify.py
            is_auto_submit: If TRUE, answers are automatically submitted for annotation mode
            session: Database session

        Returns:
            Created QuestionGroup

        Raises:
            ValueError: If title already exists or validation fails
        """
        if display_title is None:
            display_title = title
        
        # First, verify all parameters (will raise ValueError if validation fails)
        QuestionGroupService.verify_create_group(
            title, display_title, description, is_reusable, question_ids,
            verification_function, is_auto_submit, session
        )

        # Create group object
        group = QuestionGroup(
            title=title,
            display_title=display_title,
            description=description,
            is_reusable=is_reusable,
            verification_function=verification_function,
            is_auto_submit=is_auto_submit
        )

        session.add(group)
        session.flush()  # Get the group ID

        # Add questions to group in specified order
        for i, question_id in enumerate(question_ids):
            session.add(QuestionGroupQuestion(
                question_group_id=group.id,
                question_id=question_id,
                display_order=i
            ))

        session.commit()
        return group

    @staticmethod
    def get_group_by_name(name: str, session: Session) -> Optional[QuestionGroup]:
        """Get a question group by its name.
        
        Args:
            name: Group name
            session: Database session
            
        Returns:
            Question group if found, None otherwise
        """
        group = session.scalar(select(QuestionGroup).where(QuestionGroup.title == name))
        if not group:
            raise ValueError(f"Question group with title '{name}' not found")
        return group

    @staticmethod
    def get_group_by_id(group_id: int, session: Session) -> Optional[QuestionGroup]:
        """Get a question group by its ID.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            Question group if found, None otherwise
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        return group

    @staticmethod
    def get_available_verification_functions() -> List[str]:
        """Get all available verification functions from verify.py.
        
        Returns:
            List of function names that can be used for verification
        """
        import inspect
        
        # Get all functions from verify module
        functions = []
        for name, obj in inspect.getmembers(verify):
            if (inspect.isfunction(obj) and 
                not name.startswith('_') and  # Exclude private functions
                name != 'verify'):  # Exclude the module name itself if it exists
                functions.append(name)
        
        return sorted(functions)

    @staticmethod
    def get_verification_function_info(function_name: str) -> Dict[str, Any]:
        """Get information about a specific verification function.
        
        Args:
            function_name: Name of the verification function
            
        Returns:
            Dictionary containing function info: name, docstring, signature
            
        Raises:
            ValueError: If function not found
        """
        import inspect
        
        if not hasattr(verify, function_name):
            raise ValueError(f"Verification function '{function_name}' not found in verify.py")
        
        func = getattr(verify, function_name)
        
        return {
            "name": function_name,
            "docstring": inspect.getdoc(func) or "No documentation available",
            "signature": str(inspect.signature(func)),
            "parameters": list(inspect.signature(func).parameters.keys())
        }

    @staticmethod
    def verify_edit_group(
            group_id: int,
            new_display_title: str,
            new_description: str,
            is_reusable: bool,
            verification_function: Optional[str],
            is_auto_submit: bool = False,
            session: Session = None
    ) -> None:
        """Verify parameters for editing a question group.

        Args:
            group_id: Group ID
            new_display_title: New group display title
            new_description: New group description
            is_reusable: Whether the group is reusable
            verification_function: New verification function name (can be None to remove)
            is_auto_submit: If not None, sets the auto-submit flag
            session: Database session

        Raises:
            ValueError: If group not found or validation fails
        """
        # Get and validate group exists
        group = QuestionGroupService.get_group_by_id(group_id, session)
        # If making a group non-reusable, check if it's used in multiple schemas
        if not is_reusable and group.is_reusable:
            schemas = session.scalars(
                select(Schema)
                .join(SchemaQuestionGroup, Schema.id == SchemaQuestionGroup.schema_id)
                .where(SchemaQuestionGroup.question_group_id == group_id)
                .distinct()
            ).all()

            if len(schemas) > 1:
                raise ValueError(
                    f"Cannot make group non-reusable as it is used in multiple schemas: "
                    f"{', '.join(s.name for s in schemas)}"
                )
        
        if is_reusable and not group.is_reusable:  # Making group reusable
            # Check if any questions in this group have custom displays
            custom_displays = session.scalar(
                select(func.count())
                .select_from(ProjectVideoQuestionDisplay)
                .join(QuestionGroupQuestion, ProjectVideoQuestionDisplay.question_id == QuestionGroupQuestion.question_id)
                .where(QuestionGroupQuestion.question_group_id == group_id)
            )
            
            if custom_displays > 0:
                raise ValueError(
                    f"Cannot make group reusable. {custom_displays} questions in this group "
                    f"have existing custom displays that would conflict with reusability."
                )

        # Check if new title conflicts with existing group
        if new_display_title != group.display_title:
            existing = session.scalar(
                select(QuestionGroup).where(QuestionGroup.display_title == new_display_title)
            )
            if existing:
                raise ValueError(f"Question group with display title '{new_display_title}' already exists")

        # Validate verification function if provided
        if verification_function:
            if not hasattr(verify, verification_function):
                raise ValueError(f"Verification function '{verification_function}' not found in verify.py")

    @staticmethod
    def edit_group(group_id: int, new_display_title: str, new_description: str, is_reusable: bool,
                   verification_function: Optional[str], is_auto_submit: bool = False, session: Session = None) -> None:
        """Edit a question group including its verification function and auto-submit flag.

        Args:
            group_id: Group ID
            new_display_title: New group display title
            new_description: New group description
            is_reusable: Whether the group is reusable
            verification_function: New verification function name (can be None to remove)
            is_auto_submit: If not None, sets the auto-submit flag
            session: Database session

        Raises:
            ValueError: If group not found or validation fails
        """
        # Verify parameters
        QuestionGroupService.verify_edit_group(
            group_id, new_display_title, new_description, is_reusable,
            verification_function, is_auto_submit, session
        )

        # Get group object for updating
        group = QuestionGroupService.get_group_by_id(group_id, session)

        # Update group properties
        group.display_title = new_display_title
        group.description = new_description
        group.is_reusable = is_reusable
        group.verification_function = verification_function  # This can be None to remove verification
        group.is_auto_submit = is_auto_submit
        session.commit()

    @staticmethod
    def test_verification_function(function_name: str, test_answers: Dict[str, str]) -> Dict[str, Any]:
        """Test a verification function with sample answers.
        
        Args:
            function_name: Name of the verification function
            test_answers: Dictionary of test answers to validate
            
        Returns:
            Dictionary containing test results: success (bool), error_message (str or None)
        """
        if not hasattr(verify, function_name):
            return {
                "success": False,
                "error_message": f"Verification function '{function_name}' not found in verify.py"
            }
        
        try:
            verify_func = getattr(verify, function_name)
            verify_func(test_answers)
            return {
                "success": True,
                "error_message": None
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": str(e)
            }

    @staticmethod
    def archive_group(group_id: int, session: Session) -> None:
        """Archive a question group and its questions.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Raises:
            ValueError: If group not found
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        group.is_archived = True
        # Also archive all questions in this group
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == group_id)
        ).all()
        for q in questions:
            q.is_archived = True
        session.commit()

    @staticmethod
    def unarchive_group(group_id: int, session: Session) -> None:
        """Unarchive a question group.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Raises:
            ValueError: If group not found
        """
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
        
        group.is_archived = False
        session.commit()

    @staticmethod
    def get_question_order(group_id: int, session: Session) -> List[int]:
        """Get the ordered list of question IDs in a group.
        
        Args:
            group_id: Group ID
            session: Database session
            
        Returns:
            List of question IDs in display order
            
        Raises:
            ValueError: If group not found
        """
        # Check if group exists
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
            
        # Get all questions in group ordered by display_order
        assignments = session.scalars(
            select(QuestionGroupQuestion)
            .where(QuestionGroupQuestion.question_group_id == group_id)
            .order_by(QuestionGroupQuestion.display_order)
        ).all()
        
        return [a.question_id for a in assignments]

    @staticmethod
    def update_question_order(group_id: int, question_ids: List[int], session: Session) -> None:
        """Update the order of questions in a group.
        
        Args:
            group_id: Group ID
            question_ids: List of question IDs in desired order
            session: Database session
            
        Raises:
            ValueError: If group not found, or if any question not in group
        """
        # Check if group exists
        group = session.get(QuestionGroup, group_id)
        if not group:
            raise ValueError(f"Question group with ID {group_id} not found")
            
        # Get all current assignments
        assignments = session.scalars(
            select(QuestionGroupQuestion)
            .where(QuestionGroupQuestion.question_group_id == group_id)
        ).all()
        
        # Create lookup for assignments
        assignment_map = {a.question_id: a for a in assignments}
        
        # Validate all questions exist in group
        for question_id in question_ids:
            if question_id not in assignment_map:
                raise ValueError(f"Question {question_id} not in group {group_id}")
        
        # Update orders based on list position
        for i, question_id in enumerate(question_ids):
            assignment_map[question_id].display_order = i
            
        session.commit()


    @staticmethod  
    def search_groups_for_selection(search_term: str, limit: int = 20, session: Session = None) -> List[Dict[str, Any]]:
        """Search question groups for selection dropdowns - returns limited results including archived."""
        query = select(QuestionGroup).where(
            or_(
                QuestionGroup.title.ilike(f"%{search_term}%"),
                QuestionGroup.display_title.ilike(f"%{search_term}%"),
                QuestionGroup.description.ilike(f"%{search_term}%")
            )
        ).limit(limit)
        
        groups = session.scalars(query).all()
        return [{"id": g.id, "title": g.title, "display_title": g.display_title, "description": g.description, "archived": g.is_archived} for g in groups]

class BaseAnswerService:
    """Base class with shared functionality for answer submission services."""
    
    @staticmethod
    def _validate_project_and_user(project_id: int, user_id: int, session: Session) -> tuple[Project, User]:
        """Validate project and user exist and are active.
        
        Args:
            project_id: The ID of the project
            user_id: The ID of the user
            session: Database session
            
        Returns:
            Tuple of (Project, User) objects
            
        Raises:
            ValueError: If validation fails
        """
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        if project.is_archived:
            raise ValueError("Project is archived")
            
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        if user.is_archived:
            raise ValueError("User is archived")
            
        return project, user

    @staticmethod
    def _validate_user_role(user_id: int, project_id: int, required_role: str, session: Session) -> None:
        """Validate user has required role in project.
        
        Args:
            user_id: The ID of the user
            project_id: The ID of the project
            required_role: Required role ('annotator', 'reviewer', or 'admin')
            session: Database session
            
        Raises:
            ValueError: If validation fails
        """
        # Get all non-archived roles for the user in this project
        user_roles = session.scalars(
            select(ProjectUserRole).where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.is_archived == False
            )
        ).all()
        
        # Define role hierarchy
        role_hierarchy = {
            'annotator': ['annotator', 'reviewer', 'model', 'admin'],
            'reviewer': ['reviewer', 'admin'],
            'admin': ['admin']
        }
        
        # Check if user has any role that satisfies the requirement
        if not user_roles or not any(role.role in role_hierarchy[required_role] for role in user_roles):
            raise ValueError(f"User {user_id} does not have {required_role} role in project {project_id}")

    @staticmethod
    def _get_question_group_with_questions(
        question_group_id: int,
        session: Session
    ) -> tuple[QuestionGroup, list[Question]]:
        """Validate question group and get its questions.
        
        Args:
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            Tuple of (QuestionGroup, list[Question])
            
        Raises:
            ValueError: If validation fails
        """
        group = session.get(QuestionGroup, question_group_id)
        if not group:
            raise ValueError(f"Question group with ID {question_group_id} not found")
        if group.is_archived:
            raise ValueError(f"Question group with ID {question_group_id} is archived")
            
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == question_group_id)
        ).all()
        
        return group, questions

    @staticmethod
    def _validate_answers_match_questions(
        answers: Dict[str, str],
        questions: list[Question]
    ) -> None:
        """Validate that answers match questions in group.
        
        Args:
            answers: Dictionary mapping question text to answer value
            questions: List of questions
            
        Raises:
            ValueError: If validation fails
        """
        question_texts = {q.text for q in questions}
        if set(answers.keys()) != question_texts:
            missing = question_texts - set(answers.keys())
            extra = set(answers.keys()) - question_texts
            raise ValueError(
                f"Answers do not match questions in group. "
                f"Missing: {missing}. Extra: {extra}"
            )

    @staticmethod
    def _run_verification(
        group: QuestionGroup,
        answers: Dict[str, str]
    ) -> None:
        """Run verification function if specified.
        
        Args:
            group: Question group
            answers: Dictionary mapping question text to answer value
            
        Raises:
            ValueError: If verification fails
        """
        if group.verification_function:
            verify_func = getattr(verify, group.verification_function, None)
            if not verify_func:
                raise ValueError(f"Verification function '{group.verification_function}' not found in verify.py")
            try:
                verify_func(answers)
            except ValueError as e:
                raise ValueError(f"Answer verification failed: {str(e)}")

    @staticmethod
    def _validate_answer_value(question: Question, answer_value: str) -> None:
        """Validate answer value matches question type and options.
        
        Args:
            question: Question object
            answer_value: Answer value to validate
            
        Raises:
            ValueError: If validation fails
        """
        if question.type == "single":
            if not question.options:
                raise ValueError(f"Question '{question.text}' has no options defined")
            if answer_value not in question.options:
                raise ValueError(
                    f"Answer value '{answer_value}' not in options for '{question.text}': "
                    f"{', '.join(question.options)}"
                )
        elif question.type == "description":
            if not isinstance(answer_value, str):
                raise ValueError(f"Description answer for '{question.text}' must be a string")

    @staticmethod
    def _check_and_update_completion(
        user_id: int,
        project_id: int,
        session: Session
    ) -> float:
        """Check if user has completed all questions in project and update completion timestamp.
        
        Args:
            user_id: The ID of the user
            project_id: The ID of the project
            session: Database session
            
        Returns:
            float: Completion percentage (0-100)
        """
        # Get total non-archived questions in project's schema
        total_questions = session.scalar(
            select(func.count())
            .select_from(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .join(Project, SchemaQuestionGroup.schema_id == Project.schema_id)
            .where(
                Project.id == project_id,
                Question.is_archived == False
            )
        )
        
        # Get total non-archived videos in project
        total_videos = session.scalar(
            select(func.count())
            .select_from(ProjectVideo)
            .join(Video, ProjectVideo.video_id == Video.id)
            .where(
                ProjectVideo.project_id == project_id,
                Video.is_archived == False
            )
        )
        
        # Get user's role
        user_role = session.scalar(
            select(ProjectUserRole)
            .where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.project_id == project_id,
                ProjectUserRole.is_archived == False
            )
        )
        
        if not user_role:
            return 0.0
            
        # Get total answers submitted by user
        if user_role.role == "annotator":
            # For annotators, count their own answers for non-archived questions
            total_answers = session.scalar(
                select(func.count())
                .select_from(AnnotatorAnswer)
                .join(Question, AnnotatorAnswer.question_id == Question.id)
                .where(
                    AnnotatorAnswer.user_id == user_id,
                    AnnotatorAnswer.project_id == project_id,
                    Question.is_archived == False
                )
            )
            
            # Update completion timestamp if all questions are answered
            expected_answers = total_questions * total_videos
            completion_percentage = min((total_answers / expected_answers * 100) if expected_answers > 0 else 0.0, 100.0)
            
            if total_answers >= expected_answers:
                user_role.completed_at = datetime.now(timezone.utc)
            else:
                user_role.completed_at = None
                
        else:  # reviewer
            # For reviewers, count total ground truth answers in project for non-archived questions
            total_answers = session.scalar(
                select(func.count())
                .select_from(ReviewerGroundTruth)
                .join(Question, ReviewerGroundTruth.question_id == Question.id)
                .where(
                    ReviewerGroundTruth.project_id == project_id,
                    Question.is_archived == False
                )
            )
            
            # Calculate completion percentage
            expected_answers = total_questions * total_videos
            completion_percentage = min((total_answers / expected_answers * 100) if expected_answers > 0 else 0.0, 100.0)
            
            # If all questions are answered, update completion timestamp for all reviewers
            # Get all reviewer roles for this project
            reviewer_roles = session.scalars(
                select(ProjectUserRole)
                .where(
                    ProjectUserRole.project_id == project_id,
                    ProjectUserRole.role == "reviewer",
                )
            ).all()
            if total_answers >= expected_answers:
                
                # Update completion timestamp for all reviewers
                for role in reviewer_roles:
                    role.completed_at = datetime.now(timezone.utc)
            else:
                # Reset completion timestamp for all reviewers            
                for role in reviewer_roles:
                    role.completed_at = None
            
        session.commit()
        return completion_percentage

class AnnotatorService(BaseAnswerService):

    @staticmethod
    def get_all_annotator_data_for_video(video_id: int, project_id: int, user_id: int, session: Session) -> Dict[str, Any]:
        """Get ALL annotator related data for a video in a single batch operation"""
        try:
            # Get all annotator answers for this user and video
            answers = session.scalars(
                select(AnnotatorAnswer)
                .where(
                    AnnotatorAnswer.video_id == video_id,
                    AnnotatorAnswer.project_id == project_id,
                    AnnotatorAnswer.user_id == user_id
                )
            ).all()
            
            # Organize by question_id
            answers_by_question = {answer.question_id: answer for answer in answers}
            
            # Get all questions for this project
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .join(Project, SchemaQuestionGroup.schema_id == Project.schema_id)
                .where(
                    Project.id == project_id,
                    Question.is_archived == False
                )
            ).all()
            
            # Build answer dict by question text and check submission status by group
            answer_dict = {}
            submission_status_by_group = {}
            
            for question in questions:
                if question.id in answers_by_question:
                    answer_dict[question.text] = answers_by_question[question.id].answer_value
            
            # Get question group assignments to check submission status
            group_assignments = session.scalars(
                select(QuestionGroupQuestion)
                .where(QuestionGroupQuestion.question_id.in_([q.id for q in questions]))
            ).all()
            
            # Organize questions by group
            questions_by_group = {}
            for assignment in group_assignments:
                group_id = assignment.question_group_id
                if group_id not in questions_by_group:
                    questions_by_group[group_id] = []
                questions_by_group[group_id].append(assignment.question_id)
            
            # Check submission status for each group
            for group_id, question_ids in questions_by_group.items():
                submitted_count = sum(1 for qid in question_ids if qid in answers_by_question)
                submission_status_by_group[group_id] = submitted_count > 0
            
            return {
                "answer_dict": answer_dict,
                "answers_by_question": answers_by_question,
                "submission_status_by_group": submission_status_by_group
            }
            
        except Exception as e:
            print(f"Error in get_all_annotator_data_for_video: {e}")
            return {
                "answer_dict": {},
                "answers_by_question": {},
                "submission_status_by_group": {}
            }

    @staticmethod
    def batch_get_user_answers_for_question_groups(video_id: int, project_id: int, user_id: int, question_group_ids: List[int], session: Session) -> Dict[int, Dict[str, str]]:
        """Get user answers for multiple question groups at once"""
        try:
            if not question_group_ids:
                return {}
            
            # Get all questions for these groups
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .where(
                    QuestionGroupQuestion.question_group_id.in_(question_group_ids),
                    Question.is_archived == False
                )
            ).all()
            
            # Get all answers for these questions
            question_ids = [q.id for q in questions]
            answers = session.scalars(
                select(AnnotatorAnswer)
                .where(
                    AnnotatorAnswer.video_id == video_id,
                    AnnotatorAnswer.project_id == project_id,
                    AnnotatorAnswer.user_id == user_id,
                    AnnotatorAnswer.question_id.in_(question_ids)
                )
            ).all()
            
            # Build answer dict by question_id
            answers_by_question_id = {answer.question_id: answer.answer_value for answer in answers}
            
            # Organize by group
            result = {}
            for group_id in question_group_ids:
                result[group_id] = {}
                
                group_questions = session.scalars(
                    select(Question)
                    .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                    .where(
                        QuestionGroupQuestion.question_group_id == group_id,
                        Question.is_archived == False
                    )
                ).all()
                
                for question in group_questions:
                    if question.id in answers_by_question_id:
                        result[group_id][question.text] = answers_by_question_id[question.id]
            
            return result
            
        except Exception as e:
            print(f"Error in batch_get_user_answers_for_question_groups: {e}")
            return {}

    @staticmethod
    def get_all_project_answers(project_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all annotator answers for a project in bulk.
        
        Args:
            project_id: The ID of the project
            session: Database session
            
        Returns:
            List of answer dictionaries with keys:
            - video_id, question_id, user_id, answer_value, confidence_score, created_at, modified_at, notes
        """
        answers = session.scalars(
            select(AnnotatorAnswer)
            .where(AnnotatorAnswer.project_id == project_id)
        ).all()
        
        return [
            {
                "video_id": answer.video_id,
                "question_id": answer.question_id,
                "user_id": answer.user_id,
                "answer_value": answer.answer_value,
                "confidence_score": answer.confidence_score,
                "created_at": answer.created_at,
                "modified_at": answer.modified_at,
                "notes": answer.notes
            }
            for answer in answers
        ]

    @staticmethod
    def get_all_text_answers_for_project(project_id: int, description_question_ids: List[int], session: Session) -> List[Dict[str, Any]]:
        """Get all text answers for description questions in a project.
        
        Args:
            project_id: The ID of the project
            description_question_ids: List of question IDs for description questions
            session: Database session
            
        Returns:
            List of text answer dictionaries with keys:
            - video_id, question_id, user_id, answer_value, user_name
        """
        if not description_question_ids:
            return []
        
        # Get text answers with user info in one query
        results = session.execute(
            select(
                AnnotatorAnswer.video_id,
                AnnotatorAnswer.question_id,
                AnnotatorAnswer.user_id,
                AnnotatorAnswer.answer_value,
                User.user_id_str
            ).select_from(
                AnnotatorAnswer
            ).join(
                User, AnnotatorAnswer.user_id == User.id
            ).where(
                AnnotatorAnswer.project_id == project_id,
                AnnotatorAnswer.question_id.in_(description_question_ids)
            )
        ).all()
        
        return [
            {
                "video_id": row.video_id,
                "question_id": row.question_id,
                "user_id": row.user_id,
                "answer_value": row.answer_value,
                "user_name": row.user_id_str
            }
            for row in results
        ]

    @staticmethod
    def verify_submit_answer_to_question_group(
        video_id: int,
        project_id: int,
        user_id: int,
        question_group_id: int,
        answers: Dict[str, str],
        session: Session,
        confidence_scores: Optional[Dict[str, float]] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> None:
        """Verify parameters for submitting answers to a question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            user_id: The ID of the user submitting the answers
            question_group_id: The ID of the question group
            answers: Dictionary mapping question text to answer value
            session: Database session
            confidence_scores: Optional dictionary mapping question text to confidence score
            notes: Optional dictionary mapping question text to notes
            
        Raises:
            ValueError: If validation fails or verification fails
        """
        # Validate project and user
        project, user = AnnotatorService._validate_project_and_user(project_id=project_id, user_id=user_id, session=session)
        
        # Validate user role
        AnnotatorService._validate_user_role(user_id=user_id, project_id=project_id, required_role="annotator", session=session)
            
        # Validate question group and get questions
        group, questions = AnnotatorService._get_question_group_with_questions(question_group_id=question_group_id, session=session)
        
        # Validate answers match questions
        AnnotatorService._validate_answers_match_questions(answers=answers, questions=questions)
            
        # Run verification if specified
        AnnotatorService._run_verification(group=group, answers=answers)
        
        # Validate confidence scores if provided
        if confidence_scores:
            for question_text, confidence_score in confidence_scores.items():
                if not isinstance(confidence_score, float):
                    raise ValueError(f"Confidence score for question '{question_text}' must be a float")
        
        # Validate answer values for each question
        for question in questions:
            answer_value = answers[question.text]
            AnnotatorService._validate_answer_value(question=question, answer_value=answer_value)

    @staticmethod
    def submit_answer_to_question_group(
        video_id: int,
        project_id: int,
        user_id: int,
        question_group_id: int,
        answers: Dict[str, str],  # Maps question text to answer value
        session: Session,
        confidence_scores: Optional[Dict[str, float]] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> None:
        """Submit answers for all questions in a question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            user_id: The ID of the user submitting the answers
            question_group_id: The ID of the question group
            answers: Dictionary mapping question text to answer value
            session: Database session
            confidence_scores: Optional dictionary mapping question text to confidence score
            notes: Optional dictionary mapping question text to notes
            
        Raises:
            ValueError: If validation fails or verification fails
        """
        # Verify input parameters
        AnnotatorService.verify_submit_answer_to_question_group(
            video_id=video_id,
            project_id=project_id,
            user_id=user_id,
            question_group_id=question_group_id,
            answers=answers,
            session=session,
            confidence_scores=confidence_scores,
            notes=notes
        )
        
        # Get questions for submission (already validated in verify method)
        group, questions = AnnotatorService._get_question_group_with_questions(question_group_id=question_group_id, session=session)
            
        # Submit each answer
        for question in questions:
            answer_value = answers[question.text]
            confidence_score = confidence_scores.get(question.text) if confidence_scores else None
            note = notes.get(question.text) if notes else None
            
            # Check for existing answer
            existing = session.scalar(
                select(AnnotatorAnswer).where(
                    AnnotatorAnswer.video_id == video_id,
                    AnnotatorAnswer.question_id == question.id,
                    AnnotatorAnswer.user_id == user_id,
                    AnnotatorAnswer.project_id == project_id
                )
            )
        
            if existing:
                # Update existing answer
                existing.answer_value = answer_value
                existing.modified_at = datetime.now(timezone.utc)
                existing.confidence_score = confidence_score
                existing.notes = note
            else:
                # Create new answer
                answer = AnnotatorAnswer(
                    video_id=video_id,
                    question_id=question.id,
                    project_id=project_id,
                    user_id=user_id,
                    answer_type=question.type,
                    answer_value=answer_value,
                    confidence_score=confidence_score,
                    notes=note
                )
                session.add(answer)
        session.commit()
        
        # Check and update completion status
        AnnotatorService._check_and_update_completion(user_id=user_id, project_id=project_id, session=session)

    @staticmethod
    def get_answers(video_id: int, project_id: int, session: Session) -> pd.DataFrame:
        """Get all answers for a video in a project.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            session: Database session
            
        Returns:
            DataFrame containing answers with columns:
            - Question ID
            - User ID
            - Answer ID
            - Answer Value
            - Created At
            - Modified At
            - Confidence Score
            - Notes
        """
        answers = session.scalars(
            select(AnnotatorAnswer)
            .where(
                AnnotatorAnswer.video_id == video_id,
                AnnotatorAnswer.project_id == project_id
            )
        ).all()
        
        return pd.DataFrame([
            {
                "Question ID": a.question_id,
                "User ID": a.user_id,
                "Answer ID": a.id,
                "Answer Value": a.answer_value,
                "Confidence Score": a.confidence_score,
                "Created At": a.created_at,
                "Modified At": a.modified_at,
                "Notes": a.notes
            }
            for a in answers
        ])

    @staticmethod
    def get_question_answers(question_id: int, project_id: int, session: Session) -> pd.DataFrame:
        """Get all answers for a question in a project.
        
        Args:
            question_id: The ID of the question
            project_id: The ID of the project
            session: Database session
            
        Returns:
            DataFrame containing answers
        """
        answers = session.scalars(
            select(AnnotatorAnswer)
            .where(
                AnnotatorAnswer.question_id == question_id,
                AnnotatorAnswer.project_id == project_id
            )
        ).all()
        
        return pd.DataFrame([
            {
                "Video ID": a.video_id,
                "User ID": a.user_id,
                "Answer Value": a.answer_value,
                "Confidence Score": a.confidence_score,
                "Created At": a.created_at,
                "Modified At": a.modified_at,
                "Notes": a.notes
            }
            for a in answers
        ])
    
    @staticmethod
    def get_user_answers_for_question_group(video_id: int, project_id: int, user_id: int, question_group_id: int, session: Session) -> Dict[str, str]:
        """Get user's existing answers for a video and question group as a dictionary.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            user_id: The ID of the user
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            Dictionary mapping question text to answer value
        """
        answers = session.scalars(
            select(AnnotatorAnswer)
            .join(Question, AnnotatorAnswer.question_id == Question.id)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(
                AnnotatorAnswer.video_id == video_id,
                AnnotatorAnswer.project_id == project_id,
                AnnotatorAnswer.user_id == user_id,
                QuestionGroupQuestion.question_group_id == question_group_id
            )
        ).all()
        
        result = {}
        for answer in answers:
            question = session.get(Question, answer.question_id)
            if question:
                result[question.text] = answer.answer_value
        
        return result

    @staticmethod
    def check_user_has_submitted_answers(video_id: int, project_id: int, user_id: int, question_group_id: int, session: Session) -> bool:
        """Check if user has submitted any answers for a question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            user_id: The ID of the user
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            True if user has submitted answers, False otherwise
        """
        answer_count = session.scalar(
            select(func.count())
            .select_from(AnnotatorAnswer)
            .join(Question, AnnotatorAnswer.question_id == Question.id)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(
                AnnotatorAnswer.video_id == video_id,
                AnnotatorAnswer.project_id == project_id,
                AnnotatorAnswer.user_id == user_id,
                QuestionGroupQuestion.question_group_id == question_group_id
            )
        )
        
        return answer_count > 0
    
    @staticmethod
    def calculate_user_overall_progress(user_id: int, project_id: int, session: Session) -> float:
        """Calculate user's overall progress across all question groups in a project.
        
        Args:
            user_id: The ID of the user
            project_id: The ID of the project
            session: Database session
            
        Returns:
            Progress percentage (0-100)
            
        Raises:
            ValueError: If project or user not found
        """
        # Validate project and user
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        
        # Get total questions across all groups in schema
        total_questions = session.scalar(
            select(func.count())
            .select_from(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(
                SchemaQuestionGroup.schema_id == project.schema_id,
                Question.is_archived == False
            )
        )
        
        # Get total videos in project
        total_videos = session.scalar(
            select(func.count())
            .select_from(ProjectVideo)
            .join(Video, ProjectVideo.video_id == Video.id)
            .where(
                ProjectVideo.project_id == project_id,
                Video.is_archived == False
            )
        )
        
        # Get user's total answers
        answered = session.scalar(
            select(func.count())
            .select_from(AnnotatorAnswer)
            .join(Question, AnnotatorAnswer.question_id == Question.id)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
            .where(
                AnnotatorAnswer.user_id == user_id,
                AnnotatorAnswer.project_id == project_id,
                SchemaQuestionGroup.schema_id == project.schema_id,
                Question.is_archived == False
            )
        )
        
        total_possible = total_questions * total_videos
        return (answered / total_possible * 100) if total_possible > 0 else 0.0
    
    @staticmethod
    def get_all_user_answers_for_question_group(video_id: int, project_id: int, question_group_id: int, session: Session) -> List[Dict[str, Any]]:
        """Get all users' answers for a video and question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            List of answer dictionaries
        """
        answers = session.scalars(
            select(AnnotatorAnswer)
            .join(Question, AnnotatorAnswer.question_id == Question.id)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(
                AnnotatorAnswer.video_id == video_id,
                AnnotatorAnswer.project_id == project_id,
                QuestionGroupQuestion.question_group_id == question_group_id
            )
        ).all()
        
        result = []
        for answer in answers:
            question = session.get(Question, answer.question_id)
            if question:
                result.append({
                    "user_id": answer.user_id,
                    "question_id": question.id,
                    "question_text": question.text,
                    "answer_value": answer.answer_value,
                    "created_at": answer.created_at,
                    "modified_at": answer.modified_at
                })
        
        return result

class GroundTruthService(BaseAnswerService):

    @staticmethod
    def get_complete_video_data_for_display(video_id: int, project_id: int, user_id: int, role: str, session: Session) -> Dict[str, Any]:
        """Get ALL data needed to display a video with all its question groups in one massive batch operation"""
        try:
            # Get project and schema info
            project = ProjectService.get_project_by_id(project_id=project_id, session=session)
            
            # Get all question groups for this schema
            question_groups = SchemaService.get_schema_question_groups_list(
                schema_id=project.schema_id, session=session
            )
            
            # Get ALL questions for this project at once
            all_questions = ProjectService.get_project_questions(project_id=project_id, session=session)
            
            # Organize questions by group
            questions_by_group = {}
            question_id_to_group = {}
            all_question_ids = set()
            
            for question in all_questions:
                all_question_ids.add(question["id"])
                
                # Find which groups this question belongs to
                group_assignments = session.scalars(
                    select(QuestionGroupQuestion)
                    .where(QuestionGroupQuestion.question_id == question["id"])
                ).all()
                
                for assignment in group_assignments:
                    group_id = assignment.question_group_id
                    if group_id not in questions_by_group:
                        questions_by_group[group_id] = []
                    questions_by_group[group_id].append(question)
                    question_id_to_group[question["id"]] = group_id
            
            # BATCH 1: Get ALL ground truth data for this video
            gt_data = {}
            admin_modifications = {}
            gt_status_by_question = {}
            
            if role in ["reviewer", "meta_reviewer"]:
                gt_batch = GroundTruthService.get_all_ground_truth_data_for_video(
                    video_id=video_id, project_id=project_id, session=session
                )
                gt_data = gt_batch["ground_truth_dict"]
                admin_modifications = gt_batch["admin_modifications"]
                gt_status_by_question = gt_batch["gt_status_by_question"]
            
            # BATCH 2: Get ALL annotator data for this video
            annotator_data = {}
            if role == "annotator":
                annotator_batch = AnnotatorService.get_all_annotator_data_for_video(
                    video_id=video_id, project_id=project_id, user_id=user_id, session=session
                )
                annotator_data = annotator_batch
            
            # BATCH 3: Get ALL completion statuses for all groups at once
            completion_status_by_group = {}
            
            if role == "annotator":
                # Check which groups have any answers
                for group_id in questions_by_group.keys():
                    group_questions = questions_by_group[group_id]
                    group_question_ids = [q["id"] for q in group_questions]
                    
                    has_answers = any(
                        qid in annotator_data.get("answers_by_question", {}) 
                        for qid in group_question_ids
                    )
                    completion_status_by_group[group_id] = has_answers
            else:
                # Check which groups have ground truth
                for group_id in questions_by_group.keys():
                    group_questions = questions_by_group[group_id]
                    group_question_ids = [q["id"] for q in group_questions]
                    
                    has_gt = any(qid in gt_data for qid in group_question_ids)
                    completion_status_by_group[group_id] = has_gt
            
            # BATCH 4: Get training mode ground truth if needed
            training_gt_by_group = {}
            if role == "annotator":
                # Check if this is training mode
                is_training = ProjectService.check_project_has_full_ground_truth(
                    project_id=project_id, session=session
                )
                
                if is_training:
                    # Get ALL ground truth for training mode
                    all_gt = session.scalars(
                        select(ReviewerGroundTruth)
                        .where(
                            ReviewerGroundTruth.video_id == video_id,
                            ReviewerGroundTruth.project_id == project_id
                        )
                    ).all()
                    
                    gt_by_question_id = {gt.question_id: gt.answer_value for gt in all_gt}
                    
                    # Organize by group
                    for group_id, group_questions in questions_by_group.items():
                        training_gt_by_group[group_id] = {}
                        for question in group_questions:
                            if question["id"] in gt_by_question_id:
                                training_gt_by_group[group_id][question["text"]] = gt_by_question_id[question["id"]]
            
            # BATCH 5: Get ALL answer reviews for description questions (FIXED)
            answer_reviews_by_group = {}
            if role in ["reviewer", "meta_reviewer"]:
                description_questions = [q for q in all_questions if q["type"] == "description"]
                
                if description_questions:
                    # 🚀 FIXED: Use AnnotatorService.get_answers() to get the correct "Answer ID" column
                    try:
                        answers_df = AnnotatorService.get_answers(video_id=video_id, project_id=project_id, session=session)
                        
                        if not answers_df.empty:
                            # Filter for description questions only
                            description_question_ids = [q["id"] for q in description_questions]
                            desc_answers = answers_df[answers_df["Question ID"].isin(description_question_ids)]
                            
                            if not desc_answers.empty:
                                # Get all answer IDs for this video's description questions
                                answer_ids = desc_answers["Answer ID"].tolist()
                                
                                # Get all reviews for these answers in one query
                                reviews = session.scalars(
                                    select(AnswerReview)
                                    .where(AnswerReview.answer_id.in_(answer_ids))
                                ).all()
                                
                                reviews_by_answer_id = {review.answer_id: review for review in reviews}
                                
                                # Get all user info for reviewers in one query
                                reviewer_ids = {review.reviewer_id for review in reviews if review.reviewer_id}
                                user_info_map = {}
                                if reviewer_ids:
                                    users = session.scalars(
                                        select(User).where(User.id.in_(reviewer_ids))
                                    ).all()
                                    user_info_map = {user.id: user.user_id_str for user in users}
                                
                                # Get all annotator user info for answer owners
                                annotator_ids = desc_answers["User ID"].unique().tolist()
                                annotator_info_map = {}
                                if annotator_ids:
                                    annotators = session.scalars(
                                        select(User).where(User.id.in_(annotator_ids))
                                    ).all()
                                    annotator_info_map = {user.id: user.user_id_str for user in annotators}
                                
                                # Organize by group and question
                                for question in description_questions:
                                    group_id = question_id_to_group.get(question["id"])
                                    if group_id:
                                        if group_id not in answer_reviews_by_group:
                                            answer_reviews_by_group[group_id] = {}
                                        
                                        question_answers = desc_answers[desc_answers["Question ID"] == question["id"]]
                                        
                                        if not question_answers.empty:
                                            answer_reviews_by_group[group_id][question["text"]] = {}
                                            
                                            for _, answer_row in question_answers.iterrows():
                                                answer_id = answer_row["Answer ID"]
                                                user_id_for_answer = answer_row["User ID"]
                                                
                                                # Get user display name
                                                user_name = annotator_info_map.get(user_id_for_answer, f"User {user_id_for_answer}")
                                                display_name, _ = AuthService.get_user_display_name_with_initials(user_name)
                                                
                                                if answer_id in reviews_by_answer_id:
                                                    review = reviews_by_answer_id[answer_id]
                                                    reviewer_name = user_info_map.get(review.reviewer_id)
                                                    answer_reviews_by_group[group_id][question["text"]][display_name] = {
                                                        "status": review.status,
                                                        "reviewer_id": review.reviewer_id,
                                                        "reviewer_name": reviewer_name
                                                    }
                                                else:
                                                    answer_reviews_by_group[group_id][question["text"]][display_name] = {
                                                        "status": "pending",
                                                        "reviewer_id": None,
                                                        "reviewer_name": None
                                                    }
                    except Exception as e:
                        print(f"Error getting answer reviews: {e}")
                        # Don't fail the whole function, just leave answer_reviews_by_group empty
                        answer_reviews_by_group = {}
            
            return {
                "project": project,
                "question_groups": question_groups,
                "questions_by_group": questions_by_group,
                "gt_data": gt_data,
                "admin_modifications": admin_modifications,
                "gt_status_by_question": gt_status_by_question,
                "annotator_data": annotator_data,
                "completion_status_by_group": completion_status_by_group,
                "training_gt_by_group": training_gt_by_group,
                "answer_reviews_by_group": answer_reviews_by_group,
                "is_training_mode": role == "annotator" and bool(training_gt_by_group)
            }
            
        except Exception as e:
            print(f"Error in get_complete_video_data_for_display: {e}")
            return {}

    @staticmethod
    def get_all_ground_truth_data_for_video(video_id: int, project_id: int, session: Session) -> Dict[str, Any]:
        """Get ALL ground truth related data for a video in a single batch operation"""
        try:
            # Get all ground truth records for this video
            ground_truths = session.scalars(
                select(ReviewerGroundTruth)
                .where(
                    ReviewerGroundTruth.video_id == video_id,
                    ReviewerGroundTruth.project_id == project_id
                )
            ).all()
            
            # Organize by question_id
            gt_by_question = {}
            gt_dict = {}
            all_question_ids = set()
            
            for gt in ground_truths:
                question_id = gt.question_id
                all_question_ids.add(question_id)
                
                gt_by_question[question_id] = gt
                gt_dict[question_id] = gt.answer_value
            
            # Get all questions for this project to check completion
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .join(Project, SchemaQuestionGroup.schema_id == Project.schema_id)
                .where(
                    Project.id == project_id,
                    Question.is_archived == False
                )
            ).all()
            
            project_question_ids = {q.id for q in questions}
            
            # Check admin modifications for all questions at once
            admin_modifications = {}
            if all_question_ids:
                admin_modifications = GroundTruthService.batch_check_admin_modifications(
                    video_id=video_id, project_id=project_id, 
                    question_ids=list(all_question_ids), session=session
                )
            
            # Get user info for all reviewers at once
            reviewer_ids = {gt.reviewer_id for gt in ground_truths}
            admin_ids = {gt.modified_by_admin_id for gt in ground_truths if gt.modified_by_admin_id}
            all_user_ids = reviewer_ids | admin_ids
            
            user_info = {}
            if all_user_ids:
                users = session.scalars(
                    select(User).where(User.id.in_(all_user_ids))
                ).all()
                user_info = {user.id: user.user_id_str for user in users}
            
            # Build GT status strings
            gt_status_by_question = {}
            for question_id in project_question_ids:
                if question_id in gt_by_question:
                    gt = gt_by_question[question_id]
                    reviewer_name = user_info.get(gt.reviewer_id, f"User {gt.reviewer_id}")
                    
                    if gt.modified_by_admin_id:
                        admin_name = user_info.get(gt.modified_by_admin_id, f"Admin {gt.modified_by_admin_id}")
                        gt_status_by_question[question_id] = f"🏆 GT by: {reviewer_name} (Overridden by {admin_name})"
                    else:
                        gt_status_by_question[question_id] = f"🏆 GT by: {reviewer_name}"
                else:
                    gt_status_by_question[question_id] = "📭 No GT"
            
            return {
                "ground_truth_dict": gt_dict,
                "admin_modifications": admin_modifications,
                "gt_status_by_question": gt_status_by_question,
                "gt_records": gt_by_question,
                "has_any_admin_modifications": any(mod["is_modified"] for mod in admin_modifications.values())
            }
            
        except Exception as e:
            print(f"Error in get_all_ground_truth_data_for_video: {e}")
            return {
                "ground_truth_dict": {},
                "admin_modifications": {},
                "gt_status_by_question": {},
                "gt_records": {},
                "has_any_admin_modifications": False
            }

    @staticmethod
    def batch_get_ground_truth_for_question_groups(video_id: int, project_id: int, question_group_ids: List[int], session: Session) -> Dict[int, Dict[str, str]]:
        """Get ground truth for multiple question groups at once"""
        try:
            if not question_group_ids:
                return {}
            
            # Get all questions for these groups
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .where(
                    QuestionGroupQuestion.question_group_id.in_(question_group_ids),
                    Question.is_archived == False
                )
            ).all()
            
            # Organize questions by group
            questions_by_group = {}
            for question in questions:
                group_assignments = session.scalars(
                    select(QuestionGroupQuestion)
                    .where(
                        QuestionGroupQuestion.question_id == question.id,
                        QuestionGroupQuestion.question_group_id.in_(question_group_ids)
                    )
                ).all()
                
                for assignment in group_assignments:
                    group_id = assignment.question_group_id
                    if group_id not in questions_by_group:
                        questions_by_group[group_id] = []
                    questions_by_group[group_id].append(question)
            
            # Get ground truth for all questions at once
            question_ids = [q.id for q in questions]
            ground_truths = session.scalars(
                select(ReviewerGroundTruth)
                .where(
                    ReviewerGroundTruth.video_id == video_id,
                    ReviewerGroundTruth.project_id == project_id,
                    ReviewerGroundTruth.question_id.in_(question_ids)
                )
            ).all()
            
            # Build GT dict by question
            gt_dict = {gt.question_id: gt.answer_value for gt in ground_truths}
            
            # Organize by group
            result = {}
            for group_id, group_questions in questions_by_group.items():
                result[group_id] = {}
                for question in group_questions:
                    if question.id in gt_dict:
                        result[group_id][question.text] = gt_dict[question.id]
            
            return result
            
        except Exception as e:
            print(f"Error in batch_get_ground_truth_for_question_groups: {e}")
            return {}

    @staticmethod
    def verify_submit_ground_truth_to_question_group(
        video_id: int,
        project_id: int,
        reviewer_id: int,
        question_group_id: int,
        answers: Dict[str, str],
        session: Session,
        confidence_scores: Optional[Dict[str, float]] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> None:
        """Verify parameters for submitting ground truth answers to a question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            reviewer_id: The ID of the reviewer
            question_group_id: The ID of the question group
            answers: Dictionary mapping question text to answer value
            session: Database session
            confidence_scores: Optional dictionary mapping question text to confidence score
            notes: Optional dictionary mapping question text to notes
            
        Raises:
            ValueError: If validation fails or verification fails
        """
        # Validate project and reviewer
        project, reviewer = GroundTruthService._validate_project_and_user(project_id=project_id, user_id=reviewer_id, session=session)
        
        # Validate reviewer role
        GroundTruthService._validate_user_role(user_id=reviewer_id, project_id=project_id, required_role="reviewer", session=session)
            
        # Validate question group and get questions
        group, questions = GroundTruthService._get_question_group_with_questions(question_group_id=question_group_id, session=session)
        
        # Validate answers match questions
        GroundTruthService._validate_answers_match_questions(answers=answers, questions=questions)
            
        # Run verification if specified
        GroundTruthService._run_verification(group=group, answers=answers)
        
        # Validate confidence scores if provided
        if confidence_scores:
            for question_text, confidence_score in confidence_scores.items():
                if not isinstance(confidence_score, float):
                    raise ValueError(f"Confidence score for question '{question_text}' must be a float")
        
        # Validate answer values for each question
        for question in questions:
            answer_value = answers[question.text]
            GroundTruthService._validate_answer_value(question=question, answer_value=answer_value)

    @staticmethod
    def submit_ground_truth_to_question_group(
        video_id: int,
        project_id: int,
        reviewer_id: int,
        question_group_id: int,
        answers: Dict[str, str],  # Maps question text to answer value
        session: Session,
        confidence_scores: Optional[Dict[str, float]] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> None:
        """Submit ground truth answers for all questions in a question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            reviewer_id: The ID of the reviewer
            question_group_id: The ID of the question group
            answers: Dictionary mapping question text to answer value
            session: Database session
            confidence_scores: Optional dictionary mapping question text to confidence score
            notes: Optional dictionary mapping question text to notes
            
        Raises:
            ValueError: If validation fails or verification fails
        """
        # Verify input parameters
        GroundTruthService.verify_submit_ground_truth_to_question_group(
            video_id=video_id,
            project_id=project_id,
            reviewer_id=reviewer_id,
            question_group_id=question_group_id,
            answers=answers,
            session=session,
            confidence_scores=confidence_scores,
            notes=notes
        )
        
        # Get questions for submission (already validated in verify method)
        group, questions = GroundTruthService._get_question_group_with_questions(question_group_id=question_group_id, session=session)
            
        existing_gts = session.scalars(
            select(ReviewerGroundTruth).where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id.in_([q.id for q in questions])
            )
        ).all()

        existing_map = {gt.question_id: gt for gt in existing_gts}

        # Submit each ground truth answer
        for question in questions:
            answer_value = answers[question.text]
            confidence_score = confidence_scores.get(question.text) if confidence_scores else None
            note = notes.get(question.text) if notes else None
            
            # Check for existing ground truth
            existing = existing_map.get(question.id)
        
            if existing:
                # Update existing ground truth
                existing.answer_value = answer_value
                existing.answer_type = question.type
                existing.confidence_score = confidence_score
                existing.notes = note
                existing.modified_at = datetime.now(timezone.utc)
            else:
                # Create new ground truth
                gt = ReviewerGroundTruth(
                    video_id=video_id,
                    question_id=question.id,
                    project_id=project_id,
                    reviewer_id=reviewer_id,
                    answer_type=question.type,
                    answer_value=answer_value,
                    original_answer_value=answer_value,
                    confidence_score=confidence_score,
                    notes=note
                )
                session.add(gt)
            
        session.commit()
        
        # Check and update completion status
        GroundTruthService._check_and_update_completion(user_id=reviewer_id, project_id=project_id, session=session)

    @staticmethod
    def get_ground_truth(video_id: int, project_id: int, session: Session) -> pd.DataFrame:
        """Get ground truth answers for a video in a project.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            session: Database session
            
        Returns:
            DataFrame containing ground truth answers
        """
        gts = session.scalars(
            select(ReviewerGroundTruth)
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id
            )
        ).all()
        
        return pd.DataFrame([
            {
                "Question ID": gt.question_id,
                "Answer Value": gt.answer_value,
                "Original Value": gt.original_answer_value,
                "Reviewer ID": gt.reviewer_id,
                "Modified At": gt.modified_at,
                "Modified By Admin": gt.modified_by_admin_id,
                "Modified By Admin At": gt.modified_by_admin_at,
                "Confidence Score": gt.confidence_score,
                "Created At": gt.created_at,
                "Notes": gt.notes
            }
            for gt in gts
        ])

    @staticmethod
    def get_ground_truth_for_question(video_id: int, project_id: int, question_id: int, session: Session) -> Optional[Dict]:
        """Get ground truth for a single question, returns None if no ground truth exists."""
        gt = session.scalar(
            select(ReviewerGroundTruth)
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id == question_id
            )
        )
        
        if not gt:
            return None
            
        return {
            "Question ID": gt.question_id,
            "Answer Value": gt.answer_value,
            "Original Value": gt.original_answer_value,
            "Reviewer ID": gt.reviewer_id,
            "Modified At": gt.modified_at,
            "Modified By Admin": gt.modified_by_admin_id,
            "Modified By Admin At": gt.modified_by_admin_at,
            "Confidence Score": gt.confidence_score,
            "Created At": gt.created_at,
            "Notes": gt.notes
        }

    @staticmethod
    def get_ground_truth_for_question_group(video_id: int, project_id: int, question_group_id: int, session: Session) -> pd.DataFrame:
        """Get ground truth for all questions in a specific question group."""
        
        # Use existing service method to get questions
        try:
            questions = QuestionService.get_questions_by_group_id(group_id=question_group_id, session=session)
            if not questions:
                return pd.DataFrame()
            
            question_ids = [q["id"] for q in questions]
            
            # Get ground truth for these specific questions
            gts = session.scalars(
                select(ReviewerGroundTruth)
                .where(
                    ReviewerGroundTruth.video_id == video_id,
                    ReviewerGroundTruth.project_id == project_id,
                    ReviewerGroundTruth.question_id.in_(question_ids)
                )
            ).all()
            
            return pd.DataFrame([
                {
                    "Question ID": gt.question_id,
                    "Answer Value": gt.answer_value,
                    "Original Value": gt.original_answer_value,
                    "Reviewer ID": gt.reviewer_id,
                    "Modified At": gt.modified_at,
                    "Modified By Admin": gt.modified_by_admin_id,
                    "Modified By Admin At": gt.modified_by_admin_at,
                    "Confidence Score": gt.confidence_score,
                    "Created At": gt.created_at,
                    "Notes": gt.notes
                }
                for gt in gts
            ])
        except Exception as e:
            print(f"Error in get_ground_truth_for_question_group: {e}")
            return pd.DataFrame()

    @staticmethod
    def check_ground_truth_exists_for_question(video_id: int, project_id: int, question_id: int, session: Session) -> bool:
        """Check if ground truth exists for a single question (most efficient for existence checks)."""
        return session.scalar(
            select(func.count(ReviewerGroundTruth.question_id))
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id == question_id
            )
        ) > 0

    # @staticmethod
    # TODO: See if we want to use this instead
    # def check_question_modified_by_admin_optimized(video_id: int, project_id: int, question_id: int, session: Session) -> bool:
    #     """Optimized version that doesn't fetch full ground truth data."""
    #     return session.scalar(
    #         select(func.count(ReviewerGroundTruth.question_id))
    #         .where(
    #             ReviewerGroundTruth.video_id == video_id,
    #             ReviewerGroundTruth.project_id == project_id,
    #             ReviewerGroundTruth.question_id == question_id,
    #             ReviewerGroundTruth.modified_by_admin_id.is_not(None)
    #         )
    #     ) > 0

    @staticmethod
    def check_all_questions_have_ground_truth_for_group(video_id: int, project_id: int, question_group_id: int, session: Session) -> bool:
        """Check if all questions in a question group have ground truth for a specific video.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            True if all questions in the group have ground truth, False otherwise
        """
        try:
            # Get all questions in the group
            questions = QuestionService.get_questions_by_group_id(group_id=question_group_id, session=session)
            if not questions:
                return False
            
            question_ids = [q["id"] for q in questions]
            
            # Count how many of these questions have ground truth
            gt_count = session.scalar(
                select(func.count(ReviewerGroundTruth.question_id))
                .where(
                    ReviewerGroundTruth.video_id == video_id,
                    ReviewerGroundTruth.project_id == project_id,
                    ReviewerGroundTruth.question_id.in_(question_ids)
                )
            )
            
            # Return True if all questions have ground truth
            return gt_count == len(question_ids)
            
        except Exception as e:
            print(f"Error in check_all_questions_have_ground_truth_for_group: {e}")
            return False

    @staticmethod
    def get_ground_truth_for_question(video_id: int, project_id: int, question_id: int, session: Session) -> Optional[Dict]:
        """Get ground truth answer for a specific question on a video.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project  
            question_id: The ID of the specific question
            session: Database session
            
        Returns:
            Dictionary with ground truth data for the question, or None if not found
        """
        gt = session.scalar(
            select(ReviewerGroundTruth)
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id == question_id
            )
        )
        
        if not gt:
            return None
            
        return {
            "Question ID": gt.question_id,
            "Answer Value": gt.answer_value,
            "Original Value": gt.original_answer_value,
            "Reviewer ID": gt.reviewer_id,
            "Modified At": gt.modified_at,
            "Modified By Admin": gt.modified_by_admin_id,
            "Modified By Admin At": gt.modified_by_admin_at,
            "Confidence Score": gt.confidence_score,
            "Created At": gt.created_at,
            "Notes": gt.notes
        }

    # @staticmethod
    # def get_reviewer_accuracy(project_id: int, session: Session) -> Dict[int, Dict[int, Dict[str, int]]]:
    #     """Get accuracy data for all reviewers in a project.
        
    #     Args:
    #         project_id: The ID of the project
    #         session: Database session
            
    #     Returns:
    #         Dictionary: {reviewer_id: {question_id: {"total": int, "correct": int}}}
    #         A question is correct if it was NOT modified by admin.
    #         If no ground truth exists for a question, both total and correct are 0.
            
    #     Raises:
    #         ValueError: If project not found
    #     """
    #     # Validate project exists
    #     project = session.get(Project, project_id)
    #     if not project:
    #         raise ValueError(f"Project with ID {project_id} not found")
        
    #     # Get all reviewers and questions
    #     reviewers = ProjectService.get_project_reviewers(project_id, session)
    #     questions = ProjectService.get_project_questions(project_id, session)
        
    #     # Initialize result structure
    #     accuracy_data = {}
    #     for reviewer in reviewers:
    #         reviewer_id = reviewer['id']
    #         accuracy_data[reviewer_id] = {}
    #         for question in questions:
    #             question_id = question['id']
    #             accuracy_data[reviewer_id][question_id] = {"total": 0, "correct": 0}
        
    #     # Get all ground truth answers in the project
    #     ground_truths = session.scalars(
    #         select(ReviewerGroundTruth)
    #         .where(ReviewerGroundTruth.project_id == project_id)
    #     ).all()
        
    #     # Process each ground truth answer
    #     for gt in ground_truths:
    #         reviewer_id = gt.reviewer_id
    #         question_id = gt.question_id
            
    #         # Skip if reviewer or question not in our tracking
    #         if reviewer_id not in accuracy_data or question_id not in accuracy_data[reviewer_id]:
    #             continue
                
    #         # Count total
    #         accuracy_data[reviewer_id][question_id]["total"] += 1
            
    #         # Count correct (not modified by admin)
    #         if gt.modified_by_admin_id is None:
    #             accuracy_data[reviewer_id][question_id]["correct"] += 1
        
    #     return accuracy_data

    # @staticmethod
    # def get_annotator_accuracy(project_id: int, session: Session) -> Dict[int, Dict[int, Dict[str, int]]]:
    #     """Get accuracy data for all annotators in a project.
        
    #     Args:
    #         project_id: The ID of the project
    #         session: Database session
            
    #     Returns:
    #         Dictionary: {annotator_id: {question_id: {"total": int, "correct": int}}}
    #         For single choice: correct if answer matches ground truth
    #         For description: only count if reviewed (not pending), correct if approved
            
    #     Raises:
    #         ValueError: If project not found, no videos, no answers, or incomplete ground truth
    #     """
    #     # Validate project exists
    #     project = session.get(Project, project_id)
    #     if not project:
    #         raise ValueError(f"Project with ID {project_id} not found")
        
    #     # Check if project has complete ground truth
    #     if not ProjectService.check_project_has_full_ground_truth(project_id, session):
    #         raise ValueError(f"Project {project_id} does not have complete ground truth")
        
    #     # Get all videos in project
    #     videos = session.scalars(
    #         select(ProjectVideo.video_id)
    #         .join(Video, ProjectVideo.video_id == Video.id)
    #         .where(
    #             ProjectVideo.project_id == project_id,
    #             Video.is_archived == False
    #         )
    #     ).all()
        
    #     if not videos:
    #         raise ValueError(f"No videos found in project {project_id}")
        
    #     # Get all annotator answers
    #     annotator_answers = session.scalars(
    #         select(AnnotatorAnswer)
    #         .where(AnnotatorAnswer.project_id == project_id)
    #     ).all()
        
    #     if not annotator_answers:
    #         raise ValueError(f"No annotator answers found in project {project_id}")
        
    #     # Get annotators and questions
    #     annotators = ProjectService.get_project_annotators(project_id, session)
    #     questions = ProjectService.get_project_questions(project_id, session)
        
    #     # Initialize result structure
    #     accuracy_data = {}
    #     for display_name, annotator_info in annotators.items():
    #         annotator_id = annotator_info['id']
    #         accuracy_data[annotator_id] = {}
    #         for question in questions:
    #             question_id = question['id']
    #             accuracy_data[annotator_id][question_id] = {"total": 0, "correct": 0}
        
    #     # Process each annotator answer
    #     for answer in annotator_answers:
    #         annotator_id = answer.user_id
    #         question_id = answer.question_id
    #         video_id = answer.video_id
            
    #         # Skip if annotator or question not in our tracking
    #         if annotator_id not in accuracy_data or question_id not in accuracy_data[annotator_id]:
    #             continue
            
    #         # Get question details
    #         question = session.get(Question, question_id)
    #         if not question:
    #             continue
                
    #         if question.type == "single":
    #             # For single choice questions
    #             # Get ground truth
    #             gt = session.get(ReviewerGroundTruth, (video_id, question_id, project_id))
    #             if gt:
    #                 accuracy_data[annotator_id][question_id]["total"] += 1
    #                 if answer.answer_value == gt.answer_value:
    #                     accuracy_data[annotator_id][question_id]["correct"] += 1
                        
    #         elif question.type == "description":
    #             # For description questions, check review status
    #             review = session.scalar(
    #                 select(AnswerReview)
    #                 .where(AnswerReview.answer_id == answer.id)
    #             )
                
    #             # Only count if reviewed (not pending or missing)
    #             if review and review.status != "pending":
    #                 accuracy_data[annotator_id][question_id]["total"] += 1
    #                 if review.status == "approved":
    #                     accuracy_data[annotator_id][question_id]["correct"] += 1
        
    #     return accuracy_data

    # OPTIMIZED VERSIONS - Drop-in replacements for GroundTruthService methods
    # These should be added to the GroundTruthService class in your main file
    # Requires these imports at top of file:
    # from sqlalchemy import select, func, distinct, case, and_

    @staticmethod
    def get_reviewer_accuracy(project_id: int, session: Session) -> Dict[int, Dict[int, Dict[str, int]]]:
        """Get accuracy data for all reviewers in a project.
        
        Uses a single complex query with JOINs instead of multiple queries + loops.
        Should be 10-100x faster than the original implementation.
        """
        try:
            # Single query to get all reviewer accuracy data with JOINs
            query = select(
                ReviewerGroundTruth.reviewer_id,
                ReviewerGroundTruth.question_id,
                func.count().label('total_count'),
                func.sum(
                    case(
                        (ReviewerGroundTruth.modified_by_admin_id.is_(None), 1),
                        else_=0
                    )
                ).label('correct_count')
            ).select_from(
                ReviewerGroundTruth
            ).join(
                Question, ReviewerGroundTruth.question_id == Question.id
            ).join(
                User, ReviewerGroundTruth.reviewer_id == User.id
            ).where(
                ReviewerGroundTruth.project_id == project_id,
                Question.is_archived == False,
                User.is_archived == False
            ).group_by(
                ReviewerGroundTruth.reviewer_id,
                ReviewerGroundTruth.question_id
            )
            
            result = session.execute(query).all()
            
            # Get all reviewers and questions that should be in the result
            # Use original logic: only reviewers who actually submitted ground truth
            reviewers_query = select(
                ReviewerGroundTruth.reviewer_id
            ).where(
                ReviewerGroundTruth.project_id == project_id
            ).distinct()
            
            questions_query = select(
                Question.id
            ).select_from(
                Question
            ).join(
                QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id
            ).join(
                SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id
            ).join(
                Project, SchemaQuestionGroup.schema_id == Project.schema_id
            ).where(
                Project.id == project_id,
                Question.is_archived == False
            )
            
            reviewer_ids = session.scalars(reviewers_query).all()
            question_ids = session.scalars(questions_query).all()
            
            # Initialize result structure with all combinations
            accuracy_data = {}
            for reviewer_id in reviewer_ids:
                accuracy_data[reviewer_id] = {}
                for question_id in question_ids:
                    accuracy_data[reviewer_id][question_id] = {"total": 0, "correct": 0}
            
            # Fill in actual data from query result
            for row in result:
                reviewer_id = row.reviewer_id
                question_id = row.question_id
                
                if reviewer_id in accuracy_data and question_id in accuracy_data[reviewer_id]:
                    accuracy_data[reviewer_id][question_id] = {
                        "total": int(row.total_count),
                        "correct": int(row.correct_count)
                    }
            
            return accuracy_data
            
        except Exception as e:
            raise ValueError(f"Error getting reviewer accuracy: {str(e)}")


    @staticmethod  
    def get_annotator_accuracy(project_id: int, session: Session) -> Dict[int, Dict[int, Dict[str, int]]]:
        """Get accuracy data for all annotators in a project.
        
        Uses bulk queries with JOINs instead of per-item queries.
        Should be 50-500x faster than the original implementation.
        """
        try:
            # Validate project exists
            project = session.get(Project, project_id)
            if not project:
                raise ValueError(f"Project with ID {project_id} not found")
            
            # Check if project has complete ground truth using original method
            if not ProjectService.check_project_has_full_ground_truth(project_id, session):
                raise ValueError(f"Project {project_id} does not have complete ground truth")

            # Get all annotators for this project in one query
            annotators_query = select(
                User.id,
                User.user_id_str
            ).select_from(
                User
            ).join(
                AnnotatorAnswer, User.id == AnnotatorAnswer.user_id
            ).where(
                AnnotatorAnswer.project_id == project_id,
                User.is_archived == False
            ).distinct()
            
            annotators_result = session.execute(annotators_query).all()
            if not annotators_result:
                raise ValueError(f"No annotator answers found in project {project_id}")
            
            # Get all questions for this project
            questions_query = select(
                Question.id,
                Question.type
            ).select_from(
                Question
            ).join(
                QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id
            ).join(
                SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id
            ).where(
                SchemaQuestionGroup.schema_id == project.schema_id,
                Question.is_archived == False
            )
            
            questions_result = session.execute(questions_query).all()
            question_types = {q.id: q.type for q in questions_result}
            
            # Initialize result structure
            accuracy_data = {}
            for annotator in annotators_result:
                accuracy_data[annotator.id] = {}
                for question in questions_result:
                    accuracy_data[annotator.id][question.id] = {"total": 0, "correct": 0}
            
            # SINGLE CHOICE QUESTIONS: Get all data with one big JOIN query
            single_choice_query = select(
                AnnotatorAnswer.user_id,
                AnnotatorAnswer.question_id,
                AnnotatorAnswer.video_id,
                AnnotatorAnswer.answer_value.label('annotator_answer'),
                ReviewerGroundTruth.answer_value.label('ground_truth_answer')
            ).select_from(
                AnnotatorAnswer
            ).join(
                ReviewerGroundTruth, 
                and_(
                    AnnotatorAnswer.video_id == ReviewerGroundTruth.video_id,
                    AnnotatorAnswer.question_id == ReviewerGroundTruth.question_id,
                    AnnotatorAnswer.project_id == ReviewerGroundTruth.project_id
                )
            ).join(
                Question, AnnotatorAnswer.question_id == Question.id
            ).where(
                AnnotatorAnswer.project_id == project_id,
                Question.type == "single",
                Question.is_archived == False
            )
            
            single_choice_result = session.execute(single_choice_query).all()
            
            # Process single choice results
            for row in single_choice_result:
                user_id = row.user_id
                question_id = row.question_id
                
                if user_id in accuracy_data and question_id in accuracy_data[user_id]:
                    accuracy_data[user_id][question_id]["total"] += 1
                    if row.annotator_answer == row.ground_truth_answer:
                        accuracy_data[user_id][question_id]["correct"] += 1
            
            # DESCRIPTION QUESTIONS: Get all review data with one query
            description_query = select(
                AnnotatorAnswer.id.label('answer_id'),
                AnnotatorAnswer.user_id,
                AnnotatorAnswer.question_id,
                AnswerReview.status
            ).select_from(
                AnnotatorAnswer
            ).join(
                Question, AnnotatorAnswer.question_id == Question.id
            ).outerjoin(  # LEFT JOIN to include answers without reviews
                AnswerReview, AnnotatorAnswer.id == AnswerReview.answer_id
            ).where(
                AnnotatorAnswer.project_id == project_id,
                Question.type == "description",
                Question.is_archived == False
            )
            
            description_result = session.execute(description_query).all()
            
            # Process description results
            for row in description_result:
                user_id = row.user_id
                question_id = row.question_id
                status = row.status
                
                if user_id in accuracy_data and question_id in accuracy_data[user_id]:
                    # Only count if reviewed (not pending or missing)
                    if status and status != "pending":
                        accuracy_data[user_id][question_id]["total"] += 1
                        if status == "approved":
                            accuracy_data[user_id][question_id]["correct"] += 1
            
            return accuracy_data
            
        except Exception as e:
            raise ValueError(f"Error getting annotator accuracy: {str(e)}")


    @staticmethod
    def get_project_accuracy_summary(project_id: int, session: Session) -> Dict[str, Any]:
        """Get complete accuracy summary for a project in one call.
        
        Returns both reviewer and annotator accuracy plus summary statistics.
        Much faster than calling the individual methods separately.
        """
        try:
            reviewer_accuracy = GroundTruthService.get_reviewer_accuracy(project_id, session)
            annotator_accuracy = GroundTruthService.get_annotator_accuracy(project_id, session)
            
            # Calculate summary statistics
            reviewer_stats = {}
            for reviewer_id, questions in reviewer_accuracy.items():
                total_answers = sum(q["total"] for q in questions.values())
                correct_answers = sum(q["correct"] for q in questions.values())
                accuracy_pct = (correct_answers / total_answers * 100) if total_answers > 0 else 0
                
                reviewer_stats[reviewer_id] = {
                    "total_answers": total_answers,
                    "correct_answers": correct_answers,
                    "accuracy_percentage": round(accuracy_pct, 2)
                }
            
            annotator_stats = {}
            for annotator_id, questions in annotator_accuracy.items():
                total_answers = sum(q["total"] for q in questions.values())
                correct_answers = sum(q["correct"] for q in questions.values())
                accuracy_pct = (correct_answers / total_answers * 100) if total_answers > 0 else 0
                
                annotator_stats[annotator_id] = {
                    "total_answers": total_answers,
                    "correct_answers": correct_answers,
                    "accuracy_percentage": round(accuracy_pct, 2)
                }
            
            return {
                "reviewer_accuracy": reviewer_accuracy,
                "annotator_accuracy": annotator_accuracy,
                "reviewer_summary": reviewer_stats,
                "annotator_summary": annotator_stats,
                "project_id": project_id
            }
            
        except Exception as e:
            raise ValueError(f"Error getting project accuracy summary: {str(e)}")

    @staticmethod
    def override_ground_truth_to_question_group(
        video_id: int,
        project_id: int,
        question_group_id: int,
        admin_id: int,
        answers: Dict[str, str],  # Maps question text to answer value
        session: Session
    ) -> None:
        """Override ground truth answers for all questions in a question group (admin only).
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_group_id: The ID of the question group
            admin_id: The ID of the admin
            answers: Dictionary mapping question text to answer value
            session: Database session
            
        Raises:
            ValueError: If validation fails
        """
        # Validate project and admin
        project, admin = GroundTruthService._validate_project_and_user(project_id=project_id, user_id=admin_id, session=session)
        
        # Validate project admin role
        GroundTruthService._validate_user_role(user_id=admin_id, project_id=project_id, required_role="admin", session=session)
            
        # Validate question group and get questions
        group, questions = GroundTruthService._get_question_group_with_questions(question_group_id=question_group_id, session=session)
        
        # Validate answers match questions
        GroundTruthService._validate_answers_match_questions(answers=answers, questions=questions)
            
        # Run verification if specified
        GroundTruthService._run_verification(group=group, answers=answers)
            
        # Override each ground truth answer
        for question in questions:
            answer_value = answers[question.text]
            
            # Validate answer value
            GroundTruthService._validate_answer_value(question=question, answer_value=answer_value)
            
            # Get ground truth
            gt = session.get(ReviewerGroundTruth, (video_id, question.id, project_id))
            if not gt:
                raise ValueError(f"No ground truth found for video {video_id}, question {question.id}, project {project_id}")
            
            # Only update if answer value actually changes
            if gt.answer_value != answer_value:
                gt.answer_value = answer_value
                gt.modified_by_admin_id = admin_id
                gt.modified_by_admin_at = datetime.now(timezone.utc)
        
        session.commit()
    

    @staticmethod
    def batch_check_admin_modifications(video_id: int, project_id: int, question_ids: List[int], session: Session) -> Dict[int, Dict]:
        """Batch check admin modifications for multiple questions - MUCH FASTER"""
        try:
            from sqlalchemy import select
            from label_pizza.models import ReviewerGroundTruth, User
            
            admin_modifications = {}
            
            if not question_ids:
                return {}
            
            # Single query to get all ground truth records for these questions
            query = select(
                ReviewerGroundTruth.question_id,
                ReviewerGroundTruth.modified_by_admin_id,
                ReviewerGroundTruth.modified_by_admin_at,
                ReviewerGroundTruth.original_answer_value,
                ReviewerGroundTruth.answer_value
            ).where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id.in_(question_ids)
            )
            
            results = session.execute(query).all()
            
            # Get all admin user names in one query if needed
            admin_user_ids = [r.modified_by_admin_id for r in results if r.modified_by_admin_id]
            admin_users = {}
            
            if admin_user_ids:
                users_query = select(User.id, User.user_id_str).where(User.id.in_(admin_user_ids))
                user_results = session.execute(users_query).all()
                admin_users = {r.id: r.user_id_str for r in user_results}
            
            # Build result dictionary
            for result in results:
                question_id = result.question_id
                is_modified = result.modified_by_admin_id is not None
                
                admin_modifications[question_id] = {
                    "is_modified": is_modified,
                    "admin_info": {
                        "current_value": result.answer_value,
                        "original_value": result.original_answer_value,
                        "admin_id": result.modified_by_admin_id,
                        "admin_name": admin_users.get(result.modified_by_admin_id, f"Admin {result.modified_by_admin_id}") if result.modified_by_admin_id else None,
                        "modified_at": result.modified_by_admin_at
                    } if is_modified else None
                }
            
            # Fill in False for questions not found (no ground truth)
            for question_id in question_ids:
                if question_id not in admin_modifications:
                    admin_modifications[question_id] = {
                        "is_modified": False,
                        "admin_info": None
                    }
            
            return admin_modifications
            
        except Exception as e:
            print(f"Error in batch_check_admin_modifications: {e}")
            return {qid: {"is_modified": False, "admin_info": None} for qid in question_ids}

    @staticmethod
    def submit_answer_review(
        answer_id: int,
        reviewer_id: int,
        status: str,  # "approved"/"rejected"/"pending"
        session: Session,
        comment: Optional[str] = None  # Optional comment for the review
    ) -> None:
        """Submit a review for a description-type answer.
        
        Args:
            answer_id: The ID of the answer to review
            reviewer_id: The ID of the reviewer
            status: Review status ("approved"/"rejected"/"pending")
            session: Database session
            comment: Optional review comment
            
        Raises:
            ValueError: If validation fails
        """
        # Get the answer
        answer = session.get(AnnotatorAnswer, answer_id)
        if not answer:
            raise ValueError(f"Answer with ID {answer_id} not found")
            
        # Get the question
        question = session.get(Question, answer.question_id)
        if not question:
            raise ValueError(f"Question with ID {answer.question_id} not found")
            
        # Verify question is description type
        if question.type != "description":
            raise ValueError(f"Question '{question.text}' is not a description type question")
            
        # Validate reviewer role
        GroundTruthService._validate_user_role(user_id=reviewer_id, project_id=answer.project_id, required_role="reviewer", session=session)
        
        # Validate review status
        valid_statuses = {"approved", "rejected", "pending"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid review status: {status}. Must be one of {valid_statuses}")
        
        # Create or update review
        review = session.scalar(
            select(AnswerReview)
            .where(AnswerReview.answer_id == answer_id)
        )
        
        if review:
            # Update existing review
            review.status = status
            review.comment = comment
            review.reviewer_id = reviewer_id  # Update reviewer in case it's different
            review.reviewed_at = datetime.now(timezone.utc)
        else:
            # Create new review
            review = AnswerReview(
                answer_id=answer_id,
                reviewer_id=reviewer_id,
                status=status,
                comment=comment
            )
            session.add(review)
            
        session.commit()

    @staticmethod
    def get_answer_review(answer_id: int, session: Session) -> Optional[dict]:
        """Get the review for a specific answer.
        
        Args:
            answer_id: The ID of the answer
            session: Database session
            
        Returns:
            Dictionary containing the review with keys:
            - status: The review status ("approved"/"rejected"/"pending")
            - comment: The review comment
            - reviewer_id: The ID of the reviewer
            - reviewed_at: When the review was made
            Returns None if no review exists
        """
        review = session.scalar(
            select(AnswerReview)
            .where(AnswerReview.answer_id == int(answer_id))
        )
        
        if not review:
            return None
            
        return {
            "status": review.status,
            "comment": review.comment,
            "reviewer_id": review.reviewer_id,
            "reviewed_at": review.reviewed_at
        }
    
    @staticmethod
    def check_question_modified_by_admin(video_id: int, project_id: int, question_id: int, session: Session) -> bool:
        """Check if a question's ground truth has been modified by admin.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project  
            question_id: The ID of the question
            session: Database session
            
        Returns:
            True if the question has been modified by admin, False otherwise
        """
        gt = session.get(ReviewerGroundTruth, (video_id, question_id, project_id))
        return gt is not None and gt.modified_by_admin_id is not None

    @staticmethod
    def get_admin_modification_details(video_id: int, project_id: int, question_id: int, session: Session) -> Optional[Dict[str, Any]]:
        """Get admin modification details for a question.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_id: The ID of the question
            session: Database session
            
        Returns:
            Dictionary with admin modification details or None if not modified by admin.
            Contains: original_value, current_value, admin_id, admin_name, modified_at
        """
        gt = session.get(ReviewerGroundTruth, (video_id, question_id, project_id))
        if not gt or gt.modified_by_admin_id is None:
            return None
        
        # Get admin user info
        admin_user = session.get(User, gt.modified_by_admin_id)
        admin_name = admin_user.user_id_str if admin_user else f"Admin {gt.modified_by_admin_id}"
        
        return {
            "original_value": gt.original_answer_value,
            "current_value": gt.answer_value,
            "admin_id": gt.modified_by_admin_id,
            "admin_name": admin_name,
            "modified_at": gt.modified_by_admin_at
        }

    @staticmethod
    def check_all_questions_modified_by_admin(video_id: int, project_id: int, question_group_id: int, session: Session) -> bool:
        """Check if all questions in a group have been modified by admin.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            True if all questions in the group have been modified by admin, False otherwise
        """
        # Get all questions in the group
        questions = session.scalars(
            select(Question)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(QuestionGroupQuestion.question_group_id == question_group_id)
        ).all()
        
        if not questions:
            return False
        
        # Check if all questions have been modified by admin
        for question in questions:
            if not GroundTruthService.check_question_modified_by_admin(video_id=video_id, project_id=project_id, question_id=question.id, session=session):
                return False
        
        return True

    @staticmethod
    def get_question_option_selections(video_id: int, project_id: int, question_id: int, annotator_user_ids: List[int], session: Session) -> Dict[str, List[Dict[str, str]]]:
        """Get who selected which option for a single-choice question.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_id: The ID of the question
            annotator_user_ids: List of annotator user IDs to include
            session: Database session
            
        Returns:
            Dictionary mapping option values to list of selector info dicts.
            Each selector dict contains: name, initials, type ('annotator' or 'ground_truth')
        """
        option_selections = {}
        
        # Track which users we've already processed to avoid duplicates
        processed_users = set()
        
        # Get annotator answers
        for user_id in annotator_user_ids:
            # Skip if we've already processed this user
            if user_id in processed_users:
                continue
            processed_users.add(user_id)
            
            answer = session.scalar(
                select(AnnotatorAnswer)
                .where(
                    AnnotatorAnswer.video_id == video_id,
                    AnnotatorAnswer.project_id == project_id,
                    AnnotatorAnswer.question_id == question_id,
                    AnnotatorAnswer.user_id == user_id
                )
            )
            
            if answer:
                user = session.get(User, user_id)
                if user:
                    name_parts = user.user_id_str.split()
                    if len(name_parts) >= 2:
                        initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
                    else:
                        initials = user.user_id_str[:2].upper()
                    
                    option_value = answer.answer_value
                    if option_value not in option_selections:
                        option_selections[option_value] = []
                    
                    # Check if this user is already in the list for this option
                    user_already_listed = any(
                        selector["name"] == user.user_id_str and selector["type"] == "annotator"
                        for selector in option_selections[option_value]
                    )
                    
                    if not user_already_listed:
                        option_selections[option_value].append({
                            "name": user.user_id_str,
                            "initials": initials,
                            "type": "annotator"
                        })
        
        # Get ground truth answer
        gt_answer = session.scalar(
            select(ReviewerGroundTruth)
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                ReviewerGroundTruth.question_id == question_id
            )
        )
        
        if gt_answer:
            option_value = gt_answer.answer_value
            if option_value not in option_selections:
                option_selections[option_value] = []
            
            # Check if ground truth is already listed
            gt_already_listed = any(
                selector["type"] == "ground_truth"
                for selector in option_selections[option_value]
            )
            
            if not gt_already_listed:
                option_selections[option_value].append({
                    "name": "Ground Truth",
                    "initials": "GT",
                    "type": "ground_truth"
                })
        
        return option_selections

    @staticmethod
    def get_question_text_answers(video_id: int, project_id: int, question_id: int, annotator_user_ids: List[int], session: Session) -> List[Dict[str, str]]:
        """Get text answers for a description question.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_id: The ID of the question
            annotator_user_ids: List of annotator user IDs to include
            session: Database session
            
        Returns:
            List of answer dictionaries containing: name, initials, answer_value
        """
        text_answers = []
        
        for user_id in annotator_user_ids:
            answer = session.scalar(
                select(AnnotatorAnswer)
                .where(
                    AnnotatorAnswer.video_id == video_id,
                    AnnotatorAnswer.project_id == project_id,
                    AnnotatorAnswer.question_id == question_id,
                    AnnotatorAnswer.user_id == user_id
                )
            )
            
            if answer:
                user = session.get(User, user_id)
                if user:
                    name_parts = user.user_id_str.split()
                    if len(name_parts) >= 2:
                        initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
                    else:
                        initials = user.user_id_str[:2].upper()
                    
                    text_answers.append({
                        "name": user.user_id_str,
                        "initials": initials,
                        "answer_value": answer.answer_value
                    })
        
        return text_answers
    
    @staticmethod
    def get_ground_truth_dict_for_question_group(video_id: int, project_id: int, question_group_id: int, session: Session) -> Dict[str, str]:
        """Get existing ground truth for a video and question group.
        
        Args:
            video_id: The ID of the video
            project_id: The ID of the project
            question_group_id: The ID of the question group
            session: Database session
            
        Returns:
            Dictionary mapping question text to answer value
        """
        gts = session.scalars(
            select(ReviewerGroundTruth)
            .join(Question, ReviewerGroundTruth.question_id == Question.id)
            .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
            .where(
                ReviewerGroundTruth.video_id == video_id,
                ReviewerGroundTruth.project_id == project_id,
                QuestionGroupQuestion.question_group_id == question_group_id
            )
        ).all()
        
        result = {}
        for gt in gts:
            question = session.get(Question, gt.question_id)
            if question:
                result[question.text] = gt.answer_value
        
        return result
    
    @staticmethod
    def search_videos_by_criteria_optimized(
        criteria: List[Dict], 
        match_all: bool, 
        session: Session,
        progress_callback=None
    ) -> List[Dict]:
        """Optimized search for videos matching ground truth criteria with progress tracking"""
        
        if not criteria:
            return []
        
        try:
            # Step 1: Get all relevant project-video mappings in batch
            if progress_callback:
                progress_callback(1, 5, "Loading project-video mappings...")
            
            project_ids = list(set(c["project_id"] for c in criteria))
            project_video_map = GroundTruthService._get_project_video_mappings(project_ids, session)
            
            # Step 2: Get all relevant ground truth data in batch  
            if progress_callback:
                progress_callback(2, 5, "Loading ground truth data...")
            
            question_ids = list(set(c["question_id"] for c in criteria))
            ground_truth_map = GroundTruthService._get_ground_truth_mappings(
                project_ids, question_ids, session
            )
            
            # Step 3: Get all video info
            if progress_callback:
                progress_callback(3, 5, "Loading video information...")
            
            all_video_ids = set()
            for project_videos in project_video_map.values():
                all_video_ids.update(project_videos)
            
            video_info_map = GroundTruthService._get_video_info_mappings(all_video_ids, session)
            
            # Step 4: Process matches
            if progress_callback:
                progress_callback(4, 5, "Processing criteria matches...")
            
            matching_videos = []
            processed_videos = set()
            
            total_combinations = len(all_video_ids)
            current_combination = 0
            
            for video_id in all_video_ids:
                current_combination += 1
                
                if progress_callback and current_combination % 10 == 0:
                    progress_callback(
                        4, 5, 
                        f"Processing video {current_combination}/{total_combinations}: {video_info_map.get(video_id, {}).get('uid', 'Unknown')}"
                    )
                
                if video_id in processed_videos:
                    continue
                processed_videos.add(video_id)
                
                video_info = video_info_map.get(video_id)
                if not video_info:
                    continue
                
                # Check all criteria for this video
                matches = []
                
                for criterion in criteria:
                    project_id = criterion["project_id"]
                    question_id = criterion["question_id"]
                    required_answer = criterion["required_answer"]
                    
                    # Check if video is in this project
                    if video_id not in project_video_map.get(project_id, set()):
                        matches.append(False)
                        continue
                    
                    # Check ground truth
                    gt_key = (video_id, project_id, question_id)
                    if gt_key in ground_truth_map:
                        actual_answer = ground_truth_map[gt_key]
                        matches.append(str(actual_answer) == str(required_answer))
                    else:
                        matches.append(False)
                
                # Apply match logic
                if match_all and all(matches):
                    matching_videos.append({
                        "video_info": video_info,
                        "matches": matches,
                        "criteria": criteria
                    })
                elif not match_all and any(matches):
                    matching_videos.append({
                        "video_info": video_info,
                        "matches": matches,
                        "criteria": criteria
                    })
            
            # Step 5: Complete
            if progress_callback:
                progress_callback(5, 5, f"Search complete! Found {len(matching_videos)} matching videos.")
            
            return matching_videos
            
        except Exception as e:
            raise ValueError(f"Error in optimized ground truth search: {str(e)}")
    
    @staticmethod
    def _get_project_video_mappings(project_ids: List[int], session: Session) -> Dict[int, set]:
        """Get all project-video mappings in batch"""
        
        project_video_map = {}
        
        # Single query to get all project-video relationships
        query = select(
            ProjectVideo.project_id,
            ProjectVideo.video_id
        ).select_from(
            ProjectVideo
        ).join(
            Video, ProjectVideo.video_id == Video.id
        ).where(
            ProjectVideo.project_id.in_(project_ids),
            Video.is_archived == False
        )
        
        result = session.execute(query).all()
        
        for row in result:
            project_id = row.project_id
            video_id = row.video_id
            
            if project_id not in project_video_map:
                project_video_map[project_id] = set()
            project_video_map[project_id].add(video_id)
        
        return project_video_map
    
    @staticmethod
    def _get_ground_truth_mappings(
        project_ids: List[int], 
        question_ids: List[int], 
        session: Session
    ) -> Dict[Tuple[int, int, int], str]:
        """Get all relevant ground truth data in batch"""
        
        ground_truth_map = {}
        
        # Single query to get all relevant ground truth answers
        query = select(
            ReviewerGroundTruth.video_id,
            ReviewerGroundTruth.project_id,
            ReviewerGroundTruth.question_id,
            ReviewerGroundTruth.answer_value
        ).where(
            ReviewerGroundTruth.project_id.in_(project_ids),
            ReviewerGroundTruth.question_id.in_(question_ids)
        )
        
        result = session.execute(query).all()
        
        for row in result:
            key = (row.video_id, row.project_id, row.question_id)
            ground_truth_map[key] = row.answer_value
        
        return ground_truth_map
    
    @staticmethod
    def _get_video_info_mappings(video_ids: set, session: Session) -> Dict[int, Dict]:
        """Get video info for all videos in batch"""
        
        video_info_map = {}
        
        if not video_ids:
            return video_info_map
        
        # Single query to get all video info
        query = select(
            Video.id,
            Video.video_uid,
            Video.url,
            Video.video_metadata,
            Video.created_at,
            Video.is_archived
        ).where(
            Video.id.in_(video_ids),
            Video.is_archived == False
        )
        
        result = session.execute(query).all()
        
        for row in result:
            video_info_map[row.id] = {
                "id": row.id,
                "uid": row.video_uid,
                "url": row.url,
                "metadata": row.video_metadata or {},
                "created_at": row.created_at,
                "is_archived": row.is_archived
            }
        
        return video_info_map

    @staticmethod
    def search_projects_by_completion_optimized(
        project_ids: List[int],
        completion_filter: str,
        session: Session,
        progress_callback=None
    ) -> List[Dict]:
        """Optimized search for projects by completion status with progress tracking"""
        
        try:
            results = []
            
            if progress_callback:
                progress_callback(1, 4, "Loading project data...")
            
            # Get all project info in batch
            projects = session.execute(
                select(Project.id, Project.name)
                .where(Project.id.in_(project_ids))
            ).all()
            
            project_map = {p.id: p.name for p in projects}
            
            if progress_callback:
                progress_callback(2, 4, "Loading project videos and questions...")
            
            # Get all project videos and questions in batch
            project_video_map = {}
            project_question_counts = {}
            
            # Get videos for all projects
            video_query = select(
                ProjectVideo.project_id,
                ProjectVideo.video_id,
                Video.video_uid,
                Video.url
            ).select_from(
                ProjectVideo
            ).join(
                Video, ProjectVideo.video_id == Video.id
            ).where(
                ProjectVideo.project_id.in_(project_ids),
                Video.is_archived == False
            )
            
            video_results = session.execute(video_query).all()
            
            for row in video_results:
                project_id = row.project_id
                if project_id not in project_video_map:
                    project_video_map[project_id] = []
                
                project_video_map[project_id].append({
                    "id": row.video_id,
                    "uid": row.video_uid,
                    "url": row.url
                })
            
            # Get question counts for all projects
            for project_id in project_ids:
                project = session.get(Project, project_id)
                if project:
                    question_count = session.scalar(
                        select(func.count())
                        .select_from(Question)
                        .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                        .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                        .where(
                            SchemaQuestionGroup.schema_id == project.schema_id,
                            Question.is_archived == False
                        )
                    )
                    project_question_counts[project_id] = question_count
            
            if progress_callback:
                progress_callback(3, 4, "Loading ground truth data...")
            
            # Get all ground truth data in batch
            gt_query = select(
                ReviewerGroundTruth.video_id,
                ReviewerGroundTruth.project_id,
                func.count().label('gt_count')
            ).where(
                ReviewerGroundTruth.project_id.in_(project_ids)
            ).group_by(
                ReviewerGroundTruth.video_id,
                ReviewerGroundTruth.project_id
            )
            
            gt_results = session.execute(gt_query).all()
            gt_map = {}  # (video_id, project_id) -> count
            
            for row in gt_results:
                key = (row.video_id, row.project_id)
                gt_map[key] = row.gt_count
            
            if progress_callback:
                progress_callback(4, 4, "Processing completion status...")
            
            # Process each project-video combination
            total_combinations = sum(len(videos) for videos in project_video_map.values())
            current_combination = 0
            
            for project_id in project_ids:
                project_name = project_map.get(project_id, f"Project {project_id}")
                videos = project_video_map.get(project_id, [])
                total_questions = project_question_counts.get(project_id, 0)
                
                for video in videos:
                    current_combination += 1
                    
                    if progress_callback and current_combination % 20 == 0:
                        progress_callback(
                            4, 4,
                            f"Processing {current_combination}/{total_combinations}: {video['uid']}"
                        )
                    
                    video_id = video["id"]
                    
                    # Get completion status
                    gt_count = gt_map.get((video_id, project_id), 0)
                    
                    if total_questions == 0:
                        completion_status = "no_questions"
                    elif gt_count == 0:
                        completion_status = "missing"
                    elif gt_count == total_questions:
                        completion_status = "complete"
                    else:
                        completion_status = "partial"
                    
                    # Apply filter
                    include_video = False
                    
                    if completion_filter == "All videos":
                        include_video = True
                    elif completion_filter == "Complete ground truth" and completion_status == "complete":
                        include_video = True
                    elif completion_filter == "Missing ground truth" and completion_status == "missing":
                        include_video = True
                    elif completion_filter == "Partial ground truth" and completion_status == "partial":
                        include_video = True
                    
                    if include_video:
                        results.append({
                            "video_id": video_id,
                            "video_uid": video["uid"],
                            "video_url": video["url"],
                            "project_id": project_id,
                            "project_name": project_name,
                            "completion_status": completion_status,
                            "completed_questions": gt_count,
                            "total_questions": total_questions
                        })
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error in optimized completion search: {str(e)}")


class ProjectGroupService:
    @staticmethod
    def verify_create_project_group(name: str, description: str, project_ids: list[int] | None, session: Session) -> ProjectGroup:
        """Verify create project group with optional list of project IDs, enforcing uniqueness constraints."""
        # Check for unique name
        existing = session.scalar(select(ProjectGroup).where(ProjectGroup.name == name))
        if existing:
            raise ValueError(f"Project group with name '{name}' already exists")

        if project_ids:
            ProjectGroupService._validate_project_group_uniqueness(project_ids=project_ids, session=session)
        
    @staticmethod
    def create_project_group(name: str, description: str, project_ids: list[int] | None, session: Session) -> ProjectGroup:
        """Create a new project group with optional list of project IDs, enforcing uniqueness constraints."""
        # Check for unique name
        ProjectGroupService.verify_create_project_group(name=name, description=description, project_ids=project_ids, session=session)

        group = ProjectGroup(
            name=name,
            description=description,
        )
        session.add(group)
        session.flush()  # get group.id

        if project_ids:
            for pid in project_ids:
                session.add(ProjectGroupProject(project_group_id=group.id, project_id=pid))

        session.commit()
        return group

    @staticmethod
    def verify_edit_project_group(group_id: int, name: str | None, description: str | None, add_project_ids: list[int] | None, remove_project_ids: list[int] | None, session: Session) -> ProjectGroup:
        """Verify edit project group with optional list of project IDs, enforcing uniqueness constraints when adding."""
        group = session.get(ProjectGroup, group_id)
        if not group:
            raise ValueError(f"Project group with ID {group_id} not found")
        if name:
            # Check for unique name
            existing = session.scalar(select(ProjectGroup).where(ProjectGroup.name == name, ProjectGroup.id != group_id))
            if existing and existing.id != group_id:
                raise ValueError(f"Project group with name '{name}' already exists")
        if add_project_ids:
            # Get current project IDs
            current_ids = set(row.project_id for row in session.scalars(select(ProjectGroupProject).where(ProjectGroupProject.project_group_id == group_id)).all())
            new_ids = set(add_project_ids)
            all_ids = list(current_ids | new_ids)
            ProjectGroupService._validate_project_group_uniqueness(project_ids=all_ids, session=session)
        if remove_project_ids:
            current_ids = set(row.project_id for row in session.scalars(select(ProjectGroupProject).where(ProjectGroupProject.project_group_id == group_id)).all())
            for pid in remove_project_ids:
                if pid not in current_ids:
                    raise ValueError(f"Project with ID {pid} not found in group")
        
    
    @staticmethod
    def edit_project_group(group_id: int, name: str | None, description: str | None, add_project_ids: list[int] | None, remove_project_ids: list[int] | None, session: Session) -> ProjectGroup:
        """Edit group name/description, add/remove projects, enforcing uniqueness constraints when adding."""
        ProjectGroupService.verify_edit_project_group(group_id=group_id, name=name, description=description, add_project_ids=add_project_ids, remove_project_ids=remove_project_ids, session=session)
        group = session.get(ProjectGroup, group_id)
        if name:
            group.name = name
        if description:
            group.description = description
        if add_project_ids:
            # Get current project IDs
            current_ids = set(row.project_id for row in session.scalars(select(ProjectGroupProject).where(ProjectGroupProject.project_group_id == group_id)).all())
            new_ids = set(add_project_ids)
            all_ids = list(current_ids | new_ids)
            ProjectGroupService._validate_project_group_uniqueness(project_ids=all_ids, session=session)
            for pid in new_ids - current_ids:
                session.add(ProjectGroupProject(project_group_id=group_id, project_id=pid))
        if remove_project_ids:
            current_ids = set(row.project_id for row in session.scalars(select(ProjectGroupProject).where(ProjectGroupProject.project_group_id == group_id)).all())
            for pid in remove_project_ids:
                row = session.scalar(select(ProjectGroupProject).where(ProjectGroupProject.project_group_id == group_id, ProjectGroupProject.project_id == pid))
                if row:
                    session.delete(row)
        session.commit()
        return group

    @staticmethod
    def get_project_group_by_id(group_id: int, session: Session):
        group = session.get(ProjectGroup, group_id)
        if not group:
            raise ValueError(f"Project group with ID {group_id} not found")
        projects = session.scalars(
            select(Project).join(ProjectGroupProject, Project.id == ProjectGroupProject.project_id)
            .where(ProjectGroupProject.project_group_id == group_id)
        ).all()
        # Convert group to dict
        group = {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "created_at": group.created_at
        }
        # Convert projects to dicts
        projects = [
            {
                "id": p.id,
                "name": p.name,
                "schema_id": p.schema_id,
                "description": p.description,
                "created_at": p.created_at,
                "is_archived": p.is_archived
            }
            for p in projects
        ]
        return {"group": group, "projects": projects}
    
    @staticmethod
    def get_project_group_by_name(name: str, session: Session):
        """Get a project group by name."""
        group = session.scalar(select(ProjectGroup).where(ProjectGroup.name == name))
        if not group:
            raise ValueError(f"Project group with name '{name}' not found")
        return group

    
    @staticmethod
    def list_project_groups(session: Session):
        groups = session.scalars(select(ProjectGroup)).all()
        groups = [
            {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "created_at": g.created_at
            }
            for g in groups
        ]
        return groups

    @staticmethod
    def _validate_project_group_uniqueness(project_ids: list[int], session: Session):
        # For every pair of projects, check uniqueness constraint
        projects = [session.get(Project, pid) for pid in project_ids]
        # Get schema questions for each project
        project_questions = {}
        project_videos = {}
        for p in projects:
            if not p:
                raise ValueError(f"Project not found")
            # Get all questions in schema
            qids = set(session.scalars(
                select(Question.id)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .join(SchemaQuestionGroup, QuestionGroupQuestion.question_group_id == SchemaQuestionGroup.question_group_id)
                .where(SchemaQuestionGroup.schema_id == p.schema_id)
            ).all())
            vids = set(session.scalars(
                select(ProjectVideo.video_id)
                .where(ProjectVideo.project_id == p.id)
            ).all())
            # Only consider non-archived videos
            vids = set(
                v for v in vids if not session.get(Video, v).is_archived
            )
            project_questions[p.id] = qids
            project_videos[p.id] = vids
        # Check all pairs
        n = len(projects)
        for i in range(n):
            for j in range(i+1, n):
                q_overlap = project_questions[projects[i].id] & project_questions[projects[j].id]
                if not q_overlap:
                    continue  # No conflict
                v_overlap = project_videos[projects[i].id] & project_videos[projects[j].id]
                if v_overlap:
                    # Get project names
                    project1_name = projects[i].name
                    project2_name = projects[j].name
                    
                    # Get video UIDs for better readability
                    overlapping_video_uids = []
                    for video_id in v_overlap:
                        video = session.get(Video, video_id)
                        if video:
                            overlapping_video_uids.append(video.video_uid)
                    
                    # Format the error message with better readability
                    if len(overlapping_video_uids) <= 5:
                        # Show all video UIDs if 5 or fewer
                        video_list = ", ".join(overlapping_video_uids)
                        error_msg = (f"Cannot group projects '{project1_name}' and '{project2_name}' together. "
                                   f"They have {len(overlapping_video_uids)} overlapping videos with shared questions: {video_list}")
                    else:
                        # Show first 3 and summary if more than 5
                        preview_videos = ", ".join(overlapping_video_uids[:3])
                        remaining_count = len(overlapping_video_uids) - 3
                        error_msg = (f"Cannot group projects '{project1_name}' and '{project2_name}' together. "
                                   f"They have {len(overlapping_video_uids)} overlapping videos with shared questions. "
                                   f"Examples: {preview_videos} (and {remaining_count} more)")
                    
                    raise ValueError(error_msg)

    
    @staticmethod
    def get_grouped_projects_for_user(user_id: int, role: str, session: Session) -> Dict[str, List[Dict[str, Any]]]:
        """Get project groups with their projects for a user.
        
        Args:
            user_id: The ID of the user
            role: The role to filter by
            session: Database session
            
        Returns:
            Dictionary mapping group names to lists of project info dicts
        """
        # Get all projects assigned to user
        user_projects = session.scalars(
            select(Project)
            .join(ProjectUserRole, Project.id == ProjectUserRole.project_id)
            .where(
                ProjectUserRole.user_id == user_id,
                ProjectUserRole.role == role,
                ProjectUserRole.is_archived == False,
                Project.is_archived == False
            )
        ).all()
        
        if not user_projects:
            return {"Unassigned": []}
        
        # Get all project groups
        project_groups = session.scalars(select(ProjectGroup).where(ProjectGroup.is_archived == False)).all()
        
        # Group projects
        grouped_projects = {"Unassigned": []}
        
        # Add named groups
        for group in project_groups:
            grouped_projects[group.name] = []
        
        # Categorize projects
        for project in user_projects:
            # Check if project is in any group
            group_assignment = session.scalar(
                select(ProjectGroupProject)
                .join(ProjectGroup, ProjectGroupProject.project_group_id == ProjectGroup.id)
                .where(
                    ProjectGroupProject.project_id == project.id,
                    ProjectGroup.is_archived == False
                )
            )
            
            project_dict = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at
            }
            
            if group_assignment:
                group_name = session.get(ProjectGroup, group_assignment.project_group_id).name
                if group_name in grouped_projects:
                    grouped_projects[group_name].append(project_dict)
            else:
                grouped_projects["Unassigned"].append(project_dict)
        
        # Remove empty groups
        return {name: projects for name, projects in grouped_projects.items() if projects}

    @staticmethod
    def get_project_group_counts(session: Session) -> Dict[str, int]:
        """Get count statistics for project groups and their projects."""
        from sqlalchemy import func, select
        
        total_groups = session.scalar(select(func.count()).select_from(ProjectGroup))
        
        # Count total projects in groups using join
        total_projects_in_groups = session.scalar(
            select(func.count()).select_from(ProjectGroupProject)
        )
        
        # Calculate average (will be 0 if no groups)
        avg_projects_per_group = round(total_projects_in_groups / total_groups, 1) if total_groups > 0 else 0
        
        return {
            "total_groups": total_groups,
            "total_projects_in_groups": total_projects_in_groups,
            "avg_projects_per_group": avg_projects_per_group
        }
    
    @staticmethod
    def get_export_counts(user_id: int, role: str, session: Session) -> Dict[str, int]:
        """Get count statistics for export without loading full data."""
        from sqlalchemy import func, select, and_
        
        # Get project groups count for this user
        project_groups_count = session.scalar(
            select(func.count(func.distinct(ProjectGroup.id)))
            .select_from(ProjectGroup)
            .join(ProjectGroupProject, ProjectGroup.id == ProjectGroupProject.project_group_id)
            .join(Project, ProjectGroupProject.project_id == Project.id)
            .join(ProjectUserRole, Project.id == ProjectUserRole.project_id)  # FIXED: ProjectUserRole
            .where(
                and_(
                    ProjectUserRole.user_id == user_id,
                    ProjectUserRole.role == role,
                    ProjectUserRole.is_archived == False
                )
            )
        )
        
        # Get total projects accessible to this user
        total_projects_count = session.scalar(
            select(func.count(func.distinct(Project.id)))
            .select_from(Project)
            .join(ProjectUserRole, Project.id == ProjectUserRole.project_id)  # FIXED: ProjectUserRole
            .where(
                and_(
                    ProjectUserRole.user_id == user_id,
                    ProjectUserRole.role == role,
                    ProjectUserRole.is_archived == False
                )
            )
        )
        
        # Count projects ready for export (this is still expensive but only runs once)
        ready_for_export_count = 0
        try:
            accessible_project_ids = session.scalars(
                select(Project.id)
                .select_from(Project)
                .join(ProjectUserRole, Project.id == ProjectUserRole.project_id)  # FIXED: ProjectUserRole
                .where(
                    and_(
                        ProjectUserRole.user_id == user_id,
                        ProjectUserRole.role == role,
                        ProjectUserRole.is_archived == False
                    )
                )
            ).all()
            
            # This is still expensive but only runs once for counts
            for project_id in accessible_project_ids:
                try:
                    if check_project_has_full_ground_truth(project_id=project_id, session=session):
                        ready_for_export_count += 1
                except:
                    pass  # Skip projects that can't be checked
        except Exception:
            ready_for_export_count = 0
        
        return {
            "project_groups": project_groups_count,
            "total_projects": total_projects_count,
            "ready_for_export": ready_for_export_count
        }

class AutoSubmitService:
    @staticmethod
    def get_weighted_votes_for_question(
        video_id: int, 
        project_id: int, 
        question_id: int,
        include_user_ids: List[int],
        virtual_responses: List[Dict],
        session: Session,
        user_weights: Dict[int, float] = None
    ) -> Dict[str, float]:
        """Calculate weighted votes for a single question with user and option weights"""
        try:
            from label_pizza.models import User, ProjectUserRole
            from sqlalchemy import select
            
            # Get question details
            question = QuestionService.get_question_by_id(question_id=question_id, session=session)
            if not question:
                return {}
            
            user_weights = user_weights or {}
            vote_weights = {}
            
            # Get real user answers
            answers_df = AnnotatorService.get_question_answers(
                question_id=question_id, project_id=project_id, session=session
            )
            
            if not answers_df.empty:
                video_answers = answers_df[
                    (answers_df["Video ID"] == video_id) & 
                    (answers_df["User ID"].isin(include_user_ids))
                ]
                
                for _, answer_row in video_answers.iterrows():
                    user_id = int(answer_row["User ID"])
                    answer_value = str(answer_row["Answer Value"])
                    
                    # Get user weight - prioritize passed weights, then database, then default
                    user_weight = user_weights.get(user_id)
                    if user_weight is None:
                        assignment = session.execute(
                            select(ProjectUserRole).where(
                                ProjectUserRole.user_id == user_id,
                                ProjectUserRole.project_id == project_id,
                                ProjectUserRole.role == "annotator"
                            )
                        ).first()
                        user_weight = float(assignment[0].user_weight) if assignment else 1.0
                    
                    # Get option weight
                    option_weight = 1.0
                    if question["type"] == "single" and question["option_weights"]:
                        try:
                            option_index = question["options"].index(answer_value)
                            option_weight = float(question["option_weights"][option_index])
                        except (ValueError, IndexError):
                            option_weight = 1.0
                    
                    # Combined weight = user_weight * option_weight
                    combined_weight = user_weight * option_weight
                    vote_weights[answer_value] = vote_weights.get(answer_value, 0.0) + combined_weight
            
            # Add virtual responses
            for virtual_response in virtual_responses:
                answer_value = str(virtual_response["answer"])
                user_weight = float(virtual_response["user_weight"])
                
                option_weight = 1.0
                if question["type"] == "single" and question["option_weights"]:
                    try:
                        option_index = question["options"].index(answer_value)
                        option_weight = float(question["option_weights"][option_index])
                    except (ValueError, IndexError):
                        option_weight = 1.0
                
                # Combined weight = user_weight * option_weight
                combined_weight = user_weight * option_weight
                vote_weights[answer_value] = vote_weights.get(answer_value, 0.0) + combined_weight
            
            return vote_weights
            
        except Exception as e:
            raise ValueError(f"Error calculating weighted votes: {str(e)}")
    
    @staticmethod
    def calculate_auto_submit_answers(
        video_id: int,
        project_id: int, 
        question_group_id: int,
        include_user_ids: List[int],
        virtual_responses_by_question: Dict[int, List[Dict]],
        thresholds: Dict[int, float],
        session: Session,
        user_weights: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """Calculate which answers would be auto-submitted"""
        try:
            questions = QuestionService.get_questions_by_group_id(group_id=question_group_id, session=session)
            results = {
                "answers": {},
                "skipped": [],
                "threshold_failures": [],
                "vote_details": {},
                "voting_summary": {}
            }
            
            # Check existing answers
            existing_answers = {}
            try:
                existing_answers = AnnotatorService.get_user_answers_for_question_group(
                    video_id=video_id, project_id=project_id, user_id=include_user_ids[0] if include_user_ids else 1,
                    question_group_id=question_group_id, session=session
                )
            except:
                pass
            
            total_votes = 0
            annotator_count = len(include_user_ids)
            total_confidence = 0.0
            confidence_count = 0
            consensus_scores = []
            
            for question in questions:
                question_id = question["id"]
                question_text = question["text"]
                question_type = question["type"]
                
                # Skip if answer already exists
                if question_text in existing_answers and existing_answers[question_text]:
                    results["skipped"].append(question_text)
                    continue
                
                virtual_responses = virtual_responses_by_question.get(question_id, [])
                # ✅ CONSISTENCY: Same logic for description questions
                if question_type == "description":
                    if virtual_responses:
                        # For description questions with virtual responses, use directly
                        if len(virtual_responses) > 1:
                            raise ValueError(f"Description question {question_id} has multiple virtual responses")
                        selected_answer = virtual_responses[0]["answer"]
                        results["answers"][question_text] = selected_answer
                        
                        # Add to vote details for transparency
                        results["vote_details"][question_text] = {
                            selected_answer: virtual_responses[0]["user_weight"]
                        }
                        continue
                    # If no virtual response, fall through to weighted voting
                

                vote_weights = AutoSubmitService.get_weighted_votes_for_question(
                    video_id=video_id, project_id=project_id, question_id=question_id,
                    include_user_ids=include_user_ids, virtual_responses=virtual_responses, 
                    session=session, user_weights=user_weights
                )
                
                results["vote_details"][question_text] = vote_weights
                
                if not vote_weights:
                    continue
                
                total_votes += sum(vote_weights.values())
                
                # Calculate consensus score for this question
                total_weight = sum(vote_weights.values())
                if total_weight > 0:
                    max_weight = max(vote_weights.values())
                    consensus_score = (max_weight / total_weight) * 100
                    consensus_scores.append(consensus_score)
                
                # Find winning option
                if total_weight == 0:
                    continue
                
                winning_option = max(vote_weights.keys(), key=lambda k: vote_weights[k])
                winning_weight = vote_weights[winning_option]
                winning_percentage = (winning_weight / total_weight) * 100
                
                # Check threshold
                threshold = thresholds.get(question_id, 100.0)
                if winning_percentage >= threshold:
                    results["answers"][question_text] = winning_option
                else:
                    results["threshold_failures"].append({
                        "question": question_text,
                        "percentage": winning_percentage,
                        "threshold": threshold
                    })
            
            # Add voting summary
            results["voting_summary"] = {
                "total_votes": total_votes,
                "annotator_count": annotator_count,
                "avg_confidence": total_confidence / confidence_count if confidence_count > 0 else 0,
                "consensus_score": sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error calculating auto-submit answers: {str(e)}")
    
    @staticmethod
    def auto_submit_question_group(
        video_id: int,
        project_id: int,
        question_group_id: int, 
        user_id: int,
        include_user_ids: List[int],
        virtual_responses_by_question: Dict[int, List[Dict]],
        thresholds: Dict[int, float],
        session: Session,
        user_weights: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """Actually submit the auto-calculated answers"""
        try:
            # Calculate what would be submitted
            calculation_results = AutoSubmitService.calculate_auto_submit_answers(
                video_id=video_id, project_id=project_id, question_group_id=question_group_id,
                include_user_ids=include_user_ids, virtual_responses_by_question=virtual_responses_by_question,
                thresholds=thresholds, session=session, user_weights=user_weights
            )
            
            answers = calculation_results["answers"]
            
            if not answers:
                return {
                    "success": True,
                    "submitted_count": 0,
                    "skipped_count": len(calculation_results["skipped"]),
                    "threshold_failures": len(calculation_results["threshold_failures"]),
                    "verification_failed": False,
                    "details": calculation_results
                }
            
            # Run verification if the group has one
            try:
                group_details = QuestionGroupService.get_group_details_with_verification(
                    group_id=question_group_id, session=session
                )
                verification_function = group_details.get("verification_function")
                
                if verification_function:
                    from label_pizza.models import QuestionGroup
                    from sqlalchemy import select
                    
                    group = session.execute(
                        select(QuestionGroup).where(QuestionGroup.id == question_group_id)
                    ).scalar_one()
                    
                    AnnotatorService._run_verification(group, answers)
                    
            except ValueError as verification_error:
                return {
                    "success": False,
                    "submitted_count": 0,
                    "skipped_count": len(calculation_results["skipped"]),
                    "threshold_failures": len(calculation_results["threshold_failures"]),
                    "verification_failed": True,
                    "verification_error": str(verification_error),
                    "details": calculation_results
                }
            
            # Submit the answers
            AnnotatorService.submit_answer_to_question_group(
                video_id=video_id, project_id=project_id, user_id=user_id,
                question_group_id=question_group_id, answers=answers, session=session
            )
            
            return {
                "success": True,
                "submitted_count": len(answers),
                "skipped_count": len(calculation_results["skipped"]),
                "threshold_failures": len(calculation_results["threshold_failures"]),
                "verification_failed": False,
                "details": calculation_results
            }
            
        except Exception as e:
            raise ValueError(f"Error in auto-submit: {str(e)}")


class ReviewerAutoSubmitService:

    @staticmethod
    def auto_submit_ground_truth_group_with_custom_weights(
        video_id: int,
        project_id: int,
        question_group_id: int, 
        reviewer_id: int,
        include_user_ids: List[int],
        virtual_responses_by_question: Dict[int, List[Dict]],
        thresholds: Dict[int, float],
        session: Session,
        user_weights: Dict[int, float] = None,
        custom_option_weights: Dict[int, Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Auto-submit ground truth using weighted voting with custom option weights - FIXED LOGIC"""
        try:
            # Calculate what would be submitted
            calculation_results = ReviewerAutoSubmitService.calculate_auto_submit_ground_truth_with_custom_weights(
                video_id=video_id, project_id=project_id, question_group_id=question_group_id,
                include_user_ids=include_user_ids, virtual_responses_by_question=virtual_responses_by_question,
                thresholds=thresholds, session=session, user_weights=user_weights,
                custom_option_weights=custom_option_weights
            )
            
            answers = calculation_results["answers"]
            skipped = calculation_results["skipped"]
            threshold_failures = calculation_results["threshold_failures"]
            
            # CASE 1: All questions already have GT (entire group skipped)
            if skipped and not answers:
                return {
                    "success": True,
                    "submitted_count": 0,
                    "skipped_count": len(skipped),  # Number of questions that were skipped
                    "threshold_failures": 0,
                    "verification_failed": False,
                    "details": calculation_results
                }
            
            # CASE 2: Any threshold failures (don't submit anything)
            if threshold_failures:
                return {
                    "success": True,  # Not an error, just didn't meet criteria
                    "submitted_count": 0,
                    "skipped_count": 0,
                    "threshold_failures": len(threshold_failures),  # Number of questions that failed
                    "verification_failed": False,
                    "details": calculation_results
                }
            
            # CASE 3: No answers ready to submit (edge case)
            if not answers:
                return {
                    "success": True,
                    "submitted_count": 0,
                    "skipped_count": len(skipped),
                    "threshold_failures": len(threshold_failures),
                    "verification_failed": False,
                    "details": calculation_results
                }
            
            # CASE 4: We have answers to submit - run verification
            try:
                group_details = QuestionGroupService.get_group_details_with_verification(
                    group_id=question_group_id, session=session
                )
                verification_function = group_details.get("verification_function")
                
                if verification_function:
                    from label_pizza.models import QuestionGroup
                    from sqlalchemy import select
                    
                    group = session.execute(
                        select(QuestionGroup).where(QuestionGroup.id == question_group_id)
                    ).scalar_one()
                    AnnotatorService._run_verification(group, answers)
                    
            except ValueError as verification_error:
                return {
                    "success": False,
                    "submitted_count": 0,
                    "skipped_count": len(skipped),
                    "threshold_failures": len(threshold_failures),
                    "verification_failed": True,
                    "verification_error": str(verification_error),
                    "details": calculation_results
                }
            
            # CASE 5: Submit the answers
            GroundTruthService.submit_ground_truth_to_question_group(
                video_id=video_id, project_id=project_id, reviewer_id=reviewer_id,
                question_group_id=question_group_id, answers=answers, session=session
            )
            
            return {
                "success": True,
                "submitted_count": len(answers),  # Number of questions actually submitted
                "skipped_count": len(skipped),   # Number of questions that already had GT
                "threshold_failures": len(threshold_failures),  # Should be 0 if we got here
                "verification_failed": False,
                "details": calculation_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "submitted_count": 0,
                "error": f"Error in reviewer auto-submit: {str(e)}",
                "skipped_count": 0,
                "threshold_failures": 0,
                "verification_failed": False
            }

    @staticmethod
    def calculate_auto_submit_ground_truth_with_custom_weights(
        video_id: int,
        project_id: int, 
        question_group_id: int,
        include_user_ids: List[int],
        virtual_responses_by_question: Dict[int, List[Dict]],
        thresholds: Dict[int, float],
        session: Session,
        user_weights: Dict[int, float] = None,
        custom_option_weights: Dict[int, Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate ground truth with custom option weights - FIXED LOGIC"""
        try:
            questions = QuestionService.get_questions_by_group_id(group_id=question_group_id, session=session)
            results = {
                "answers": {},
                "skipped": [],
                "threshold_failures": [],
                "vote_details": {},
                "voting_summary": {}
            }
            
            # Get existing GROUND TRUTH answers for entire group
            existing_answers = {}
            try:
                existing_answers = GroundTruthService.get_ground_truth_dict_for_question_group(
                    video_id=video_id, project_id=project_id, 
                    question_group_id=question_group_id, session=session
                )
            except:
                pass
            
            # BEHAVIOR 1: Check if ALL questions already have GT → skip entire group
            total_questions = len(questions)
            questions_with_gt = len([q for q in questions if q["text"] in existing_answers and existing_answers[q["text"]]])
            
            if questions_with_gt == total_questions:
                # All questions have GT - skip entire group
                results["skipped"] = [q["text"] for q in questions]
                return results
            
            # BEHAVIOR 2: Process questions - use existing GT where available
            any_threshold_failure = False
            
            for question in questions:
                question_id = question["id"]
                question_text = question["text"]
                question_type = question["type"]
                
                # If question already has GT, use it directly (ignore thresholds/virtual responses)
                if question_text in existing_answers and existing_answers[question_text]:
                    results["answers"][question_text] = existing_answers[question_text]
                    # Don't add to skipped - we're using the existing value
                    continue
                
                # Question needs new GT - apply normal calculation logic
                virtual_responses = virtual_responses_by_question.get(question_id, [])

                # Handle description questions with virtual responses
                if question_type == "description":
                    if virtual_responses:
                        if len(virtual_responses) > 1:
                            raise ValueError(f"Description question {question_id} has multiple virtual responses")
                        selected_answer = virtual_responses[0]["answer"]
                        results["answers"][question_text] = selected_answer
                        results["vote_details"][question_text] = {
                            selected_answer: virtual_responses[0]["user_weight"]
                        }
                        continue
                
                # Calculate weighted votes for this question
                question_custom_weights = None
                if custom_option_weights and question_id in custom_option_weights:
                    question_custom_weights = custom_option_weights[question_id]
                
                vote_weights = ReviewerAutoSubmitService.get_weighted_votes_for_question_with_custom_weights(
                    video_id=video_id, project_id=project_id, question_id=question_id,
                    include_user_ids=include_user_ids, virtual_responses=virtual_responses, 
                    session=session, user_weights=user_weights,
                    custom_option_weights=question_custom_weights
                )
                
                results["vote_details"][question_text] = vote_weights
                
                if not vote_weights:
                    # No votes available - this counts as threshold failure
                    any_threshold_failure = True
                    results["threshold_failures"].append({
                        "question": question_text,
                        "percentage": 0.0,
                        "threshold": thresholds.get(question_id, 100.0)
                    })
                    continue
                
                total_weight = sum(vote_weights.values())
                if total_weight == 0:
                    any_threshold_failure = True
                    results["threshold_failures"].append({
                        "question": question_text,
                        "percentage": 0.0,
                        "threshold": thresholds.get(question_id, 100.0)
                    })
                    continue
                
                winning_option = max(vote_weights.keys(), key=lambda k: vote_weights[k])
                winning_weight = vote_weights[winning_option]
                winning_percentage = (winning_weight / total_weight) * 100
                
                # Check threshold
                threshold = thresholds.get(question_id, 100.0)
                if winning_percentage >= threshold:
                    results["answers"][question_text] = winning_option
                else:
                    any_threshold_failure = True
                    results["threshold_failures"].append({
                        "question": question_text,
                        "percentage": winning_percentage,
                        "threshold": threshold
                    })
            
            # BEHAVIOR 3: If ANY question failed threshold, clear answers (skip entire group)
            if any_threshold_failure:
                results["answers"] = {}  # Clear all answers - don't submit anything
            
            return results
            
        except Exception as e:
            print(f"Error calculating auto-submit ground truth: {str(e)}")
            raise ValueError(f"Error calculating auto-submit ground truth: {str(e)}")

    @staticmethod
    def get_weighted_votes_for_question_with_custom_weights(
        video_id: int, 
        project_id: int, 
        question_id: int,
        include_user_ids: List[int],
        virtual_responses: List[Dict],
        session: Session,
        user_weights: Dict[int, float] = None,
        custom_option_weights: Dict[str, float] = None,
        cache_data: Dict = None  # ADD THIS PARAMETER
    ) -> Dict[str, float]:
        """
        OPTIMIZED: Uses cached data when available to avoid database queries
        """
        try:
            # Get question details - still need this for option weights
            question = QuestionService.get_question_by_id(question_id=question_id, session=session)
            if not question:
                return {}
            
            user_weights = user_weights or {}
            vote_weights = {}
            
            # OPTIMIZED: Use cached data if available
            if cache_data and question_id in cache_data.get("annotator_answers", {}):
                # Use cached answers
                for answer_record in cache_data["annotator_answers"][question_id]:
                    user_id = int(answer_record["User ID"])
                    answer_value = str(answer_record["Answer Value"])
                    
                    # Get user weight - prioritize passed weights, then cached, then default
                    user_weight = user_weights.get(user_id)
                    if user_weight is None:
                        # Try to get from cache first
                        if user_id in cache_data.get("user_weights", {}):
                            user_weight = float(cache_data["user_weights"][user_id])
                        else:
                            # Fallback to service method
                            try:
                                project_user_weights = AuthService.get_user_weights_for_project(
                                    project_id=project_id, session=session
                                )
                                user_weight = float(project_user_weights.get(user_id, 1.0))
                            except:
                                user_weight = 1.0
                    
                    # Use custom option weights if provided (for reviewers)
                    option_weight = 1.0
                    if question["type"] == "single":
                        if custom_option_weights and answer_value in custom_option_weights:
                            option_weight = float(custom_option_weights[answer_value])
                        elif question["option_weights"]:
                            try:
                                option_index = question["options"].index(answer_value)
                                option_weight = float(question["option_weights"][option_index])
                            except (ValueError, IndexError):
                                option_weight = 1.0
                    
                    # Combined weight = user_weight * option_weight
                    combined_weight = user_weight * option_weight
                    vote_weights[answer_value] = vote_weights.get(answer_value, 0.0) + combined_weight
            else:
                # Fallback to original method if no cache
                answers_df = AnnotatorService.get_question_answers(
                    question_id=question_id, project_id=project_id, session=session
                )
                
                if not answers_df.empty:
                    video_answers = answers_df[
                        (answers_df["Video ID"] == video_id) & 
                        (answers_df["User ID"].isin(include_user_ids))
                    ]
                    
                    for _, answer_row in video_answers.iterrows():
                        user_id = int(answer_row["User ID"])
                        answer_value = str(answer_row["Answer Value"])
                        
                        # Get user weight - prioritize passed weights, then service, then default
                        user_weight = user_weights.get(user_id)
                        if user_weight is None:
                            try:
                                project_user_weights = AuthService.get_user_weights_for_project(
                                    project_id=project_id, session=session
                                )
                                user_weight = float(project_user_weights.get(user_id, 1.0))
                            except:
                                user_weight = 1.0
                        
                        # Use custom option weights if provided (for reviewers)
                        option_weight = 1.0
                        if question["type"] == "single":
                            if custom_option_weights and answer_value in custom_option_weights:
                                option_weight = float(custom_option_weights[answer_value])
                            elif question["option_weights"]:
                                try:
                                    option_index = question["options"].index(answer_value)
                                    option_weight = float(question["option_weights"][option_index])
                                except (ValueError, IndexError):
                                    option_weight = 1.0
                        
                        # Combined weight = user_weight * option_weight
                        combined_weight = user_weight * option_weight
                        vote_weights[answer_value] = vote_weights.get(answer_value, 0.0) + combined_weight
            
            # Add virtual responses (unchanged)
            for virtual_response in virtual_responses:
                answer_value = str(virtual_response["answer"])
                user_weight = float(virtual_response["user_weight"])
                
                option_weight = 1.0
                if question["type"] == "single":
                    if custom_option_weights and answer_value in custom_option_weights:
                        option_weight = float(custom_option_weights[answer_value])
                    elif question["option_weights"]:
                        try:
                            option_index = question["options"].index(answer_value)
                            option_weight = float(question["option_weights"][option_index])
                        except (ValueError, IndexError):
                            option_weight = 1.0
                
                # Combined weight = user_weight * option_weight
                combined_weight = user_weight * option_weight
                vote_weights[answer_value] = vote_weights.get(answer_value, 0.0) + combined_weight
            
            return vote_weights
            
        except Exception as e:
            raise ValueError(f"Error calculating weighted votes: {str(e)}")

class GroundTruthExportService:
    @staticmethod
    def export_ground_truth_data(project_ids: List[int], session: Session) -> List[Dict[str, Any]]:
        """Export ground truth data from a list of projects.
        
        Args:
            project_ids: List of project IDs to export from
            session: Database session
            
        Returns:
            List of video dictionaries with video_uid, url, and answers
            
        Raises:
            ValueError: If projects not found or reusable question groups have inconsistent answers
        """
        # Validate all projects exist
        for project_id in project_ids:
            project = session.get(Project, project_id)
            if not project:
                raise ValueError(f"Project with ID {project_id} not found")
            if project.is_archived:
                raise ValueError(f"Project with ID {project_id} is archived")
        
        # Check reusable question group consistency before proceeding
        GroundTruthExportService._validate_reusable_question_groups(project_ids, session)

        # Check non-reusable question group constraints
        GroundTruthExportService._validate_non_reusable_question_groups(project_ids, session)
        
        # Get all unique videos across projects
        all_videos = set()
        for project_id in project_ids:
            from label_pizza.models import ProjectVideo
            videos = session.scalars(
                select(ProjectVideo.video_id)
                .where(ProjectVideo.project_id == project_id)
            ).all()
            all_videos.update(videos)
        
        # Export data for each video
        export_data = []
        for video_id in sorted(all_videos):  # Sort for consistent ordering
            video = session.get(Video, video_id)
            if not video or video.is_archived:
                continue
                
            video_data = {
                "video_uid": video.video_uid,
                "url": video.url,
                "answers": {}, # question.text -> raw answer value
                "question_display_text": {}, # question.text -> display text shown
                "answer_display_values": {} # question.text -> display value shown
            }
            
            # Get all ground truth answers for this video across all projects
            # Use a set to track which questions we've already processed to avoid duplicates
            processed_questions = set()
            
            for project_id in project_ids:
                gts = session.scalars(
                    select(ReviewerGroundTruth)
                    .where(
                        ReviewerGroundTruth.video_id == video_id,
                        ReviewerGroundTruth.project_id == project_id
                    )
                ).all()
                
                for gt in gts:
                    question = session.get(Question, gt.question_id)
                    if question and not question.is_archived:
                        # For reusable groups, we've already validated consistency,
                        # so we can safely take the first answer we encounter
                        if question.text not in processed_questions:
                            # video_data["answers"][question.text] = gt.answer_value
                            # processed_questions.add(question.text)
                            try:
                                question_with_display = CustomDisplayService.get_custom_display(
                                    question_id=gt.question_id,
                                    project_id=project_id,
                                    video_id=video_id,
                                    session=session
                                )
                                display_text = question_with_display["display_text"]
                                
                                # Get display value for the selected answer
                                if question_with_display["type"] == "single" and question_with_display["display_values"]:
                                    try:
                                        option_index = question_with_display["options"].index(gt.answer_value)
                                        display_value = question_with_display["display_values"][option_index]
                                    except (ValueError, IndexError):
                                        display_value = gt.answer_value  # Fallback to raw value
                                else:
                                    display_value = gt.answer_value  # For description type or no display values
                                    
                            except Exception:
                                # Fallback to original display text if custom display fails
                                display_text = question.display_text
                                display_value = gt.answer_value
                            
                            # Store all the information
                            video_data["answers"][question.text] = gt.answer_value
                            video_data["question_display_text"][question.text] = display_text
                            video_data["answer_display_values"][question.text] = display_value
                            processed_questions.add(question.text)
                            
            if video_data["answers"]:  # Only include videos with answers
                export_data.append(video_data)
        
        return export_data
    
    @staticmethod
    def _validate_reusable_question_groups(project_ids: List[int], session: Session) -> None:
        """Validate that reusable question groups have consistent answers across projects.
        
        Raises:
            ValueError: If inconsistencies are found
        """
        # Get all reusable question groups used in these projects
        reusable_groups = session.scalars(
            select(QuestionGroup)
            .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
            .join(Project, SchemaQuestionGroup.schema_id == Project.schema_id)
            .where(
                Project.id.in_(project_ids),
                QuestionGroup.is_reusable == True,
                QuestionGroup.is_archived == False
            )
            .distinct()
        ).all()
        
        if not reusable_groups:
            return  # No reusable groups to validate
        
        # Check consistency for each reusable group
        group_inconsistencies = {}
        
        for group in reusable_groups:
            # Get all questions in this group
            questions = session.scalars(
                select(Question)
                .join(QuestionGroupQuestion, Question.id == QuestionGroupQuestion.question_id)
                .where(
                    QuestionGroupQuestion.question_group_id == group.id,
                    Question.is_archived == False
                )
            ).all()
            
            # Find which projects use this reusable group
            projects_using_group = session.scalars(
                select(Project.id)
                .join(SchemaQuestionGroup, Project.schema_id == SchemaQuestionGroup.schema_id)
                .where(
                    SchemaQuestionGroup.question_group_id == group.id,
                    Project.id.in_(project_ids),
                    Project.is_archived == False
                )
            ).all()
            
            if len(projects_using_group) < 2:
                continue  # No conflict possible with only one project
            
            # For each question, check consistency across projects
            for question in questions:
                # Get all videos that have answers for this question across the relevant projects
                video_answer_map = {}  # video_id -> {project_id -> answer_value}
                
                for project_id in projects_using_group:
                    gts = session.scalars(
                        select(ReviewerGroundTruth)
                        .where(
                            ReviewerGroundTruth.question_id == question.id,
                            ReviewerGroundTruth.project_id == project_id
                        )
                    ).all()
                    
                    for gt in gts:
                        video_id = gt.video_id
                        if video_id not in video_answer_map:
                            video_answer_map[video_id] = {}
                        video_answer_map[video_id][project_id] = gt.answer_value
                
                # Check for inconsistencies - videos that appear in multiple projects with different answers
                inconsistent_videos = []
                for video_id, project_answers in video_answer_map.items():
                    if len(project_answers) > 1:  # Video appears in multiple projects
                        unique_answers = set(project_answers.values())
                        if len(unique_answers) > 1:  # Different answers across projects
                            video = session.get(Video, video_id)
                            if video:
                                inconsistent_videos.append(video.video_uid)
                
                if inconsistent_videos:
                    if group.title not in group_inconsistencies:
                        group_inconsistencies[group.title] = []
                    group_inconsistencies[group.title].append({
                        "question": question.text,
                        "videos": inconsistent_videos
                    })
        
        if group_inconsistencies:
            error_parts = []
            total_failing_videos = set()
            
            for group_name, inconsistencies in group_inconsistencies.items():
                group_videos = set()
                question_details = []
                
                for inc in inconsistencies:
                    group_videos.update(inc['videos'])
                    question_details.append(f"    Question '{inc['question']}': {len(inc['videos'])} videos")
                
                total_failing_videos.update(group_videos)
                
                error_parts.append(
                    f"Reusable question group '{group_name}' ({len(group_videos)} failing videos):\n" +
                    "\n".join(question_details)
                )
            
            failing_video_list = sorted(list(total_failing_videos))
            
            raise ValueError(
                f"Cannot export due to inconsistent answers in reusable question groups.\n"
                f"Total videos with inconsistencies: {len(failing_video_list)}\n"
                f"Failing videos: {', '.join(failing_video_list[:10])}"
                f"{'...' if len(failing_video_list) > 10 else ''}\n\n"
                f"Details:\n" + "\n\n".join(error_parts)
            )

    @staticmethod
    def _validate_non_reusable_question_groups(project_ids: List[int], session: Session) -> None:
        """Validate that non-reusable question groups don't have videos appearing in multiple projects.
        
        Raises:
            ValueError: If any video appears in multiple projects sharing non-reusable question groups
        """
        # Get all non-reusable question groups used in these projects
        non_reusable_groups = session.scalars(
            select(QuestionGroup)
            .join(SchemaQuestionGroup, QuestionGroup.id == SchemaQuestionGroup.question_group_id)
            .join(Project, SchemaQuestionGroup.schema_id == Project.schema_id)
            .where(
                Project.id.in_(project_ids),
                QuestionGroup.is_reusable == False,
                QuestionGroup.is_archived == False
            )
            .distinct()
        ).all()
        
        if not non_reusable_groups:
            return  # No non-reusable groups to validate
        
        # Check for violations for each non-reusable group
        group_violations = {}
        
        for group in non_reusable_groups:
            # Find which projects use this non-reusable group
            projects_using_group = session.scalars(
                select(Project.id)
                .join(SchemaQuestionGroup, Project.schema_id == SchemaQuestionGroup.schema_id)
                .where(
                    SchemaQuestionGroup.question_group_id == group.id,
                    Project.id.in_(project_ids),
                    Project.is_archived == False
                )
            ).all()
            
            if len(projects_using_group) < 2:
                continue  # No violation possible with only one project
            
            # Get all videos for each project using this group
            project_videos = {}  # project_id -> set of video_ids
            for project_id in projects_using_group:
                videos = set(session.scalars(
                    select(ProjectVideo.video_id)
                    .where(ProjectVideo.project_id == project_id)
                ).all())
                project_videos[project_id] = videos
            
            # Check for overlapping videos between projects
            overlapping_videos = set()
            project_list = list(projects_using_group)
            
            for i in range(len(project_list)):
                for j in range(i + 1, len(project_list)):
                    project1_id = project_list[i]
                    project2_id = project_list[j]
                    
                    overlap = project_videos[project1_id] & project_videos[project2_id]
                    if overlap:
                        overlapping_videos.update(overlap)
            
            if overlapping_videos:
                # Get video UIDs for better error messages
                video_uids = []
                for video_id in overlapping_videos:
                    video = session.get(Video, video_id)
                    if video:
                        video_uids.append(video.video_uid)
                
                group_violations[group.title] = {
                    "videos": sorted(video_uids),
                    "projects": projects_using_group
                }
        
        if group_violations:
            error_parts = []
            total_failing_videos = set()
            
            for group_name, violation in group_violations.items():
                total_failing_videos.update(violation['videos'])
                
                # Get project names for better error messages
                project_names = []
                for project_id in violation['projects']:
                    project = session.get(Project, project_id)
                    if project:
                        project_names.append(f"'{project.name}' (ID: {project_id})")
                
                error_parts.append(
                    f"Non-reusable question group '{group_name}' is used in multiple projects:\n"
                    f"    Projects: {', '.join(project_names)}\n"
                    f"    Overlapping videos ({len(violation['videos'])}): {', '.join(violation['videos'][:10])}"
                    f"{'...' if len(violation['videos']) > 10 else ''}"
                )
            
            raise ValueError(
                f"Cannot export due to non-reusable question groups being shared across projects.\n"
                f"Non-reusable groups should only be used in one project each.\n"
                f"Total videos affected: {len(total_failing_videos)}\n\n"
                f"Violations:\n" + "\n\n".join(error_parts)
            )

def save_export_as_json(export_data: List[Dict[str, Any]], filepath: str) -> None:
    """Save export data as JSON file with pretty formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        import json
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def save_export_as_excel(export_data: List[Dict[str, Any]], filepath_or_buffer) -> None:
    """Save export data as Excel file with pretty formatting."""
    if not export_data:
        # Create empty Excel file
        pd.DataFrame({"Message": ["No data to export"]}).to_excel(filepath_or_buffer, index=False)
        return
    
    # Collect all unique question texts
    all_questions = set()
    for video in export_data:
        all_questions.update(video["answers"].keys())
    
    all_questions = sorted(list(all_questions))
    
    # Prepare data for DataFrame
    rows = []
    for video in export_data:
        row = {
            "Video UID": video["video_uid"],
            "URL": video["url"]
        }
        
        # Add answers for each question
        for i, question in enumerate(all_questions):
            display_text = video["question_display_text"].get(question, question)
            display_value = video["answer_display_values"].get(question, "")
            
            row[f"Q{i}: {question}"] = display_text      # Custom display question text
            row[f"Q{i} Answer"] = display_value          # Custom display answer value
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to Excel with formatting
    try:
        with pd.ExcelWriter(filepath_or_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Ground Truth Export', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Ground Truth Export']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                # Set width with some padding, cap at reasonable size
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Style the header row
            try:
                from openpyxl.styles import Font, PatternFill, Alignment
                
                header_font = Font(bold=True)
                header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                center_alignment = Alignment(horizontal="center", vertical="center")
                
                for cell in worksheet[1]:  # First row
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = center_alignment
                    
                # Freeze the header row
                worksheet.freeze_panes = "A2"
                
            except ImportError:
                # If openpyxl styling modules aren't available, skip formatting
                pass
                
    except Exception as e:
        # Fallback to basic Excel export if formatting fails
        df.to_excel(filepath_or_buffer, index=False)
        if hasattr(filepath_or_buffer, 'write'):  # It's a buffer, not a file path
            pass  # Don't print for buffers
        else:
            print(f"Warning: Excel formatting failed ({e}), saved with basic formatting")


# Example usage function
def export_projects_ground_truth(project_ids: List[int], output_path: str, format_type: str = "json", session: Session = None):
    """Convenience function to export ground truth data.
    
    Args:
        project_ids: List of project IDs to export
        output_path: Path for output file
        format_type: "json" or "excel"
        session: Database session
    """
    if session is None:
        raise ValueError("Database session is required")
    
    try:
        # Export the data
        export_data = GroundTruthExportService.export_ground_truth_data(project_ids, session)
        
        # Ensure correct file extension
        output_path = Path(output_path)
        if format_type == "json" and not output_path.suffix.lower() == ".json":
            output_path = output_path.with_suffix(".json")
        elif format_type == "excel" and not output_path.suffix.lower() in [".xlsx", ".xls"]:
            output_path = output_path.with_suffix(".xlsx")
        
        # Save the data
        if format_type == "json":
            save_export_as_json(export_data, str(output_path))
        elif format_type == "excel":
            save_export_as_excel(export_data, str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"Successfully exported {len(export_data)} videos to {output_path}")
        return str(output_path)
        
    except ValueError as e:
        print(f"Export failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during export: {e}")
        raise