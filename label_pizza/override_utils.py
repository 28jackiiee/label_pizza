#!/usr/bin/env python3
"""
Database Override Utilities for Label Pizza
==========================================

This module provides cascading delete operations and name management for Label Pizza database tables.
All delete operations follow the dependency analysis and automatically handle child row deletions.

IMPORTANT: Always run these two lines first before using any functions:

    from label_pizza.db import init_database
    init_database("DBURL")

All delete functions support backup parameters:
- backup_first=True - Create automatic backup before deletion
- backup_dir="./backups" - Default backup directory  
- backup_file=None - Auto-generated timestamp filename
- compress=True - Enable gzip compression
"""

from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import label_pizza.db
from sqlalchemy.orm import Session
from sqlalchemy import text, and_
from label_pizza.manage_db import create_backup_if_requested
import os


def get_db_url_for_backup():
    """Get database URL using the same environment variable used in init_database"""
    import label_pizza.db
    from dotenv import load_dotenv
    
    # Get the environment variable name that was used during init_database
    env_var_name = getattr(label_pizza.db, 'current_database_url_name', 'DBURL')
    
    load_dotenv()
    db_url = os.getenv(env_var_name)
    if not db_url:
        raise ValueError(f"Environment variable '{env_var_name}' not found")
    
    return db_url


class DeleteOperation(NamedTuple):
    """Represents a single delete operation"""
    table: str
    operation_type: str  # 'using_id' or 'using_name' 
    identifier: Tuple[Any, ...]  # The ID(s) or name(s) to delete
    backup_params: Dict[str, Any]


# Define proper deletion order based on dependency levels
DELETION_ORDER = [
    # Leaf nodes (no dependencies)
    'AnswerReview', 'ReviewerGroundTruth', 'ProjectVideoQuestionDisplay', 'ProjectGroupProject', 'VideoTag',
    # Level 1 dependencies  
    'AnnotatorAnswer', 'ProjectGroup',
    # Level 2 dependencies
    'ProjectVideo', 'ProjectUserRole', 'QuestionGroupQuestion', 'SchemaQuestionGroup',
    # Level 3 dependencies
    'Video', 'Question', 'QuestionGroup', 'Project', 'User', 
    # Level 4 dependencies
    'Schema'
]


def _get_deletion_priority(table: str) -> int:
    """Get deletion priority (lower number = delete first)"""
    try:
        return DELETION_ORDER.index(table)
    except ValueError:
        return 999  # Unknown tables go last


def _collect_delete_operations(operations: List[DeleteOperation]) -> List[DeleteOperation]:
    """Sort and deduplicate delete operations by dependency order"""
    # Sort by deletion priority
    sorted_ops = sorted(operations, key=lambda op: _get_deletion_priority(op.table))
    
    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    
    for op in sorted_ops:
        # Create unique key for deduplication
        key = (op.table, op.operation_type, op.identifier)
        if key not in seen:
            seen.add(key)
            deduplicated.append(op)
    
    return deduplicated


def _confirm_cascade_deletion(operations: List[DeleteOperation]) -> bool:
    """Ask user to confirm cascade deletion operations"""
    if not operations:
        print("No operations to perform.")
        return True
    
    print("🚨 CASCADE DELETE OPERATIONS")
    print("=" * 50)
    print("The following delete operations will be executed in sequence:")
    print()
    
    for i, op in enumerate(operations, 1):
        table = op.table
        readable_name = _get_readable_operation_name(op)
        print(f"{i:2d}. {readable_name}")
    
    print()
    print(f"Total operations: {len(operations)}")
    print("⚠️  This will permanently delete data from the database!")
    print()
    
    response = input("Confirm cascade deletion? Type 'DELETE' to proceed: ")
    return response.strip() == 'DELETE'


def _get_readable_operation_name(op: DeleteOperation) -> str:
    """Convert delete operation to human-readable name"""
    table = op.table
    identifier = op.identifier
    
    try:
        with label_pizza.db.SessionLocal() as session:
            if table == 'User':
                user_name = _get_name_from_id(session, 'users', 'user_id_str', identifier[0])
                return f"Delete user '{user_name}'"
                
            elif table == 'Video':
                video_name = _get_name_from_id(session, 'videos', 'video_uid', identifier[0])
                return f"Delete video '{video_name}'"
                
            elif table == 'QuestionGroup':
                group_name = _get_name_from_id(session, 'question_groups', 'title', identifier[0])
                return f"Delete question group '{group_name}'"
                
            elif table == 'Question':
                question_text = _get_name_from_id(session, 'questions', 'text', identifier[0])
                return f"Delete question '{question_text[:50]}{'...' if len(question_text) > 50 else ''}'"
                
            elif table == 'Schema':
                schema_name = _get_name_from_id(session, 'schemas', 'name', identifier[0])
                return f"Delete schema '{schema_name}'"
                
            elif table == 'Project':
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[0])
                return f"Delete project '{project_name}'"
                
            elif table == 'ProjectGroup':
                group_name = _get_name_from_id(session, 'project_groups', 'name', identifier[0])
                return f"Delete project group '{group_name}'"
                
            elif table == 'ProjectVideo':
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[0])
                video_name = _get_name_from_id(session, 'videos', 'video_uid', identifier[1])
                return f"Remove video '{video_name}' from project '{project_name}'"
                
            elif table == 'ProjectUserRole':
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[0])
                user_name = _get_name_from_id(session, 'users', 'user_id_str', identifier[1])
                role = identifier[2]
                return f"Remove {role} '{user_name}' from project '{project_name}'"
                
            elif table == 'ProjectGroupProject':
                group_name = _get_name_from_id(session, 'project_groups', 'name', identifier[0])
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[1])
                return f"Remove project '{project_name}' from group '{group_name}'"
                
            elif table == 'QuestionGroupQuestion':
                group_name = _get_name_from_id(session, 'question_groups', 'title', identifier[0])
                question_text = _get_name_from_id(session, 'questions', 'text', identifier[1])
                return f"Remove question from group '{group_name}'"
                
            elif table == 'SchemaQuestionGroup':
                schema_name = _get_name_from_id(session, 'schemas', 'name', identifier[0])
                group_name = _get_name_from_id(session, 'question_groups', 'title', identifier[1])
                return f"Remove group '{group_name}' from schema '{schema_name}'"
                
            elif table == 'ProjectVideoQuestionDisplay':
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[0])
                video_name = _get_name_from_id(session, 'videos', 'video_uid', identifier[1])
                question_text = _get_name_from_id(session, 'questions', 'text', identifier[2])
                return f"Delete custom display for '{question_text[:30]}...' in '{project_name}'"
                
            elif table == 'AnnotatorAnswer':
                # Get detailed info about the annotation
                result = session.execute(text("""
                    SELECT u.user_id_str, v.video_uid, q.text, p.name
                    FROM annotator_answers aa
                    JOIN users u ON aa.user_id = u.id
                    JOIN videos v ON aa.video_id = v.id  
                    JOIN questions q ON aa.question_id = q.id
                    JOIN projects p ON aa.project_id = p.id
                    WHERE aa.id = :id
                """), {"id": identifier[0]})
                row = result.fetchone()
                if row:
                    user_name, video_name, question_text, project_name = row
                    return f"Delete '{user_name}' annotation for '{question_text[:30]}...' in '{project_name}'"
                return f"Delete annotation (ID: {identifier[0]})"
                
            elif table == 'ReviewerGroundTruth':
                video_name = _get_name_from_id(session, 'videos', 'video_uid', identifier[0])
                question_text = _get_name_from_id(session, 'questions', 'text', identifier[1])
                project_name = _get_name_from_id(session, 'projects', 'name', identifier[2])
                return f"Delete ground truth for '{question_text[:30]}...' in '{project_name}'"
                
            elif table == 'AnswerReview':
                # Get detailed info about the review
                result = session.execute(text("""
                    SELECT reviewer.user_id_str, annotator.user_id_str, q.text
                    FROM answer_reviews ar
                    JOIN annotator_answers aa ON ar.answer_id = aa.id
                    JOIN users reviewer ON ar.reviewer_id = reviewer.id
                    JOIN users annotator ON aa.user_id = annotator.id
                    JOIN questions q ON aa.question_id = q.id
                    WHERE ar.id = :id
                """), {"id": identifier[0]})
                row = result.fetchone()
                if row:
                    reviewer_name, annotator_name, question_text = row
                    return f"Delete '{reviewer_name}' review of '{annotator_name}' annotation"
                return f"Delete answer review (ID: {identifier[0]})"
                
            else:
                # Fallback for unknown table types
                return f"Delete {table.lower()} (ID: {identifier[0]})"
                
    except Exception as e:
        # Fallback if name lookup fails
        return f"Delete {table.lower()} (ID: {identifier[0]})"


def _get_name_from_id(session: Session, table: str, name_column: str, id_value: int) -> str:
    """Helper function to get name from ID for any table"""
    try:
        result = session.execute(text(f"SELECT {name_column} FROM {table} WHERE id = :id"), {"id": id_value})
        row = result.fetchone()
        return row[0] if row else f"ID_{id_value}"
    except:
        return f"ID_{id_value}"


def _execute_delete_operations(operations: List[DeleteOperation]) -> bool:
    """Execute the delete operations in order"""
    if not operations:
        return True
    
    # Create backup if any operation requests it
    backup_created = None
    for op in operations:
        if op.backup_params.get('backup_first', False):
            try:
                db_url = get_db_url_for_backup()
                backup_created = create_backup_if_requested(
                    str(db_url),
                    op.backup_params.get('backup_dir', './backups'),
                    op.backup_params.get('backup_file', None),
                    op.backup_params.get('compress', True)
                )
                print(f"💾 Backup created successfully: {backup_created}")
                break  # Only need one backup
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Deletion cancelled due to backup failure")
                    return False
                break
    
    success_count = 0
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Handle admin reversion operations first
            for op in operations:
                if op.table == 'ProjectUserRole' and len(op.identifier) == 3 and op.identifier[2] == 'admin':
                    # This is an admin role deletion, need to revert ground truths first
                    # This logic reverts any ground truth modifications made by this admin
                    project_id, user_id, role = op.identifier
                    try:
                        result = session.execute(text("""
                            UPDATE reviewer_ground_truth 
                            SET answer_value = original_answer_value, 
                                modified_at = NULL, 
                                modified_by_admin_id = NULL, 
                                modified_by_admin_at = NULL 
                            WHERE project_id = :p_id AND modified_by_admin_id = :u_id
                        """), {"p_id": project_id, "u_id": user_id})
                        rows_affected = result.rowcount
                        if rows_affected > 0:
                            print(f"✅ Reverted {rows_affected} admin modifications for user {user_id} in project {project_id}")
                    except Exception as e:
                        print(f"❌ Failed to revert admin modifications: {e}")
                        session.rollback()
                        return False
            
            # Now execute the actual delete operations
            for op in operations:
                try:
                    _execute_single_delete_operation(session, op)
                    success_count += 1
                    readable_name = _get_readable_operation_name(op)
                    print(f"✅ {readable_name}")
                except Exception as e:
                    readable_name = _get_readable_operation_name(op)
                    print(f"❌ Failed: {readable_name} - {e}")
                    session.rollback()
                    return False
            
            session.commit()
            
    except Exception as e:
        print(f"❌ Transaction failed: {e}")
        return False
    
    print(f"\n🎉 Successfully completed {success_count} delete operations")
    if backup_created:
        print(f"💾 Backup saved to: {backup_created}")
    
    return True


def _execute_single_delete_operation(session: Session, op: DeleteOperation):
    """Execute a single delete operation"""
    table = op.table.lower()
    
    if op.operation_type == 'using_id':
        if table == 'user':
            session.execute(text("DELETE FROM users WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'video':
            session.execute(text("DELETE FROM videos WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'videotag':
            session.execute(text("DELETE FROM video_tags WHERE video_id = :video_id AND tag = :tag"), 
                          {"video_id": op.identifier[0], "tag": op.identifier[1]})
        elif table == 'questiongroup':
            session.execute(text("DELETE FROM question_groups WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'question':
            session.execute(text("DELETE FROM questions WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'questiongroupquestion':
            session.execute(text("DELETE FROM question_group_questions WHERE question_group_id = :qg_id AND question_id = :q_id"),
                          {"qg_id": op.identifier[0], "q_id": op.identifier[1]})
        elif table == 'schema':
            session.execute(text("DELETE FROM schemas WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'schemaquestiongroup':
            session.execute(text("DELETE FROM schema_question_groups WHERE schema_id = :s_id AND question_group_id = :qg_id"),
                          {"s_id": op.identifier[0], "qg_id": op.identifier[1]})
        elif table == 'project':
            session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'projectvideo':
            session.execute(text("DELETE FROM project_videos WHERE project_id = :p_id AND video_id = :v_id"),
                          {"p_id": op.identifier[0], "v_id": op.identifier[1]})
        elif table == 'projectuserrole':
            session.execute(text("DELETE FROM project_user_roles WHERE project_id = :p_id AND user_id = :u_id AND role = :role"),
                          {"p_id": op.identifier[0], "u_id": op.identifier[1], "role": op.identifier[2]})
        elif table == 'projectgroup':
            session.execute(text("DELETE FROM project_groups WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'projectgroupproject':
            session.execute(text("DELETE FROM project_group_projects WHERE project_group_id = :pg_id AND project_id = :p_id"),
                          {"pg_id": op.identifier[0], "p_id": op.identifier[1]})
        elif table == 'projectvideoquestiondisplay':
            session.execute(text("DELETE FROM project_video_question_displays WHERE project_id = :p_id AND video_id = :v_id AND question_id = :q_id"),
                          {"p_id": op.identifier[0], "v_id": op.identifier[1], "q_id": op.identifier[2]})
        elif table == 'annotatoranswer':
            session.execute(text("DELETE FROM annotator_answers WHERE id = :id"), {"id": op.identifier[0]})
        elif table == 'reviewergroundtruth':
            session.execute(text("DELETE FROM reviewer_ground_truth WHERE video_id = :v_id AND question_id = :q_id AND project_id = :p_id"),
                          {"v_id": op.identifier[0], "q_id": op.identifier[1], "p_id": op.identifier[2]})
        elif table == 'answerreview':
            session.execute(text("DELETE FROM answer_reviews WHERE id = :id"), {"id": op.identifier[0]})
        else:
            raise ValueError(f"Unknown table for using_id deletion: {table}")


# ============================================================================
# HELPER FUNCTIONS FOR COLLECTING DEPENDENCIES
# ============================================================================

def _get_user_id_from_user_id_str(session: Session, user_id_str: str) -> int:
    """Get user ID from user_id_str"""
    result = session.execute(text("SELECT id FROM users WHERE user_id_str = :user_id_str"), {"user_id_str": user_id_str})
    row = result.fetchone()
    if not row:
        raise ValueError(f"User not found: {user_id_str}")
    return row[0]


def _get_video_id_from_video_uid(session: Session, video_uid: str) -> int:
    """Get video ID from video_uid"""
    result = session.execute(text("SELECT id FROM videos WHERE video_uid = :video_uid"), {"video_uid": video_uid})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Video not found: {video_uid}")
    return row[0]


def _get_question_group_id_from_title(session: Session, title: str) -> int:
    """Get question group ID from title"""
    result = session.execute(text("SELECT id FROM question_groups WHERE title = :title"), {"title": title})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Question group not found: {title}")
    return row[0]


def _get_question_id_from_text(session: Session, text_param: str) -> int:
    """Get question ID from text"""
    result = session.execute(text("SELECT id FROM questions WHERE text = :text"), {"text": text_param})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Question not found: {text_param}")
    return row[0]


def _get_schema_id_from_name(session: Session, name: str) -> int:
    """Get schema ID from name"""
    result = session.execute(text("SELECT id FROM schemas WHERE name = :name"), {"name": name})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Schema not found: {name}")
    return row[0]


def _get_project_id_from_name(session: Session, name: str) -> int:
    """Get project ID from name"""
    result = session.execute(text("SELECT id FROM projects WHERE name = :name"), {"name": name})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Project not found: {name}")
    return row[0]


def _get_project_group_id_from_name(session: Session, name: str) -> int:
    """Get project group ID from name"""
    result = session.execute(text("SELECT id FROM project_groups WHERE name = :name"), {"name": name})
    row = result.fetchone()
    if not row:
        raise ValueError(f"Project group not found: {name}")
    return row[0]


def _collect_answer_review_dependencies(session: Session, answer_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for AnnotatorAnswer deletion"""
    operations = []
    
    # Find all AnswerReviews for this answer
    result = session.execute(text("SELECT id FROM answer_reviews WHERE answer_id = :answer_id"), {"answer_id": answer_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('AnswerReview', 'using_id', (row[0],), backup_params))
    
    return operations


def _collect_project_group_dependencies(session: Session, project_group_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for ProjectGroup deletion"""
    operations = []
    
    # Find all ProjectGroupProjects for this project group
    result = session.execute(text("SELECT project_id FROM project_group_projects WHERE project_group_id = :pg_id"), {"pg_id": project_group_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('ProjectGroupProject', 'using_id', (project_group_id, row[0]), backup_params))
    
    return operations


def _collect_project_video_dependencies(session: Session, project_id: int, video_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for ProjectVideo deletion"""
    operations = []
    
    # Find AnnotatorAnswers for this project+video
    result = session.execute(text("SELECT id FROM annotator_answers WHERE project_id = :p_id AND video_id = :v_id"), 
                           {"p_id": project_id, "v_id": video_id})
    for row in result.fetchall():
        operations.extend(_collect_annotator_answer_dependencies(session, row[0], backup_params))
        operations.append(DeleteOperation('AnnotatorAnswer', 'using_id', (row[0],), backup_params))
    
    # Find ReviewerGroundTruth for this project+video
    result = session.execute(text("SELECT question_id FROM reviewer_ground_truth WHERE project_id = :p_id AND video_id = :v_id"),
                           {"p_id": project_id, "v_id": video_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('ReviewerGroundTruth', 'using_id', (video_id, row[0], project_id), backup_params))
    
    # Find ProjectVideoQuestionDisplays for this project+video
    result = session.execute(text("SELECT question_id FROM project_video_question_displays WHERE project_id = :p_id AND video_id = :v_id"),
                           {"p_id": project_id, "v_id": video_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('ProjectVideoQuestionDisplay', 'using_id', (project_id, video_id, row[0]), backup_params))
    
    return operations


def _collect_project_user_role_dependencies(session: Session, project_id: int, user_id: int, role: str, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for ProjectUserRole deletion with role-specific logic"""
    operations = []
    
    if role == "admin":
        # If deleting admin role, ground truth reversion is handled automatically
        # in _execute_delete_operations() before any actual deletions occur
        pass
        
    elif role == "reviewer":
        # If deleting reviewer role, delete their ground truths and reviews
        # First check if they also have admin role
        admin_role_result = session.execute(text("""
            SELECT 1 FROM project_user_roles 
            WHERE project_id = :p_id AND user_id = :u_id AND role = 'admin'
        """), {"p_id": project_id, "u_id": user_id})
        
        if admin_role_result.fetchone():
            operations.append(DeleteOperation('ProjectUserRole', 'using_id', (project_id, user_id, 'admin'), backup_params))
        
        # Delete reviewer ground truths
        result = session.execute(text("SELECT video_id, question_id FROM reviewer_ground_truth WHERE project_id = :p_id AND reviewer_id = :u_id"),
                               {"p_id": project_id, "u_id": user_id})
        for row in result.fetchall():
            operations.append(DeleteOperation('ReviewerGroundTruth', 'using_id', (row[0], row[1], project_id), backup_params))
        
        # Delete answer reviews by this reviewer
        result = session.execute(text("""
            SELECT ar.id FROM answer_reviews ar
            JOIN annotator_answers aa ON ar.answer_id = aa.id
            WHERE aa.project_id = :p_id AND ar.reviewer_id = :u_id
        """), {"p_id": project_id, "u_id": user_id})
        for row in result.fetchall():
            operations.append(DeleteOperation('AnswerReview', 'using_id', (row[0],), backup_params))
            
    elif role in ["annotator", "model"]:
        # If deleting annotator/model role, delete their annotations
        # First check if they also have reviewer role
        reviewer_role_result = session.execute(text("""
            SELECT 1 FROM project_user_roles 
            WHERE project_id = :p_id AND user_id = :u_id AND role = 'reviewer'
        """), {"p_id": project_id, "u_id": user_id})
        
        if reviewer_role_result.fetchone():
            operations.extend(_collect_project_user_role_dependencies(session, project_id, user_id, 'reviewer', backup_params))
            operations.append(DeleteOperation('ProjectUserRole', 'using_id', (project_id, user_id, 'reviewer'), backup_params))
        
        # Delete annotator answers
        result = session.execute(text("SELECT id FROM annotator_answers WHERE project_id = :p_id AND user_id = :u_id"),
                               {"p_id": project_id, "u_id": user_id})
        for row in result.fetchall():
            operations.extend(_collect_annotator_answer_dependencies(session, row[0], backup_params))
            operations.append(DeleteOperation('AnnotatorAnswer', 'using_id', (row[0],), backup_params))
    
    return operations


def _collect_annotator_answer_dependencies(session: Session, answer_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for AnnotatorAnswer deletion"""
    return _collect_answer_review_dependencies(session, answer_id, backup_params)


def _collect_question_group_question_dependencies(session: Session, question_group_id: int, question_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for QuestionGroupQuestion deletion"""
    operations = []
    
    # Step 1: Get all schemas using this question group
    schema_result = session.execute(text("SELECT schema_id FROM schema_question_groups WHERE question_group_id = :qg_id"), 
                                  {"qg_id": question_group_id})
    schema_ids = [row[0] for row in schema_result.fetchall()]
    
    # Step 2: Get all projects using those schemas
    project_ids = []
    for schema_id in schema_ids:
        project_result = session.execute(text("SELECT id FROM projects WHERE schema_id = :s_id"), {"s_id": schema_id})
        project_ids.extend([row[0] for row in project_result.fetchall()])
    
    # Step 3: Delete related answers and displays for each project
    for project_id in project_ids:
        # Delete AnnotatorAnswers
        result = session.execute(text("SELECT id FROM annotator_answers WHERE project_id = :p_id AND question_id = :q_id"),
                               {"p_id": project_id, "q_id": question_id})
        for row in result.fetchall():
            operations.extend(_collect_annotator_answer_dependencies(session, row[0], backup_params))
            operations.append(DeleteOperation('AnnotatorAnswer', 'using_id', (row[0],), backup_params))
        
        # Delete ReviewerGroundTruth
        result = session.execute(text("SELECT video_id FROM reviewer_ground_truth WHERE project_id = :p_id AND question_id = :q_id"),
                               {"p_id": project_id, "q_id": question_id})
        for row in result.fetchall():
            operations.append(DeleteOperation('ReviewerGroundTruth', 'using_id', (row[0], question_id, project_id), backup_params))
        
        # Delete ProjectVideoQuestionDisplay
        result = session.execute(text("SELECT video_id FROM project_video_question_displays WHERE project_id = :p_id AND question_id = :q_id"),
                               {"p_id": project_id, "q_id": question_id})
        for row in result.fetchall():
            operations.append(DeleteOperation('ProjectVideoQuestionDisplay', 'using_id', (project_id, row[0], question_id), backup_params))
    
    return operations


def _collect_schema_question_group_dependencies(session: Session, schema_id: int, question_group_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for SchemaQuestionGroup deletion"""
    operations = []
    
    # Step 1: Get all projects using this schema
    project_result = session.execute(text("SELECT id FROM projects WHERE schema_id = :s_id"), {"s_id": schema_id})
    project_ids = [row[0] for row in project_result.fetchall()]
    
    # Step 2: Get all questions from this question group
    question_result = session.execute(text("SELECT question_id FROM question_group_questions WHERE question_group_id = :qg_id"), 
                                    {"qg_id": question_group_id})
    question_ids = [row[0] for row in question_result.fetchall()]
    
    # Step 3: Delete related answers and displays for each project+question combination
    for project_id in project_ids:
        for question_id in question_ids:
            # Delete AnnotatorAnswers
            result = session.execute(text("SELECT id FROM annotator_answers WHERE project_id = :p_id AND question_id = :q_id"),
                                   {"p_id": project_id, "q_id": question_id})
            for row in result.fetchall():
                operations.extend(_collect_annotator_answer_dependencies(session, row[0], backup_params))
                operations.append(DeleteOperation('AnnotatorAnswer', 'using_id', (row[0],), backup_params))
            
            # Delete ReviewerGroundTruth
            result = session.execute(text("SELECT video_id FROM reviewer_ground_truth WHERE project_id = :p_id AND question_id = :q_id"),
                                   {"p_id": project_id, "q_id": question_id})
            for row in result.fetchall():
                operations.append(DeleteOperation('ReviewerGroundTruth', 'using_id', (row[0], question_id, project_id), backup_params))
            
            # Delete ProjectVideoQuestionDisplay
            result = session.execute(text("SELECT video_id FROM project_video_question_displays WHERE project_id = :p_id AND question_id = :q_id"),
                                   {"p_id": project_id, "q_id": question_id})
            for row in result.fetchall():
                operations.append(DeleteOperation('ProjectVideoQuestionDisplay', 'using_id', (project_id, row[0], question_id), backup_params))
    
    return operations


def _collect_video_dependencies(session: Session, video_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for Video deletion"""
    operations = []
    
    # Delete VideoTags
    result = session.execute(text("SELECT tag FROM video_tags WHERE video_id = :v_id"), {"v_id": video_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('VideoTag', 'using_id', (video_id, row[0]), backup_params))
    
    # Delete ProjectVideos (and their dependencies)
    result = session.execute(text("SELECT project_id FROM project_videos WHERE video_id = :v_id"), {"v_id": video_id})
    for row in result.fetchall():
        operations.extend(_collect_project_video_dependencies(session, row[0], video_id, backup_params))
        operations.append(DeleteOperation('ProjectVideo', 'using_id', (row[0], video_id), backup_params))
    
    return operations


def _collect_question_dependencies(session: Session, question_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for Question deletion"""
    operations = []
    
    # Delete QuestionGroupQuestions (and their dependencies)
    result = session.execute(text("SELECT question_group_id FROM question_group_questions WHERE question_id = :q_id"), {"q_id": question_id})
    for row in result.fetchall():
        operations.extend(_collect_question_group_question_dependencies(session, row[0], question_id, backup_params))
        operations.append(DeleteOperation('QuestionGroupQuestion', 'using_id', (row[0], question_id), backup_params))
    
    return operations


def _collect_question_group_dependencies(session: Session, question_group_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for QuestionGroup deletion"""
    operations = []
    
    # Delete QuestionGroupQuestions (and their dependencies)
    result = session.execute(text("SELECT question_id FROM question_group_questions WHERE question_group_id = :qg_id"), {"qg_id": question_group_id})
    for row in result.fetchall():
        operations.extend(_collect_question_group_question_dependencies(session, question_group_id, row[0], backup_params))
        operations.append(DeleteOperation('QuestionGroupQuestion', 'using_id', (question_group_id, row[0]), backup_params))
    
    # Delete SchemaQuestionGroups (and their dependencies)
    result = session.execute(text("SELECT schema_id FROM schema_question_groups WHERE question_group_id = :qg_id"), {"qg_id": question_group_id})
    for row in result.fetchall():
        operations.extend(_collect_schema_question_group_dependencies(session, row[0], question_group_id, backup_params))
        operations.append(DeleteOperation('SchemaQuestionGroup', 'using_id', (row[0], question_group_id), backup_params))
    
    return operations


def _collect_project_dependencies(session: Session, project_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for Project deletion"""
    operations = []
    
    # Delete ProjectVideos (and their dependencies)
    result = session.execute(text("SELECT video_id FROM project_videos WHERE project_id = :p_id"), {"p_id": project_id})
    for row in result.fetchall():
        operations.extend(_collect_project_video_dependencies(session, project_id, row[0], backup_params))
        operations.append(DeleteOperation('ProjectVideo', 'using_id', (project_id, row[0]), backup_params))
    
    # Delete ProjectUserRoles (and their dependencies)
    result = session.execute(text("SELECT user_id, role FROM project_user_roles WHERE project_id = :p_id"), {"p_id": project_id})
    for row in result.fetchall():
        operations.extend(_collect_project_user_role_dependencies(session, project_id, row[0], row[1], backup_params))
        operations.append(DeleteOperation('ProjectUserRole', 'using_id', (project_id, row[0], row[1]), backup_params))
    
    # Delete ProjectGroupProjects
    result = session.execute(text("SELECT project_group_id FROM project_group_projects WHERE project_id = :p_id"), {"p_id": project_id})
    for row in result.fetchall():
        operations.append(DeleteOperation('ProjectGroupProject', 'using_id', (row[0], project_id), backup_params))
    
    return operations


def _collect_user_dependencies(session: Session, user_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for User deletion"""
    operations = []
    
    # Delete ProjectUserRoles (and their dependencies)
    result = session.execute(text("SELECT project_id, role FROM project_user_roles WHERE user_id = :u_id"), {"u_id": user_id})
    for row in result.fetchall():
        operations.extend(_collect_project_user_role_dependencies(session, row[0], user_id, row[1], backup_params))
        operations.append(DeleteOperation('ProjectUserRole', 'using_id', (row[0], user_id, row[1]), backup_params))
    
    return operations


def _collect_schema_dependencies(session: Session, schema_id: int, backup_params: Dict[str, Any]) -> List[DeleteOperation]:
    """Collect dependencies for Schema deletion"""
    operations = []
    
    # Delete SchemaQuestionGroups (and their dependencies)
    result = session.execute(text("SELECT question_group_id FROM schema_question_groups WHERE schema_id = :s_id"), {"s_id": schema_id})
    for row in result.fetchall():
        operations.extend(_collect_schema_question_group_dependencies(session, schema_id, row[0], backup_params))
        operations.append(DeleteOperation('SchemaQuestionGroup', 'using_id', (schema_id, row[0]), backup_params))
    
    # Delete Projects (and their dependencies)
    result = session.execute(text("SELECT id FROM projects WHERE schema_id = :s_id"), {"s_id": schema_id})
    for row in result.fetchall():
        operations.extend(_collect_project_dependencies(session, row[0], backup_params))
        operations.append(DeleteOperation('Project', 'using_id', (row[0],), backup_params))
    
    return operations


# ============================================================================
# DELETE USING ID FUNCTIONS
# ============================================================================

def delete_user_using_id(user_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                        backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete user by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if user exists
            result = session.execute(text("SELECT user_id_str FROM users WHERE id = :id"), {"id": user_id})
            row = result.fetchone()
            if not row:
                print(f"❌ User with ID {user_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for User ID {user_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_user_dependencies(session, user_id, backup_params)
            operations.append(DeleteOperation('User', 'using_id', (user_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return False


def delete_video_using_id(video_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                         backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete video by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if video exists
            result = session.execute(text("SELECT video_uid FROM videos WHERE id = :id"), {"id": video_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Video with ID {video_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for Video ID {video_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_video_dependencies(session, video_id, backup_params)
            operations.append(DeleteOperation('Video', 'using_id', (video_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting video: {e}")
        return False


def delete_video_tag_using_id(video_id: int, tag: str, backup_first: bool = True, backup_dir: str = "./backups", 
                             backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete video tag by video_id and tag"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if video tag exists
            result = session.execute(text("SELECT 1 FROM video_tags WHERE video_id = :v_id AND tag = :tag"), {"v_id": video_id, "tag": tag})
            if not result.fetchone():
                print(f"❌ Video tag not found: video_id={video_id}, tag='{tag}'")
                return False
            
            print(f"🔍 Deleting VideoTag: video_id={video_id}, tag='{tag}'")
            
            operations = [DeleteOperation('VideoTag', 'using_id', (video_id, tag), backup_params)]
            sorted_operations = _collect_delete_operations(operations)
            
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting video tag: {e}")
        return False


def delete_question_group_using_id(question_group_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                                  backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question group by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if question group exists
            result = session.execute(text("SELECT title FROM question_groups WHERE id = :id"), {"id": question_group_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Question group with ID {question_group_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for QuestionGroup ID {question_group_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_question_group_dependencies(session, question_group_id, backup_params)
            operations.append(DeleteOperation('QuestionGroup', 'using_id', (question_group_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting question group: {e}")
        return False


def delete_question_using_id(question_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                            backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if question exists
            result = session.execute(text("SELECT text FROM questions WHERE id = :id"), {"id": question_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Question with ID {question_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for Question ID {question_id}")
            
            # Collect all dependent operations
            operations = _collect_question_dependencies(session, question_id, backup_params)
            operations.append(DeleteOperation('Question', 'using_id', (question_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting question: {e}")
        return False


def delete_question_group_question_using_id(question_group_id: int, question_id: int, backup_first: bool = True, 
                                           backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question group question relationship by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if relationship exists
            result = session.execute(text("SELECT 1 FROM question_group_questions WHERE question_group_id = :qg_id AND question_id = :q_id"), 
                                   {"qg_id": question_group_id, "q_id": question_id})
            if not result.fetchone():
                print(f"❌ QuestionGroupQuestion not found: question_group_id={question_group_id}, question_id={question_id}")
                return False
            
            print(f"🔍 Collecting dependencies for QuestionGroupQuestion: question_group_id={question_group_id}, question_id={question_id}")
            
            # Collect all dependent operations
            operations = _collect_question_group_question_dependencies(session, question_group_id, question_id, backup_params)
            operations.append(DeleteOperation('QuestionGroupQuestion', 'using_id', (question_group_id, question_id), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting question group question: {e}")
        return False


def delete_schema_using_id(schema_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                          backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete schema by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if schema exists
            result = session.execute(text("SELECT name FROM schemas WHERE id = :id"), {"id": schema_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Schema with ID {schema_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for Schema ID {schema_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_schema_dependencies(session, schema_id, backup_params)
            operations.append(DeleteOperation('Schema', 'using_id', (schema_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting schema: {e}")
        return False


def delete_schema_question_group_using_id(schema_id: int, question_group_id: int, backup_first: bool = True, 
                                         backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete schema question group relationship by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if relationship exists
            result = session.execute(text("SELECT 1 FROM schema_question_groups WHERE schema_id = :s_id AND question_group_id = :qg_id"), 
                                   {"s_id": schema_id, "qg_id": question_group_id})
            if not result.fetchone():
                print(f"❌ SchemaQuestionGroup not found: schema_id={schema_id}, question_group_id={question_group_id}")
                return False
            
            print(f"🔍 Collecting dependencies for SchemaQuestionGroup: schema_id={schema_id}, question_group_id={question_group_id}")
            
            # Collect all dependent operations
            operations = _collect_schema_question_group_dependencies(session, schema_id, question_group_id, backup_params)
            operations.append(DeleteOperation('SchemaQuestionGroup', 'using_id', (schema_id, question_group_id), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting schema question group: {e}")
        return False


def delete_project_using_id(project_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                           backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if project exists
            result = session.execute(text("SELECT name FROM projects WHERE id = :id"), {"id": project_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Project with ID {project_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for Project ID {project_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_project_dependencies(session, project_id, backup_params)
            operations.append(DeleteOperation('Project', 'using_id', (project_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project: {e}")
        return False


def delete_project_video_using_id(project_id: int, video_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                                 backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project video relationship by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if relationship exists
            result = session.execute(text("SELECT 1 FROM project_videos WHERE project_id = :p_id AND video_id = :v_id"), 
                                   {"p_id": project_id, "v_id": video_id})
            if not result.fetchone():
                print(f"❌ ProjectVideo not found: project_id={project_id}, video_id={video_id}")
                return False
            
            print(f"🔍 Collecting dependencies for ProjectVideo: project_id={project_id}, video_id={video_id}")
            
            # Collect all dependent operations
            operations = _collect_project_video_dependencies(session, project_id, video_id, backup_params)
            operations.append(DeleteOperation('ProjectVideo', 'using_id', (project_id, video_id), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project video: {e}")
        return False


def delete_project_user_role_using_id(project_id: int, user_id: int, role: str, backup_first: bool = True, 
                                     backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project user role by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if relationship exists
            result = session.execute(text("SELECT 1 FROM project_user_roles WHERE project_id = :p_id AND user_id = :u_id AND role = :role"), 
                                   {"p_id": project_id, "u_id": user_id, "role": role})
            if not result.fetchone():
                print(f"❌ ProjectUserRole not found: project_id={project_id}, user_id={user_id}, role='{role}'")
                return False
            
            print(f"🔍 Collecting dependencies for ProjectUserRole: project_id={project_id}, user_id={user_id}, role='{role}'")
            
            # Collect all dependent operations
            operations = _collect_project_user_role_dependencies(session, project_id, user_id, role, backup_params)
            operations.append(DeleteOperation('ProjectUserRole', 'using_id', (project_id, user_id, role), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project user role: {e}")
        return False


def delete_project_group_using_id(project_group_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                                 backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project group by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if project group exists
            result = session.execute(text("SELECT name FROM project_groups WHERE id = :id"), {"id": project_group_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Project group with ID {project_group_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for ProjectGroup ID {project_group_id} ('{row[0]}')")
            
            # Collect all dependent operations
            operations = _collect_project_group_dependencies(session, project_group_id, backup_params)
            operations.append(DeleteOperation('ProjectGroup', 'using_id', (project_group_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project group: {e}")
        return False


def delete_project_group_project_using_id(project_group_id: int, project_id: int, backup_first: bool = True, 
                                         backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project group project relationship by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if relationship exists
            result = session.execute(text("SELECT 1 FROM project_group_projects WHERE project_group_id = :pg_id AND project_id = :p_id"), 
                                   {"pg_id": project_group_id, "p_id": project_id})
            if not result.fetchone():
                print(f"❌ ProjectGroupProject not found: project_group_id={project_group_id}, project_id={project_id}")
                return False
            
            print(f"🔍 Deleting ProjectGroupProject: project_group_id={project_group_id}, project_id={project_id}")
            
            operations = [DeleteOperation('ProjectGroupProject', 'using_id', (project_group_id, project_id), backup_params)]
            sorted_operations = _collect_delete_operations(operations)
            
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project group project: {e}")
        return False


def delete_project_video_question_display_using_id(project_id: int, video_id: int, question_id: int, backup_first: bool = True, 
                                                   backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project video question display by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if display exists
            result = session.execute(text("SELECT 1 FROM project_video_question_displays WHERE project_id = :p_id AND video_id = :v_id AND question_id = :q_id"), 
                                   {"p_id": project_id, "v_id": video_id, "q_id": question_id})
            if not result.fetchone():
                print(f"❌ ProjectVideoQuestionDisplay not found: project_id={project_id}, video_id={video_id}, question_id={question_id}")
                return False
            
            print(f"🔍 Deleting ProjectVideoQuestionDisplay: project_id={project_id}, video_id={video_id}, question_id={question_id}")
            
            operations = [DeleteOperation('ProjectVideoQuestionDisplay', 'using_id', (project_id, video_id, question_id), backup_params)]
            sorted_operations = _collect_delete_operations(operations)
            
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting project video question display: {e}")
        return False


def delete_annotator_answer_using_id(answer_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                                    backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete annotator answer by ID with cascade deletion of all dependent data"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if answer exists
            result = session.execute(text("SELECT user_id FROM annotator_answers WHERE id = :id"), {"id": answer_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Annotator answer with ID {answer_id} not found")
                return False
            
            print(f"🔍 Collecting dependencies for AnnotatorAnswer ID {answer_id}")
            
            # Collect all dependent operations
            operations = _collect_annotator_answer_dependencies(session, answer_id, backup_params)
            operations.append(DeleteOperation('AnnotatorAnswer', 'using_id', (answer_id,), backup_params))
            
            # Sort and deduplicate
            sorted_operations = _collect_delete_operations(operations)
            
            # Confirm and execute
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting annotator answer: {e}")
        return False


def delete_reviewer_ground_truth_using_id(video_id: int, question_id: int, project_id: int, backup_first: bool = True, 
                                         backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete reviewer ground truth by IDs"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if ground truth exists
            result = session.execute(text("SELECT reviewer_id FROM reviewer_ground_truth WHERE video_id = :v_id AND question_id = :q_id AND project_id = :p_id"), 
                                   {"v_id": video_id, "q_id": question_id, "p_id": project_id})
            row = result.fetchone()
            if not row:
                print(f"❌ ReviewerGroundTruth not found: video_id={video_id}, question_id={question_id}, project_id={project_id}")
                return False
            
            print(f"🔍 Deleting ReviewerGroundTruth: video_id={video_id}, question_id={question_id}, project_id={project_id}")
            
            operations = [DeleteOperation('ReviewerGroundTruth', 'using_id', (video_id, question_id, project_id), backup_params)]
            sorted_operations = _collect_delete_operations(operations)
            
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting reviewer ground truth: {e}")
        return False


def delete_answer_review_using_id(answer_review_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                                 backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete answer review by ID"""
    backup_params = {"backup_first": backup_first, "backup_dir": backup_dir, "backup_file": backup_file, "compress": compress}
    
    try:
        with label_pizza.db.SessionLocal() as session:
            # Check if review exists
            result = session.execute(text("SELECT answer_id FROM answer_reviews WHERE id = :id"), {"id": answer_review_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Answer review with ID {answer_review_id} not found")
                return False
            
            print(f"🔍 Deleting AnswerReview ID {answer_review_id}")
            
            operations = [DeleteOperation('AnswerReview', 'using_id', (answer_review_id,), backup_params)]
            sorted_operations = _collect_delete_operations(operations)
            
            if not _confirm_cascade_deletion(sorted_operations):
                print("❌ Deletion cancelled")
                return False
            
            return _execute_delete_operations(sorted_operations)
            
    except Exception as e:
        print(f"❌ Error deleting answer review: {e}")
        return False


# ============================================================================
# DELETE USING NAME FUNCTIONS  
# ============================================================================

def delete_user(user_id_str: str, backup_first: bool = True, backup_dir: str = "./backups", 
               backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete user by user_id_str with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            user_id = _get_user_id_from_user_id_str(session, user_id_str)
            return delete_user_using_id(user_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting user '{user_id_str}': {e}")
        return False


def delete_video(video_uid: str, backup_first: bool = True, backup_dir: str = "./backups", 
                backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete video by video_uid with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            video_id = _get_video_id_from_video_uid(session, video_uid)
            return delete_video_using_id(video_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting video '{video_uid}': {e}")
        return False


def delete_video_tag(video_uid: str, tag: str, backup_first: bool = True, backup_dir: str = "./backups", 
                    backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete video tag by video_uid and tag"""
    try:
        with label_pizza.db.SessionLocal() as session:
            video_id = _get_video_id_from_video_uid(session, video_uid)
            return delete_video_tag_using_id(video_id, tag, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting video tag '{video_uid}'/'{tag}': {e}")
        return False


def delete_question_group(title: str, backup_first: bool = True, backup_dir: str = "./backups", 
                         backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question group by title with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            question_group_id = _get_question_group_id_from_title(session, title)
            return delete_question_group_using_id(question_group_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting question group '{title}': {e}")
        return False


def delete_question(text: str, backup_first: bool = True, backup_dir: str = "./backups", 
                   backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question by text with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            question_id = _get_question_id_from_text(session, text)
            return delete_question_using_id(question_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting question '{text}': {e}")
        return False


def delete_question_group_question(question_group_title: str, question_text: str, backup_first: bool = True, 
                                  backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete question group question relationship by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            question_group_id = _get_question_group_id_from_title(session, question_group_title)
            question_id = _get_question_id_from_text(session, question_text)
            return delete_question_group_question_using_id(question_group_id, question_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting question group question '{question_group_title}'/'{question_text}': {e}")
        return False


def delete_schema(name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                 backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete schema by name with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            schema_id = _get_schema_id_from_name(session, name)
            return delete_schema_using_id(schema_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting schema '{name}': {e}")
        return False


def delete_schema_question_group(schema_name: str, question_group_title: str, backup_first: bool = True, 
                                backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete schema question group relationship by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            schema_id = _get_schema_id_from_name(session, schema_name)
            question_group_id = _get_question_group_id_from_title(session, question_group_title)
            return delete_schema_question_group_using_id(schema_id, question_group_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting schema question group '{schema_name}'/'{question_group_title}': {e}")
        return False


def delete_project(name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                  backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project by name with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_id = _get_project_id_from_name(session, name)
            return delete_project_using_id(project_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project '{name}': {e}")
        return False


def delete_project_video(project_name: str, video_uid: str, backup_first: bool = True, backup_dir: str = "./backups", 
                        backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project video relationship by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_id = _get_project_id_from_name(session, project_name)
            video_id = _get_video_id_from_video_uid(session, video_uid)
            return delete_project_video_using_id(project_id, video_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project video '{project_name}'/'{video_uid}': {e}")
        return False


def delete_project_user_role(project_name: str, user_id_str: str, role: str, backup_first: bool = True, 
                            backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project user role by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_id = _get_project_id_from_name(session, project_name)
            user_id = _get_user_id_from_user_id_str(session, user_id_str)
            return delete_project_user_role_using_id(project_id, user_id, role, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project user role '{project_name}'/'{user_id_str}'/'{role}': {e}")
        return False


def delete_project_group(name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                        backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project group by name with cascade deletion of all dependent data"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_group_id = _get_project_group_id_from_name(session, name)
            return delete_project_group_using_id(project_group_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project group '{name}': {e}")
        return False


def delete_project_group_project(project_group_name: str, project_name: str, backup_first: bool = True, 
                                backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project group project relationship by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_group_id = _get_project_group_id_from_name(session, project_group_name)
            project_id = _get_project_id_from_name(session, project_name)
            return delete_project_group_project_using_id(project_group_id, project_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project group project '{project_group_name}'/'{project_name}': {e}")
        return False


def delete_project_video_question_display(project_name: str, video_uid: str, question_text: str, backup_first: bool = True, 
                                         backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete project video question display by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_id = _get_project_id_from_name(session, project_name)
            video_id = _get_video_id_from_video_uid(session, video_uid)
            question_id = _get_question_id_from_text(session, question_text)
            return delete_project_video_question_display_using_id(project_id, video_id, question_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting project video question display '{project_name}'/'{video_uid}'/'{question_text}': {e}")
        return False


def delete_annotator_answer(video_uid: str, question_text: str, user_id_str: str, project_name: str, 
                           backup_first: bool = True, backup_dir: str = "./backups", 
                           backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete annotator answer by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            video_id = _get_video_id_from_video_uid(session, video_uid)
            question_id = _get_question_id_from_text(session, question_text)
            user_id = _get_user_id_from_user_id_str(session, user_id_str)
            project_id = _get_project_id_from_name(session, project_name)
            
            # Find the answer ID
            result = session.execute(text("""
                SELECT id FROM annotator_answers 
                WHERE video_id = :v_id AND question_id = :q_id AND user_id = :u_id AND project_id = :p_id
            """), {"v_id": video_id, "q_id": question_id, "u_id": user_id, "p_id": project_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Annotator answer not found for video '{video_uid}', question '{question_text}', user '{user_id_str}', project '{project_name}'")
                return False
                
            return delete_annotator_answer_using_id(row[0], backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting annotator answer: {e}")
        return False


def delete_reviewer_ground_truth(video_uid: str, question_text: str, project_name: str, backup_first: bool = True, 
                                backup_dir: str = "./backups", backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete reviewer ground truth by names"""
    try:
        with label_pizza.db.SessionLocal() as session:
            video_id = _get_video_id_from_video_uid(session, video_uid)
            question_id = _get_question_id_from_text(session, question_text)
            project_id = _get_project_id_from_name(session, project_name)
            return delete_reviewer_ground_truth_using_id(video_id, question_id, project_id, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error deleting reviewer ground truth '{video_uid}'/'{question_text}'/'{project_name}': {e}")
        return False


def delete_answer_review(answer_id: int, backup_first: bool = True, backup_dir: str = "./backups", 
                        backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Delete answer review by answer_id - this function takes answer_id directly as specified in the table"""
    return delete_answer_review_using_id(answer_id, backup_first, backup_dir, backup_file, compress)


# ============================================================================
# NAME MANAGEMENT FUNCTIONS - GET FUNCTIONS
# ============================================================================

def get_user_name(user_id: int) -> str:
    """Get user_id_str from user ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT user_id_str FROM users WHERE id = :id"), {"id": user_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"User with ID {user_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting user name: {e}")


def get_video_name(video_id: int) -> str:
    """Get video_uid from video ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT video_uid FROM videos WHERE id = :id"), {"id": video_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Video with ID {video_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting video name: {e}")


def get_question_group_name(question_group_id: int) -> str:
    """Get title from question group ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT title FROM question_groups WHERE id = :id"), {"id": question_group_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Question group with ID {question_group_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting question group name: {e}")


def get_question_name(question_id: int) -> str:
    """Get text from question ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT text FROM questions WHERE id = :id"), {"id": question_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Question with ID {question_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting question name: {e}")


def get_schema_name(schema_id: int) -> str:
    """Get name from schema ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT name FROM schemas WHERE id = :id"), {"id": schema_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Schema with ID {schema_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting schema name: {e}")


def get_project_name(project_id: int) -> str:
    """Get name from project ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT name FROM projects WHERE id = :id"), {"id": project_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Project with ID {project_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting project name: {e}")


def get_project_group_name(project_group_id: int) -> str:
    """Get name from project group ID"""
    try:
        with label_pizza.db.SessionLocal() as session:
            result = session.execute(text("SELECT name FROM project_groups WHERE id = :id"), {"id": project_group_id})
            row = result.fetchone()
            if not row:
                raise ValueError(f"Project group with ID {project_group_id} not found")
            return row[0]
    except Exception as e:
        raise ValueError(f"Error getting project group name: {e}")


# ============================================================================
# NAME MANAGEMENT FUNCTIONS - SET USING ID FUNCTIONS
# ============================================================================

def set_user_name_using_id(user_id: int, user_id_str: str, backup_first: bool = True, backup_dir: str = "./backups", 
                          backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set user_id_str for user ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if user exists
            result = session.execute(text("SELECT user_id_str FROM users WHERE id = :id"), {"id": user_id})
            row = result.fetchone()
            if not row:
                print(f"❌ User with ID {user_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM users WHERE user_id_str = :user_id_str AND id != :id"), 
                                   {"user_id_str": user_id_str, "id": user_id})
            if result.fetchone():
                print(f"❌ User ID string '{user_id_str}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE users SET user_id_str = :user_id_str WHERE id = :id"), 
                          {"user_id_str": user_id_str, "id": user_id})
            session.commit()
            
            print(f"✅ Updated user ID {user_id}: '{old_name}' → '{user_id_str}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting user name: {e}")
        return False


def set_video_name_using_id(video_id: int, video_uid: str, backup_first: bool = True, backup_dir: str = "./backups", 
                           backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set video_uid for video ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if video exists
            result = session.execute(text("SELECT video_uid FROM videos WHERE id = :id"), {"id": video_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Video with ID {video_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM videos WHERE video_uid = :video_uid AND id != :id"), 
                                   {"video_uid": video_uid, "id": video_id})
            if result.fetchone():
                print(f"❌ Video UID '{video_uid}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE videos SET video_uid = :video_uid WHERE id = :id"), 
                          {"video_uid": video_uid, "id": video_id})
            session.commit()
            
            print(f"✅ Updated video ID {video_id}: '{old_name}' → '{video_uid}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting video name: {e}")
        return False


def set_question_group_name_using_id(question_group_id: int, title: str, backup_first: bool = True, backup_dir: str = "./backups", 
                                    backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set title for question group ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if question group exists
            result = session.execute(text("SELECT title FROM question_groups WHERE id = :id"), {"id": question_group_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Question group with ID {question_group_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM question_groups WHERE title = :title AND id != :id"), 
                                   {"title": title, "id": question_group_id})
            if result.fetchone():
                print(f"❌ Question group title '{title}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE question_groups SET title = :title WHERE id = :id"), 
                          {"title": title, "id": question_group_id})
            session.commit()
            
            print(f"✅ Updated question group ID {question_group_id}: '{old_name}' → '{title}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting question group name: {e}")
        return False

def set_question_name_using_id(question_id: int, question_text: str, backup_first: bool = True, backup_dir: str = "./backups", 
                              backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set text for question ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if question exists
            result = session.execute(text("SELECT text FROM questions WHERE id = :id"), {"id": question_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Question with ID {question_id} not found")
                return False
                
            old_text = row[0]
            
            # Check if new text is unique
            result = session.execute(text("SELECT id FROM questions WHERE text = :text AND id != :id"), 
                                   {"text": question_text, "id": question_id})  # Fixed: use question_text parameter
            if result.fetchone():
                print(f"❌ Question text already exists")
                return False
            
            # Update the text
            session.execute(text("UPDATE questions SET text = :text WHERE id = :id"), 
                          {"text": question_text, "id": question_id})  # Fixed: use question_text parameter
            session.commit()
            
            print(f"✅ Updated question ID {question_id} text")
            return True
            
    except Exception as e:
        print(f"❌ Error setting question name: {e}")
        return False


def set_schema_name_using_id(schema_id: int, name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                            backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name for schema ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if schema exists
            result = session.execute(text("SELECT name FROM schemas WHERE id = :id"), {"id": schema_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Schema with ID {schema_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM schemas WHERE name = :name AND id != :id"), 
                                   {"name": name, "id": schema_id})
            if result.fetchone():
                print(f"❌ Schema name '{name}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE schemas SET name = :name WHERE id = :id"), 
                          {"name": name, "id": schema_id})
            session.commit()
            
            print(f"✅ Updated schema ID {schema_id}: '{old_name}' → '{name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting schema name: {e}")
        return False


def set_project_name_using_id(project_id: int, name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                             backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name for project ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if project exists
            result = session.execute(text("SELECT name FROM projects WHERE id = :id"), {"id": project_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Project with ID {project_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM projects WHERE name = :name AND id != :id"), 
                                   {"name": name, "id": project_id})
            if result.fetchone():
                print(f"❌ Project name '{name}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE projects SET name = :name WHERE id = :id"), 
                          {"name": name, "id": project_id})
            session.commit()
            
            print(f"✅ Updated project ID {project_id}: '{old_name}' → '{name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting project name: {e}")
        return False


def set_project_group_name_using_id(project_group_id: int, name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                                   backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name for project group ID"""
    try:
        if backup_first:
            try:
                db_url = get_db_url_for_backup()
                backup_file_created = create_backup_if_requested(str(db_url), backup_dir, backup_file, compress)
                print(f"💾 Backup created: {backup_file_created}")
            except Exception as e:
                print(f"❌ Failed to create backup: {e}")
                print("⚠️  Backup creation failed, but backup_first=True was requested.")
                response = input("Continue without backup? Type 'CONTINUE' to proceed: ")
                if response.strip() != 'CONTINUE':
                    print("❌ Operation cancelled due to backup failure")
                    return False
            
        with label_pizza.db.SessionLocal() as session:
            # Check if project group exists
            result = session.execute(text("SELECT name FROM project_groups WHERE id = :id"), {"id": project_group_id})
            row = result.fetchone()
            if not row:
                print(f"❌ Project group with ID {project_group_id} not found")
                return False
                
            old_name = row[0]
            
            # Check if new name is unique
            result = session.execute(text("SELECT id FROM project_groups WHERE name = :name AND id != :id"), 
                                   {"name": name, "id": project_group_id})
            if result.fetchone():
                print(f"❌ Project group name '{name}' already exists")
                return False
            
            # Update the name
            session.execute(text("UPDATE project_groups SET name = :name WHERE id = :id"), 
                          {"name": name, "id": project_group_id})
            session.commit()
            
            print(f"✅ Updated project group ID {project_group_id}: '{old_name}' → '{name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error setting project group name: {e}")
        return False


# ============================================================================
# NAME MANAGEMENT FUNCTIONS - SET USING NAME FUNCTIONS (NEW!)
# ============================================================================

def set_user_name(old_user_id_str: str, new_user_id_str: str, backup_first: bool = True, backup_dir: str = "./backups", 
                 backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set user_id_str by current user_id_str"""
    try:
        with label_pizza.db.SessionLocal() as session:
            user_id = _get_user_id_from_user_id_str(session, old_user_id_str)
            return set_user_name_using_id(user_id, new_user_id_str, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting user name '{old_user_id_str}': {e}")
        return False


def set_video_name(old_video_uid: str, new_video_uid: str, backup_first: bool = True, backup_dir: str = "./backups", 
                  backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set video_uid by current video_uid"""
    try:
        with label_pizza.db.SessionLocal() as session:
            video_id = _get_video_id_from_video_uid(session, old_video_uid)
            return set_video_name_using_id(video_id, new_video_uid, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting video name '{old_video_uid}': {e}")
        return False


def set_question_group_name(old_title: str, new_title: str, backup_first: bool = True, backup_dir: str = "./backups", 
                           backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set title by current title"""
    try:
        with label_pizza.db.SessionLocal() as session:
            question_group_id = _get_question_group_id_from_title(session, old_title)
            return set_question_group_name_using_id(question_group_id, new_title, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting question group name '{old_title}': {e}")
        return False


def set_question_name(old_text: str, new_text: str, backup_first: bool = True, backup_dir: str = "./backups", 
                     backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set text by current text"""
    try:
        with label_pizza.db.SessionLocal() as session:
            question_id = _get_question_id_from_text(session, old_text)
            return set_question_name_using_id(question_id, new_text, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting question name '{old_text}': {e}")
        return False


def set_schema_name(old_name: str, new_name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                   backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name by current name"""
    try:
        with label_pizza.db.SessionLocal() as session:
            schema_id = _get_schema_id_from_name(session, old_name)
            return set_schema_name_using_id(schema_id, new_name, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting schema name '{old_name}': {e}")
        return False


def set_project_name(old_name: str, new_name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                    backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name by current name"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_id = _get_project_id_from_name(session, old_name)
            return set_project_name_using_id(project_id, new_name, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting project name '{old_name}': {e}")
        return False


def set_project_group_name(old_name: str, new_name: str, backup_first: bool = True, backup_dir: str = "./backups", 
                          backup_file: Optional[str] = None, compress: bool = True) -> bool:
    """Set name by current name"""
    try:
        with label_pizza.db.SessionLocal() as session:
            project_group_id = _get_project_group_id_from_name(session, old_name)
            return set_project_group_name_using_id(project_group_id, new_name, backup_first, backup_dir, backup_file, compress)
    except Exception as e:
        print(f"❌ Error setting project group name '{old_name}': {e}")
        return False


if __name__ == "__main__":
    print("Label Pizza Override Utils")
    print("=========================")
    print()
    print("This module provides cascade delete and name management functions.")
    print("Before using any functions, always run:")
    print()
    print("    from label_pizza.db import init_database")
    print("    init_database('DBURL')")
    print()
    print("See the module docstring for full usage examples.")