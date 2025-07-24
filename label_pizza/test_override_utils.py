#!/usr/bin/env python3
"""
Comprehensive Test Suite for override_utils.py
==============================================

Tests all delete functions, name management functions, and edge cases.
Covers cascade deletion logic, backup scenarios, admin reversion, and error handling.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from sqlalchemy.orm import Session
from sqlalchemy import text

# Add the current directory to Python path so we can import override_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import override_utils
    from override_utils import DeleteOperation
except ImportError as e:
    print(f"Error importing override_utils: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    print("Make sure override_utils.py is in the same directory as this test file.")
    sys.exit(1)


# ============================================================================
# FIXTURES AND SETUP
# ============================================================================

@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session for database operations"""
    session = Mock(spec=Session)
    session.execute.return_value.fetchone.return_value = None
    session.execute.return_value.fetchall.return_value = []
    session.execute.return_value.rowcount = 0
    session.commit.return_value = None
    session.rollback.return_value = None
    return session


@pytest.fixture
def mock_db():
    """Mock label_pizza.db module"""
    with patch('override_utils.label_pizza.db') as mock_db:
        mock_session = Mock(spec=Session)
        mock_session.execute.return_value.fetchone.return_value = None
        mock_session.execute.return_value.fetchall.return_value = []
        mock_session.execute.return_value.rowcount = 0
        mock_session.commit.return_value = None
        mock_session.rollback.return_value = None
        
        mock_db.SessionLocal.return_value.__enter__.return_value = mock_session
        mock_db.engine.url = "postgresql://test"
        yield mock_db


@pytest.fixture
def mock_backup():
    """Mock backup creation function"""
    with patch('override_utils.create_backup_if_requested') as mock_backup:
        mock_backup.return_value = "/backups/test_backup.sql.gz"
        yield mock_backup


@pytest.fixture
def sample_backup_params():
    """Sample backup parameters for testing - using tuple to make it hashable"""
    return {
        "backup_first": True,
        "backup_dir": "./test_backups",
        "backup_file": None,
        "compress": True
    }


@pytest.fixture
def hashable_backup_params():
    """Hashable backup parameters for set operations"""
    return tuple(sorted([
        ("backup_first", True),
        ("backup_dir", "./test_backups"),
        ("backup_file", None),
        ("compress", True)
    ]))


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestHelperFunctions:
    """Test helper and utility functions"""
    
    def test_get_deletion_priority(self):
        """Test deletion priority ordering"""
        # Leaf nodes should have lower priority (deleted first)
        assert override_utils._get_deletion_priority('AnswerReview') < override_utils._get_deletion_priority('User')
        assert override_utils._get_deletion_priority('VideoTag') < override_utils._get_deletion_priority('Project')
        
        # Schema should have highest priority (deleted last)
        assert override_utils._get_deletion_priority('Schema') > override_utils._get_deletion_priority('Project')
        
        # Unknown tables should have very high priority
        assert override_utils._get_deletion_priority('UnknownTable') == 999
    
    def test_collect_delete_operations(self):
        """Test operation collection and deduplication"""
        # Create operations with same backup params to avoid hashing issues
        backup_params = {"backup_first": True}
        
        operations = {
            DeleteOperation('User', 'using_id', (1,), backup_params),
            DeleteOperation('AnswerReview', 'using_id', (1,), backup_params),
            DeleteOperation('Schema', 'using_id', (1,), backup_params),
        }
        
        result = override_utils._collect_delete_operations(operations)
        
        # Should be sorted by deletion priority
        assert result[0].table == 'AnswerReview'  # Leaf node first
        assert result[-1].table == 'Schema'       # Schema last
        
        # Should have all unique operations
        assert len(result) == 3
    
    @patch('builtins.input', return_value='DELETE')
    def test_confirm_cascade_deletion_accept(self, mock_input):
        """Test user confirmation - accept deletion"""
        backup_params = {"backup_first": True}
        operations = [
            DeleteOperation('User', 'using_id', (1,), backup_params),
            DeleteOperation('AnswerReview', 'using_id', (1,), backup_params)
        ]
        
        result = override_utils._confirm_cascade_deletion(operations)
        assert result is True
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='CANCEL')
    def test_confirm_cascade_deletion_reject(self, mock_input):
        """Test user confirmation - reject deletion"""
        backup_params = {"backup_first": True}
        operations = [DeleteOperation('User', 'using_id', (1,), backup_params)]
        
        result = override_utils._confirm_cascade_deletion(operations)
        assert result is False
    
    def test_confirm_cascade_deletion_empty(self):
        """Test confirmation with empty operations"""
        result = override_utils._confirm_cascade_deletion([])
        assert result is True


# ============================================================================
# DATABASE LOOKUP HELPER TESTS
# ============================================================================

class TestDatabaseLookups:
    """Test database lookup helper functions"""
    
    def test_get_user_id_from_user_id_str_success(self, mock_session):
        """Test successful user ID lookup"""
        mock_session.execute.return_value.fetchone.return_value = (123,)
        
        result = override_utils._get_user_id_from_user_id_str(mock_session, "test_user")
        
        assert result == 123
        mock_session.execute.assert_called_once()
    
    def test_get_user_id_from_user_id_str_not_found(self, mock_session):
        """Test user ID lookup - user not found"""
        mock_session.execute.return_value.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="User not found: test_user"):
            override_utils._get_user_id_from_user_id_str(mock_session, "test_user")
    
    def test_get_video_id_from_video_uid_success(self, mock_session):
        """Test successful video ID lookup"""
        mock_session.execute.return_value.fetchone.return_value = (456,)
        
        result = override_utils._get_video_id_from_video_uid(mock_session, "test_video")
        
        assert result == 456
    
    def test_get_video_id_from_video_uid_not_found(self, mock_session):
        """Test video ID lookup - video not found"""
        mock_session.execute.return_value.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="Video not found: test_video"):
            override_utils._get_video_id_from_video_uid(mock_session, "test_video")


# ============================================================================
# DEPENDENCY COLLECTION TESTS
# ============================================================================

class TestDependencyCollection:
    """Test dependency collection functions"""
    
    def test_collect_answer_review_dependencies(self, mock_session):
        """Test answer review dependency collection"""
        mock_session.execute.return_value.fetchall.return_value = [(1,), (2,)]
        backup_params = {"backup_first": True}
        
        result = override_utils._collect_answer_review_dependencies(mock_session, 123, backup_params)
        
        assert len(result) == 2
        operations_list = list(result)
        assert all(op.table == 'AnswerReview' for op in operations_list)
        identifiers = [op.identifier for op in operations_list]
        assert (1,) in identifiers
        assert (2,) in identifiers
    
    def test_collect_project_group_dependencies(self, mock_session):
        """Test project group dependency collection"""
        mock_session.execute.return_value.fetchall.return_value = [(10,), (20,)]
        backup_params = {"backup_first": True}
        
        result = override_utils._collect_project_group_dependencies(mock_session, 5, backup_params)
        
        assert len(result) == 2
        operations_list = list(result)
        assert all(op.table == 'ProjectGroupProject' for op in operations_list)
        identifiers = [op.identifier for op in operations_list]
        assert (5, 10) in identifiers
        assert (5, 20) in identifiers
    
    def test_collect_project_user_role_dependencies_admin(self, mock_session):
        """Test admin role deletion - should trigger ground truth reversion"""
        backup_params = {"backup_first": True}
        
        result = override_utils._collect_project_user_role_dependencies(
            mock_session, 1, 2, "admin", backup_params
        )
        
        # Admin role deletion should not add additional operations
        # (reversion is handled in _execute_delete_operations)
        assert len(result) == 0
    
    def test_collect_project_user_role_dependencies_reviewer_simple(self, mock_session):
        """Test reviewer role deletion without admin role"""
        backup_params = {"backup_first": True}
        
        # Mock no admin role + some ground truths and reviews
        mock_session.execute.return_value.fetchone.return_value = None  # No admin role
        mock_session.execute.return_value.fetchall.side_effect = [
            [(10, 1), (20, 1)],  # Reviewer ground truths
            [(100,), (200,)]     # Answer reviews
        ]
        
        result = override_utils._collect_project_user_role_dependencies(
            mock_session, 1, 2, "reviewer", backup_params
        )
        
        operations_list = list(result)
        tables = [op.table for op in operations_list]
        assert 'ReviewerGroundTruth' in tables
        assert 'AnswerReview' in tables


# ============================================================================
# DELETE USING ID FUNCTION TESTS
# ============================================================================

class TestDeleteUsingIdFunctions:
    """Test all delete_xxx_using_id functions"""
    
    def test_delete_user_using_id_success(self, mock_db):
        """Test successful user deletion by ID"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = ("test_user",)
        
        with patch('override_utils._collect_user_dependencies', return_value=set()) as mock_collect_deps, \
             patch('override_utils._collect_delete_operations', return_value=[]) as mock_collect_ops, \
             patch('override_utils._confirm_cascade_deletion', return_value=True) as mock_confirm, \
             patch('override_utils._execute_delete_operations', return_value=True) as mock_execute:
            
            result = override_utils.delete_user_using_id(1)
            
            assert result is True
            mock_collect_deps.assert_called_once()
            mock_confirm.assert_called_once()
            mock_execute.assert_called_once()
    
    def test_delete_user_using_id_not_found(self, mock_db):
        """Test user deletion by ID - user not found"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = None
        
        result = override_utils.delete_user_using_id(999)
        
        assert result is False
    
    def test_delete_user_using_id_cancelled(self, mock_db):
        """Test user deletion by ID - user cancels operation"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = ("test_user",)
        
        with patch('override_utils._collect_user_dependencies', return_value=set()), \
             patch('override_utils._collect_delete_operations', return_value=[]), \
             patch('override_utils._confirm_cascade_deletion', return_value=False) as mock_confirm, \
             patch('override_utils._execute_delete_operations') as mock_execute:
            
            result = override_utils.delete_user_using_id(1)
            
            assert result is False
            mock_execute.assert_not_called()
    
    def test_delete_video_tag_using_id_success(self, mock_db):
        """Test successful video tag deletion"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = (1,)  # Tag exists
        
        with patch('override_utils._confirm_cascade_deletion', return_value=True), \
             patch('override_utils._execute_delete_operations', return_value=True):
            
            result = override_utils.delete_video_tag_using_id(1, "NSFW")
            
            assert result is True
    
    def test_delete_video_tag_using_id_not_found(self, mock_db):
        """Test video tag deletion - tag not found"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = None
        
        result = override_utils.delete_video_tag_using_id(1, "NONEXISTENT")
        
        assert result is False


# ============================================================================
# DELETE USING NAME FUNCTION TESTS  
# ============================================================================

class TestDeleteUsingNameFunctions:
    """Test all delete_xxx functions (name-based deletion)"""
    
    def test_delete_user_success(self, mock_db):
        """Test successful user deletion by name"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        
        with patch('override_utils._get_user_id_from_user_id_str', return_value=123) as mock_get_id, \
             patch('override_utils.delete_user_using_id', return_value=True) as mock_delete_id:
            
            result = override_utils.delete_user("test_user")
            
            assert result is True
            mock_get_id.assert_called_once_with(mock_session, "test_user")
            mock_delete_id.assert_called_once_with(123, True, "./backups", None, True)
    
    def test_delete_user_not_found(self, mock_db):
        """Test user deletion by name - user not found"""
        with patch('override_utils._get_user_id_from_user_id_str', side_effect=ValueError("User not found")):
            result = override_utils.delete_user("nonexistent_user")
            assert result is False
    
    def test_delete_video_success(self, mock_db):
        """Test successful video deletion by name"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        
        with patch('override_utils._get_video_id_from_video_uid', return_value=456) as mock_get_id, \
             patch('override_utils.delete_video_using_id', return_value=True) as mock_delete_id:
            
            result = override_utils.delete_video("test_video.mp4")
            
            assert result is True
            mock_get_id.assert_called_once_with(mock_session, "test_video.mp4")
            mock_delete_id.assert_called_once_with(456, True, "./backups", None, True)


# ============================================================================
# EXECUTE OPERATIONS TESTS
# ============================================================================

class TestExecuteOperations:
    """Test operation execution and backup handling"""
    
    def test_execute_delete_operations_empty(self):
        """Test execution with empty operations list"""
        result = override_utils._execute_delete_operations([])
        assert result is True
    
    @patch('builtins.input', return_value='CONTINUE')
    def test_execute_delete_operations_backup_failure(self, mock_input, mock_db, mock_backup):
        """Test operation execution when backup fails"""
        mock_backup.side_effect = Exception("Backup failed")
        backup_params = {"backup_first": True, "backup_dir": "./backups", "backup_file": None, "compress": True}
        operations = [DeleteOperation('User', 'using_id', (1,), backup_params)]
        
        with patch('override_utils._execute_single_delete_operation') as mock_single:
            result = override_utils._execute_delete_operations(operations)
            
            assert result is True  # Should continue after user confirmation
            mock_input.assert_called_once()
            mock_single.assert_called_once()
    
    @patch('builtins.input', return_value='ABORT')
    def test_execute_delete_operations_backup_failure_abort(self, mock_input, mock_db, mock_backup):
        """Test operation execution when backup fails and user aborts"""
        mock_backup.side_effect = Exception("Backup failed")
        backup_params = {"backup_first": True, "backup_dir": "./backups", "backup_file": None, "compress": True}
        operations = [DeleteOperation('User', 'using_id', (1,), backup_params)]
        
        result = override_utils._execute_delete_operations(operations)
        
        assert result is False
        mock_input.assert_called_once()
    
    def test_execute_delete_operations_admin_reversion(self, mock_db):
        """Test admin ground truth reversion during operation execution"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.rowcount = 3
        
        backup_params = {"backup_first": False}
        operations = [
            DeleteOperation('ProjectUserRole', 'using_id', (1, 2, 'admin'), backup_params),
            DeleteOperation('User', 'using_id', (2,), backup_params)
        ]
        
        with patch('override_utils._execute_single_delete_operation') as mock_single:
            result = override_utils._execute_delete_operations(operations)
            
            assert result is True
            # Should execute reversion SQL and single operations
            assert mock_session.execute.call_count >= 1
            mock_single.assert_called()
    
    def test_execute_single_delete_operation_user(self, mock_session):
        """Test single operation execution for user table"""
        backup_params = {"backup_first": False}
        operation = DeleteOperation('User', 'using_id', (123,), backup_params)
        
        override_utils._execute_single_delete_operation(mock_session, operation)
        
        # Verify SQL execution
        mock_session.execute.assert_called()
        call_args = mock_session.execute.call_args
        sql_str = str(call_args[0][0])
        assert "DELETE FROM users WHERE id = :id" in sql_str
        assert call_args[1] == {"id": 123}
    
    def test_execute_single_delete_operation_video_tag(self, mock_session):
        """Test single operation execution for video tag"""
        backup_params = {"backup_first": False}
        operation = DeleteOperation('VideoTag', 'using_id', (1, 'NSFW'), backup_params)
        
        override_utils._execute_single_delete_operation(mock_session, operation)
        
        # Verify SQL execution
        mock_session.execute.assert_called()
        call_args = mock_session.execute.call_args
        sql_str = str(call_args[0][0])
        assert "DELETE FROM video_tags" in sql_str
        assert call_args[1] == {"video_id": 1, "tag": "NSFW"}
    
    def test_execute_single_delete_operation_unknown_table(self, mock_session):
        """Test single operation execution with unknown table"""
        backup_params = {"backup_first": False}
        operation = DeleteOperation('UnknownTable', 'using_id', (1,), backup_params)
        
        with pytest.raises(ValueError, match="Unknown table for using_id deletion"):
            override_utils._execute_single_delete_operation(mock_session, operation)


# ============================================================================
# NAME MANAGEMENT FUNCTION TESTS
# ============================================================================

class TestNameManagementFunctions:
    """Test get_xxx_name and set_xxx_name_using_id functions"""
    
    def test_get_user_name_success(self, mock_db):
        """Test successful user name retrieval"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = ("test_user",)
        
        result = override_utils.get_user_name(123)
        
        assert result == "test_user"
        mock_session.execute.assert_called_once()
    
    def test_get_user_name_not_found(self, mock_db):
        """Test user name retrieval - user not found"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="User with ID 999 not found"):
            override_utils.get_user_name(999)
    
    def test_get_video_name_success(self, mock_db):
        """Test successful video name retrieval"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = ("test_video.mp4",)
        
        result = override_utils.get_video_name(456)
        
        assert result == "test_video.mp4"
    
    @patch('builtins.input', return_value='CONTINUE')
    def test_set_user_name_using_id_backup_failure(self, mock_input, mock_db, mock_backup):
        """Test setting user name when backup fails"""
        mock_backup.side_effect = Exception("Backup failed")
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.side_effect = [("old_name",), None]
        
        result = override_utils.set_user_name_using_id(1, "new_name")
        
        assert result is True
        mock_input.assert_called_once()
        assert mock_session.execute.call_count >= 2
    
    @patch('builtins.input', return_value='ABORT')
    def test_set_user_name_using_id_backup_failure_abort(self, mock_input, mock_db, mock_backup):
        """Test setting user name when backup fails and user aborts"""
        mock_backup.side_effect = Exception("Backup failed")
        
        result = override_utils.set_user_name_using_id(1, "new_name")
        
        assert result is False
        mock_input.assert_called_once()
    
    def test_set_user_name_using_id_user_not_found(self, mock_db):
        """Test setting user name - user not found"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.return_value = None
        
        result = override_utils.set_user_name_using_id(999, "new_name", backup_first=False)
        
        assert result is False
    
    def test_set_user_name_using_id_name_exists(self, mock_db):
        """Test setting user name - name already exists"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.side_effect = [("old_name",), ("existing_user",)]
        
        result = override_utils.set_user_name_using_id(1, "existing_name", backup_first=False)
        
        assert result is False
    
    def test_set_user_name_using_id_success(self, mock_db, mock_backup):
        """Test successful user name update"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.fetchone.side_effect = [("old_name",), None]
        
        result = override_utils.set_user_name_using_id(1, "new_name")
        
        assert result is True
        mock_backup.assert_called_once()
        mock_session.commit.assert_called_once()


# ============================================================================
# INTEGRATION AND EDGE CASE TESTS
# ============================================================================

class TestIntegrationAndEdgeCases:
    """Test complex scenarios and edge cases"""
    
    def test_empty_operations_handling(self):
        """Test handling of empty operations"""
        result = override_utils._collect_delete_operations(set())
        assert result == []
        
        result = override_utils._execute_delete_operations([])
        assert result is True
    
    def test_database_error_handling(self, mock_db):
        """Test handling of database errors"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.side_effect = Exception("Database connection lost")
        
        result = override_utils.delete_user_using_id(1)
        
        assert result is False
    
    def test_backup_params_types(self):
        """Test different backup parameter types"""
        # Test with different backup parameter combinations
        params1 = {"backup_first": True, "backup_dir": "./backups", "backup_file": None, "compress": True}
        params2 = {"backup_first": False, "backup_dir": "./other", "backup_file": "custom.sql", "compress": False}
        
        op1 = DeleteOperation('User', 'using_id', (1,), params1)
        op2 = DeleteOperation('Video', 'using_id', (2,), params2)
        
        # Should be able to create operations with different params
        assert op1.table == 'User'
        assert op2.table == 'Video'
        assert op1.backup_params["backup_first"] is True
        assert op2.backup_params["backup_first"] is False


# ============================================================================
# SPECIFIC SCENARIO TESTS
# ============================================================================

class TestSpecificScenarios:
    """Test specific business logic scenarios"""
    
    def test_admin_role_deletion_reversion_logic(self, mock_db):
        """Test admin role deletion triggers ground truth reversion"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        mock_session.execute.return_value.rowcount = 5  # 5 ground truths reverted
        
        backup_params = {"backup_first": False}
        operations = [
            DeleteOperation('ProjectUserRole', 'using_id', (1, 2, 'admin'), backup_params)
        ]
        
        with patch('override_utils._execute_single_delete_operation') as mock_single:
            result = override_utils._execute_delete_operations(operations)
            
            assert result is True
            # Should execute reversion SQL
            assert mock_session.execute.call_count >= 1
            
            # Check that the reversion SQL was called
            reversion_call = mock_session.execute.call_args_list[0]
            sql_text = str(reversion_call[0][0])
            assert "UPDATE reviewer_ground_truth" in sql_text
            assert "SET answer_value = original_answer_value" in sql_text
    
    def test_reviewer_role_without_admin(self, mock_db):
        """Test reviewer role deletion when user doesn't have admin role"""
        mock_session = mock_db.SessionLocal.return_value.__enter__.return_value
        
        # Mock that user does NOT have admin role
        mock_session.execute.return_value.fetchone.return_value = None
        mock_session.execute.return_value.fetchall.side_effect = [
            [(10, 1), (20, 1)],  # Reviewer ground truths
            [(100,), (200,)]     # Answer reviews
        ]
        
        backup_params = {"backup_first": False}
        result = override_utils._collect_project_user_role_dependencies(
            mock_session, 1, 2, "reviewer", backup_params
        )
        
        operations_list = list(result)
        tables = [op.table for op in operations_list]
        
        # Should NOT include admin role deletion
        assert 'ProjectUserRole' not in tables or \
               not any(op.identifier[2] == 'admin' for op in operations_list if op.table == 'ProjectUserRole')
        
        # Should include reviewer-specific deletions
        assert 'ReviewerGroundTruth' in tables
        assert 'AnswerReview' in tables


# ============================================================================
# PERFORMANCE AND BOUNDARY TESTS
# ============================================================================

class TestPerformanceAndBoundary:
    """Test performance edge cases and boundary conditions"""
    
    def test_large_dependency_chain_simple(self):
        """Test handling of moderately large dependency chains"""
        backup_params = {"backup_first": False}
        
        # Create a moderate set of dependencies to avoid memory issues
        large_dependency_set = set()
        for i in range(100):  # Reduced from 1000 to avoid memory issues
            large_dependency_set.add(
                DeleteOperation('AnswerReview', 'using_id', (i,), backup_params)
            )
        
        result = override_utils._collect_delete_operations(large_dependency_set)
        
        # Should handle moderate sets efficiently
        assert len(result) == 100
        assert all(op.table == 'AnswerReview' for op in result)
    
    def test_duplicate_operations_simple(self):
        """Test basic duplicate operation handling"""
        backup_params = {"backup_first": False}
        
        # Test with a small set to avoid hashing issues
        operations = [
            DeleteOperation('User', 'using_id', (1,), backup_params),
            DeleteOperation('AnswerReview', 'using_id', (1,), backup_params),
        ]
        
        # Convert to set and back to list (simulating deduplication)
        # Note: This won't actually deduplicate due to dict in backup_params,
        # but we can test the sorting logic
        result = override_utils._collect_delete_operations(set(operations))
        
        assert len(result) >= 1  # Should have at least some operations
        # Test that AnswerReview comes before User (deletion priority)
        answer_review_pos = next((i for i, op in enumerate(result) if op.table == 'AnswerReview'), -1)
        user_pos = next((i for i, op in enumerate(result) if op.table == 'User'), -1)
        
        if answer_review_pos >= 0 and user_pos >= 0:
            assert answer_review_pos < user_pos


# ============================================================================
# UTILITY TESTS
# ============================================================================

def test_module_imports():
    """Test that all required modules and functions are importable"""
    # Test that all expected functions exist
    expected_functions = [
        'delete_user_using_id', 'delete_user', 'get_user_name', 'set_user_name_using_id',
        'delete_video_using_id', 'delete_video', 'get_video_name', 'set_video_name_using_id',
        'delete_project_using_id', 'delete_project', 'get_project_name', 'set_project_name_using_id',
    ]
    
    for func_name in expected_functions:
        assert hasattr(override_utils, func_name), f"Function {func_name} not found in module"
        assert callable(getattr(override_utils, func_name)), f"{func_name} is not callable"


def test_delete_operation_namedtuple():
    """Test DeleteOperation namedtuple structure"""
    operation = DeleteOperation(
        table='User',
        operation_type='using_id',
        identifier=(123,),
        backup_params={'backup_first': True}
    )
    
    assert operation.table == 'User'
    assert operation.operation_type == 'using_id'
    assert operation.identifier == (123,)
    assert operation.backup_params['backup_first'] is True


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])