# Label Pizza - Override Workflows

This guide explains how to rename or delete data in the database.

## ⚠️ Danger Zone: Rename or Delete Data (Use with Caution)

By default, **Label Pizza does not allow deleting** users, videos, questions, question groups, schemas, or projects through standard syncing. These records are meant to be **archived**, not removed. This prevents accidental data loss and avoids breaking dependencies—for example, deleting a schema that is still used by a project.

Similarly, **you cannot rename core identifiers** like a video UID, question text, schema name, or project name through the web interface. These are used as stable, unique identifiers when syncing and updating the database.

That said, if you **really** need to delete or rename something, you can use the force override functions below. These bypass all safeguards and give you full control.

### ⚠️ What These Functions Do

The override utilities provide two dangerous capabilities:

1. **🗑️ Cascade Delete Operations** - **PERMANENTLY DELETE** any data with automatic handling of all dependent records. This is irreversible and will destroy data across multiple tables.

2. **📝 Name Management** - **FORCE RENAME** the unique identifiers that sync operations depend on. This can break existing sync workflows and cause confusion.

### ⚠️ Why This Is Dangerous

**Cascade Deletion Risks:**
- Deletes data **permanently** from the database (not archived)
- Automatically deletes **all dependent data** without additional confirmation
- Can delete hundreds of records in a single operation
- May break references in external systems or scripts
- **Cannot be undone** without restoring from backup

**Name Change Risks:**
- Breaks sync workflows that depend on stable identifiers
- Can cause duplicate entries if sync files aren't updated
- May confuse team members who know the old names
- External integrations may fail if they reference old names

### ⚠️ Before You Proceed

**✅ Required Checklist:**
- [ ] You have or will create a **full database backup**
- [ ] You understand this will **permanently change or delete data**
- [ ] You have informed your team about identifier changes

**🚫 Do NOT use these functions if:**
- You just want to hide old data (use archiving instead)
- You're unsure about dependencies
- You do not intend to create a backup first

---

## API Reference

Before diving into examples, here are the complete override functions available:

### Name Management Operations

| Table | Get Name Function | Set Name Function (Using ID) | Set Name Function (Using Name) |
|-------|-------------------|-------------------------------|--------------------------------|
| **User** | `get_user_name(id) → user_id_str` | `set_user_name_using_id(id, new_user_id_str, **backup_params)` | `set_user_name(old_user_id_str, new_user_id_str, **backup_params)` |
| **Video** | `get_video_name(id) → video_uid` | `set_video_name_using_id(id, new_video_uid, **backup_params)` | `set_video_name(old_video_uid, new_video_uid, **backup_params)` |
| **Question Group** | `get_question_group_name(id) → title` | `set_question_group_name_using_id(id, new_title, **backup_params)` | `set_question_group_name(old_title, new_title, **backup_params)` |
| **Question** | `get_question_name(id) → text` | `set_question_name_using_id(id, new_text, **backup_params)` | `set_question_name(old_text, new_text, **backup_params)` |
| **Schema** | `get_schema_name(id) → name` | `set_schema_name_using_id(id, new_name, **backup_params)` | `set_schema_name(old_name, new_name, **backup_params)` |
| **Project** | `get_project_name(id) → name` | `set_project_name_using_id(id, new_name, **backup_params)` | `set_project_name(old_name, new_name, **backup_params)` |
| **Project Group** | `get_project_group_name(id) → name` | `set_project_group_name_using_id(id, new_name, **backup_params)` | `set_project_group_name(old_name, new_name, **backup_params)` |

**Notes:**
- **Get operations** are read-only lookups that return the current name for an ID
- **Set operations using ID** update the name by database ID and include backup parameters
- **Set operations using name** update the name by current name (more user-friendly)
- The new name must be unique, otherwise an error will be raised

### Delete Operations Available

| What to Delete | Using ID Function | Using Name Function |
|----------------|-------------------|-------------------|
| **User** | `delete_user_using_id(id, **backup_params)` | `delete_user(user_id_str, **backup_params)` |
| **Video** | `delete_video_using_id(id, **backup_params)` | `delete_video(video_uid, **backup_params)` |
| **Question Group** | `delete_question_group_using_id(id, **backup_params)` | `delete_question_group(title, **backup_params)` |
| **Question** | `delete_question_using_id(id, **backup_params)` | `delete_question(text, **backup_params)` |
| **Schema** | `delete_schema_using_id(id, **backup_params)` | `delete_schema(name, **backup_params)` |
| **Project** | `delete_project_using_id(id, **backup_params)` | `delete_project(name, **backup_params)` |
| **Project Group** | `delete_project_group_using_id(id, **backup_params)` | `delete_project_group(name, **backup_params)` |

**Relationship Deletions**:

| Relationship | Using ID Function | Using Name Function |
|--------------|-------------------|-------------------|
| **Question ↔ Group** | `delete_question_group_question_using_id(group_id, question_id, **backup_params)` | `delete_question_group_question(group_title, question_text, **backup_params)` |
| **Schema ↔ Group** | `delete_schema_question_group_using_id(schema_id, group_id, **backup_params)` | `delete_schema_question_group(schema_name, group_title, **backup_params)` |
| **Project ↔ Video** | `delete_project_video_using_id(project_id, video_id, **backup_params)` | `delete_project_video(project_name, video_uid, **backup_params)` |
| **Project ↔ User** | `delete_project_user_role_using_id(project_id, user_id, role, **backup_params)` | `delete_project_user_role(project_name, user_id_str, role, **backup_params)` |
| **Project Group ↔ Project** | `delete_project_group_project_using_id(group_id, project_id, **backup_params)` | `delete_project_group_project(group_name, project_name, **backup_params)` |

**Advanced Deletions**:

| Advanced Cases | Using ID Function | Using Name Function |
|----------------|-------------------|-------------------|
| **Custom Displays** | `delete_project_video_question_display_using_id(project_id, video_id, question_id, **backup_params)` | `delete_project_video_question_display(project_name, video_uid, question_text, **backup_params)` |
| **Annotations** | `delete_annotator_answer_using_id(id, **backup_params)` | `delete_annotator_answer(video_uid, question_text, user_id_str, project_name, **backup_params)` |
| **Ground Truth** | `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id, **backup_params)` | `delete_reviewer_ground_truth(video_uid, question_text, project_name, **backup_params)` |
| **Answer Reviews** | `delete_answer_review_using_id(id, **backup_params)` | `delete_answer_review(answer_id, **backup_params)` |

**Backup Parameters for All Functions**:
- `backup_first=True` - Create automatic backup before operation
- `backup_dir="./backups"` - Default backup directory  
- `backup_file=None` - Auto-generated timestamp filename
- `compress=True` - Enable gzip compression

---

## Getting Started with Override Operations

**IMPORTANT: Always test these functions on a test database first.**

### Step-by-step guide

> **Before you start:** make sure the **`DBURL`** key in your [.env](.env) file contains the correct PostgreSQL connection string.

1. **Initialize a test database and seed the first admin account**

   ```bash
   python label_pizza/manage_db.py \
       --mode init \
       --database-url-name DBURL \
       --email admin1@example.com \
       --password admin111 \
       --user-id "Admin 1"
   ```

2. **Set up test data for practicing override operations**

3. **Start with name management (safer than deletion)**

4. **Practice deletion operations (most dangerous)**

---

## Hands-On Examples

Let's practice override operations with realistic test data. Each example is **self-contained and automatically restores** the database to its original state after demonstrating the operation.

### 0. Prerequisite: `init_database`

**CRITICAL:** You must run these lines before using any override functions, otherwise you will get an error.

```python
from label_pizza.db import init_database
init_database("DBURL")  # or change DBURL to another key in .env
```

### 1. Helper Functions

First, let's set up helper functions for test data and backup restoration. **Run this once before trying any examples:**

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import (
    sync_videos, sync_users, sync_question_groups, sync_schemas, 
    sync_projects, sync_users_to_projects, sync_project_groups,
    sync_annotations, sync_ground_truths
)

def restore_from_backup(backup_file: str, backup_dir: str = "./backups") -> bool:
    """Restore database from backup file using the proper backup format parser"""
    try:
        import os
        from pathlib import Path
        
        # Handle input file path
        if not os.path.sep in backup_file and not os.path.isabs(backup_file):
            input_file = os.path.join(backup_dir, backup_file)
        else:
            input_file = backup_file
            
        if not os.path.exists(input_file):
            print(f"❌ Backup file not found: {input_file}")
            return False
        
        # Get database URL using the same env var as init_database
        import label_pizza.db
        from dotenv import load_dotenv
        
        env_var_name = getattr(label_pizza.db, 'current_db_env_var', 'DBURL')
        load_dotenv()
        db_url = os.getenv(env_var_name)
        if not db_url:
            print(f"❌ Environment variable '{env_var_name}' not found")
            return False
        
        print(f"📥 Restoring database from: {input_file}")
        
        # Use the proper DatabaseBackupRestore class from backup_restore.py
        from label_pizza.backup_restore import DatabaseBackupRestore
        
        handler = DatabaseBackupRestore(db_url)
        success = handler.restore_backup(input_file=input_file, force=True)
        
        if success:
            print("   ✅ Database restored successfully")
            return True
        else:
            print("   ❌ Database restore failed")
            return False
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def setup_test_data():
    """Create test dataset combining example/ and example_custom_question/ folders"""
    
    print("🏗️ Creating test data...")
    
    # 1. Create videos 
    videos_data = [
        {
            "video_uid": "human.mp4",
            "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/human.mp4",
            "is_active": True,
            "metadata": {}
        },
        {
            "video_uid": "pizza.mp4",
            "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/pizza.mp4",
            "is_active": True,
            "metadata": {}
        },
        {
            "video_uid": "human2.mp4",
            "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/human2.mp4",
            "is_active": True,
            "metadata": {}
        }
    ]
    sync_videos(videos_data=videos_data)
    print("✅ Created 3 videos")
    
    # 2. Create users 
    users_data = [
        {
            "user_id": "User 1",
            "email": "user1@example.com",
            "password": "user111",
            "user_type": "human",
            "is_active": True
        },
        {
            "user_id": "Robot 1",
            "password": "",
            "email": None,
            "user_type": "model",
            "is_active": True
        },
        {
            "user_id": "Admin 2",
            "email": "admin2@example.com", 
            "password": "admin222",
            "user_type": "human",
            "is_active": True
        }
    ]
    sync_users(users_data=users_data)
    print("✅ Created 3 users")
    
    # 3. Create question groups 
    question_groups_data = [
        {
            "title": "Human",
            "display_title": "Human",
            "description": "Detect and describe all humans in the video.",
            "is_reusable": False,
            "is_auto_submit": False,
            "verification_function": "check_human_description",
            "questions": [
                {
                    "qtype": "single",
                    "text": "Number of people?",
                    "display_text": "Number of people?",
                    "options": ["0", "1", "2", "3 or more"],
                    "display_values": ["0", "1", "2", "3 or more"],
                    "option_weights": [1.0, 1.0, 1.0, 1.0],
                    "default_option": "0"
                },
                {
                    "qtype": "description",
                    "text": "If there are people, describe them.",
                    "display_text": "If there are people, describe them."
                }
            ]
        },
        {
            "title": "Pizza",
            "display_title": "Pizza",
            "description": "Detect and describe all pizzas in the video.",
            "is_reusable": False,
            "is_auto_submit": False,
            "verification_function": "check_pizza_description",
            "questions": [
                {
                    "qtype": "single",
                    "text": "Number of pizzas?",
                    "display_text": "Number of pizzas?",
                    "options": ["0", "1", "2", "3 or more"],
                    "display_values": ["0", "1", "2", "3 or more"],
                    "option_weights": [1.0, 1.0, 1.0, 1.0],
                    "default_option": "0"
                },
                {
                    "qtype": "description",
                    "text": "If there are pizzas, describe them.",
                    "display_text": "If there are pizzas, describe them."
                }
            ]
        },
        {
            "title": "NSFW",
            "display_title": "NSFW",
            "description": "Check whether the video is not safe for work.",
            "is_reusable": True,
            "is_auto_submit": True,
            "verification_function": "",
            "questions": [
                {
                    "qtype": "single",
                    "text": "Is the video not safe for work?",
                    "display_text": "Is the video not safe for work?",
                    "options": ["No", "Yes"],
                    "display_values": ["No", "Yes"],
                    "option_weights": [1.0, 99.0],
                    "default_option": "No"
                }
            ]
        },
        {
            "title": "Pizza Custom",
            "display_title": "Pizza Custom",
            "description": "Detect and describe all pizzas in the video with customized questions per video.",
            "is_reusable": False,
            "is_auto_submit": False,
            "verification_function": "",
            "questions": [
                {
                    "qtype": "single",
                    "text": "Pick one option",
                    "display_text": "Pick one option",
                    "options": ["Option A", "Option B"],
                    "display_values": ["Option A", "Option B"],
                    "option_weights": [1.0, 1.0],
                    "default_option": "Option A"
                },
                {
                    "qtype": "description",
                    "text": "Describe one aspect of the video",
                    "display_text": "Describe one aspect of the video"
                }
            ]
        }
    ]
    sync_question_groups(question_groups_data=question_groups_data)
    print("✅ Created 4 question groups")
    
    # 4. Create schemas 
    schemas_data = [
        {
            "schema_name": "Questions about Humans",
            "instructions_url": "https://en.wikipedia.org/wiki/Human",
            "question_group_names": ["Human", "NSFW"],
            "has_custom_display": False,
            "is_active": True
        },
        {
            "schema_name": "Questions about Pizzas",
            "instructions_url": "https://en.wikipedia.org/wiki/Pizza",
            "question_group_names": ["Pizza", "NSFW"],
            "has_custom_display": False,
            "is_active": True
        },
        {
            "schema_name": "Questions about Pizzas Custom",
            "instructions_url": "",
            "question_group_names": ["Pizza Custom"],
            "has_custom_display": True,
            "is_active": True
        }
    ]
    sync_schemas(schemas_data=schemas_data)
    print("✅ Created 3 schemas")
    
    # 5. Create projects 
    projects_data = [
        {
            "project_name": "Human Test 0",
            "schema_name": "Questions about Humans",
            "description": "Test project for humans",
            "is_active": True,
            "videos": ["human.mp4", "pizza.mp4"]
        },
        {
            "project_name": "Pizza Test 0",
            "schema_name": "Questions about Pizzas",
            "description": "Test project for pizzas",
            "is_active": True,
            "videos": ["human.mp4", "pizza.mp4"]
        },
        {
            "project_name": "Pizza Test 0 Custom",
            "schema_name": "Questions about Pizzas Custom",
            "description": "Test project with customized questions per video",
            "is_active": True,
            "videos": [
                {
                    "video_uid": "human.mp4",
                    "questions": [
                        {
                            "question_text": "Pick one option",
                            "custom_question": "Is there a pizza in the video?",
                            "custom_option": {
                                "Option A": "No",
                                "Option B": "Yes, there is."
                            }
                        },
                        {
                            "question_text": "Describe one aspect of the video",
                            "custom_question": "If no pizza is shown, describe what is present instead."
                        }
                    ]
                },
                {
                    "video_uid": "pizza.mp4",
                    "questions": [
                        {
                            "question_text": "Pick one option",
                            "custom_question": "What type of pizza is shown?",
                            "custom_option": {
                                "Option A": "Pepperoni",
                                "Option B": "Veggie"
                            }
                        },
                        {
                            "question_text": "Describe one aspect of the video",
                            "custom_question": "Describe the type of pizza shown in the video."
                        }
                    ]
                }
            ]
        }
    ]
    sync_projects(projects_data=projects_data)
    print("✅ Created 3 projects")
    
    # 6. Create user-project assignments (admins are auto-assigned to all projects)
    assignments_data = [
        {
            "user_name": "User 1",
            "project_name": "Pizza Test 0",
            "role": "annotator",
            "user_weight": 1.0,
            "is_active": True
        },
        {
            "user_name": "User 1",
            "project_name": "Human Test 0",
            "role": "annotator",
            "user_weight": 1.0,
            "is_active": True
        },
        {
            "user_name": "Robot 1",
            "project_name": "Human Test 0",
            "role": "model",
            "user_weight": 1.0,
            "is_active": True
        },
        {
            "user_name": "User 1",
            "project_name": "Pizza Test 0 Custom",
            "role": "annotator",
            "user_weight": 1.0,
            "is_active": True
        },
        {
            "user_name": "Admin 2",
            "project_name": "Human Test 0",
            "role": "reviewer",
            "user_weight": 2.0,
            "is_active": True
        }
    ]
    sync_users_to_projects(assignments_data=assignments_data)
    print("✅ Created 5 user-project assignments (admins auto-assigned)")
    
    # 7. Create project groups 
    project_groups_data = [
        {
            "project_group_name": "Human Test Projects",
            "description": "This is a project group for human test",
            "projects": ["Human Test 0"]
        },
        {
            "project_group_name": "Pizza Test Projects",
            "description": "This is a project group for pizza test",
            "projects": ["Pizza Test 0"]
        },
        {
            "project_group_name": "Pizza Test Custom Projects",
            "description": "This is a project group for pizza test with customized questions",
            "projects": ["Pizza Test 0 Custom"]
        }
    ]
    sync_project_groups(project_groups_data=project_groups_data)
    print("✅ Created 3 project groups")
    
    # 8. Create annotations 
    annotations_data = [
        {
            "question_group_title": "Human",
            "project_name": "Human Test 0",
            "user_name": "User 1",
            "video_uid": "human.mp4",
            "answers": {
                "Number of people?": "1",
                "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
            },
            "is_ground_truth": False
        },
        {
            "question_group_title": "Human",
            "project_name": "Human Test 0",
            "user_name": "Robot 1",
            "video_uid": "human.mp4",
            "answers": {
                "Number of people?": "2",
                "If there are people, describe them.": "The two people are a man and a woman."
            },
            "confidence_scores": {
                "Number of people?": 0.3,
                "If there are people, describe them.": 0.2
            },
            "is_ground_truth": False
        },
        {
            "question_group_title": "Pizza",
            "project_name": "Pizza Test 0",
            "user_name": "User 1",
            "video_uid": "pizza.mp4",
            "answers": {
                "Number of pizzas?": "1",
                "If there are pizzas, describe them.": "The huge pizza looks delicious."
            },
            "is_ground_truth": False
        },
        {
            "question_group_title": "NSFW",
            "project_name": "Human Test 0",
            "user_name": "User 1",
            "video_uid": "human.mp4",
            "answers": {
                "Is the video not safe for work?": "No"
            },
            "is_ground_truth": False
        },
        {
            "question_group_title": "Pizza Custom",
            "project_name": "Pizza Test 0 Custom",
            "user_name": "User 1",
            "video_uid": "human.mp4",
            "answers": {
                "Pick one option": "Option A",
                "Describe one aspect of the video": ""
            },
            "is_ground_truth": False
        }
    ]
    sync_annotations(annotations_data=annotations_data)
    print("✅ Created 5 annotations")
    
    # 9. Create ground truths 
    ground_truths_data = [
        {
            "question_group_title": "Human",
            "project_name": "Human Test 0",
            "user_name": "Admin 2",
            "video_uid": "human.mp4",
            "answers": {
                "Number of people?": "1",
                "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
            },
            "is_ground_truth": True
        },
        {
            "question_group_title": "Pizza",
            "project_name": "Pizza Test 0",
            "user_name": "Admin 1",
            "video_uid": "pizza.mp4",
            "answers": {
                "Number of pizzas?": "1",
                "If there are pizzas, describe them.": "The huge pizza looks delicious."
            },
            "is_ground_truth": True
        }
    ]
    sync_ground_truths(ground_truths_data=ground_truths_data)
    print("✅ Created 2 ground truth entries")
    
    # 10. Create admin-modified ground truth and answer reviews
    import label_pizza.db
    from sqlalchemy import text
    
    with label_pizza.db.SessionLocal() as session:
        # Simulate admin override of ground truth
        admin2_id_result = session.execute(text("SELECT id FROM users WHERE user_id_str = 'Admin 2'"))
        admin2_id = admin2_id_result.fetchone()[0]
        
        # Update the ground truth to show it was modified by admin
        session.execute(text("""
            UPDATE reviewer_ground_truth 
            SET original_answer_value = answer_value,
                answer_value = '{"Number of people?": "2", "If there are people, describe them.": "Actually there are two people visible in the background"}',
                modified_by_admin_id = :admin_id,
                modified_by_admin_at = NOW(),
                modified_at = NOW()
            WHERE project_id = (SELECT id FROM projects WHERE name = 'Human Test 0')
            AND video_id = (SELECT id FROM videos WHERE video_uid = 'human.mp4')
            AND question_id = (SELECT id FROM questions WHERE text = 'Number of people?')
        """), {"admin_id": admin2_id})
        
        # Create some answer reviews
        result = session.execute(text("""
            SELECT aa.id, 
                   (SELECT id FROM users WHERE user_id_str = 'Admin 1' LIMIT 1) as reviewer_id,
                   CASE WHEN u.user_type = 'model' 
                        THEN 'approved' 
                        ELSE 'pending'
                   END as status
            FROM annotator_answers aa
            JOIN users u ON aa.user_id = u.id
            WHERE u.is_archived = false
            LIMIT 3
        """))
        
        for row in result.fetchall():
            answer_id, reviewer_id, status = row
            comment = f"Review of annotation - status: {status}"
            session.execute(text("""
                INSERT INTO answer_reviews (answer_id, reviewer_id, status, comment)
                VALUES (:answer_id, :reviewer_id, :status, :comment)
            """), {"answer_id": answer_id, "reviewer_id": reviewer_id, "status": status, "comment": comment})
        
        session.commit()
    print("✅ Created admin-modified ground truth and answer reviews")
    
    print("\n🎉 Test data setup complete!")
    print("📊 Final Summary:")
    print("   • 3 videos with simplified metadata")
    print("   • 3 users (1 annotator, 1 model, 1 admin)") 
    print("   • 4 question groups (including custom display)")
    print("   • 3 schemas (2 standard, 1 with customized questions per video)")
    print("   • 3 projects with various configurations")
    print("   • 5 user-project assignments (admins auto-assigned to all projects)") 
    print("   • 3 project groups organizing the projects")
    print("   • 5 annotations across multiple projects and users")
    print("   • 2 ground truth entries, one modified by admin")
    print("   • Answer reviews with varying statuses")
    print()
    print("⚠️  This dataset demonstrates ALL override scenarios!")
    print("🧪 You can now safely test any override operation.")

# Run the setup once
setup_test_data()
```

### 2. Practice Name Management (Start Here - Safer Operations)

These examples demonstrate all name management functions. Each example is self-contained and restores the database afterward.

#### Example 1: Rename Users

```python
def example_user_name_management():
    print("\n📝 EXAMPLE 1: User Name Management")
    print("=" * 40)
    
    from label_pizza.override_utils import set_user_name
    
    # Rename user directly by name (much simpler!)
    backup_file = "example1_rename_user.sql.gz"
    success = set_user_name(
        old_user_id_str="User 1",
        new_user_id_str="User 1 (Promoted)",
        backup_first=True,
        backup_file=backup_file
    )
    
    if success:
        print("✅ User renamed successfully")
    else:
        print("❌ User rename failed")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_user_name_management()
```

#### Example 2: Rename Projects

```python
def example_project_name_management():
    print("\n📝 EXAMPLE 2: Project Name Management")
    print("=" * 42)
    
    from label_pizza.override_utils import set_project_name
    
    # Rename project directly by name
    backup_file = "example2_rename_project.sql.gz"
    success = set_project_name(
        old_name="Pizza Test 0",
        new_name="Advanced Pizza Test 0",
        backup_first=True,
        backup_file=backup_file
    )
    
    if success:
        print("✅ Project renamed successfully")
    else:
        print("❌ Project rename failed")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_project_name_management()
```

#### Example 3: Rename Multiple Items (All Name Functions)

```python
def example_all_name_management():
    print("\n📝 EXAMPLE 3: All Name Management Functions")
    print("=" * 45)
    
    from label_pizza.override_utils import (
        set_user_name, set_video_name, set_question_group_name,
        set_question_name, set_schema_name, set_project_name, set_project_group_name
    )
    
    backup_file = "example3_all_name_functions.sql.gz"
    
    # 1. Rename user
    print("1. Renaming user...")
    success1 = set_user_name("User 1", "User 1 (Senior)", backup_first=True, backup_file=backup_file)
    print(f"   User rename: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    
    # 2. Rename video
    print("2. Renaming video...")
    success2 = set_video_name("human2.mp4", "senior_human.mp4", backup_first=False)
    print(f"   Video rename: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    # 3. Rename question group
    print("3. Renaming question group...")
    success3 = set_question_group_name("Pizza Custom", "Advanced Pizza Analysis", backup_first=False)
    print(f"   Question group rename: {'✅ SUCCESS' if success3 else '❌ FAILED'}")
    
    # 4. Rename individual question
    print("4. Renaming individual question...")
    success4 = set_question_name("Pick one option", "Select best option", backup_first=False)
    print(f"   Question rename: {'✅ SUCCESS' if success4 else '❌ FAILED'}")
    
    # 5. Rename schema
    print("5. Renaming schema...")
    success5 = set_schema_name("Questions about Humans", "Enhanced Human Detection Schema", backup_first=False)
    print(f"   Schema rename: {'✅ SUCCESS' if success5 else '❌ FAILED'}")
    
    # 6. Rename project
    print("6. Renaming project...")
    success6 = set_project_name("Pizza Test 0", "Advanced Pizza Analysis Project", backup_first=False)
    print(f"   Project rename: {'✅ SUCCESS' if success6 else '❌ FAILED'}")
    
    # 7. Rename project group
    print("7. Renaming project group...")
    success7 = set_project_group_name("Human Test Projects", "Enhanced Human Test Suite", backup_first=False)
    print(f"   Project group rename: {'✅ SUCCESS' if success7 else '❌ FAILED'}")
    
    # Summary
    successes = sum([success1, success2, success3, success4, success5, success6, success7])
    print(f"\n📋 Summary: {successes}/7 name management functions completed successfully")
    print("   ✅ All 7 name-based set functions demonstrated:")
    print("      • set_user_name() • set_video_name() • set_question_group_name()")
    print("      • set_question_name() • set_schema_name() • set_project_name()")
    print("      • set_project_group_name()")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_all_name_management()
```

### 3. What You'll See During Delete Operations

**Understanding the Delete Process:** Each delete function follows a consistent pattern:

#### Phase 1: Dependency Analysis
```
🔍 Collecting dependencies for User ID 5 ('User 1')
```
The system analyzes what other data depends on what you're deleting.

#### Phase 2: Operation Planning  
```
🚨 CASCADE DELETE OPERATIONS
==================================================
The following delete operations will be executed in sequence:

 1. Delete 'Admin 1' review of 'User 1' annotation
 2. Delete 'Admin 1' review of 'Robot 1' annotation  
 3. Delete 'User 1' annotation for 'Number of people?...' in 'Human Test 0'
 4. Delete 'User 1' annotation for 'If there are people, describe...' in 'Human Test 0'
 5. Remove annotator 'User 1' from project 'Human Test 0'
 6. Delete user 'User 1'

Total operations: 6
⚠️  This will permanently delete data from the database!
```

#### Phase 3: Confirmation
```
Confirm cascade deletion? Type 'DELETE' to proceed: 
```
You must type exactly **'DELETE'** (in caps) to proceed.

#### Phase 4: Backup Creation (if requested)
```
💾 Backup created successfully: ./backups/example_delete_user.sql.gz
```

#### Phase 5: Execution
```
✅ Delete 'Admin 1' review of 'User 1' annotation
✅ Delete 'Admin 1' review of 'Robot 1' annotation
✅ Delete 'User 1' annotation for 'Number of people?...' in 'Human Test 0'
✅ Delete 'User 1' annotation for 'If there are people, describe them...' in 'Human Test 0'
✅ Remove annotator 'User 1' from project 'Human Test 0'
✅ Delete user 'User 1'

🎉 Successfully completed 6 delete operations
💾 Backup saved to: ./backups/example_delete_user.sql.gz
```

**Key Points:**
- Operation planning shows **meaningful names** instead of database IDs
- Operations are executed in dependency order (safest first)
- Each operation shows **meaningful success messages** with names
- Backups are created before any deletions
- The entire operation is wrapped in a database transaction

### 4. Understanding Cascade Deletion Hierarchy

Before practicing deletions, understand the hierarchy:

**Cascade Deletion Levels:**
- **🍃 LEAF NODES** (safest): ProjectGroupProject, ProjectVideoQuestionDisplay, ReviewerGroundTruth, AnswerReview
- **📊 LEVEL 1**: AnnotatorAnswer → AnswerReview; ProjectGroup → ProjectGroupProject  
- **📈 LEVEL 2**: ProjectVideo, ProjectUserRole, QuestionGroupQuestion, SchemaQuestionGroup
- **🏗️ LEVEL 3**: Video, Question, QuestionGroup, Project, User
- **☢️ LEVEL 4**: Schema (most dangerous)

To better understand the cascade deletion hierarchy, refer to the [dependency](dependency.md) page for relationships between the tables.

### 5. Practice Deletion Functions - Leaf Nodes (Safest)

**What You'll See:** Leaf node deletions show minimal cascade operations since they have few or no dependencies. You'll see a confirmation prompt, then 1-2 delete operations, followed by success messages.

#### Example 4: Delete Project Group Project Relationships

```python
def example_delete_project_group_project():
    print("\n🗑️ EXAMPLE 4: Delete Project Group Project Relationship")
    print("=" * 55)
    
    from label_pizza.override_utils import delete_project_group_project
    
    # Delete relationship between group and project
    backup_file = "example4_delete_group_project.sql.gz"
    success = delete_project_group_project(
        project_group_name="Human Test Projects",
        project_name="Human Test 0",
        backup_first=True,
        backup_file=backup_file
    )
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Remove project 'Human Test 0' from group 'Human Test Projects'")
    print("📋 Expected child rows deleted:")
    print("    • ProjectGroupProject: 'Human Test Projects' ↔ 'Human Test 0' relationship")
    print("    • Both 'Human Test Projects' group and 'Human Test 0' project remain intact")
    print("📋 Verification: Check that 'Human Test 0' no longer appears in 'Human Test Projects' group")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_project_group_project()
```

#### Example 5: Delete Custom Display and Ground Truth

```python
def example_delete_display_and_ground_truth():
    print("\n🗑️ EXAMPLE 5: Delete Custom Display and Ground Truth")
    print("=" * 52)
    
    from label_pizza.override_utils import (
        delete_project_video_question_display,
        delete_reviewer_ground_truth
    )

    # Delete customized question display
    print("Deleting customized question display...")
    backup_file = "example5_delete_custom_display.sql.gz"
    success1 = delete_project_video_question_display(
        project_name="Pizza Test 0 Custom",
        video_uid="human.mp4",
        question_text="Pick one option",
        backup_first=True,
        backup_file=backup_file
    )
    print(f"Custom display: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("📋 Verification: Custom question display for 'human.mp4' in Pizza Test 0 Custom should be removed")

    # Delete ground truth
    print("Deleting ground truth entry...")
    success2 = delete_reviewer_ground_truth(
        video_uid="human.mp4",
        question_text="Number of people?",
        project_name="Human Test 0",
        backup_first=False  # Use same backup
    )
    print(f"Ground truth: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete ground truth for 'Number of people?...' in 'Human Test 0'")
    print("📋 Verification: Admin 2's ground truth for 'Number of people?' in 'human.mp4' should be removed")

    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_display_and_ground_truth()
```

#### Example 6: Delete Answer Review

```python
def example_delete_answer_review():
    print("\n🗑️ EXAMPLE 6: Delete Answer Review")
    print("=" * 35)
    
    from label_pizza.override_utils import delete_answer_review

    # Find an answer review to delete
    import label_pizza.db
    from sqlalchemy import text

    backup_file = "example6_delete_answer_review.sql.gz"
    with label_pizza.db.SessionLocal() as session:
        result = session.execute(text("""
            SELECT ar.id, reviewer.user_id_str, annotator.user_id_str, q.text 
            FROM answer_reviews ar
            JOIN annotator_answers aa ON ar.answer_id = aa.id
            JOIN users reviewer ON ar.reviewer_id = reviewer.id
            JOIN users annotator ON aa.user_id = annotator.id  
            JOIN questions q ON aa.question_id = q.id
            LIMIT 1
        """))
        row = result.fetchone()
        if row:
            answer_review_id, reviewer_name, annotator_name, question_text = row
            success = delete_answer_review(
                answer_id=answer_review_id,
                backup_first=True,
                backup_file=backup_file
            )
            print(f"Answer review deletion: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print("📋 Expected operations shown in confirmation:")
            print(f"    1. Delete '{reviewer_name}' review of '{annotator_name}' annotation")
            print("📋 Expected child rows deleted:")
            print(f"    • AnswerReview: {reviewer_name}'s review of {annotator_name}'s '{question_text[:30]}...' answer")
            print(f"📋 Verification: {annotator_name}'s original annotation remains, only {reviewer_name}'s review is deleted")
        else:
            print("⚠️ No answer reviews found to delete")
            print("📋 Expected: This message if no reviews exist in test data")
            success = False
    
    # Restore database to original state
    if success:
        print("\n🔄 Restoring database to original state...")
        restore_success = restore_from_backup(backup_file)
        print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_answer_review()
```

### 6. Practice Deletion Functions - Level 1 Dependencies

**What You'll See:** Level 1 deletions show small cascade operations (2-5 operations). The system automatically finds and deletes dependent records, showing each step clearly with actual child row names.

#### Example 7: Delete Annotator Answer

```python
def example_delete_annotator_answer():
    print("\n🗑️ EXAMPLE 7: Delete Annotator Answer (Cascades to Reviews)")
    print("=" * 58)
    
    from label_pizza.override_utils import delete_annotator_answer

    # Delete annotator answer (will cascade to any reviews)
    backup_file = "example7_delete_annotator_answer.sql.gz"
    success = delete_annotator_answer(
        video_uid="human.mp4",
        question_text="Number of people?",
        user_id_str="User 1",
        project_name="Human Test 0",
        backup_first=True,
        backup_file=backup_file
    )
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'Admin 1' review of 'User 1' annotation")
    print("    2. Delete 'User 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("📋 Expected child rows deleted:")
    print("    • AnswerReview: Admin 1's review of User 1's 'Number of people?' answer")
    print("    • AnnotatorAnswer: User 1's answer '1' for 'Number of people?' in 'human.mp4'")
    print("📋 Verification: Robot 1's answer for same question remains, User 1's description answer remains")
    print("    • Cascade: 2-3 operations total, preserving other users' answers")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_annotator_answer()
```

#### Example 8: Delete Project Group

```python
def example_delete_project_group():
    print("\n🗑️ EXAMPLE 8: Delete Project Group (Cascades to Relationships)")
    print("=" * 62)
    
    from label_pizza.override_utils import delete_project_group

    # Delete entire project group
    backup_file = "example8_delete_project_group.sql.gz"
    success = delete_project_group(
        name="Pizza Test Custom Projects",
        backup_first=True,
        backup_file=backup_file
    )
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Remove project 'Pizza Test 0 Custom' from group 'Pizza Test Custom Projects'")
    print("    2. Delete project group 'Pizza Test Custom Projects'")
    print("📋 Expected child rows deleted:")
    print("    • ProjectGroupProject: 'Pizza Test Custom Projects' ↔ 'Pizza Test 0 Custom' relationship")
    print("    • ProjectGroup: 'Pizza Test Custom Projects' group itself")
    print("📋 Verification: 'Pizza Test 0 Custom' project remains intact but no longer in any group")
    print("    • Other project groups ('Human Test Projects', 'Pizza Test Projects') remain")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_project_group()
```

### 7. Practice Deletion Functions - Level 2 Dependencies

**What You'll See:** Level 2 deletions show moderate cascade operations (5-15 operations). You'll see more complex dependency analysis as the system traces through multiple related tables with specific child row names.

#### Example 9: Delete Project User Role

```python
def example_delete_project_user_role():
    print("\n🗑️ EXAMPLE 9: Delete Project User Role (Complex Logic)")
    print("=" * 56)
    
    from label_pizza.override_utils import delete_project_user_role

    # Delete reviewer role (admins are auto-assigned so we demonstrate reviewer role deletion)
    backup_file = "example9_delete_reviewer_role.sql.gz"
    success = delete_project_user_role(
        project_name="Human Test 0",
        user_id_str="Admin 2",
        role="reviewer",
        backup_first=True,
        backup_file=backup_file
    )
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete ground truth for 'Number of people?...' in 'Human Test 0'")
    print("    2. Delete ground truth for 'If there are people, describe them...' in 'Human Test 0'")
    print("    3. Remove reviewer 'Admin 2' from project 'Human Test 0'")
    print("📋 Expected child rows deleted:")
    print("    • ReviewerGroundTruth: Admin 2's ground truths for both 'Number of people?' and 'If there are people, describe them.'")
    print("    • ProjectUserRole: Admin 2's 'reviewer' role in 'Human Test 0'")
    print("📋 Verification: Admin 2's auto-assigned 'admin' role remains intact")
    print("    • Admin modifications: Reverts Admin 2's ground truth changes to original values")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_project_user_role()
```

#### Example 10: Delete Project Video and Relationships

```python
def example_delete_project_video_relationships():
    print("\n🗑️ EXAMPLE 10: Delete Project-Video and Schema-Group Relationships")
    print("=" * 71)
    
    from label_pizza.override_utils import (
        delete_project_video,
        delete_schema_question_group,
        delete_question_group_question
    )

    # Delete project-video relationship
    print("Deleting project-video relationship...")
    backup_file = "example10_delete_relationships.sql.gz"
    success1 = delete_project_video(
        project_name="Human Test 0",
        video_uid="pizza.mp4",
        backup_first=True,
        backup_file=backup_file
    )
    print(f"Project-video: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Remove video 'pizza.mp4' from project 'Human Test 0'")
    print("📋 Verification: 'pizza.mp4' remains in 'Pizza Test 0' project, removed only from 'Human Test 0'")
    print("    • No annotations exist for 'pizza.mp4' in 'Human Test 0' to delete")
    
    # Delete schema-question group relationship
    print("Deleting schema-question group relationship...")
    success2 = delete_schema_question_group(
        schema_name="Questions about Humans",
        question_group_title="NSFW",
        backup_first=False  # Use same backup
    )
    print(f"Schema-group: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'User 1' annotation for 'Is the video not safe for work?...' in 'Human Test 0'")
    print("    2. Remove group 'NSFW' from schema 'Questions about Humans'")
    print("📋 Verification: 'NSFW' group remains available, 'Human' group stays in schema")
    
    # Delete question-group relationship
    print("Deleting question from group...")
    success3 = delete_question_group_question(
        question_group_title="Human",
        question_text="If there are people, describe them.",
        backup_first=False  # Use same backup
    )
    print(f"Question-group: {'✅ SUCCESS' if success3 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'User 1' annotation for 'If there are people, describe them...' in 'Human Test 0'")
    print("    2. Delete 'Robot 1' annotation for 'If there are people, describe them...' in 'Human Test 0'")
    print("    3. Remove question from group 'Human'")
    print("📋 Verification: Description question remains available, 'Number of people?' stays in 'Human' group")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_project_video_relationships()
```

### 8. Practice Deletion Functions - Level 3 (Major Impact)

**What You'll See:** Level 3 deletions show extensive cascade operations (15-50+ operations). These affect many tables and you'll see comprehensive dependency analysis with detailed lists of specific child rows being deleted.

#### Example 11: Delete Video

```python
def example_delete_video():
    print("\n🗑️ EXAMPLE 11: Delete Video (Extensive Cascade Operations)")
    print("=" * 57)
    
    from label_pizza.override_utils import delete_video
    
    # Delete video with extensive dependencies
    backup_file = "example11_delete_video.sql.gz"
    success = delete_video(
        video_uid="human.mp4",
        backup_first=True,
        backup_file=backup_file
    )
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'Admin 1' review of 'User 1' annotation")
    print("    2. Delete 'Admin 1' review of 'User 1' annotation")
    print("    3. Delete 'Admin 1' review of 'User 1' annotation")
    print("    4. Delete ground truth for 'Number of people?...' in 'Human Test 0'")
    print("    5. Delete ground truth for 'If there are people, describe ...' in 'Human Test 0'")
    print("    6. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    7. Delete custom display for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    8. Delete 'User 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("    9. Delete 'Robot 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("   10. Delete 'User 1' annotation for 'If there are people, describe ...' in 'Human Test 0'")
    print("   11. Delete 'Robot 1' annotation for 'If there are people, describe ...' in 'Human Test 0'")
    print("   12. Delete 'User 1' annotation for 'Is the video not safe for work...' in 'Human Test 0'")
    print("   13. Delete 'User 1' annotation for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("   14. Delete 'User 1' annotation for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("   15. Remove video 'human.mp4' from project 'Human Test 0'")
    print("   16. Remove video 'human.mp4' from project 'Pizza Test 0'")
    print("   17. Remove video 'human.mp4' from project 'Pizza Test 0 Custom'")
    print("   18. Delete video 'human.mp4'")
    print("📋 Expected child rows deleted:")
    print("    • AnswerReview: 3 Admin 1 reviews (all for User 1 annotations)")
    print("    • ReviewerGroundTruth: Admin 2's ground truths for 'human.mp4' in Human Test 0")
    print("    • ProjectVideoQuestionDisplay: Custom displays for 'human.mp4' in Pizza Test 0 Custom")
    print("    • AnnotatorAnswer: All annotations for 'human.mp4' (User 1, Robot 1 in Human Test 0; User 1 in Pizza Test 0 Custom)")
    print("    • ProjectVideo: Remove 'human.mp4' from Human Test 0, Pizza Test 0, and Pizza Test 0 Custom")
    print("    • Video: The 'human.mp4' video record itself")
    print("📋 Verification: 'human.mp4' completely removed, other videos ('pizza.mp4', 'human2.mp4') remain")
    print("    • Users, projects, questions, and schemas remain intact")
    print("    • This demonstrates 18 cascade operations following DELETION_ORDER sequence")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_video()
```

#### Example 12: Delete Question and Question Group

```python
def example_delete_question_and_group():
    print("\n🗑️ EXAMPLE 12: Delete Question and Question Group")
    print("=" * 49)
    
    from label_pizza.override_utils import delete_question, delete_question_group

    # Delete individual question
    print("Deleting individual question...")
    backup_file = "example12_delete_question_and_group.sql.gz"
    success1 = delete_question(
        text="If there are pizzas, describe them.",
        backup_first=True,
        backup_file=backup_file
    )
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'Admin 1' review of 'User 1' annotation")
    print("    2. Delete ground truth for 'If there are pizzas, describe them...' in 'Pizza Test 0'")
    print("    3. Delete 'User 1' annotation for 'If there are pizzas, describe them...' in 'Pizza Test 0'")
    print("    4. Remove question from group 'Pizza'")
    print("    5. Delete question 'If there are pizzas, describe them.'")
    print("📋 Verification: 'Pizza' group and 'Number of pizzas?' question remain intact")
    
    # Delete entire question group
    print("Deleting entire question group...")
    success2 = delete_question_group(
        title="Pizza Custom",
        backup_first=False  # Use same backup
    )
    print(f"Question group deletion: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    2. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    3. Delete custom display for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    4. Delete custom display for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    5. Delete 'User 1' annotation for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    6. Delete 'User 1' annotation for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    7. Remove question from group 'Pizza Custom'")
    print("    8. Remove question from group 'Pizza Custom'")
    print("    9. Remove group 'Pizza Custom' from schema 'Questions about Pizzas Custom'")
    print("   10. Delete question group 'Pizza Custom'")
    print("📋 Verification: Individual questions ('Pick one option', 'Describe one aspect') remain unassigned")
    print("    • 'Questions about Pizzas Custom' schema remains but with no question groups")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_question_and_group()
```

#### Example 13: Delete Project and User

```python
def example_delete_project_and_user():
    print("\n🗑️ EXAMPLE 13: Delete Project and User")
    print("=" * 39)
    
    from label_pizza.override_utils import delete_project, delete_user

    # Delete entire project
    print("Deleting entire project...")
    backup_file = "example13_delete_project_and_user.sql.gz"
    success1 = delete_project(
        name="Pizza Test 0",
        backup_first=True,
        backup_file=backup_file
    )
    print(f"Project deletion: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'Admin 1' review of 'User 1' annotation")
    print("    2. Delete 'Admin 1' review of 'User 1' annotation")
    print("    3. Delete ground truth for 'Number of pizzas?...' in 'Pizza Test 0'")
    print("    4. Delete ground truth for 'If there are pizzas, describe ...' in 'Pizza Test 0'")
    print("    5. Remove project 'Pizza Test 0' from group 'Pizza Test Projects'")
    print("    6. Delete 'User 1' annotation for 'Number of pizzas?...' in 'Pizza Test 0'")
    print("    7. Delete 'User 1' annotation for 'If there are pizzas, describe ...' in 'Pizza Test 0'")
    print("    8. Remove video 'human.mp4' from project 'Pizza Test 0'")
    print("    9. Remove video 'pizza.mp4' from project 'Pizza Test 0'")
    print("   10. Remove admin 'Admin 1' from project 'Pizza Test 0'")
    print("   11. Remove reviewer 'Admin 1' from project 'Pizza Test 0'")
    print("   12. Remove annotator 'Admin 1' from project 'Pizza Test 0'")
    print("   13. Remove annotator 'User 1' from project 'Pizza Test 0'")
    print("   14. Delete project 'Pizza Test 0'")
    print("📋 Verification: Videos, users, questions remain intact, 'Pizza Test Projects' group remains")
    
    # Delete user from all projects
    print("Deleting user from all projects...")
    success2 = delete_user(
        user_id_str="Robot 1",
        backup_first=False  # Use same backup
    )
    print(f"User deletion: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation:")
    print("    1. Delete 'Robot 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("    2. Delete 'Robot 1' annotation for 'If there are people, describe them...' in 'Human Test 0'")
    print("    3. Remove model 'Robot 1' from project 'Human Test 0'")
    print("    4. Delete user 'Robot 1'")
    print("📋 Verification: All projects, videos, and questions remain intact")
    print("    • User 1 and Admin accounts remain with their annotations")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_project_and_user()
```

### 9. Practice Deletion Functions - Level 4 (Nuclear Option)

**What You'll See:** Level 4 schema deletion shows the most extensive cascade operations (50-100+ operations). This affects nearly every table in the database and you'll see the longest lists of specific child rows being deleted with the most comprehensive dependency analysis.

#### Example 14: Delete Schema (Most Dangerous)

```python
def example_delete_schema():
    print("\n🗑️ EXAMPLE 14: Delete Schema (☢️ NUCLEAR OPTION)")
    print("=" * 51)
    
    from label_pizza.override_utils import delete_schema

    # Delete simple schema first
    print("Deleting simple schema...")
    backup_file = "example14_delete_schema.sql.gz"
    success1 = delete_schema(
        name="Questions about Pizzas Custom",
        backup_first=True,
        backup_file=backup_file
    )
    print(f"Simple schema: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation (simple schema):")
    print("    1. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    2. Delete custom display for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    3. Delete custom display for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    4. Delete custom display for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    5. Remove project 'Pizza Test 0 Custom' from group 'Pizza Test Custom Projects'")
    print("    6. Delete 'User 1' annotation for 'Pick one option...' in 'Pizza Test 0 Custom'")
    print("    7. Delete 'User 1' annotation for 'Describe one aspect of the vid...' in 'Pizza Test 0 Custom'")
    print("    8. Remove video 'human.mp4' from project 'Pizza Test 0 Custom'")
    print("    9. Remove video 'pizza.mp4' from project 'Pizza Test 0 Custom'")
    print("   10. Remove admin 'Admin 1' from project 'Pizza Test 0 Custom'")
    print("   11. Remove reviewer 'Admin 1' from project 'Pizza Test 0 Custom'")
    print("   12. Remove annotator 'Admin 1' from project 'Pizza Test 0 Custom'")
    print("   13. Remove annotator 'User 1' from project 'Pizza Test 0 Custom'")
    print("   14. Remove group 'Pizza Custom' from schema 'Questions about Pizzas Custom'")
    print("   15. Delete project 'Pizza Test 0 Custom'")
    print("   16. Delete schema 'Questions about Pizzas Custom'")
    print("📋 Verification: 'Pizza Custom' question group and its questions remain available")
    
    # Delete complex schema (affects multiple projects)
    print("Deleting complex schema (affects multiple projects)...")
    success2 = delete_schema(
        name="Questions about Humans",
        backup_first=False  # Use same backup
    )
    print(f"Complex schema: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("📋 Expected operations shown in confirmation (complex schema):")
    print("    1. Delete 'Admin 1' review of 'User 1' annotation")
    print("    2. Delete ground truth for 'Number of people?...' in 'Human Test 0'")
    print("    3. Delete ground truth for 'If there are people, describe ...' in 'Human Test 0'")
    print("    4. Remove project 'Human Test 0' from group 'Human Test Projects'")
    print("    5. Delete 'Robot 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("    6. Delete 'User 1' annotation for 'Number of people?...' in 'Human Test 0'")
    print("    7. Delete 'Robot 1' annotation for 'If there are people, describe ...' in 'Human Test 0'")
    print("    8. Delete 'User 1' annotation for 'If there are people, describe ...' in 'Human Test 0'")
    print("    9. Delete 'User 1' annotation for 'Is the video not safe for work...' in 'Human Test 0'")
    print("   10. Remove video 'human.mp4' from project 'Human Test 0'")
    print("   11. Remove video 'pizza.mp4' from project 'Human Test 0'")
    print("   12. Remove admin 'Admin 1' from project 'Human Test 0'")
    print("   13. Remove reviewer 'Admin 1' from project 'Human Test 0'")
    print("   14. Remove annotator 'Admin 1' from project 'Human Test 0'")
    print("   15. Remove annotator 'User 1' from project 'Human Test 0'")
    print("   16. Remove model 'Robot 1' from project 'Human Test 0'")
    print("   17. Remove reviewer 'Admin 2' from project 'Human Test 0'")
    print("   18. Remove annotator 'Admin 2' from project 'Human Test 0'")
    print("   19. Remove group 'Human' from schema 'Questions about Humans'")
    print("   20. Remove group 'NSFW' from schema 'Questions about Humans'")
    print("   21. Delete project 'Human Test 0'")
    print("   22. Delete schema 'Questions about Humans'")
    print("📋 Verification: All videos, users, question groups, and questions remain available")
    print("    • 'Human Test Projects' group remains but empty")
    print("    • 'NSFW' and 'Human' question groups remain available for other schemas")
    
    print("\n⚠️ WARNING: Schema deletion is the most destructive operation!")
    print("    Complex schema deletion shows 15-20+ named operations in confirmation")
    print("    Affects nearly every table and deletes massive amounts of data")
    print("    In production, this operation might take several minutes")
    print("    Always ensure you have verified backups before proceeding!")
    
    # Restore database to original state
    print("\n🔄 Restoring database to original state...")
    restore_success = restore_from_backup(backup_file)
    print(f"Restore: {'✅ SUCCESS' if restore_success else '❌ FAILED'}")

example_delete_schema()
```

## 🛡️ Recovery and Safety

### If Something Goes Wrong

**Restore from backup:**
```bash
# List available backups
ls -la ./backups/

# Restore from a specific backup
python label_pizza/manage_db.py \
    --mode restore \
    --backup-file ./backups/delete_complex_schema.sql.gz \
    --email admin1@example.com \
    --password admin111 \
    --user-id "Admin 1"
```

### Best Practices

**✅ Always Do This:**
- Test on non-production database first
- Each example automatically creates backups and restores afterward
- Use descriptive backup filenames for manual operations
- Understand cascade implications before proceeding
- Use the name-based functions for cleaner code

**❌ Never Do This:**
- Delete data without backup on production
- Rename identifiers without updating sync files
- Skip confirmation prompts
- Use override functions for normal data management

## ⚠️ Final Reminder

These functions bypass Label Pizza's normal safeguards. **Use archiving instead of deletion whenever possible.** Only use override functions when you have a specific need that cannot be met through normal operations.

**Every delete operation is permanent and cascades automatically. Always create backups and proceed with caution.**

---

[← Back to start](start_here.md)