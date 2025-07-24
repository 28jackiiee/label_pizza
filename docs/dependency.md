# Database Table Dependencies Analysis

## API Reference Tables

### Delete APIs
**CRITICAL: All delete operations perform CASCADE DELETE and MUST first clear all child rows that depend on the parent row, as detailed in the dependency analysis below.**

**CASCADE DELETE BEHAVIOR:**
- All delete functions automatically identify and delete dependent child rows in the correct sequence
- Before performing any deletion, the system will print all child row delete function calls that will be executed in sequence
- The system will always ask for confirmation before proceeding with any deletion
- All `delete_xxx(name, **backup_params)` functions use `delete_xxx_using_id(id, **backup_params)` under the hood after resolving the name to ID

**All delete functions include these backup parameters (imported from manage_db.py):**
- `backup_first=True` - Create automatic backup before deletion
- `backup_dir="./backups"` - Default backup directory  
- `backup_file=None` - Auto-generated timestamp filename
- `compress=True` - Enable gzip compression

| Table | Delete Using ID | Delete Using Name |
|-------|----------------|-------------------|
| User | `delete_user_using_id(id, **backup_params)` | `delete_user(user_id_str, **backup_params)` |
| Video | `delete_video_using_id(id, **backup_params)` | `delete_video(video_uid, **backup_params)` |
| QuestionGroup | `delete_question_group_using_id(id, **backup_params)` | `delete_question_group(title, **backup_params)` |
| Question | `delete_question_using_id(id, **backup_params)` | `delete_question(text, **backup_params)` |
| QuestionGroupQuestion | `delete_question_group_question_using_id(question_group_id, question_id, **backup_params)` | `delete_question_group_question(question_group_title, question_text, **backup_params)` |
| Schema | `delete_schema_using_id(id, **backup_params)` | `delete_schema(name, **backup_params)` |
| SchemaQuestionGroup | `delete_schema_question_group_using_id(schema_id, question_group_id, **backup_params)` | `delete_schema_question_group(schema_name, question_group_title, **backup_params)` |
| Project | `delete_project_using_id(id, **backup_params)` | `delete_project(name, **backup_params)` |
| ProjectVideo | `delete_project_video_using_id(project_id, video_id, **backup_params)` | `delete_project_video(project_name, video_uid, **backup_params)` |
| ProjectUserRole | `delete_project_user_role_using_id(project_id, user_id, role, **backup_params)` | `delete_project_user_role(project_name, user_id_str, role, **backup_params)` |
| ProjectGroup | `delete_project_group_using_id(id, **backup_params)` | `delete_project_group(name, **backup_params)` |
| ProjectGroupProject | `delete_project_group_project_using_id(project_group_id, project_id, **backup_params)` | `delete_project_group_project(project_group_name, project_name, **backup_params)` |
| ProjectVideoQuestionDisplay | `delete_project_video_question_display_using_id(project_id, video_id, question_id, **backup_params)` | `delete_project_video_question_display(project_name, video_uid, question_text, **backup_params)` |
| AnnotatorAnswer | `delete_annotator_answer_using_id(id, **backup_params)` | `delete_annotator_answer(video_uid, question_text, user_id_str, project_name, **backup_params)` |
| ReviewerGroundTruth | `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id, **backup_params)` | `delete_reviewer_ground_truth(video_uid, question_text, project_name, **backup_params)` |
| AnswerReview | `delete_answer_review_using_id(id, **backup_params)` | `delete_answer_review(answer_id, **backup_params)` |

### Name Management APIs

| Table | Get Name (ID → Name) | Set Name (ID + Name → Update) | Set Name (Name + Name → Update) |
|-------|---------------------|------------------------------|----------------------------------|
| User | `get_user_name(id) → user_id_str` | `set_user_name_using_id(id, user_id_str, **backup_params)` | `set_user_name(old_user_id_str, new_user_id_str, **backup_params)` |
| Video | `get_video_name(id) → video_uid` | `set_video_name_using_id(id, video_uid, **backup_params)` | `set_video_name(old_video_uid, new_video_uid, **backup_params)` |
| QuestionGroup | `get_question_group_name(id) → title` | `set_question_group_name_using_id(id, title, **backup_params)` | `set_question_group_name(old_title, new_title, **backup_params)` |
| Question | `get_question_name(id) → text` | `set_question_name_using_id(id, text, **backup_params)` | `set_question_name(old_text, new_text, **backup_params)` |
| Schema | `get_schema_name(id) → name` | `set_schema_name_using_id(id, name, **backup_params)` | `set_schema_name(old_name, new_name, **backup_params)` |
| Project | `get_project_name(id) → name` | `set_project_name_using_id(id, name, **backup_params)` | `set_project_name(old_name, new_name, **backup_params)` |
| ProjectGroup | `get_project_group_name(id) → name` | `set_project_group_name_using_id(id, name, **backup_params)` | `set_project_group_name(old_name, new_name, **backup_params)` |

**Notes:**
- **Get operations** are read-only lookups that return the current name for an ID
- **Set operations (using ID)** update the name by database ID and include backup parameters
- **Set operations (using name)** update the name by current name (more user-friendly)
- The new name must be unique, otherwise an error will be raised
- **Set operations** use same backup parameters as delete APIs above

## Direct Dependency Graph

This shows what tables would be directly affected (have orphaned/inconsistent data) if we delete a row from each table. **All delete operations will automatically execute the dependency chain in the correct sequence after user confirmation.**

## LEAF NODES (No dependencies - safe to delete)

```
ProjectGroupProject → must delete first: []
ProjectVideoQuestionDisplay → must delete first: []
ReviewerGroundTruth → must delete first: []
AnswerReview → must delete first: []
```

## LEVEL 1 DEPENDENCIES

### AnnotatorAnswer
**Must delete first:** `[AnswerReview]`
- Call `delete_answer_review_using_id(id)` where `(answer_id = id)` from `AnswerReview`

### ProjectGroup
**Must delete first:** `[ProjectGroupProject]`
- Call `delete_project_group_project_using_id(project_group_id, project_id)` where `(project_group_id = id)` from `ProjectGroupProject`

## LEVEL 2 DEPENDENCIES

### ProjectVideo
**Must delete first:** `[AnnotatorAnswer, ReviewerGroundTruth, ProjectVideoQuestionDisplay]`
- Call `delete_annotator_answer_using_id(id)` where `(project_id = project_id, video_id = video_id)` from `AnnotatorAnswer`
- Call `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id)` where `(project_id = project_id, video_id = video_id)` from `ReviewerGroundTruth`
- Call `delete_project_video_question_display_using_id(project_id, video_id, question_id)` where `(project_id = project_id, video_id = video_id)` from `ProjectVideoQuestionDisplay`

### ProjectUserRole
**Conditional check:** `[AnnotatorAnswer, ReviewerGroundTruth]`

**If deleting admin role:**
- Call `update_reviewer_ground_truth_using_id(video_id, question_id, project_id)` where `(project_id = project_id, modified_by_admin_id = user_id)` to revert ground truth from `ReviewerGroundTruth` -- importantly, need set `answer_value` to `original_answer_value` and set to `NULL` for `(modified_at, modified_by_admin_id, and modified_by_admin_at)`

**If deleting reviewer role:**
- Must delete first from `ReviewerGroundTruth`
- If admin role exists for this user in this project, call `delete_project_user_role_using_id(project_id, user_id, "admin")`
- Call `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id)` where `(project_id = project_id, reviewer_id = user_id)` from `ReviewerGroundTruth`
- Gather all answers `(answer_id)` from `AnnotatorAnswer` and call `delete_answer_review_using_id(id)` where `(answer_id, reviewer_id = user_id)` from `AnswerReview`

**If deleting annotator/model role:**
- Must delete first from `AnnotatorAnswer`
- If reviewer role exists for this user in this project, call `delete_project_user_role_using_id(project_id, user_id, "reviewer")`
- Call `delete_annotator_answer_using_id(id)` where `(project_id = project_id, user_id = user_id)` from `AnnotatorAnswer`

### QuestionGroupQuestion
**Must delete first:** `[AnnotatorAnswer, ReviewerGroundTruth, ProjectVideoQuestionDisplay]`

1. **Step 1:** Gather all schemas `(schema_id)` using this question group `(question_group_id)`
2. **Step 2:** Gather all projects `(project_id)` using those schemas `(schema_id)`
3. Call `delete_annotator_answer_using_id(id)`, `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id)`, and `delete_project_video_question_display_using_id(project_id, video_id, question_id)` where `(project_id = project_id, question_id = question_id)` tuple matches

### SchemaQuestionGroup
**Must delete first:** `[AnnotatorAnswer, ReviewerGroundTruth, ProjectVideoQuestionDisplay]`

1. **Step 1:** Gather all projects `(project_id)` using this schema `(schema_id)`
2. **Step 2:** Gather all questions `(question_id)` from `QuestionGroupQuestion` using this question group `(question_group_id)`
3. Call `delete_annotator_answer_using_id(id)`, `delete_reviewer_ground_truth_using_id(video_id, question_id, project_id)`, and `delete_project_video_question_display_using_id(project_id, video_id, question_id)` where `(project_id = project_id, question_id = question_id)` tuple matches

## LEVEL 3 DEPENDENCIES

### Video
**Must delete first:** `[ProjectVideo]`
- Call `delete_project_video_using_id(project_id, video_id)` where `(video_id = video_id)` from `ProjectVideo`

### Question
**Must delete first:** `[QuestionGroupQuestion]`
- Call `delete_question_group_question_using_id(question_group_id, question_id)` where `(question_id = question_id)` from `QuestionGroupQuestion`

### QuestionGroup
**Must delete first:** `[QuestionGroupQuestion, SchemaQuestionGroup]`
- Call `delete_question_group_question_using_id(question_group_id, question_id)` where `(question_group_id = question_group_id)` from `QuestionGroupQuestion`
- Call `delete_schema_question_group_using_id(schema_id, question_group_id)` where `(question_group_id = question_group_id)` from `SchemaQuestionGroup`

### Project
**Must delete first:** `[ProjectVideo, ProjectUserRole, ProjectGroupProject]`
- Call `delete_project_video_using_id(project_id, video_id)` where `(project_id = project_id)` from `ProjectVideo`
- Call `delete_project_user_role_using_id(project_id, user_id, role)` where `(project_id = project_id)` from `ProjectUserRole`
- Call `delete_project_group_project_using_id(project_group_id, project_id)` where `(project_id = project_id)` from `ProjectGroupProject`

### User
**Must delete first:** `[ProjectUserRole]`
- Call `delete_project_user_role_using_id(project_id, user_id, role)` where `(user_id = user_id)` from `ProjectUserRole`

## LEVEL 4 DEPENDENCIES

### Schema
**Must delete first:** `[SchemaQuestionGroup, Project]`
- Call `delete_schema_question_group_using_id(schema_id, question_group_id)` where `(schema_id = schema_id)` from `SchemaQuestionGroup`
- Call `delete_project_using_id(id)` where `(schema_id = schema_id)` from `Project`