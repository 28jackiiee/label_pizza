# Label Pizza - Sync Workflows

This guide explains how to sync data from JSON dictionaries or folders into the database.

## Getting your data into Label Pizza

Now that you understand the key components of Label Pizza (videos, users, question groups, schemas, projects, and project-user assignments), you can dive in right away with our provided demo (`example/`) or bring in your own data (`workspace/`):

| Goal                                           | Command                                                | What happens                                                 |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| **Explore a working demo**                     | `python sync_from_folder.py --folder-path ./example`   | Loads the sample **example/** folder into a fresh database so you can click around immediately. |
| **Start from scratch or import your own data** | `python sync_from_folder.py --folder-path ./workspace` | The empty **workspace/** folder has the correct structure. Run the command as-is for a blank database, **or** fill the JSON files first to bulk-import all your videos, users, schemas, and projects in one shot. |


### Step-by-step guide


> **Before you start:** make sure the **`DBURL`** key in your [.env](.env) file contains the correct PostgreSQL connection string you got from Supabase (or any other Postgres host).

1. **Initialize the database once and seed the first admin account**

   ```bash
   python label_pizza/manage_db.py \
       --mode init \
       --database-url-name DBURL \
       --email admin1@example.com \
       --password admin111 \
       --user-id "Admin 1"
   ```

2. **Sync the folder**

   ```bash
   # Our provided demo
   python sync_from_folder.py --folder-path ./example
   
   # Blank or custom project
   python sync_from_folder.py --folder-path ./workspace
   ```

3. **Launch the web UI**

   ```bash
   streamlit run label_pizza/label_pizza_app.py \
       --server.port 8000 --server.address 0.0.0.0 \
       -- --database-url-name DBURL
   ```

   Open **[http://localhost:8000](http://localhost:8000)** and log in with the admin credentials from step 1.

4. **Explore the interface**

   * **Admin portal** – view all videos, users, question groups, schemas, and projects.
   * **Annotation portal** – try answering a few demo tasks.
   * **Reviewer portal** – try setting a few ground truths by looking at the annotator answers.
   * **Meta-reviewer portal** – override reviewer decisions for final ground truth.
   * **Search portal** – look up any video by UID, answer pattern, or completion status.

   You can walk through the full workflow: coaching annotators, collecting ground truth, auditing reviews, and inspecting accuracy analytics.

### Reset or backup the database

Want to save your current work or start over from a blank database? Use the commands below—always back up first so you can restore any time.

| Action                                                       | One-liner to run                                             | Result                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Create a backup**                                          | `python label_pizza/manage_db.py --mode backup --backup-file my_sql.sql.gz` | Saves `./backups/my_sql.sql.gz` with every table.            |
| **Reset the database**<br>(nuclear option, makes its own backup) | `python label_pizza/manage_db.py --database-url-name DBURL --mode reset --auto-backup --backup-file my_sql.sql.gz --email admin1@example.com --password admin111 --user-id "Admin 1"` | Backs up first, then drops every table and recreates them from scratch (all tables start empty). |
| **Restore from backup**                                      | `python label_pizza/manage_db.py --database-url-name DBURL --mode restore --backup-file my_sql.sql.gz --email admin1@example.com --password admin111 --user-id "Admin 1"` | Loads `my_sql.sql.gz` into a freshly reset database, repopulating all tables with the saved data. |

### Syncing the database

Throughout your work, you can always synchronize the database—adding new items, updating existing ones, or archiving anything you no longer need—using one of the three methods below:

| Method                                                       | When to use it                                               | Typical examples                                             | Pros                                                         | Cons                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Web UI**                                                   | Quick, single-item edits                                     | Add one user and set a password • Assign that user to a project • Change the URL or metadata of one video | Instant; no code                                             | Tedious for hundreds of items                                |
| **`sync_from_folder.py`** (whole-folder sync)                | First import of the demo or your own workspace • One-time migration of a full dataset | Point the script at **workspace/** (or your custom folder) after you’ve prepared JSON for every table | One command; the JSON in **workspace/** is the canonical source of truth you can commit to Git | Scans every file (slow on very large folders) • Can’t handle some cross-table actions such as archiving a schema *and* its projects in the same run (explained later) |
| **Helpers in `sync_utils.py`** (`sync_videos`, `sync_question_groups`, …) | Day-to-day batch jobs or surgical edits                      | Add or update hundreds of videos/question groups/users/schemas/projects/assignments in one call • Import model answers and their confidence scores | Fast; touches only the table you call • Lets you script complex sequences (e.g., archive projects, then archive their now-unused schema) | You must run helpers in dependency order and manage the JSON yourself |

**Rule of thumb**

* **Web UI** → most day-to-day quick operations (e.g., one user, one video, one project assignment).
* **`sync_from_folder.py`** → load the demo, start a fresh workspace, or migrate an entire dataset the first time.
* **`sync_utils.py` helpers** → large or repeatable work once you’re up and running.

> **Note:** Most items in Label Pizza use a fixed identifier (e.g., `video_uid`, `user_id`, `project_name`, `schema_name`, etc.) and **cannot be renamed** once created. In most cases, you can simply update the display name shown in the UI (e.g., `display_text` for a question). But if you truly need to change the identifier, you can use the rename APIs described in **[admin_override.md](admin_override.md)**.

> **Warning:** This guide only supports **soft deletion** using `is_active=False`, which archives the item without removing its records. If you need to **hard delete** items—such as removing a video from all projects or deleting a question group from a schema—see **[admin_override.md](admin_override.md)** for details.


## Customizing your sync workflow with `sync_utils.py`

Once your database is live you will likely need to add or update hundreds of videos, projects, question groups, and model answers. The simplest route is to modify the JSON  inside **[`workspace/`](workspace/)** and rerun **[`sync_from_folder.py`](sync_from_folder.py)**, but be aware of how that script works:

* Reads **every** JSON file in the folder – videos, question groups, schemas, users, projects, assignments, annotations, and ground-truths.
* Compares each record to what's already in the database.
* Flags *any* difference as an update.

On a large workspace that full-folder scan can be slow.

Because **[`sync_from_folder.py`](sync_from_folder.py)** is just a thin wrapper around the helper functions in **[`label_pizza/sync_utils.py`](label_pizza/sync_utils.py)**, you can call those helpers directly and update **just** the data you need. 

### Overview of the helpers

| Helper                   | What it syncs                                           | Common tasks                                                 |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| `sync_videos`            | Videos                                                  | Add new clips, update video URLs or metadata, archive videos |
| `sync_question_groups`   | Question Groups **plus** their Questions                | Create or update question groups, add verification rules, archive question groups |
| `sync_schemas`           | Schemas                                                 | Add or update schemas, update instruction urls, toggle `has_custom_display` for per-video questions/options, archive schemas |
| `sync_users`             | Users                                                   | Bulk-create or archive users, reset passwords                |
| `sync_projects`          | Projects (and per-video custom question text / options) | Add new projects, add or update per-video custom question text / options, archive projects |
| `sync_users_to_projects` | User ↔ Project assignments                              | Grant or revoke project assignments, update user roles in a project, adjust user weights |
| `sync_project_groups`    | Project Groups                                          | Organize projects into groups for easier management, archive project groups |
| `sync_annotations`       | Annotator answers                                       | Import existing human or model predictions                   |
| `sync_ground_truths`     | Reviewer ground-truth answers                           | Import ground-truth answers so new annotators can start a project in Training mode with immediate feedback |

### How the helpers take input

By default the examples in this guide pass a **Python list of dictionaries**:

```python
sync_videos(videos_data=[{...}, {...}])
```

Some of the helpers can also take:

* **A path to a single `.json` file** like
  `sync_videos(videos_path="workspace/videos.json")`
* **A folder of many `.json` files** (for `sync_annotations` and `sync_ground_truths`) like 
  `sync_annotations(annotations_folder="workspace/annotations")`

For this guide, we'll use the Python list of dictionaries.

### Tip — what should I keep in the `workspace/` folder?

You don’t need to keep everything in `workspace/` — just the parts that are helpful to version-control or sync in bulk. Our recommendation is to keep the following:

<!-- * **✅ Videos & Projects**
  These often involve hundreds or thousands of entries. It’s much easier to manage them in JSON and upload via helper scripts than to enter them manually.

* **✅ Question Groups & Schemas**
  While you *can* create or edit these in the browser, it’s a good idea to keep them in JSON too — so you always have a version-controlled record of what’s been used, and you can easily apply the same schemas on another database.

* **🟡 Users & Assignments**
  For small teams, the Admin UI is the simplest way to manage users and assign them to projects. Use the helpers only if you're onboarding dozens of users at once.

* **🚫 Annotations & Ground-Truths**
  These should be collected directly in the web UI. Only use the helpers if you're importing existing labels (e.g., from a model or legacy dataset). -->

| Keep in JSON?                     | Why                                                          |
| --------------------------------- | ------------------------------------------------------------ |
| **Videos & Projects ✅**           | usually hundreds—JSON + helper is far faster than hand entry |
| **Question Groups & Schemas ✅**   | version-control your labeling policy and reuse it elsewhere  |
| **Users & Assignments 🟡**         | UI is fine for small teams; use helpers only for large batches |
| **Annotations & Ground-Truths 🚫** | collect via Web UI; import here only for existing labels (e.g., legacy dataset or from a model) |


Keeping the important JSON files in `workspace/` lets you:

* back them up, version-control them, and reuse them across machines or databases
* feed them directly into helper functions without rewriting scripts or paths


### Tip — order matters

```
Add / update :  videos → question groups → schemas → users → projects → assignments → annotations / ground_truths
Archive      :  annotations / ground_truths → assignments → projects → schemas → question groups → videos
```

Upload things **before** anything that depends on them, and archive in the reverse direction to avoid dependency errors.


### Starter Example: Syncing Helpers in Action

This section walks you through a small, working example to help you understand how to use the syncing helpers. You’ll upload a few videos, users, projects, and annotations to get familiar with the workflow.

Some steps depend on earlier ones, so we recommend running the commands in the order shown to avoid errors.

**Before you begin**, make sure to back up your current database (if you’ve already been using Label Pizza) and reset the database with the following command:

```bash
python label_pizza/manage_db.py \
  --auto-backup \
  --mode reset \
  --database-url-name DBURL \
  --email admin1@example.com \
  --password admin111 \
  --user-id "Admin 1"
```

This will clear all existing data and create a fresh admin account.


#### 0. Prerequisite: `init_database`

**Important:** You must run this once before **importing** any other helper functions, otherwise you will get an error.

```python
from label_pizza.db import init_database
init_database("DBURL")  # or change DBURL to another key in .env
```


#### 1. `sync_videos`

Function for adding / updating / archiving videos

1.1 - Add videos

To add videos, you must provide a `video_uid` that does not already exist in the database.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_videos
videos_data = [
  {
    "video_uid": "human.mp4", # must NOT exist in the database
    "url": "https://your-repo/human.mp4",
    "is_active": True,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=L3wKzyIN1yk",
      "license": "Standard YouTube License"
    }
  },
  {
    "video_uid": "pizza.mp4",
    "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/pizza.mp4",
    "is_active": True,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=8J1NzjA9jNg",
      "description": "A video of a pizza being made.",
      "license": "Standard YouTube License"
    }
  }
]

sync_videos(videos_data=videos_data)
```

1.2 - Update or archive a video

To update a video record, keep the `video_uid` fixed and modify other fields such as `url`, `is_active`, or `metadata`.

```python
# ...After step 1.1

videos_data = [
  {
    "video_uid": "human.mp4",                       # must already exist in the database
    "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/human.mp4",           # update url (Must not exist in the database)
    "is_active": True,                              # keep this True (set to False if you want to archive the video)
    "metadata": {
       # update metadata to be an empty dictionary
    }
  }
]

sync_videos(videos_data=videos_data)
```

### 2. `sync_users`

Function for adding / updating / archiving users

2.1 - Add a user

To add a user, provide a unique `user_id` that does not already exist in the database. If specifying an `email`, it must also be unique.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_users

users_data = [
    {
        "user_id": "User 1",          # must NOT exist in the database
        "email": "user1@example.com", # must NOT exist in the database
        "password": "user111",
        "user_type": "human",
        "is_active": True
    }
]

sync_users(users_data=users_data)
```

2.2 - Update or archive a user

To update a user, provide the `user_id` that already exists in the database to update the `email`, `password`, `user_type`, or `is_active`.

```python
# ...After step 2.1

# update email using user_id
users_data = [
    {
        "user_id": "User 1",              # the existing user_id
        "email": "user1-new@example.com", # new email
        "password": "user111-new",        # new password
        "user_type": "human",             # could only select from "admin", "human"
        "is_active": False                # set to False to archive the user
    }
]

sync_users(users_data=users_data)

# Then update user_id using email + archive the user
users_data = [
    {
        "user_id": "User 1 New",          # new user_id
        "email": "user1-new@example.com", # the existing email
        "password": "user111-new",        
        "user_type": "human",             # could only select from "admin", "human"
        "is_active": True                 # set to True to unarchive the user
    }
]

sync_users(users_data=users_data)

# You can add a model user. Model users do not have `email`
users_data = [
    {
        "user_id": "Model 1",          # new user_id
        "email": None,                 # model user could not have email
        "password": "",                # model user do not have password
        "user_type": "model",
        "is_active": True
    }
]

sync_users(users_data=users_data)
```

> **Note:** You cannot change a human/admin user to a model user because model users do not have `email` and might have `confidence_score` in the `annotations` table.

### 3. Sync Question Groups and Their Questions

Before creating question groups, it is important to decide whether you need a **verification function**. This optional Python function (which you can define in `label_pizza/verify.py`) checks that all answers in a group are logically consistent *before* they are submitted.

> **For example:** Annotators must describe a person if there are more than 0 people in the video; otherwise, the description must be blank. The `check_human_description` function enforces this rule.

```python
def check_human_description(answers: Dict[str, str]) -> None:
    num_people = answers.get("Number of people?")
    description = answers.get("If there are people, describe them.")
    
    if num_people == "0" and description:
        raise ValueError("Description cannot be provided when there are no people")
    if num_people != "0" and not description:
        raise ValueError("Description must be provided when there are people")
```

If you do not need such a check, set `verification_function` to `None` (`null` in JSON).

3.1 - Add question groups and their questions

To add a question group, use a unique `title` that does not yet exist.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_question_groups
question_groups_data = [
    {
        "title": "Human",           # must NOT exist in the database
        "display_title": "Human",
        "description": "Detect and describe all humans in the video.",
        "is_reusable": False,
        "is_auto_submit": False,
        "verification_function": None,
        "questions": [
            {
                "qtype": "single",
                "text": "Number of people?",
                "display_text": "Number of people?",
                "options": [
                    "0",
                    "1",
                    "2",
                    "3 or more"
                ],
                "display_values": [
                    "0",
                    "1",
                    "2",
                    "3 or more"
                ],
                "option_weights": [
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                "default_option": "0"
            },
            {
                "qtype": "description",
                "text": "If there are people, describe them.",
                "display_text": "If there are people, describe them.",
                "default_option": None
            }
        ]
    },
    {
        "title": "NSFW",
        "display_title": "NSFW",
        "description": "Check whether the video is not safe for work.",
        "is_reusable": True,
        "is_auto_submit": True,
        "verification_function": None,
        "questions": [
            {
                "qtype": "single",
                "text": "Is the video not safe for work?",
                "display_text": "Is the video not safe for work?",
                "options": [
                    "No",
                    "Yes"
                ],
                "display_values": [
                    "No",
                    "Yes"
                ],
                "option_weights": [
                    1.0,
                    99.0 # If any annotator selects “Yes,” most of the weight is given to it.
                ],
                "default_option": "No",
            }
        ]
    }
]

sync_question_groups(question_groups_data=question_groups_data)
```

3.2 - Update a question group and its questions

To update an existing question group, keep the same `title` and include the complete current list of `questions`. You may modify

- `display_title`, `description`, `is_reusable`, `is_auto_submit`, `verification_function`
- The **order** of questions. 
- For each question, its `display_text`, `display_values`, and `default_option`. For `single` questions, you may also reorder `options` and adjust their `option_weights`.

> **Note 1:** You cannot add or remove questions from a question group. This ensures existing annotations aren’t affected.

> **Note 2:** You cannot change the `qtype` of a question. For `single` questions, you may append new options, but you may not delete existing ones. Updating a question (using `text` as the unique identifier) in one group will automatically update it in every other group that includes that question.

```python
#...After step 3.1
question_groups_data = [
    {
        "title": "Human",                  # must exist in the database
        "display_title": "Human (Updated)",  # update display_title
        "description": "Detect and describe all humans in the video. (Updated)", # update description here
        "is_reusable": True,               # update is_reusable
        "is_auto_submit": True,            # update is_auto_submit
        "verification_function": "check_human_description",     # add verification_function
        "questions": [                     # update the question order to move description to the first question
            {
                "qtype": "description",
                "text": "If there are people, describe them.",
                "display_text": "If there are people, describe them.",
                "default_option": "Auto‑submit requires a default answer." # update the default answer (for description questions)
            },
            {
                "qtype": "single",
                "text": "Number of people?",
                "display_text": "Number of people (Updated)",     # update the display_text of a question
                "options": [ # reorder the options
                    "3 or more",
                    "2",
                    "1",
                    "0",
                ],
                "display_values": [ # be sure to update the display_values to match the new order of options
                    "3+",
                    "2",
                    "1",
                    "0",
                ],
                "option_weights": [
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                "default_option": "3 or more" # update the default option for single questions
            }
        ]
    }
]

sync_question_groups(question_groups_data=question_groups_data)
```

### 4. Sync Schemas

Function for adding / updating / archiving schemas (a schema collects one or more question groups into a task)

4.1 - Add schemas

To add a schema, use a unique `schema_name` that is not yet in the database. For `question_group_names`, use the `title` of the question groups you have already added.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_schemas

schemas_data = [
  {
    "schema_name": "Questions about Humans",   # must NOT exist in the database
    "instructions_url": "",
    "question_group_names": [
       "NSFW", "Human"
    ],
    "has_custom_display": False,
    "is_active": True
  }
]

sync_schemas(schemas_data=schemas_data)
```

4.2 - Update or archive schemas

To update a schema, use the `schema_name` that exists in the database. You may update `instructions_url`, `has_custom_display`, `is_active`, and reorder `question_group_names`. You cannot add or remove question groups from a schema.

> **Note:** You cannot archive a schema if it is still used by an active project.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_schemas

schemas_data = [
  {
    "schema_name": "Questions about Humans",                    # must exist in the database
    "instructions_url": "https://en.wikipedia.org/wiki/Human",  # update instruction_url
    "question_group_names": [                                   # new question group order
      "Human", "NSFW"
    ],
    "has_custom_display": True,                          # update has_custom_display
    "is_active": True                                    # keep this True (set to False if you want to archive the schema)
  }
]

sync_schemas(schemas_data=schemas_data)
```

### 5. Sync Projects

Function for adding / updating / archiving projects (a project combines a set of videos and a schema)

5.1 - Add projects

To add a project, use a unique `project_name` that is not yet in the database.

> **Note:** We recommend using at most 50-100 videos per project for better user experience.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_projects

projects_data = [
  {
    "project_name": "Human Test 0",            # must NOT exist in the database
    "schema_name": "Questions about Humans",
    "description": "Test project for human questions",
    "is_active": True,
    "videos": [
      "human.mp4",
      "pizza.mp4"
    ]
  }
]

sync_projects(projects_data=projects_data)
```

5.2 - Update / archive projects

You can update a project using the `project_name` that already exists in the database. Only `description` and `is_active` can be changed. If you want to remove videos, you can do so by archiving them (across all projects). If you want to add videos, we recommend creating a new project.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_projects

projects_data = [
  {
    "project_name": "Human Test 0",            # must exist in the database
    "schema_name": "Questions about Humans",   # must exist in the database
    "description": "Test project for human questions (Updated)",   # you could only update the description
    "is_active": True,                         # keep this True (set to False if you want to archive the project)
    "videos": [ # You cannot add or remove videos from a project
      "human.mp4",
      "pizza.mp4"
    ]
  }
]

sync_projects(projects_data=projects_data)
```

### 6. Sync Users to Projects

Function for adding / updating / archiving project-user assignments

6.1 - Add user to project

To assign a user to a project, provide a unique (`user_name, project_name`) pair that does not already exist in the database.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_users_to_projects

assignments_data = [
  {
    "user_name": "Model 1",          
    "project_name": "Human Test 0", 
    "role": "model",                 # `model` users could only be assigned as `model`
    "user_weight": 1.0,              
    "is_active": True
  },
  {
    "user_name": "User 1 New",            
    "project_name": "Human Test 0",
    "role": "annotator",             # `human` users could only be assigned as `reviewer` or `annotator`
    "user_weight": 1.0,              
    "is_active": True
  }
]
sync_users_to_projects(assignments_data=assignments_data)
```


6.2 - Update a user's role in a project

You can modify the user's `role` and `user_weight` in a project.

```python
#...After step 6.1
assignments_data = [
  {
    "user_name": "User 1 New",            
    "project_name": "Human Test 0",
    "role": "reviewer", # update the role to reviewer
    "user_weight": 2.0, # update the user weight              
    "is_active": True
  }
]

sync_users_to_projects(assignments_data=assignments_data)
```

6.3 - Remove a user from a project

Set `is_active` to `False` to remove a user from a project. Note that this will not delete any of their existing annotations or ground truth answers within the project.

> **Note:** If you want to remove a user completely from a project, you must set `is_active` to `False` and `role` to `annotator`. If you only want to remove a user's reviewer role, you can set `is_active` to `False` and `role` to `reviewer`.

```python
#...After step 6.2
assignments_data = [
  {
    "user_name": "Model 1",
    "project_name": "Human Test 0",
    "role": "model",
    "user_weight": 2.0,
    "is_active": False  # set False to remove user from this project
  }
]

sync_users_to_projects(assignments_data=assignments_data)
```

### 7. Sync Project Groups

Function for adding / updating / archiving project groups (a project group collects one or more projects for better organization)

7.1 - Adding project group

To add a project group, use a unique `project_group_name` that does not already exist in the database.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_project_groups
project_groups_data = [
    {
        "project_group_name": "Human Test Projects",
        "description": "This is a project group for human test with a single project",
        "projects": [
            "Human Test 0"
        ]
    }
]

sync_project_groups(project_groups_data=project_groups_data)
```

7.2 - Update / archive project groups

You can use the `project_group_name` to update a project group's `description` and modify the `projects` list to add or remove projects.

```python
#...After step 7.1
project_groups_data = [
  {
      "project_group_name": "Human Test Projects",
      "description": "This is an empty project group",  # Update the description
      "projects": [ # Remove all projects from the project group
      ]
  }
]

sync_project_groups(project_groups_data=project_groups_data)
```

### 8. Sync Annotations (Annotator) and Ground Truths (Reviewer)


Function for adding / updating annotations (annotator) and ground truths (reviewer)

8.1 - Add annotations (annotator)

You can upload annotations by providing a unique combination of `question_group_title`, `project_name`, `user_name`, and `video_uid`.

For all annotations, `is_ground_truth` must be set to `False`.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_annotations

annotations_data = [
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "User 1 New",
    "video_uid": "human.mp4",
    "answers": {   # Answers must include all and only the questions defined in this question group
      "Number of people?": "1",
      "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
    },
    "is_ground_truth": False            # For annotations, is_ground_truth MUST be False
  },
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "Model 1",
    "video_uid": "human.mp4",
    "answers": {   # Answers must include all and only the questions defined in this question group
      "Number of people?": "1",
      "If there are people, describe them.": "The man has cropped hair."
    },
    "confidence_scores": {
      "Number of people?": 0.9,
      "If there are people, describe them.": 0.8
    },
    "is_ground_truth": False    # is_ground_truth MUST be False for annotations
  }
]

sync_annotations(annotations_data=annotations_data)
```

> **Note:** The `answers` dictionary must include **all and only** the questions defined in the specified `question_group_title`.

8.2 - Update annotations (annotator)

To update an annotation, use the same unique combination of `question_group_title`, `project_name`, `user_name`, and `video_uid`. You can modify the `answers` and `confidence_scores`.

```python
#...After step 8.1
annotations_data = [
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "Model 1",
    "video_uid": "human.mp4",
    "answers": {   # update the answers
      "Number of people?": "0",
      "If there are people, describe them.": ""
    },
    "confidence_scores": { # update the confidence scores
      "Number of people?": 0.3,
      "If there are people, describe them.": 0.2
    },
    "is_ground_truth": False    # is_ground_truth MUST be False for annotations
  }
]

sync_annotations(annotations_data=annotations_data)
```

8.3 - Add ground truths (reviewer)

You can upload ground truths by providing a unique combination of `question_group_title`, `project_name`, and `video_uid`. The `user_name` must correspond to a user with a `reviewer` or `admin` role.

For all ground truths, `is_ground_truth` must be set to `True`.

> **Note:** Each (video, question group) pair can have at most **one** ground truth.

```python
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_ground_truths

ground_truths_data = [
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "User 1 New",          # User must have at least "annotation" privileges
    "video_uid": "human.mp4",
    "answers": {   # Answers must include all and only the questions defined in the question group
      "Number of people?": "0",
      "If there are people, describe them.": ""
    },
    "is_ground_truth": True             # must be True
  }
]

sync_ground_truths(ground_truths_data=ground_truths_data)
```

8.4 - Update a ground truth

To update an existing ground truth, use the same unique combination of `question_group_title`, `project_name`, and `video_uid`. You may change the `answers`, and optionally update the `user_name`. If you supply a different `user_name`, the updated ground truth will be attributed to that user.

```python
#...After step 8.3
ground_truths_data = [
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "Admin 1",          # User must have at least "annotation" privileges
    "video_uid": "human.mp4",
    "answers": {   # Answers must include all and only the questions defined in the question group
      "Number of people?": "1",
      "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
    },
    "is_ground_truth": True             # must be True
  }
]

sync_ground_truths(ground_truths_data=ground_truths_data)
```


### 9. Launch the Labeling Platform

You can now launch the labeling platform locally to see the starter example in action. Run the following command (feel free to change the port number if needed):
```bash
# Start the Streamlit app; adjust the port if you'd like
streamlit run label_pizza/label_pizza_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  -- \
  --database-url-name DBURL
```

Once it’s running, open **[http://localhost:8000](http://localhost:8000)** in your browser and log in.


---

[← Back to start](start_here.md) | [Next → Custom Question](custom_display.md)