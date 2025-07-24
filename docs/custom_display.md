# Label Pizza - Custom Question Per Video

This guide explains how to customize the question text and options for each video in a project.

## Custom Display: Per-Video Questions and Options

Label Pizza supports an advanced feature that lets you **customize the question text and options for each video** in a project. This is useful when the same schema applies across all videos, but you want to adjust the phrasing or choices based on the video content.

### What is a custom display?

A **custom display** lets you override the question text and/or the option labels for individual videos within a project—without changing the underlying schema.

Each override is defined per video and per question, using:

* `custom_question`: the question text you want the annotator to see
* `custom_option`: a dictionary mapping each original option to a new display string (must include all options in the same order)

This is useful when the schema logic remains the same, but the user-facing language should vary by video.

> **Example:** For one video, you might ask “Is there a pizza in the video?” with Yes/No options. For another, you might ask “What type of pizza is shown?” with options like “Pepperoni” and “Veggie”.

You can explore this feature in one of two ways:

* **Option 1:** Use the provided example folder to try it out immediately
* **Option 2:** Set everything up manually to better understand how it works

**Before you begin**, you should back up your current database (if you’ve already been using Label Pizza) and reset the database with the following command:

```bash
python label_pizza/manage_db.py \
  --auto-backup \
  --mode reset \
  --database-url-name DBURL \
  --email admin1@example.com \
  --password admin111 \
  --user-id "Admin 1"
```

You can use this admin account to test the custom display feature.

---

### Option 1: Try the prepared example

We’ve created a working demo in `example_custom_question/` so you can see how per-video customization works in action.

**Step 1 – Import the example**

```bash
python sync_from_folder.py \
  --database-url-name DBURL \
  --folder-path ./example_custom_question
```

**Step 2 – Launch the app**

```bash
streamlit run label_pizza/label_pizza_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  -- \
  --database-url-name DBURL
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser and log in with the admin account you created in the previous step (e.g., email is `admin1@example.com` and password is `admin111`). You’ll see a project where each video has custom question text and options.

---

### Option 2: Set it up manually to learn the workflow

If you want to understand how custom display works under the hood, you can build a simple project yourself by following the steps below.

#### 1. Prepare videos, a question group, and a schema

The schema must have `has_custom_display = True`.

```
from label_pizza.db import init_database
init_database("DBURL")

from label_pizza.sync_utils import sync_videos, sync_question_groups, sync_schemas

# Add two videos
videos_data = [
  {
    "video_uid": "human.mp4",
    "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/human.mp4",
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

# Add a question group (the display text and values will be overridden for each video)
question_groups_data = [
  {
    "title": "Pizza Custom",
    "display_title": "Pizza Custom",
    "description": "Detect and describe all pizzas in the video.",
    "is_reusable": False,
    "is_auto_submit": False,
    "verification_function": None,
    "questions": [
      {
          "qtype": "single",
          "text": "Pick one option",
          "display_text": "Pick one option",
          "options": [
              "Option A",
              "Option B"
          ],
          "display_values": [
              "Option A",
              "Option B"
          ],
          "option_weights": [
              1.0,
              1.0
          ],
          "default_option": "Option A"
      },
      {
        "qtype": "description",
        "text": "Describe one aspect of the video",
        "display_text": "Describe one aspect of the video",
        "default_option": None
      }
    ]
  }
]

# Add a schema (the question group will be used in this schema)
schemas_data = [
  {
    "schema_name": "Questions about Pizzas Custom",
    "instructions_url": "",
    "question_group_names": [
      "Pizza Custom"
    ],
    "has_custom_display": True,
    "is_active": True
  }
]

sync_videos(videos_data=videos_data)
sync_question_groups(question_groups_data=question_groups_data)
sync_schemas(schemas_data=schemas_data)
```

#### 2. Create a project with custom questions and options per video

To set custom displays for any project:

- The schema must have `has_custom_display` == `True`.
- For each question, `custom_option` must match the number and order of the original `options`. 

```
from label_pizza.sync_utils import sync_projects

projects_data = [
  {
    "project_name": "Pizza Test 0 Custom",
    "schema_name": "Questions about Pizzas Custom",
    "description": "Test project for custom questions",
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
```


#### 3. Launch the app

```bash
streamlit run label_pizza/label_pizza_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  -- \
  --database-url-name DBURL
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser and log in with the admin account you created in the previous step (e.g., email is `admin1@example.com` and password is `admin111`).

---

[← Back to start](start_here.md) | [Next → Admin Override](admin_override.md)