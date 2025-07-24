# Label Pizza - Data Model

This guide explains the core concepts behind Label Pizza: videos, users, question groups, schemas, projects, assignments, annotations, and ground truths.

**If you want to explore the web UI while reading this guide, use the below command:**

```bash
# Initialize database using the database url (DBURL) defined in your .env (Caveat: This will delete all existing data in the database)
python label_pizza/manage_db.py \
  --mode init \
  --database-url-name DBURL \
  --email admin1@example.com \
  --password admin111 \
  --user-id "Admin 1"

# Import data from example folder
python sync_from_folder.py \
  --database-url-name DBURL \
  --folder-path ./example
```

This single command imports everything in the `example/` folder — videos, users, question groups, schemas, projects, and even sample annotations — so you get a fully‑working demo in seconds. If you just want to see Label Pizza in action, run it and explore the UI. When you’re ready to tailor the workflow to your own data, continue with the rest of this guide to learn how to batch‑upload users, videos, question groups, schemas, and projects.

---

The rest of this README explains the detailed folder structure, JSON formats, and step‑by‑step process for anyone who wants to learn how to batch‑upload their own projects.

## Folder Structure

> This directory provides a compact, end‑to‑end example of the files required to set up a video‑annotation workflow. Copy whichever pieces you already have, adjust the JSON to match your questions and videos, and import them with the project‑creation scripts. Any missing parts (e.g., annotations) can always be added later through the web interface.

```
example/
├── videos.json
├── question_groups/
│   ├── humans.json
│   ├── pizzas.json
│   └── nsfw.json
├── schemas.json
├── users.json
├── projects.json
├── project_groups.json
├── assignments.json
├── annotations/
│   ├── humans.json
│   ├── pizzas.json
│   └── nsfw.json
└── ground_truths/
    ├── humans.json
    ├── pizzas.json
    └── nsfw.json
```

## Folder Structure / JSON Format

### `videos.json`

Each entry represents one video. See the example file ([videos.json](example/videos.json)) below.

```json
[
  {
    "video_uid": "human.mp4",
    "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/human.mp4",
    "is_active": true,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=L3wKzyIN1yk",
      "license": "Standard YouTube License"
    }
  },
  {
    "video_uid": "pizza.mp4",
    "url": "https://huggingface.co/datasets/syCen/example4labelpizza/resolve/main/pizza.mp4",
    "is_active": true,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=8J1NzjA9jNg",
      "description": "A video of a pizza being made.",
      "license": "Standard YouTube License"
    }
  },
  {
    "video_uid": "human2.mp4",
    "url": "https://www.youtube.com/embed/2vjPBrBU-TM?start=33&end=36",
    "is_active": true,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=2vjPBrBU-TM",
      "license": "Standard YouTube License"
    }
  },
  {
    "video_uid": "human3.mp4",
    "url": "https://www.youtube.com/embed/2S24-y0Ij3Y?start=110&end=112",
    "is_active": true,
    "metadata": {
      "original_url": "https://www.youtube.com/watch?v=2S24-y0Ij3Y",
      "license": "Standard YouTube License"
    }
  }
]
```

### **What is a video?**

The `url` field can point to any video resource—an MP4 file hosted on Hugging Face, S3, or another server, or even a YouTube link—so long as the link resolves to playable video content.

* **`video_uid`** is a unique, permanent identifier. Once a video is created, its `video_uid` never changes.
* **`url`** may be updated later (we will provide instructions for safe updates).
* **`is_active`** defaults to `true`. Setting it to `false` cleanly archives the video across the platform while retaining its record. For robustness, we do not offer a way to delete videos (and similary for other assets we will discuss later in this guide such as users, questions, projects, and etc.).
* Everything inside **`metadata`** remains exactly as supplied for provenance.
* You may supply a YouTube link in `url`. To reference a specific time span, use the YouTube embed format with `start` and `end` parameters, as illustrated above for `human2.mp4` and `human3.mp4`.

### `question_groups/*.json`

Each JSON file under `question_groups/` defines *one* group of logically related questions. The example below ([question_groups/humans.json](example/question_groups/humans.json)) includes a single-choice question and a description question that together ask annotators to state how many people appear in a video and, if any, to describe them.

```json
{
    "title": "Human",
    "display_title": "Human",
    "description": "Detect and describe all humans in the video.",
    "is_reusable": false,
    "is_auto_submit": false,
    "verification_function": "check_human_description",
    "is_active": true,
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
            "default_option": null
        }
    ]
}
```

### **What is a question?**

Each item in the **`questions`** list defines a question shown to annotators. A question can either require choosing one answer from a list (**`single`**) or typing a free-text response (**`description`**).

#### Fields inside each item of the `questions` list

* **`qtype`** – `single` for single‑choice questions, `description` for free text.
* **`text`** – immutable unique identifier for the question; once created, it cannot be changed.
* **`display_text`** – if you want to change the phrasing of the question, update this field. (Custom per-video questions are also supported — this will be covered later in the guide.)
* **`default_option`** – the answer pre-loaded when the task opens. Use `null` if you don’t want to pre-select an answer.

For single-choice questions, you should also define the following fields:

* **`options`** – a list of internal values for each answer choice. These are immutable once created (i.e. you cannot change or delete them later; however, you can add new options, which we cover later in the guide).
* **`display_values`** – a list of human-readable labels shown to annotators. These can be edited later in the web UI.
* **`option_weights`** – an optional list of weights (floats) used when aggregating answers across multiple annotators. You can assign more weight to certain options to make them count more in majority voting. If you do not provide this list (`option_weights` is `null`), all options will have an equal weight of 1.

**Note:** These three lists must share the same length, and entries at the same index correspond to the same choice (i.e., `options[i]`, `display_values[i]`, and `option_weights[i]` all refer to the same choice).

### **What is a question group?**

A question group is an ordered list of logically related **`questions`** shown together on the annotation screen. For example, the group in the JSON example above contains one single-choice question and one description question that together ask annotators to report how many people appear in a video and, if any, to describe them.

#### Question‑group fields

* **`title`** – immutable unique identifier for the question group; once created, it cannot be changed.
* **`display_title`** – editable question group name shown to annotators (tip: use 1–3 words for best UI display).
* **`description`** – optional description text.
* **`is_reusable`** – whether this group can be reused across multiple schemas (a schema is a set of question groups, introduced next).
* **`is_auto_submit`** – if `true`, default answers are automatically submitted when the video loads. This is useful when one answer applies to the vast majority of videos — for example, if 99.9% of your videos are safe, you can auto-submit “No” to the NSFW question so annotators only need to correct it for the rare unsafe videos.
* **`is_active`** – archive flag; set to `false` to hide this question group while preserving history.
* **`verification_function`** – if not `null`, you can provide the name of a Python function defined in [verify.py](label_pizza/verify.py) that checks the consistency of answers before they’re submitted. If the check fails, the function should raise an error to block submission. For example, `check_human_description` ensures that if the number of people is greater than 0, the annotator must provide a non-empty description:

```python
def check_human_description(answers: Dict[str, str]) -> None:
    """Ensure the description answer is consistent with the count answer.

    Args:
        answers: Dictionary mapping question text to answer value
      
    Raises:
        ValueError: If the description is provided when there are no people, or if the description is not provided when there are people.
    """
    num_people = answers.get("Number of people?")
    description = answers.get("If there are people, describe them.")
    
    if num_people == "0" and description:
        raise ValueError("Description cannot be provided when there are no people")
    if num_people != "0" and not description:
        raise ValueError("Description must be provided when there are people")
```

### `schemas.json`

Each schema includes one or more question groups that together define an annotation task. The example below ([schemas.json](example/schemas.json)) includes two schemas:

```json
[
  {
    "schema_name": "Questions about Humans",
    "instructions_url": "https://en.wikipedia.org/wiki/Human",
    "question_group_names": [
      "Human", "NSFW"
    ],
    "has_custom_display": false,
    "is_active": true
  },
  {
    "schema_name": "Questions about Pizzas",
    "instructions_url": "https://en.wikipedia.org/wiki/Pizza",
    "question_group_names": [
      "Pizza", "NSFW"
    ],
    "has_custom_display": false,
    "is_active": true
  }
]
```

### **What is a schema?**

A **schema** defines the full set of questions shown to annotators for a given task. It consists of one or more previously defined question groups, bundled together in a specific order. For example, the first schema in the JSON above combines the `"Human"` and `"NSFW"` question groups into a single annotation interface for videos about humans.

Each schema also includes optional instructions and display settings.

#### Schema fields

* **`schema_name`** – unique identifier for the schema.
* **`instructions_url`** – optional URL pointing to external instructions (e.g., Google Doc, PDF, etc.) for annotators. This link will be shown at the top of the task screen.
* **`question_group_names`** – an ordered list of question group `titles`.
* **`has_custom_display`** – if `true`, allows each video to override `display_text` (question wording) and `display_values` (answer choices). In other words, different videos in the same project can show different questions or options. We explain how to use this later in the guide.
* **`is_active`** – archive flag; set to `false` to hide the schema while keeping its record.

### `users.json`

Each entry defines a user account. See the example file ([users.json](example/users.json)) below:

```json
[
    {
        "user_id": "Admin 1",
        "email": "admin1@example.com",
        "password": "admin111",
        "user_type": "admin",
        "is_active": true
    },
    {
        "user_id": "User 1",
        "email": "user1@example.com",
        "password": "user111",
        "user_type": "human",
        "is_active": true
    },
    {
        "user_id": "Robot 1",
        "password": "robot111",
        "user_type": "model",
        "is_active": true
    }
]
```

### **What is a user?**

Each user must have a unique `user_id`. If the user is a human or admin, they must also have a unique `email` for logging in. Model users do not use email and cannot log in through the UI.

* **`user_id`** – unique identifier, displayed in the UI.
* **`email`** – required only for human/admin users; must be unique.
* **`password`** – password for the user.
* **`user_type`** – must be one of: `human`, `admin`, or `model`.
  * You may switch between `human` and `admin` later.
  * Switching from `human` to `admin` gives the user access to **all** projects.
  * Downgrading from `admin` to `human` removes access to **all** previously assigned projects.
  * `model` users are fixed — you cannot switch between `model` and `human/admin`, since only model users support confidence scores.
* **`is_active`** – archive flag; set to `false` to deactivate the account while keeping its record.

> Admins automatically gain access to all projects with full privileges. Only assign `admin` to trusted stakeholders.

### `projects.json`

Each project applies a schema to a list of videos. See the example file ([projects.json](example/projects.json)) below:

```json
[
  {
    "project_name": "Human Test 0",
    "schema_name": "Questions about Humans",
    "description": "Test project for humans",
    "is_active": true,
    "videos": [
      "human.mp4",
      "pizza.mp4"
    ]
  },
  {
    "project_name": "Pizza Test 0",
    "schema_name": "Questions about Pizzas",
    "description": "Test project for pizzas",
    "is_active": true,
    "videos": [
      "human.mp4",
      "pizza.mp4"
    ]
  },
  {
    "project_name": "Human Test 1",
    "schema_name": "Questions about Humans",
    "description": "Test project for humans",
    "is_active": true,
    "videos": [
      "human2.mp4",
      "human3.mp4"
    ]
  }
]
```

### **What is a project?**

A **project** defines one labeling task: a schema applied to a set of videos.

* **`project_name`** – unique identifier for the project; once created, it cannot be changed.
* **`schema_name`** – the name of the schema used (from `schemas.json`).
* **`description`** – optional text shown in the UI.
* **`videos`** – list of video `video_uid`s included in the task.
* **`is_active`** – archive flag; set to `false` to hide this project while preserving history.

> It is also possible to customize the displayed question text (`display_text`) and answer options (`display_values`) for each video in this `videos` list. We explain this later in the guide.

### `project_groups.json`

Organizes multiple related projects into a named group. See the example file ([project\_groups.json](example/project_groups.json)) below:

```json
[
  {
    "project_group_name": "Example Project Group",
    "description": "This is a project group for human test",
    "projects": [
      "Human Test 0",
      "Human Test 1"
    ]
  }
]
```

### **What is a project group?**

A **project group** acts like a folder to organize related projects in the UI. This helps keep projects visually grouped on the website and simplifies answer export later on.

* **`project_group_name`** – unique name for the group.
* **`description`** – optional text.
* **`projects`** – list of project names (must match entries in `projects.json`).

> Two projects can only be grouped together if their schemas don’t share any non-reusable question groups. This avoids duplicate answers for the same video-question pair. Reusable question groups are allowed, but during export, their answers must match across projects. If not (e.g., one project marks a video safe and another marks it unsafe), the export will fail until the conflict is resolved.

### `assignments.json`

Grants a role to a user within a project. See the example file ([assignments.json](example/assignments.json)) below:

```json
[
  {
    "user_name": "User 1",
    "project_name": "Pizza Test 0",
    "role": "annotator",
    "user_weight": 1.0,
    "is_active": true
  },
  {
    "user_name": "User 1",
    "project_name": "Human Test 0",
    "role": "annotator",
    "user_weight": 1.0,
    "is_active": true
  }
]
```

### **What is an assignment?**

Each assignment gives a user a role (`annotator`, `reviewer`, `admin`, or `model`) for a specific project. A user can have different roles across different projects.

#### Supported roles

* **`annotator`** – submits first-pass answers. Can be a human or model user.
* **`reviewer`** – reviews annotator answers and sets ground truth. Reviewers may speed up review using majority voting among selected annotators: if all selected annotators agree on the same option and their combined weights (`user_weight` x `option_weight`) exceed a threshold, the answer can be auto-submitted without further review. Reviewers also have annotator access to the same project.
* **`admin`** (or **`meta-reviewer`**) – performs a final audit of reviewer answers. Admins automatically have annotator and reviewer access across all projects and do not require explicit assignments.
* **`model`** – behaves like an annotator but may include confidence scores. Once created, model users cannot be changed to human or admin.

#### Assignment fields

* **`user_name`** – must match a user in `users.json`.
* **`project_name`** – must match a project in `projects.json`.
* **`role`** – defines access level within the project. Admins do not need to be explicitly assigned.
* **`user_weight`** – optional float used in weighted majority voting. If not specified (`user_weight` is `null`), defaults to 1.0.
* **`is_active`** – archive flag; set to `false` to deactivate the assignment.

> For single-choice questions, we compute the weighted vote for each option as the sum of `user_weight × option_weight`. An option is auto-submitted only if it crosses a threshold, helping reviewers skip obvious cases.

### `annotations/*.json` and `ground_truths/*.json`

These folders store all answers submitted by annotators and reviewers:

* **`annotations/*.json`** holds first-pass answers from annotators (human or model).
* **`ground_truths/*.json`** stores the final reviewed answers. Only one ground-truth submission is allowed per *(video, project, question group)* triplet.

> A single *(video, project, question group)* triplet can have **many** annotator submissions, but **only one** reviewer-set ground truth.

For clarity, this guide organizes each file by question group (e.g., [annotations/humans.json](example/annotations/humans.json)). But you can also combine everything into a single file (e.g., `annotations/all.json` and `ground_truths/all.json`) or split by project or user.

#### Example folder structure

```
annotations/
├── humans.json
├── pizzas.json
└── nsfw.json

ground_truths/
├── humans.json
├── pizzas.json
└── nsfw.json
```

Each JSON file is a list of dictionaries. Each dictionary represents one submission: a user's answers to all questions in a **question group**, for a specific video and project. We show an example below from [annotations/humans.json](example/annotations/humans.json):

```json
[
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "User 1",
    "video_uid": "human.mp4",
    "answers": {
      "Number of people?": "1",
      "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
    },
    "is_ground_truth": false
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
    "is_ground_truth": false
  }
]
```

**What is an (annotator) answer?**

Each answer is a complete set of responses from a user to all questions within a single question group, for a specific video and project.

- (**`question_group_title`**, **`project_name`**, **`user_name`**, **`video_uid`**) – uniquely defines an answer submission.
- **`answers`** – a dictionary of question names and their corresponding answers.
- **`confidence_scores`** (optional; only for model users) – a dictionary of question names and their corresponding confidence scores.
- **`is_ground_truth`** - must be `false` for annotator answers.


For ground truth answers submitted by reviewers, refer to the following example [ground_truths/humans.json](example/ground_truths/humans.json):


```json
[
  {
    "question_group_title": "Human",
    "project_name": "Human Test 0",
    "user_name": "Admin 1",
    "video_uid": "human.mp4",
    "answers": {
      "Number of people?": "1",
      "If there are people, describe them.": "The person appears to be a large man with a full beard and closely cropped hair."
    },
    "is_ground_truth": true
  }
]
```

**What is a (reviewer) ground truth?**

A ground-truth record is the **single final answer** chosen by a reviewer for one video, project, and question group. Its JSON looks just like an annotator answer, but it sets `"is_ground_truth": true`. There can be only **one** ground-truth entry for each *(video, project, question group)*.

- (**`question_group_title`**, **`project_name`**, **`video_uid`**) – uniquely defines a ground truth submission.
- **`answers`** – a dictionary of question names and their corresponding answers.
- **`is_ground_truth`** - must be `true` for ground truths.


---

[← Back to start](start_here.md) | [Next → Sync Workflows](sync_workflows.md)





