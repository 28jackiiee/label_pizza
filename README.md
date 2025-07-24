# 🍕 **Label Pizza**

*A training-aware annotation platform for high-quality video-language data curation*

Label Pizza lets you upload video collections, teach annotators the exact policy you care about, and run two quality-control passes before admins export the answers—all inside one Streamlit interface. 

> Use the same workspace to **coach new annotators**, **collect fresh ground truth**, **audit reviewer decisions**, and **inspect accuracy analytics**.

---

## 📚 Table of Contents

1. [Core Concepts](#core-concepts)
2. [Why Label Pizza?](#why-label-pizza)
3. [Life of a Video](#life-of-a-video-📽️)
4. [Unique Features](#unique-features)
5. [Schema Design](#schema-design)
6. [Quick Setup](#quick-setup-in-3-minutes)
7. [Folder Layout](#folder-layout)
8. [Next Steps](#next-steps)

---

## Core Concepts

| Concept            | What it is                                                                                                                                                                                  | Why it matters                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Video**          | A single clip and its metadata.                                                                                                                                                             | Smallest unit of work.                                              |
| **Question**       | A prompt answered by either **single-choice** buttons or a **description** text box.                                                                                                        | Defines exactly what annotators must provide.                       |
| **Question Group** | A tab that bundles related questions (e.g. *Camera Motion*). It can include a **verification rule**—for example, *“If the option **Other** is chosen, the description must not be empty.”*  | Keeps tasks modular and prevents invalid answer combinations.       |
| **Schema**         | An ordered collection of question groups—the formal labeling policy.                                                                                                                        | Lets multiple projects share one policy and stay consistent.        |
| **Project**        | A schema applied to a chosen set of videos with per-user roles. A project runs in **Training** mode when every question already has ground truth; otherwise it runs in **Annotation** mode. | Tracks answers, progress, and accuracy.                             |
| **Project Group**  | A named folder that bundles related projects.                                                                                                                                               | Powers dashboards and bulk exports.                                 |

---

## Why Label Pizza?

### 1 · Use the Platform **as a Classroom with Instant Feedback**

High-stakes, post-training datasets fail if annotators interpret the policy differently, resulting in low inter-annotator agreement. Label Pizza fixes this by offering a **Training mode** for any project with existing ground-truth labels:

Annotators work through the videos and, after each submission, see an **instant ✅ / ❌ feedback** plus the ground truth answer.

Because training happens directly within the same UI they'll use for real tasks, new hires quickly become consistent with the labeling policy through hands-on annotation.

---

### 2 · **Tiered Quality Control** for Accurate Annotations

Unlike pre-training datasets, post-training datasets are used to evaluate or fine-tune models—so even small labeling errors can lead to misleading results. As such, Label Pizza enforces a tiered quality check process to ensure every answer is reliable.

| Tier                          | What happens                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| **Annotator** (human or model) | Creates first-pass answers. A model-type user can attach a confidence score, enabling active-learning workflows such as video sorting and batch auto-submit.                                                   |
| **Reviewer**                   | 	Reviews annotator answers and sets ground truth. Can accelerate the process by auto-submitting answers that meet a high consensus threshold among a selected set of annotators. |
| **Meta-Reviewer / Admin**      | Performs a final audit, overrides ground truth when necessary, and closes edge cases. The full history is stored so reviewer and annotator accuracy can be inspected at any stage.       |

---

## Life of a Video 📽️

1. **Annotator** (**human** or **model**) submits answers for videos in a project.
2. **Reviewer** preloads the most likely answers and, if satisfied, commits them with **Auto-Submit** or edit manually.
3. **Meta-Reviewer / Admin** overrides reviewer decisions where needed.
4. When every question has ground truth, the project flips to **Training** for future annotators.

---

## Unique Features

Label Pizza includes powerful features typically locked behind expensive enterprise tools—like batch auto-submit and weighted consensus—offered here for free under an MIT license. (For comparison, tools like Label Studio Enterprise can cost ≈ $100k/year.)

| #     | Feature                         | Highlights                                                                                    |
| ----- | ------------------------------- | --------------------------------------------------------------------------------------------- |
| **1** | **Auto-Training Mode**          | Projects automatically switch from *annotation* to *training mode* once ground truth is complete by a reviewer, giving new annotators instant feedback to speed up learning.      |
| **2** | **Auto-Submit Features**   | Reviewers can auto-submit answers based on weighted consensus, prioritizing expert annotators, model predictions, or high-stake options when there’s disagreement.    |
| **3** | **Performance Leaderboard** | One-click access to human and model leaderboards and per-question performance breakdowns. |
| **4** | **Answer Validation Rules** | Custom Python rules prevent invalid answers—for example, selecting “Other (to describe)” requires a non-empty description.  |
| **5** | **Search & Filter**    | Search for any video by answer pattern or view how a video was labeled across all projects.  |


---

## Schema Design

Below is an example of a schema with two question groups, each containing two questions:
```
Camera Basics
├─ Shot Size            (single-choice)
└─ Camera Angle         (single-choice)
Lighting
├─ Key Light present?   (single-choice, verification rule)
└─ Describe Key Light   (description)
```

* **Question** – A single input field, such as radio buttons or a text box.
* **Question Group** – A set of related questions shown together in one UI tab. Can be reused across projects or set to auto-submit defaults. May include optional Python verification checks triggered on submission—e.g., if “Key Light present?” is answered Yes, then “Describe Key Light” must not be empty.
* **Schema** – The full tree structure composed of one or more question groups.

---

## Quick Setup (in 3 minutes)

### 1. Clone this repo and install packages

```bash
# clone and set up
git clone https://github.com/linzhiqiu/label_pizza.git
cd label_pizza
pip install -e . # Install all required packages such as streamlit
```

### 2. Create a free or paid Postgres DB (Supabase recommended)

| Supbase Plan | Storage | Notes |
|------|---------|-------|
| **Free tier** | 0.5 GB | Ideal for pilots and small teams |
| **Pay-as-you-go** | +10 GB ≈ US $20/mo | Scale up later without migrations |

**Steps**

1. Sign up at **https://supabase.com → Start Your Project**.  
2. Create a new *Project* (choose the free tier).  
3. In **Project → Settings → Database**, copy the **Connection string** (starts with `postgresql://`).  
4. In the repo root, drop that string into an `.env` file with one command:

```bash
echo 'DBURL=postgresql://<user>:<password>@<host>:<port>/<database>' > .env
```
(Replace the angled-bracket values with the ones Supabase shows.)

### 3 · Initialize database, seed an admin user, and launch the app!

```bash
# create the first Admin account (change the args to your own)
python label_pizza/manage_db.py \
  --mode init \
  --database-url-name DBURL \
  --email admin@example.com \
  --password MyPassword! \
  --user-id "Admin User"
````

```bash
# start Streamlit; feel free to change the port
streamlit run label_pizza/label_pizza_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  -- \
  --database-url-name DBURL
```

Visit **[http://localhost:8000](http://localhost:8000)** to log in.

> **Want to share the site externally?**
> Pipe the local port through **[pinggy.io](https://pinggy.io/)** (≈ US \$3 per static URL per month)


### 4 · Load your own videos, questions, projects, and users

Ready to use your own data? Start by learning the key concepts behind Label Pizza (e.g., videos, questions, schemas, and users) in **[docs/start_here.md](docs/start_here.md)** to set up your first project.


## Folder Layout

```
label_pizza/
├─ label_pizza_app.py    # Streamlit interface
├─ models.py             # Database tables
├─ services.py           # Business logic
├─ sync_utils.py         # Upload data to the database
└─ manage_db.py          # Init/backup/reset the database with a seed admin user
sync_from_folder.py      # Sync data from a folder
workspace/               # Your own data folder
example/                 # Example data folder
example_custom_question/ # Example data folder for custom question per video
```

Enjoy your slice of perfectly-topped labels! 🍕

## Next Steps

* **Get started with the web UI** – follow the setup guide in 
  [docs/start_here.md](docs/start_here.md)
* **Understand the core concepts** – understand the terminology in 
  [docs/data_model.md](docs/data_model.md)
* **Import, update, or backup data** – follow the helper examples in  
  [docs/sync_workflows.md](docs/sync_workflows.md)
* **Custom questions or options per video** – see  
  [docs/custom_display.md](docs/custom_display.md)