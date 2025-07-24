# Label Pizza — Start Here

Welcome! In about **20 minutes** you’ll learn the core concepts behind Label Pizza and get a project up and running.
The guides below build on one another; we recommend reading the first three in order, then jumping to the advanced topics as needed.

| Step  | Guide (file)       | What you’ll get out of it                                                                                          | Est. time        |
| ----- | ------------------ | ------------------------------------------------------------------------------------------------------------------ | ---------------- |
| **1** | **[start_here.md](start_here.md)** | Install dependencies, spin up the demo database, and launch the Streamlit UI.                                      | **3 min**        |
| 2     | [data_model.md](data_model.md)     | Understand the core concepts: videos, users, question groups, schemas, projects, assignments, and answers.         | 8 min            |
| 3     | [sync_workflows.md](sync_workflows.md) | Learn how to sync data from JSON dictionaries or folders into the database.                                        | 8 min            |
| 4     | [custom_display.md](custom_display.md) | Override question wording on a per‑video basis using a shared schema.                                              | 6 min (optional) |
| 5     | [admin_override.md](admin_override.md) | Perform dangerous override operations: rename immutable IDs or permanently delete records (with automatic backup). | 5 min (optional) |


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

---

[← Back to README](../README.md) | [Next → Data Model](data_model.md)


