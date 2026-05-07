# AI Agentic YouTube Comment Sentiment Analysis and SQL Reporting System

This project implements an end‑to‑end agentic AI pipeline that:

- Accepts a YouTube channel handle, URL, or channel ID as input.  
- Fetches the latest 10 uploaded videos for that channel using the YouTube Data API.  
- Extracts top‑level comments for each video.  
- Runs Hugging Face Transformers sentiment analysis on every comment.  
- Stores all data in a SQLite database with clear table relationships.  
- Generates video‑wise and channel‑level sentiment summaries and CSV reports.  

A Hugging Face smolagents CodeAgent orchestrates the full workflow through a single tool call, making the system an example of agentic AI + NLP + SQL integration in one project.[web:3]

---

## Features

- Input: YouTube channel handle (e.g., `@MrBeast`), URL, or channel ID.  
- Agentic orchestration using smolagents CodeAgent.  
- Latest 10 uploaded videos fetched via YouTube Data API v3.  
- Top‑level comment collection with pagination and error handling.  
- Sentiment analysis with `distilbert-base-uncased-finetuned-sst-2-english` via `pipeline("sentiment-analysis")`.[web:15]  
- SQL data model with `channels`, `videos`, `comments`, and `sentiment_summary` tables.  
- CSV exports for videos, comments, and summary metrics.  
- Designed to be reproducible in VS Code or Colab.

---

## Architecture Overview

High‑level components:

### 1. User / Client

- Provides channel input (`@handle`, URL, or `channel_id`).

### 2. Agent Layer (smolagents CodeAgent)

- Exposes a tool: `run_channel_sentiment_pipeline(channel_input: str)`.  
- When invoked, runs the entire pipeline end‑to‑end.

### 3. YouTube Data API v3

- `channels.list` – resolve channel and get content details.  
- `playlistItems.list` – fetch latest uploads.  
- `commentThreads.list` – fetch top‑level comments.

### 4. Python Ingestion & Processing

- Clean comment text, paginate, handle API errors.  
- Prepare records for database and model.

### 5. Sentiment Model (Hugging Face Transformers)

- `pipeline("sentiment-analysis")` with DistilBERT SST‑2.  
- Truncation enabled (`truncation=True, max_length=512`) for long comments.[web:15]

### 6. SQLite Database

- Stores `channels`, `videos`, `comments`, `sentiment_summary`.  
- Enables SQL analytics and consistent re‑runs.

### 7. Reporting & Exports

- Aggregates sentiment metrics per video and per channel.  
- Exports `channels.csv`, `videos.csv`, `comments.csv`, `sentiment_summary.csv`.

---

## Data Model (SQL Schema)

### `channels`

- `channel_id` (PRIMARY KEY)  
- `channel_title`  
- `channel_url`  
- `retrieved_at`

### `videos`

- `video_id` (PRIMARY KEY)  
- `channel_id` (FK → `channels.channel_id`)  
- `video_title`  
- `published_at`  
- `like_count`  
- `comment_count`  
- `video_url`

### `comments`

- `comment_id` (PRIMARY KEY)  
- `video_id` (FK → `videos.video_id`)  
- `author_name`  
- `comment_text`  
- `like_count`  
- `published_at`  
- `sentiment_label` (POSITIVE / NEGATIVE)  
- `sentiment_score` (float, confidence)  
- `raw_language` (optional)  
- `is_valid` (1 = usable text, 0 = invalid/empty)

### `sentiment_summary`

- `video_id` (PRIMARY KEY)  
- `channel_id`  
- `total_comments`  
- `pos_count`, `neg_count`, `neu_count`  
- `pos_pct`, `neg_pct`, `neu_pct`  
- `avg_sentiment_score`

---

## Requirements

- Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries:

- `google-api-python-client` – YouTube Data API.  
- `transformers`, `torch`, `accelerate` – Hugging Face Transformers, PyTorch backend.[web:15]  
- `smolagents` – agent framework.[web:3]  
- `python-dotenv` – environment variable loading.  
- `pandas`, `sqlalchemy` – data processing and SQL helpers.  
- `sqlite3` (standard library) – database engine.

---

## Configuration

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=YOUR_YOUTUBE_DATA_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
```

- `YOUTUBE_API_KEY`: from Google Cloud Console, with YouTube Data API v3 enabled.  
- `HF_TOKEN`: from your Hugging Face account (for smolagents model usage).

The code reads these with `python-dotenv`.

> Important: Do **not** commit your real `.env` to GitHub. Use `.gitignore` to exclude it.

---

## Running the Pipeline (Direct Mode)

From the project folder:

```bash
python app.py --channel '@MrBeast' --videos 10
```

Or with channel ID:

```bash
python app.py --channel UCX6OQ3DkcsbYNE6H8uQQuVA --videos 10
```

What happens:

1. Initializes `youtube_sentiment.db`.  
2. Resolves channel and gets uploads playlist.  
3. Fetches latest 10 videos.  
4. Extracts top‑level comments for each video.  
5. Runs sentiment analysis on each comment.  
6. Upserts data into SQL tables.  
7. Aggregates sentiment into `sentiment_summary`.  
8. Exports CSVs for all tables.

You should see log messages such as:

- `Found 10 videos for channel 'MrBeast' (...)`  
- `Fetching comments for video: ...`  
- Final analytical summary for the channel.

---

## Running via Agent (smolagents CodeAgent)

To let the agent orchestrate the pipeline:

```bash
python app.py --channel '@MrBeast' --videos 10 --use-agent
```

Internally:

- A `CodeAgent` is built with an `InferenceClientModel` and a tool `run_channel_sentiment_pipeline(channel_input: str)`.  
- The agent receives a natural language prompt and calls the tool, which triggers the same pipeline as direct mode.

---

## Outputs

After a successful run, the project directory contains:

- `youtube_sentiment.db` – SQLite database.  
- `channels.csv` – channel metadata.  
- `videos.csv` – latest videos.  
- `comments.csv` – comments plus sentiment scores.  
- `sentiment_summary.csv` – per‑video sentiment metrics.

You can open the database with DB Browser for SQLite or query it with `sqlite3`.

---

## Evaluation Metrics

The system supports these evaluation measures:

- **Pipeline completion rate** – Whether a pipeline run successfully completes all stages (input resolution, video retrieval, comment extraction, SQL storage, sentiment scoring, and reporting).  
- **Data capture quality** – Number of videos fetched, number of comments stored, and percentage of valid non‑empty comments that received a sentiment label.  
- **SQL design quality** – Correctness of table relationships, primary keys, and ability to query cross‑video and cross‑channel sentiment.  
- **Sentiment coverage** – Percentage of collected comments with a valid `sentiment_label` and `sentiment_score`.  
- **Reporting clarity** – How clearly per‑video and overall channel sentiment is exposed through CSVs and SQL queries.

---

## Limitations and Future Work

- Uses a binary English sentiment model (positive / negative), which may not capture nuance, sarcasm, or non‑English comments.[web:15]  
- Only top‑level comments are analyzed; replies and threads are not included.  
- The current pipeline is batch/manual. Future work could add scheduling or a web dashboard (e.g., Streamlit) for continuous monitoring and visualization.

---

## Project Structure (Typical)

```text
.
├─ app.py                      # Main pipeline + agent entrypoint
├─ requirements.txt            # Python dependencies
├─ .env.example                # Example environment variables (no real keys)
├─ README.md                   # Project documentation
├─ youtube_sentiment.db        # SQLite DB (generated)
├─ channels.csv                # Export (generated)
├─ videos.csv                  # Export (generated)
├─ comments.csv                # Export (generated)
└─ sentiment_summary.csv       # Export (generated)
```

You can adjust filenames or paths as needed, but keep the core flow the same to match the project specification.
