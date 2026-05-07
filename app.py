import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from transformers import pipeline

from smolagents import CodeAgent, InferenceClientModel, tool

# =======================
#  CONFIG & ENVIRONMENT
# =======================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is not set in environment or .env")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in environment or .env")

DB_PATH = "youtube_sentiment.db"
N_VIDEOS = 10          # as per project spec: latest 10 uploads
MAX_COMMENTS = 500     # per video, you can adjust

# MrBeast main channel ID (you can also pass '@MrBeast' as input)
DEFAULT_CHANNEL_INPUT = "@MrBeast"  # or 'UCX6OQ3DkcsbYNE6H8uQQuVA'[web:16][web:21]

# =======================
#  DB SCHEMA
# =======================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS channels (
        channel_id   TEXT PRIMARY KEY,
        channel_title TEXT,
        channel_url   TEXT,
        retrieved_at  TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id      TEXT PRIMARY KEY,
        channel_id    TEXT,
        video_title   TEXT,
        published_at  TEXT,
        like_count    INTEGER,
        comment_count INTEGER,
        video_url     TEXT,
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        comment_id       TEXT PRIMARY KEY,
        video_id         TEXT,
        author_name      TEXT,
        comment_text     TEXT,
        like_count       INTEGER,
        published_at     TEXT,
        sentiment_label  TEXT,
        sentiment_score  REAL,
        raw_language     TEXT,
        is_valid         INTEGER,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_summary (
        video_id           TEXT PRIMARY KEY,
        channel_id         TEXT,
        total_comments     INTEGER,
        pos_count          INTEGER,
        neg_count          INTEGER,
        neu_count          INTEGER,
        pos_pct            REAL,
        neg_pct            REAL,
        neu_pct            REAL,
        avg_sentiment_score REAL
    );
    """)

    conn.commit()
    conn.close()


# =======================
#  YOUTUBE CLIENT
# =======================

def get_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def extract_channel_id_from_input(youtube, channel_input: str) -> str:
    """
    Accepts handle (@MrBeast), channel URL, or raw channel ID and returns channel_id.
    """
    channel_input = channel_input.strip()

    # Direct channel ID
    if channel_input.startswith("UC") and len(channel_input) >= 10:
        return channel_input

    # URL cases
    if "youtube.com" in channel_input:
        if "/channel/" in channel_input:
            m = re.search(r"/channel/([^/?]+)", channel_input)
            if m:
                return m.group(1)
        if "/@" in channel_input:
            handle = channel_input.split("/@")[-1]
        else:
            handle = channel_input.split("/")[-1]
    elif channel_input.startswith("@"):
        handle = channel_input[1:]
    else:
        handle = channel_input

    # Resolve handle via search
    request = youtube.search().list(
        part="snippet",
        q=handle,
        type="channel",
        maxResults=1
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        raise ValueError(f"Channel not found for input: {channel_input}")
    return items[0]["snippet"]["channelId"]


def get_uploads_playlist_id(youtube, channel_id: str) -> str:
    request = youtube.channels().list(
        part="contentDetails,snippet",
        id=channel_id
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        raise ValueError(f"No channel data for channel_id: {channel_id}")
    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return uploads_playlist_id


def get_latest_videos(youtube, uploads_playlist_id: str, n_videos: int = 10):
    videos = []
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=min(n_videos, 50)
    )
    while request and len(videos) < n_videos:
        response = request.execute()
        for item in response.get("items", []):
            vid = {
                "video_id": item["contentDetails"]["videoId"],
                "video_title": item["snippet"]["title"],
                "published_at": item["contentDetails"].get("videoPublishedAt")
            }
            videos.append(vid)
            if len(videos) >= n_videos:
                break
        request = youtube.playlistItems().list_next(request, response)
    return videos


def fetch_video_comments(youtube, video_id: str, max_comments: int = 500):
    """
    Fetch top-level comments for a video using commentThreads.list.
    Handles pagination until max_comments reached.[web:14]
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request:
        response = request.execute()
        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "author_name": top_comment.get("authorDisplayName"),
                "comment_text": top_comment.get("textDisplay"),
                "like_count": top_comment.get("likeCount", 0),
                "published_at": top_comment.get("publishedAt")
            })
            if len(comments) >= max_comments:
                return comments
        request = youtube.commentThreads().list_next(request, response)
    return comments


# =======================
#  SENTIMENT MODEL
# =======================

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)  # POSITIVE / NEGATIVE labels[web:15]


def clean_comment_text(text: str) -> str:
    if text is None:
        return ""
    return text.replace("\n", " ").strip()


def analyze_comment_sentiment(text: str):
    text = clean_comment_text(text)
    if not text:
        return None, None

    # Let the tokenizer truncate to the model's max length (512 tokens for DistilBERT)
    result = sentiment_pipeline(
        text,
        truncation=True,
        max_length=512  # safe for distilbert-base-uncased-finetuned-sst-2-english
    )[0]

    return result["label"], float(result["score"])


# =======================
#  DB UPSERT HELPERS
# =======================

def upsert_channel(channel_id, channel_title, channel_url):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO channels (channel_id, channel_title, channel_url, retrieved_at)
        VALUES (?, ?, ?, ?)
    """, (channel_id, channel_title, channel_url, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def upsert_video(video):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO videos
        (video_id, channel_id, video_title, published_at, like_count, comment_count, video_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        video["video_id"],
        video["channel_id"],
        video["video_title"],
        video.get("published_at"),
        video.get("like_count", 0),
        video.get("comment_count", 0),
        f"https://www.youtube.com/watch?v={video['video_id']}"
    ))
    conn.commit()
    conn.close()


def upsert_comment(comment, sentiment_label=None, sentiment_score=None, is_valid=1):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO comments
        (comment_id, video_id, author_name, comment_text, like_count, published_at,
         sentiment_label, sentiment_score, raw_language, is_valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        comment["comment_id"],
        comment["video_id"],
        comment.get("author_name"),
        clean_comment_text(comment.get("comment_text", "")),
        comment.get("like_count", 0),
        comment.get("published_at"),
        sentiment_label,
        sentiment_score,
        None,
        is_valid
    ))
    conn.commit()
    conn.close()


# =======================
#  AGGREGATION & REPORTS
# =======================

def compute_sentiment_summary():
    conn = sqlite3.connect(DB_PATH)
    comments_df = pd.read_sql_query("SELECT * FROM comments WHERE is_valid = 1", conn)
    videos_df = pd.read_sql_query("SELECT * FROM videos", conn)
    conn.close()

    if comments_df.empty:
        print("No comments to summarize.")
        return pd.DataFrame()

    def bucket(label):
        if label == "POSITIVE":
            return "pos"
        elif label == "NEGATIVE":
            return "neg"
        else:
            return "neu"

    comments_df["bucket"] = comments_df["sentiment_label"].apply(bucket)

    grouped = comments_df.groupby("video_id").agg(
        total_comments=("comment_id", "count"),
        pos_count=("bucket", lambda x: (x == "pos").sum()),
        neg_count=("bucket", lambda x: (x == "neg").sum()),
        neu_count=("bucket", lambda x: (x == "neu").sum()),
        avg_sentiment_score=("sentiment_score", "mean")
    ).reset_index()

    grouped["pos_pct"] = grouped["pos_count"] / grouped["total_comments"]
    grouped["neg_pct"] = grouped["neg_count"] / grouped["total_comments"]
    grouped["neu_pct"] = grouped["neu_count"] / grouped["total_comments"]

    merged = grouped.merge(videos_df[["video_id", "channel_id", "video_title"]], on="video_id", how="left")

    conn = sqlite3.connect(DB_PATH)
    merged.to_sql("sentiment_summary", conn, if_exists="replace", index=False)
    conn.close()

    return merged


def channel_level_summary():
    conn = sqlite3.connect(DB_PATH)
    summary_df = pd.read_sql_query("""
        SELECT channel_id,
               SUM(total_comments) AS total_comments,
               SUM(pos_count) AS pos_count,
               SUM(neg_count) AS neg_count,
               SUM(neu_count) AS neu_count
        FROM sentiment_summary
        GROUP BY channel_id
    """, conn)
    conn.close()

    if summary_df.empty:
        return summary_df

    summary_df["pos_pct"] = summary_df["pos_count"] / summary_df["total_comments"]
    summary_df["neg_pct"] = summary_df["neg_count"] / summary_df["total_comments"]
    summary_df["neu_pct"] = summary_df["neu_count"] / summary_df["total_comments"]
    return summary_df


def export_csvs():
    conn = sqlite3.connect(DB_PATH)
    for table in ["channels", "videos", "comments", "sentiment_summary"]:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(f"{table}.csv", index=False)
    conn.close()


# =======================
#  MAIN PIPELINE
# =======================

def run_pipeline_for_channel_input(channel_input: str,
                                   n_videos: int = N_VIDEOS,
                                   max_comments_per_video: int = MAX_COMMENTS):
    init_db()
    youtube = get_youtube_client()

    # Resolve channel
    channel_id = extract_channel_id_from_input(youtube, channel_input)
    channel_resp = youtube.channels().list(part="snippet", id=channel_id).execute()
    channel_item = channel_resp["items"][0]["snippet"]
    channel_title = channel_item["title"]
    channel_url = f"https://www.youtube.com/channel/{channel_id}"

    upsert_channel(channel_id, channel_title, channel_url)

    # Uploads playlist and videos
    uploads_playlist_id = get_uploads_playlist_id(youtube, channel_id)
    videos = get_latest_videos(youtube, uploads_playlist_id, n_videos=n_videos)

    print(f"Found {len(videos)} videos for channel '{channel_title}' ({channel_id}).")

    for v in videos:
        v["channel_id"] = channel_id
        upsert_video(v)

        print(f"Fetching comments for video: {v['video_title']} ({v['video_id']})")
        try:
            vid_comments = fetch_video_comments(youtube, v["video_id"], max_comments=max_comments_per_video)
        except Exception as e:
            print(f"Failed to fetch comments for {v['video_id']}: {e}")
            continue

        for c in vid_comments:
            label, score = analyze_comment_sentiment(c["comment_text"])
            is_valid = 1 if label is not None else 0
            upsert_comment(c, sentiment_label=label, sentiment_score=score, is_valid=is_valid)

    video_summary = compute_sentiment_summary()
    channel_summary = channel_level_summary()
    export_csvs()

    summary_text = f"Analyzed channel '{channel_title}' ({channel_id}). "
    if channel_summary is not None and not channel_summary.empty:
        row = channel_summary.iloc[0]
        summary_text += (
            f"Total comments: {int(row['total_comments'])}, "
            f"Positive: {row['pos_pct']:.1%}, "
            f"Negative: {row['neg_pct']:.1%}, "
            f"Neutral: {row['neu_pct']:.1%}."
        )
    else:
        summary_text += "No comments found for the latest videos."

    print(summary_text)
    return summary_text, video_summary, channel_summary


# =======================
#  SMOLAGENTS TOOL + AGENT
# =======================

from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def run_channel_sentiment_pipeline(channel_input: str) -> str:
    """
    Run the full YouTube sentiment pipeline for a given channel.

    Args:
        channel_input (str): YouTube channel handle (e.g., "@MrBeast"),
            channel URL, or channel ID.

    Returns:
        str: Short analytical summary string describing the channel sentiment.
    """
    summary_text, _, _ = run_pipeline_for_channel_input(channel_input)
    return summary_text


def build_agent():
    # Qwen2.5 Coder via InferenceClientModel (Hugging Face Inference Providers).[web:3][web:29][web:41]
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=HF_TOKEN
    )
    agent = CodeAgent(
        tools=[run_channel_sentiment_pipeline],
        model=model,
        max_steps=4,
        stream_outputs=True
    )
    return agent


# =======================
#  CLI ENTRY
# =======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic YouTube Comment Sentiment Analysis and SQL Reporting System"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL_INPUT,
        help="YouTube channel handle (@MrBeast), URL, or channel ID (default: @MrBeast)"
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="If set, run via smolagents CodeAgent; otherwise run pipeline directly."
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=N_VIDEOS,
        help="Number of latest videos to analyze (default: 10)"
    )
    args = parser.parse_args()

    if args.use_agent:
        agent = build_agent()
        prompt = (
            f"Run sentiment analysis on the YouTube channel '{args.channel}' "
            f"using the run_channel_sentiment_pipeline tool and print the summary."
        )
        agent.run(prompt)
    else:
        run_pipeline_for_channel_input(args.channel, n_videos=args.videos, max_comments_per_video=MAX_COMMENTS)