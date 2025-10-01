# app.py
import os, re, json, math, mimetypes, shutil, tempfile, time, traceback, subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------- Global config ---------------------------------
st.set_page_config(page_title="Auto Shorts Studio", page_icon="🎬", layout="wide")

APP_OUT = "downloads"
os.makedirs(APP_OUT, exist_ok=True)

# ----------------------------- Secrets + Env ---------------------------------
def _get_secret(name: str, default: str = "") -> str:
    """Prefer Streamlit secrets in cloud; fallback to env for local."""
    try:
        val = st.secrets.get(name, None)
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()

# ----------------------------- Azure config (centralized) --------------------
AZURE_API_KEY         = _get_secret("AZURE_API_KEY")
AZURE_ENDPOINT_RAW    = _get_secret("AZURE_ENDPOINT")
AZURE_API_VERSION     = _get_secret("AZURE_API_VERSION") or "2024-06-01"

# Model deployments
AZURE_CHAT_DEPLOYMENT     = _get_secret("AZURE_DEPLOYMENT") or "gpt-5-chat"
AZURE_WHISPER_DEPLOYMENT  = _get_secret("WHISPER_DEPLOYMENT") or "whisper-1"

def _normalize_openai_base(url: str) -> str:
    """Ensure endpoint uses *.openai.azure.com (not *.cognitiveservices.azure.com)."""
    if not url:
        return ""
    url = url.rstrip("/")
    if ".cognitiveservices.azure.com" in url:
        url = url.replace(".cognitiveservices.azure.com", ".openai.azure.com")
    return url

AZURE_ENDPOINT = _normalize_openai_base(AZURE_ENDPOINT_RAW)

AZURE_CHAT_URL = (
    f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_CHAT_DEPLOYMENT}/chat/completions"
    f"?api-version={AZURE_API_VERSION}"
)
AZURE_WHISPER_URL = (
    f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_WHISPER_DEPLOYMENT}/audio/transcriptions"
    f"?api-version={AZURE_API_VERSION}"
)

# Shared headers for chat calls
HEADERS = {"api-key": AZURE_API_KEY or "", "Content-Type": "application/json"}

# ----------------------------- Utilities -------------------------------------
def list_dir(path: str) -> List[str]:
    files = []
    p = Path(path)
    if not p.exists():
        return files
    for root, _, fns in os.walk(path):
        for fn in fns:
            files.append(os.path.relpath(os.path.join(root, fn), path))
    return sorted(files)

def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None

# ----------------------------- 1) YouTube Download ---------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

def yt_try_download(url: str, output_dir: str, opts: Dict[str, Any], label: str) -> Tuple[bool, str]:
    try:
        from yt_dlp import YoutubeDL
    except Exception:
        return False, "yt-dlp not installed. Run: pip install -U yt-dlp"

    os.makedirs(output_dir, exist_ok=True)
    base_opts: Dict[str, Any] = {
        "restrictfilenames": True,
        "paths": {"home": output_dir},
        "outtmpl": {"default": "%(title).180s.%(ext)s"},
        "merge_output_format": "mp4",  # let yt-dlp/ffmpeg produce mp4 when possible
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 5,
        "concurrent_fragment_downloads": 5,
        "continuedl": True,
        "nopart": False,
        "nocheckcertificate": True,
        "http_headers": {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
        "quiet": False,
        "no_warnings": False,
        "progress": True,
        "noplaylist": True,
    }
    final_opts = {**base_opts, **opts}
    try:
        with YoutubeDL(final_opts) as ydl:
            ydl.download([url])
        return True, f"✅ Success with: {label}"
    except Exception as e:
        return False, f"{label} failed: {e}\n{traceback.format_exc(limit=1)}"

def yt_build_attempts(use_cookies: bool, audio_only: bool) -> List[Tuple[str, Dict[str, Any]]]:
    fmt_best = "ba[ext=m4a]/bestaudio/best" if audio_only else "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best"
    attempts: List[Tuple[str, Dict[str, Any]]] = []
    if use_cookies:
        attempts.append(("Android client + Chrome cookies", {
            "format": fmt_best, "extractor_args": {"youtube": {"player_client": ["android"]}},
            "cookiesfrombrowser": ("chrome",),
        }))
    attempts.append(("Android client, no cookies", {
        "format": fmt_best, "extractor_args": {"youtube": {"player_client": ["android"]}},
    }))
    if use_cookies:
        attempts.append(("Web client + Chrome cookies", {"format": fmt_best, "cookiesfrombrowser": ("chrome",)}))
    attempts.append(("Web client, no cookies", {"format": fmt_best}))
    if not audio_only and use_cookies:
        attempts.append(("Legacy itag 22 MP4 + cookies", {"format": "22/best[ext=mp4]/best", "cookiesfrombrowser": ("chrome",)}))
    if not audio_only:
        attempts.append(("Legacy itag 22 MP4, no cookies", {"format": "22/best[ext=mp4]/best"}))
    # If audio-only, add explicit audio extract postprocessor
    if audio_only:
        for i in range(len(attempts)):
            lbl, opt = attempts[i]
            opt = {**opt, "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"}]}
            attempts[i] = (lbl, opt)
    return attempts

# ----------------------------- 2) Extract Audio ------------------------------
def _sh_out(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def _ffprobe_json(path: str) -> dict:
    return json.loads(_sh_out(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path]))

def _has_audio(meta: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in meta.get("streams", []))

def _duration_seconds(meta: dict) -> float:
    dur = meta.get("format", {}).get("duration")
    if dur is not None:
        try: return float(dur)
        except: pass
    for s in meta.get("streams", []):
        d = s.get("duration")
        if d:
            try: return float(d)
            except: pass
    return 0.0

def extract_audio(
    input_path: str,
    out_dir: str = "data/audio",
    rate: int = 16000,
    channels: int = 1,
    fmt: str = "wav",
    overwrite: bool = False,
) -> Tuple[bool, str, str]:
    try:
        inp = Path(input_path)
        if not inp.exists():
            return False, f"Not found: {inp}", ""
        outdir = Path(out_dir); outdir.mkdir(parents=True, exist_ok=True)
        base = inp.stem.replace(" ", "_")
        suffix = f"_{rate//1000}k_{'mono' if channels==1 else str(channels)+'ch'}"
        ext = ".wav" if fmt == "wav" else ".m4a"
        out_path = outdir / f"{base}{suffix}{ext}"
        if out_path.exists() and not overwrite:
            return True, f"Already exists: {out_path}", str(out_path)

        meta = _ffprobe_json(str(inp))
        dur = max(0.0, _duration_seconds(meta))
        got_audio = _has_audio(meta)

        ff = ["ffmpeg","-y" if overwrite else "-n","-hide_banner","-loglevel","error"]
        if got_audio:
            ff += ["-i", str(inp), "-vn", "-map", "0:a:0?"]
            if fmt == "wav":
                ff += ["-c:a","pcm_s16le","-ar",str(rate),"-ac",str(channels)]
            else:
                ff += ["-c:a","aac","-b:a","160k","-ar",str(rate),"-ac",str(channels)]
        else:
            if dur <= 0.0:
                return False, "Could not determine duration to synthesize silent audio.", ""
            ff += ["-f","lavfi","-i",f"anullsrc=channel_layout={'mono' if channels==1 else 'stereo'}:sample_rate={rate}",
                   "-t", f"{dur}"]
            if fmt == "wav":
                ff += ["-c:a","pcm_s16le"]
            else:
                ff += ["-c:a","aac","-b:a","160k"]
        ff += [str(out_path)]
        ret = subprocess.run(ff).returncode
        if ret != 0:
            return False, "ffmpeg failed.", ""
        return True, f"Audio written: {out_path}", str(out_path)
    except FileNotFoundError as e:
        return False, f"Dependency missing (ffmpeg/ffprobe?). Details: {e}", ""
    except Exception as e:
        return False, f"Error: {e}", ""

# ----------------------------- 3) Transcribe (Azure Whisper) -----------------
HINDI_BIAS_PROMPT = (
    "यह ऑडियो हिंदी (देवनागरी) में है। कृपया आउटपुट देवनागरी लिपि में लिखें, "
    "अंग्रेज़ी ट्रांसलिटरेशन का प्रयोग न करें."
)

def _guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"

def _ffprobe_duration_simple(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", path], text=True
        ).strip()
        return float(out)
    except Exception:
        return None

def _extract_audio_16k_temp(src: str) -> Tuple[str, float]:
    if not shutil.which("ffmpeg"):
        return src, _ffprobe_duration_simple(src) or 0.0
    tmpdir = tempfile.mkdtemp(prefix="whisper_")
    wav = os.path.join(tmpdir, "audio16k.wav")
    cmd = ["ffmpeg","-y","-i",src,"-ac","1","-ar","16000","-vn","-f","wav",wav]
    subprocess.run(cmd, check=True)
    dur = _ffprobe_duration_simple(wav) or 0.0
    return wav, dur

def _segments_to_srt(segs: List[dict]) -> str:
    def _fmt_ts(t: float) -> str:
        ms = int(round((t - int(t))*1000)); s = int(t) % 60; m = int(t // 60) % 60; h = int(t // 3600)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, s in enumerate(segs, 1):
        start = _fmt_ts(float(s["start"])); end = _fmt_ts(float(s["end"]))
        text  = str(s.get("text","")).strip()
        if not text: continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def _to_segments_verbose_json(data: dict) -> List[dict]:
    segs = []
    for s in data.get("segments", []):
        if "start" in s and "end" in s and "text" in s:
            segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"])})
    return segs

def _build_naive_segments(text: str, media_path: str) -> List[dict]:
    import re
    sents = [s.strip() for s in re.split(r'(?<=[\.!\?।॥])\s+', text) if s.strip()]
    if not sents: return []
    dur = _ffprobe_duration_simple(media_path) or (len(sents) * 3.0)
    per = max(2.5, dur / max(1, len(sents)))
    segs, t = [], 0.0
    for s in sents:
        start, end = t, min(dur, t+per)
        segs.append({"start": float(start), "end": float(end), "text": s})
        t = end
        if t >= dur: break
    return segs

def transcribe_audio_streamlit(
    src_media: str,
    out_dir: str,
    base_name: str,
    make_srt: bool = True,
    do_extract_audio: bool = True,
    model_lang_choice: str = "auto"   # "en" | "hi" | "auto"
) -> Tuple[bool, str, str, Optional[str], Optional[str]]:

    # Validate Azure config
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_WHISPER_DEPLOYMENT):
        return False, "Missing Azure env vars/secrets (AZURE_API_KEY, AZURE_ENDPOINT, WHISPER_DEPLOYMENT).", "", None, None

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{base_name}.json")
    raw_path = os.path.join(out_dir, f"{base_name}.raw_verbose.json")

    media_for_api = src_media
    tmp_to_cleanup = None
    if do_extract_audio:
        media_for_api, _ = _extract_audio_16k_temp(src_media)
        tmp_to_cleanup = Path(media_for_api).parent

    headers = {"api-key": AZURE_API_KEY}
    last_exc = None
    raw = None

    for attempt in range(1, 4):
        try:
            with open(media_for_api, "rb") as fh:
                files = {
                    "file": (Path(media_for_api).name, fh, _guess_mime(Path(media_for_api).name)),
                    "response_format": (None, "verbose_json"),
                    "task": (None, "transcribe"),
                    "temperature": (None, "0"),
                }
                if model_lang_choice in ("en","hi"):
                    files["language"] = (None, model_lang_choice)
                    if model_lang_choice == "hi":
                        files["prompt"] = (None, HINDI_BIAS_PROMPT)
                r = requests.post(AZURE_WHISPER_URL, headers=headers, files=files, timeout=900)
            r.raise_for_status()
            try:
                raw = r.json()
            except Exception:
                raise RuntimeError("Non-JSON response: " + r.text[:500])
            break
        except Exception as e:
            last_exc = e
            time.sleep(2 * attempt)
    else:
        return False, f"Transcription failed after retries: {last_exc}", "", None, None

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    segs = _to_segments_verbose_json(raw)
    if not segs:
        text = raw.get("text", "")
        segs = _build_naive_segments(text, src_media)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segs, f, ensure_ascii=False, indent=2)

    srt_path = None
    if make_srt and segs:
        srt_path = os.path.join(out_dir, f"{base_name}.srt")
        Path(srt_path).write_text(_segments_to_srt(segs), encoding="utf-8-sig")

    if tmp_to_cleanup:
        try:
            for p in Path(tmp_to_cleanup).glob("*"): p.unlink(missing_ok=True)
            Path(tmp_to_cleanup).rmdir()
        except Exception:
            pass

    return True, f"Wrote {len(segs)} segments", out_json, srt_path, raw_path

# ----------------------------- 4) Refine Transcript (Azure GPT) --------------
def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                try:
                    return json.loads(part)
                except Exception:
                    pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        try:
            start = s.index(open_c)
            end   = s.rindex(close_c) + 1
            return json.loads(s[start:end])
        except Exception:
            continue
    raise ValueError("Model output did not contain valid JSON.")

def _normalize_obj(obj) -> Dict:
    if isinstance(obj, dict) and "segments" in obj and isinstance(obj["segments"], list):
        return {"segments": obj["segments"]}
    if isinstance(obj, list):
        return {"segments": obj}
    raise ValueError("Parsed JSON is not in expected format.")

SYSTEM_PROMPT = (
    "You are a subtitle cleaner.\n"
    "INPUT: array of items with keys: start (number), end (number), text (string).\n"
    "TASK: Clean text only (fix punctuation/spacing, remove duplicated phrases). "
    "DO NOT translate. Keep meaning. Do not add or remove items.\n"
    "STRICT OUTPUT: JSON OBJECT ONLY:\n"
    "{\"segments\":[{\"start\":number,\"end\":number,\"text\":string}, ...]}\n"
    "Return the SAME number of items, and the SAME start/end for each item.\n"
    "No markdown. No extra keys. No prose."
)

def refine_segments_streamlit(inp_path: str, out_path: str, lang: str = "hi", batch_size: int = 10) -> Tuple[bool, str, Optional[str]]:
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_CHAT_DEPLOYMENT):
        return False, "Missing Azure env vars/secrets (AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT).", None
    try:
        with open(inp_path, "r", encoding="utf-8") as f:
            segs = json.load(f)
        if isinstance(segs, dict) and "segments" in segs:
            segs = segs["segments"]
        if not segs:
            return False, "No segments found in input JSON.", None

        cleaned_all: List[Dict] = []
        for i in range(0, len(segs), max(1, batch_size)):
            batch = segs[i:i+batch_size]
            safe_batch = []
            for b in batch:
                text = str(b.get("text", "")).strip()
                if len(text) > 280:
                    text = text[:280]
                safe_batch.append({"start": float(b["start"]), "end": float(b["end"]), "text": text})
            payload = json.dumps(safe_batch, ensure_ascii=False)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Normalization language hint: {lang}\nINPUT_SEGMENTS:\n{payload}"}
            ]
            body = {"messages": messages, "temperature": 0.0, "max_tokens": 2000}
            r = requests.post(AZURE_CHAT_URL, headers=HEADERS, json=body, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = _safe_json_loads(content)
            obj = _normalize_obj(parsed)
            cleaned = obj.get("segments", [])
            for j, orig in enumerate(batch):
                new_text = cleaned[j].get("text", orig["text"]).strip() if j < len(cleaned) else orig["text"]
                cleaned_all.append({"start": float(orig["start"]), "end": float(orig["end"]), "text": new_text})

        outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(cleaned_all, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, f"✅ Cleaned transcript saved → {outp}", str(outp)
    except Exception as e:
        return False, f"Error: {e}", None

# ----------------------------- 5) Score Highlights ---------------------------
MIN_LEN = float(os.getenv("HIGHLIGHT_MIN_LEN", 20))
MAX_LEN = float(os.getenv("HIGHLIGHT_MAX_LEN", 40))
STEP    = float(os.getenv("HIGHLIGHT_STEP", 5))
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", 0.3))

_GENERIC = ["secret","tip","mistake","hack","lesson","why","how","help","trick","warning","best","worst","avoid","finally"]
_BASE_KW = [
    ("important",1.00),("very important",1.00),("remember",0.95),("must know",0.95),
    ("note this",0.90),("common mistake",0.90),("avoid this",0.90),("warning",0.85),
    ("tip",0.85),("trick",0.85),("shortcut",0.85),("strategy",0.80),("in summary",0.80),
    ("recap",0.80),("key point",0.80),("takeaway",0.80),("do not",0.75),("final advice",0.75),
    ("exam",0.95),("board exam",0.95),("marks",0.90),("scoring",0.85),
    ("syllabus",0.80),("homework",0.75),("assignment",0.75),
    ("question",0.85),("answer",0.80),("previous year question",0.95),("pyq",0.95),
    ("cbse",0.85),("icse",0.85),("jee",0.85),("neet",0.85),("upsc",0.85),("ssc",0.80),
    ("definition",0.90),("formula",0.95),("theorem",0.90),("proof",0.85),("example",0.90),
    ("worked example",0.95),("exercise",0.80),("practice",0.80),("solution",0.85),
    ("step by step",0.95),("steps",0.80),("method",0.80),("procedure",0.80),("concept",0.80),
    ("explain",0.75),("explanation",0.75),("overview",0.70),("summary",0.80),
    ("math",0.70),("algebra",0.75),("geometry",0.75),("calculus",0.80),("trigonometry",0.80),
    ("physics",0.75),("chemistry",0.75),("biology",0.75),("english",0.70),("hindi",0.70),
    ("grammar",0.75),("essay",0.70),("comprehension",0.70),("reasoning",0.75),
    ("महत्वपूर्ण",1.00),("बहुत महत्वपूर्ण",1.00),("याद रखें",0.95),("ज़रूरी",0.95),
    ("ध्यान दें",0.90),("आम गलती",0.90),("बचें",0.90),("चेतावनी",0.85),
    ("टिप",0.85),("ट्रिक",0.85),("शॉर्टकट",0.85),("रणनीति",0.80),("सारांश",0.80),
    ("पुनरावृत्ति",0.80),("मुख्य बिंदु",0.80),("टेकअवे",0.80),
    ("परीक्षा",0.95),("बोर्ड परीक्षा",0.95),("अंक",0.90),("सिलेबस",0.80),
    ("गृह कार्य",0.75),("होमवर्क",0.75),("असाइनमेंट",0.75),
    ("प्रश्न",0.85),("उत्तर",0.80),("पिछले वर्ष के प्रश्न",0.95),
    ("सीबीएसई",0.85),("आईसीएसई",0.85),
    ("परिभाषा",0.90),("सूत्र",0.95),("प्रमेय",0.90),("सिद्ध",0.85),
    ("उदाहरण",0.90),("अभ्यास",0.80),("समाधान",0.85),
    ("कदम दर कदम",0.95),("कदम",0.80),("विधि",0.80),("प्रक्रिया",0.80),
    ("अवधारणा",0.80),("समझाएँ",0.75),("व्याख्या",0.75),("सार",0.80),
    ("गणित",0.80),("बीजगणित",0.75),("ज्यामिति",0.75),("कलन",0.80),
    ("त्रिकोणमिति",0.80),("भौतिकी",0.75),("रसायन",0.75),("जीवविज्ञान",0.75),
    ("व्याकरण",0.75),("निबंध",0.70),("बोध",0.70),("रीज़निंग",0.75),
] + [(k, 0.55) for k in _GENERIC]
_kw = {}
for k, w in _BASE_KW:
    _kw[k.lower()] = max(w, _kw.get(k.lower(), 0.0))
KEYWORD_WEIGHTS = sorted(_kw.items(), key=lambda x: -x[1])

def _run(cmd: List[str]):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def mean_loudness_db(video: str, start: float, end: float) -> float:
    try:
        p = _run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(max(0,start)),"-to",str(end),
                  "-i",video,"-vn","-filter:a","volumedetect","-f","null","-"])
        if p.returncode != 0: return -60.0
        m = re.search(r"mean_volume:\s*([-\d\.]+)\s*dB", p.stderr)
        return float(m.group(1)) if m else -60.0
    except Exception:
        return -60.0

def scene_change_times(video: str, thr: float) -> List[float]:
    if thr <= 0: return []
    p = _run(["ffmpeg","-hide_banner","-loglevel","error","-i",video,
              "-vf", f"select='gt(scene,{thr})',showinfo", "-an","-f","null","-"])
    times = []
    for line in p.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            try: times.append(float(line.split("pts_time:")[1].split()[0]))
            except: pass
    return times

def _count_kw(text: str, kw: str) -> int:
    return len(re.findall(r"\b"+re.escape(kw)+r"\b", (text or ""), flags=re.IGNORECASE))

def keyword_score(text: str) -> float:
    t = (text or "").lower()
    if not t: return 0.0
    score = 0.0
    for kw, w in KEYWORD_WEIGHTS:
        hits = _count_kw(t, kw)
        if hits:
            score += min(3, hits) * w
    return 1.0 / (1.0 + math.exp(-(score - 1.2)))

def merge_segments(segments: List[Dict], max_gap: float = 1.0) -> List[Dict]:
    merged, cur = [], None
    for s in segments:
        if not cur:
            cur = {"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","")}
        else:
            if float(s["start"]) - cur["end"] <= max_gap:
                cur["end"] = float(s["end"]); cur["text"] += " " + s.get("text","")
            else:
                merged.append(cur); cur = {"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","")}
    if cur: merged.append(cur)
    return merged

def clip_candidates(blocks: List[Dict], min_len=MIN_LEN, max_len=MAX_LEN, step=STEP):
    cands = []
    for b in blocks:
        base_s = max(0.0, float(b["start"]) - 3.0)
        base_e = float(b["end"]) + 3.0
        t = base_s
        while t + min_len <= base_e:
            w_end = min(t + max_len, base_e)
            cands.append((t, w_end, b["text"]))
            t += step
    return cands

def deoverlap_by_score(scored: List[Dict]) -> List[Dict]:
    picked = []
    def overlaps(a,b): return not (a["end"] <= b["start"] or b["end"] <= a["start"])
    for c in scored:
        if all(not overlaps(c, x) for x in picked):
            picked.append(c)
    return picked

def azure_scores(texts: List[str], batch_size: int = 12, timeout: int = 60) -> List[float]:
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_CHAT_DEPLOYMENT):
        return [0.0]*len(texts)
    headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}
    scores = [0.0]*len(texts)
    def one_batch(items):
        payload = [{"i": i, "t": (t[:1200] if len(t) > 1200 else t)} for i,t in items]
        sys_msg = ("Rate each clip transcript for TEACHING usefulness (students/teachers). "
                   "Return ONLY JSON: {\"scores\":[{\"i\":<index>,\"score\":<0..1>}, ...]}. "
                   "Higher = definitions, steps, formulae, examples, exam tips, conceptual clarity. "
                   "Lower = filler/banter/no learning. No extra keys.")
        user_msg = json.dumps({"items": payload}, ensure_ascii=False)
        body = {"messages":[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
                "temperature":0.0,"response_format":{"type":"json_object"},"max_tokens":800,"seed":11}
        try:
            r = requests.post(AZURE_CHAT_URL, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            obj = json.loads(content)
            for it in obj.get("scores", []):
                i = int(it.get("i",-1)); sc = float(it.get("score",0.0))
                if 0 <= i < len(scores):
                    scores[i] = max(0.0, min(1.0, sc))
        except Exception:
            pass
    batch = []
    for idx, t in enumerate(texts):
        batch.append((idx, t))
        if len(batch) == batch_size:
            one_batch(batch); batch = []
    if batch: one_batch(batch)
    return scores

def score_highlights_streamlit(
    video_path: str,
    transcript_json_path: str,
    out_path: str,
    use_azure: bool,
    min_score: float,
    max_clips: int,
    azure_weight: float,
    min_len: float,
    max_len: float,
    step: float,
    scene_thr: float,
) -> Tuple[bool, str, Optional[str], Optional[List[Dict]]]:
    try:
        raw = json.loads(Path(transcript_json_path).read_text(encoding="utf-8"))
        segs = raw["segments"] if isinstance(raw, dict) and "segments" in raw else raw
        blocks = merge_segments(segs)
        cands  = clip_candidates(blocks, min_len=min_len, max_len=max_len, step=step)
        scenes = scene_change_times(video_path, scene_thr)
        def scene_bonus(s,e):
            for t in scenes:
                if abs(t - s) < 1.0 or abs(t - e) < 1.0: return 0.15
            return 0.0
        base_scored = []
        for (s,e,txt) in cands:
            k = keyword_score(txt)
            loud = mean_loudness_db(video_path, s, e)
            loud_norm = max(0.0, min(1.0, (loud + 60)/60))
            scb = scene_bonus(s,e)
            base = 0.62*k + 0.28*loud_norm + scb
            base_scored.append({"start":round(s,3), "end":round(e,3), "text":(txt or "").strip(), "base":round(base,4)})
        a = azure_scores([x["text"] for x in base_scored], batch_size=12, timeout=60) if use_azure else [0.0]*len(base_scored)
        w = max(0.0, min(1.0, azure_weight))
        merged = []
        for x, az in zip(base_scored, a):
            final = (1.0 - w)*x["base"] + w*float(az)
            merged.append({**x, "azure": round(float(az),4), "score": round(final,4)})
        merged.sort(key=lambda z: z["score"], reverse=True)
        non_overlap = deoverlap_by_score(merged)
        kept = [c for c in non_overlap if c["score"] >= min_score]
        if max_clips > 0: kept = kept[:max_clips]
        out = [{"start":c["start"], "end":c["end"], "text":c["text"], "score":c["score"]} for c in kept]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, f"Wrote {len(out)} clips → {out_path}", out_path, out
    except Exception as e:
        return False, f"Error: {e}", None, None

# ----------------------------- 6) Cut Clips ----------------------------------
def cut_clips_streamlit(
    video_path: str,
    highlights_json_path: str,
    out_dir: str = "data/clips",
    crf: int = 18,
    preset: str = "fast",
    audio_bitrate: str = "128k",
    accurate_seek: bool = False,
) -> Tuple[bool, str, Optional[List[str]]]:
    try:
        outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
        highlights = json.loads(Path(highlights_json_path).read_text(encoding="utf-8"))
        if not isinstance(highlights, list):
            return False, "Invalid highlights JSON: expected a list of {start,end,text,score}.", None
        if not highlights:
            return False, "No highlights to cut.", None
        made = []
        for i, h in enumerate(highlights, 1):
            start = float(h["start"]); end = float(h["end"])
            if end <= start: continue
            out_file = outp / f"clip_{i:02d}.mp4"
            dur = end - start
            if accurate_seek:
                cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y","-i", video_path,
                       "-ss", f"{start}","-t", f"{dur}",
                       "-c:v","libx264","-crf", str(crf), "-preset", preset,
                       "-c:a","aac","-b:a", audio_bitrate, str(out_file)]
            else:
                cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
                       "-ss", f"{start}","-t", f"{dur}","-i", video_path,
                       "-c:v","libx264","-crf", str(crf), "-preset", preset,
                       "-c:a","aac","-b:a", audio_bitrate, str(out_file)]
            p = subprocess.run(cmd)
            if p.returncode != 0:
                return False, f"ffmpeg failed while cutting {out_file.name}", None
            made.append(str(out_file))
        if not made: return False, "No clips were produced (check times).", None
        return True, f"Wrote {len(made)} clips → {out_dir}", made
    except Exception as e:
        return False, f"Error: {e}", None

# ----------------------------- 7) Slice SRT per Clip -------------------------
_TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)")
def _t2s(s: str) -> float:
    h,m,sec,ms = map(int, _TIME_RE.match(s).groups()); return h*3600 + m*60 + sec + ms/1000.0
def _s2t(x: float) -> str:
    if x < 0: x = 0
    ms = int(round((x - int(x))*1000)); s = int(x) % 60; m = int(x//60) % 60; h = int(x//3600)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
def _parse_block(lines: List[str]) -> List[Tuple[float,float,str]]:
    if len(lines) < 2 or "-->" not in lines[1]: return []
    a,b = [p.strip() for p in lines[1].split("-->")]
    t0, t1 = _t2s(a), _t2s(b); text = "\n".join(lines[2:]).strip()
    return [(t0,t1,text)]
def _parse_srt(path: Path) -> List[Tuple[float,float,str]]:
    cues, block = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip(): block.append(line.rstrip("\n"))
        else:
            if block: cues.extend(_parse_block(block)); block = []
    if block: cues.extend(_parse_block(block))
    return cues
def _wrap_text(txt: str, max_chars=30, max_lines=2) -> str:
    words = re.split(r"\s+", (txt or "").strip()); lines, cur = [], ""
    for w in words:
        if not cur: cur = w
        elif len(cur) + 1 + len(w) <= max_chars: cur += " " + w
        else:
            lines.append(cur); cur = w
            if len(lines) == max_lines-1: break
    if cur and len(lines) < max_lines: lines.append(cur)
    return r"\N".join(lines)
def _split_cue(t0: float, t1: float, txt: str, max_sec: float) -> List[Tuple[float,float,str]]:
    out, t = [], t0
    while t < t1 - 1e-6:
        e = min(t1, t + max_sec); out.append((t, e, txt)); t = e
    return out
def _write_srt(cues: List[Tuple[float,float,str]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for i,(s,e,txt) in enumerate(cues,1):
            f.write(f"{i}\n{_s2t(s)} --> {_s2t(e)}\n{txt}\n\n")

def slice_srt_per_clip_streamlit(
    master_srt_path: str,
    highlights_json_path: str,
    out_dir: str,
    pad: float = 0.0,
    max_sec: float = 3.0,
    max_chars: int = 30,
    max_lines: int = 2,
) -> Tuple[bool, str, Optional[List[str]]]:
    try:
        master = Path(master_srt_path)
        if not master.exists(): return False, f"Master SRT not found: {master}", None
        outs = Path(out_dir); outs.mkdir(parents=True, exist_ok=True)
        highlights = json.loads(Path(highlights_json_path).read_text(encoding="utf-8"))
        hl = highlights["clips"] if isinstance(highlights, dict) and "clips" in highlights else highlights
        if not isinstance(hl, list) or not hl:
            return False, "Invalid highlights JSON (need list of {start,end,...}).", None
        cues = _parse_srt(master); made = []
        for idx, h in enumerate(hl, 1):
            s = float(h["start"]) - pad; e = float(h["end"]) + pad; win = []
            for (a,b,txt) in cues:
                if b <= s or a >= e: continue
                aa, bb = max(a, s), min(b, e)
                for (x0,x1,_) in _split_cue(aa, bb, txt, max_sec):
                    wrapped = _wrap_text(txt, max_chars, max_lines)
                    win.append((x0 - s, x1 - s, wrapped))
            out_path = outs / f"clip_{idx:02}.srt"
            if win: _write_srt(win, out_path)
            else: out_path.write_text("", encoding="utf-8")
            made.append(str(out_path))
        return True, f"Wrote {len(made)} per-clip SRTs → {outs}", made
    except Exception as e:
        return False, f"Error: {e}", None

# ----------------------------- 8) Add Background -----------------------------
def add_background_streamlit(
    in_dir: str,
    out_dir: str = "data/composite",
    bg_img: str = "assets/backgrounds/bgcontentlabs.jpg",
    mode: str = "landscape",
    fg_size: Optional[int] = None
) -> Tuple[bool, str, Optional[List[str]]]:
    try:
        in_dir = Path(in_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        clips = sorted(in_dir.glob("clip_*.mp4"))
        if not clips: return False, f"No clips found in {in_dir}", None

        if mode == "portrait":
            CW, CH = 1080, 1920; fg_size = fg_size or 1600
            fg_scale = f"scale=-1:{fg_size}:flags=lanczos"
        else:
            CW, CH = 1920, 1080; fg_size = fg_size or 960
            fg_scale = f"scale={fg_size}:-1:flags=lanczos"

        outs = []
        for clip in clips:
            out = out_dir / clip.name.replace(".mp4", f"_{mode}_bg.mp4")
            filter_complex = (
                f"[1:v]scale={CW}:{CH}:force_original_aspect_ratio=increase,"
                f"crop={CW}:{CH}[bg];"
                f"[0:v]{fg_scale}[vid];"
                f"[bg][vid]overlay=(W-w)/2:(H-h)/2"
            )
            cmd = ["ffmpeg","-y","-i",str(clip), "-loop","1","-i",bg_img,
                   "-filter_complex", filter_complex, "-shortest", str(out)]
            p = subprocess.run(cmd)
            if p.returncode != 0:
                return False, f"ffmpeg failed while processing {clip.name}", None
            outs.append(str(out))
        return True, f"Wrote {len(outs)} composites → {out_dir}", outs
    except Exception as e:
        return False, f"Error: {e}", None

# ----------------------------- 9) Burn Subtitles -----------------------------
def _escape_sub_path(path: Path) -> str:
    p = str(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":  # windows drive
        p = p[0] + "\\:" + p[2:]
    return p

def burn_subtitles_streamlit(
    composite_dir="data/composite",
    srt_dir="outputs/srt_per_clip",
    out_dir="data/final",
    font="Arial",
    fontsize=20,
    margin_v=30,
    outline=2,
) -> Tuple[bool, str, Optional[List[str]]]:
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for file in Path(composite_dir).glob("*.mp4"):
            parts = file.stem.split("_")
            base = "_".join(parts[:2]) if len(parts) >= 2 else file.stem
            srt_path = Path(srt_dir) / f"{base}.srt"
            if not srt_path.exists():
                continue
            out_file = Path(out_dir) / f"{base}_final.mp4"
            srt_escaped = _escape_sub_path(srt_path)
            vf = (
                f"subtitles='{srt_escaped}':"
                f"force_style='FontName={font},FontSize={fontsize},"
                f"MarginV={margin_v},Outline={outline}'"
            )
            cmd = ["ffmpeg","-y","-i",str(file), "-vf", vf, "-c:a","copy", str(out_file)]
            subprocess.run(cmd, check=True)
            results.append(str(out_file))
        return True, f"Burned subtitles into {len(results)} videos → {out_dir}", results
    except Exception as e:
        return False, f"Error: {e}", None

# =============================================================================
#                                       UI
# =============================================================================
st.sidebar.title("⚙️ Controls")
st.sidebar.info("Outputs folder: `downloads/`")

with st.expander("🔧 Azure config (debug)"):
    st.write("AZURE_ENDPOINT:", AZURE_ENDPOINT or "(unset)")
    st.write("AZURE_API_VERSION:", AZURE_API_VERSION)
    st.write("CHAT deployment:", AZURE_CHAT_DEPLOYMENT)
    st.write("WHISPER deployment:", AZURE_WHISPER_DEPLOYMENT)
    st.write("CHAT URL:", AZURE_CHAT_URL or "(n/a)")
    st.write("WHISPER URL:", AZURE_WHISPER_URL or "(n/a)")

with st.expander("🧩 System checks"):
    st.write("ffmpeg found:", _has_binary("ffmpeg"))
    st.write("ffprobe found:", _has_binary("ffprobe"))
    try:
        import yt_dlp as _y; st.write("yt-dlp version:", getattr(_y, "__version__", "ok"))
    except Exception as e:
        st.write("yt-dlp import error:", str(e))

tabs = st.tabs([
    "📥 YouTube Download",
    "🎵 Extract Audio",
    "✍️ Transcribe",
    "🧹 Refine Transcript",
    "⭐ Score Highlights",
    "✂️ Cut Clips",
    "🗂️ Slice SRT per Clip",
    "🖼️ Add Background",
    "🔤 Burn Subtitles",
    "📦 Final Output",
])

# --- Tab 1: YouTube Download ---
with tabs[0]:
    st.header("📥 YouTube Downloader (yt-dlp)")
    url = st.text_input("YouTube URL", key="ydl_url")
    col1, col2, col3 = st.columns(3)
    with col1: audio_only = st.checkbox("Audio only (M4A)", key="ydl_audio_only")
    # On Streamlit Cloud, no Chrome profile → default to False
    with col2: use_cookies = st.checkbox("Use Chrome cookies", value=False, key="ydl_cookies")
    with col3: output_dir = st.text_input("Output folder", APP_OUT, key="ydl_out_dir")
    if st.button("Download", key="ydl_download_btn"):
        if not url.strip():
            st.error("Please enter a valid URL.")
        else:
            attempts = yt_build_attempts(use_cookies=use_cookies, audio_only=audio_only)
            ok = False; errors: List[str] = []
            for label, opts in attempts:
                ok, msg = yt_try_download(url, output_dir, opts, label)
                st.write(msg)
                if ok: break
                errors.append(msg)
            if ok: st.success(f"Done! Files saved under `{output_dir}`.")
            else:
                st.error("All attempts failed. Try updating yt-dlp or checking network.")
                st.code("\n".join(errors[-3:]), language="text")
    st.subheader("📂 Files in output")
    for f in list_dir(output_dir):
        st.write("•", f)

# --- Tab 2: Extract Audio ---
with tabs[1]:
    st.header("🎵 Extract Audio")
    src_mode = st.radio("Source", ["Pick from downloads/", "Upload file"], horizontal=True, key="ea_source")
    input_path = None
    if src_mode == "Pick from downloads/":
        dl_files = [f for f in list_dir(APP_OUT) if f.lower().endswith((".mp4",".mkv",".webm",".mov",".m4v",".mp3",".wav",".m4a"))]
        picked = st.selectbox("Select a media file", ["-- choose --"] + dl_files, index=0, key="ea_media_pick")
        if picked != "-- choose --":
            input_path = str(Path(APP_OUT) / picked); st.caption(f"Selected: `{input_path}`")
    else:
        up = st.file_uploader("Upload a video/audio file", type=["mp4","mkv","webm","mov","m4v","mp3","wav","m4a"], key="ea_upload_media")
        if up is not None:
            up_dir = Path("uploads"); up_dir.mkdir(parents=True, exist_ok=True)
            saved = up_dir / up.name; saved.write_bytes(up.read())
            input_path = str(saved); st.caption(f"Uploaded to: `{input_path}`")
    col1, col2, col3, col4 = st.columns(4)
    with col1: rate = st.selectbox("Sample rate", [8000,16000,22050,24000,32000,44100,48000], index=1, key="ea_rate")
    with col2: channels = st.selectbox("Channels", [1,2], index=0, key="ea_channels")
    with col3: fmt = st.selectbox("Format", ["wav","m4a"], index=0, key="ea_fmt")
    with col4: overwrite = st.checkbox("Overwrite if exists", value=False, key="ea_overwrite")
    out_dir_audio = st.text_input("Output folder", "data/audio", key="ea_out_dir")
    do = st.button("Extract Audio", type="primary", disabled=input_path is None, key="ea_btn")
    if do:
        with st.spinner("Extracting..."):
            ok, msg, outp = extract_audio(input_path, out_dir=out_dir_audio, rate=rate, channels=channels, fmt=fmt, overwrite=overwrite)
        if ok:
            st.success(msg)
            try:
                with open(outp, "rb") as f:
                    st.download_button("Download audio", f, file_name=Path(outp).name, key="ea_dl_audio")
            except Exception:
                st.info(f"Saved at: `{outp}`")
        else:
            st.error(msg)

# --- Tab 3: Transcribe ---
with tabs[2]:
    st.header("✍️ Transcribe (Azure Whisper)")
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_WHISPER_DEPLOYMENT):
        st.warning("Set AZURE_API_KEY, AZURE_ENDPOINT, and WHISPER_DEPLOYMENT in Streamlit Secrets.")
    src_mode = st.radio("Source", ["Pick from downloads/", "Upload file"], horizontal=True, key="tr_source")
    media_path = None
    if src_mode == "Pick from downloads/":
        dl_files = [f for f in list_dir(APP_OUT) if f.lower().endswith((".mp4",".mkv",".webm",".mov",".m4v",".mp3",".wav",".m4a"))]
        picked = st.selectbox("Select media", ["-- choose --"] + dl_files, index=0, key="tr_media_pick")
        if picked != "-- choose --":
            media_path = str(Path(APP_OUT) / picked); st.caption(f"Selected: `{media_path}`")
    else:
        up = st.file_uploader("Upload media", type=["mp4","mkv","webm","mov","m4v","mp3","wav","m4a"], key="tr_upload_media")
        if up:
            p = Path("uploads") / up.name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(up.read())
            media_path = str(p); st.caption(f"Uploaded: `{media_path}`")
    col1, col2, col3 = st.columns(3)
    with col1: lang = st.selectbox("Language", ["auto","en","hi"], index=0, help="Send hint to Whisper", key="tr_lang")
    with col2: do_extract = st.checkbox("Extract 16k mono WAV before sending", value=True, key="tr_do_extract")
    with col3: make_srt = st.checkbox("Save SRT", value=True, key="tr_make_srt")
    out_dir_tr = st.text_input("Output folder", "outputs/transcripts", key="tr_out_dir")
    base_name = st.text_input("Output base name (no extension)", value="transcript", key="tr_base_name")
    run = st.button("Transcribe", type="primary", disabled=media_path is None, key="tr_btn")
    if run:
        with st.spinner("Transcribing with Azure Whisper..."):
            ok, msg, json_path, srt_path, raw_path = transcribe_audio_streamlit(
                src_media=media_path, out_dir=out_dir_tr, base_name=base_name,
                make_srt=make_srt, do_extract_audio=do_extract, model_lang_choice=lang
            )
        if ok:
            st.success(msg)
            try:
                with open(json_path, "r", encoding="utf-8") as f: segs = json.load(f)
                st.caption("Preview (first 5 segments):"); st.json(segs[:5] if isinstance(segs,list) else segs)
            except Exception as e:
                st.info(f"Saved JSON at: `{json_path}` (preview error: {e})")
            try:
                with open(json_path, "rb") as f: st.download_button("Download JSON", f, file_name=Path(json_path).name, key="tr_dl_json")
            except Exception: pass
            if srt_path and Path(srt_path).exists():
                with open(srt_path, "rb") as f: st.download_button("Download SRT", f, file_name=Path(srt_path).name, key="tr_dl_srt")
            with st.expander("Raw verbose JSON (debug)"):
                try:
                    with open(raw_path, "r", encoding="utf-8") as f: st.json(json.load(f))
                except Exception:
                    st.caption(f"Saved at: `{raw_path}`")
        else:
            st.error(msg)

# --- Tab 4: Refine Transcript ---
with tabs[3]:
    st.header("🧹 Refine Transcript (Azure GPT)")
    mode = st.radio("Input JSON", ["Pick from transcripts/", "Upload"], horizontal=True, key="rf_input_mode")
    inp_path = None
    if mode == "Pick from transcripts/":
        files = [f for f in list_dir("outputs/transcripts") if f.endswith(".json")]
        picked = st.selectbox("Choose file", ["--"] + files, key="rf_pick_file")
        if picked != "--":
            inp_path = str(Path("outputs/transcripts") / picked); st.caption(f"Selected: {inp_path}")
    else:
        up = st.file_uploader("Upload transcript JSON", type="json", key="rf_upload_json")
        if up:
            p = Path("uploads") / up.name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(up.read())
            inp_path = str(p); st.caption(f"Uploaded: {inp_path}")
    col1, col2 = st.columns(2)
    with col1: lang_ref = st.selectbox("Normalization language", ["hi","en","auto"], index=0, key="rf_lang")
    with col2: batch_size = st.number_input("Batch size", min_value=1, max_value=40, value=10, key="rf_batch")
    out_file_ref = st.text_input("Output JSON path", "outputs/transcripts/refined.json", key="rf_out_path")
    run = st.button("Refine Transcript", type="primary", disabled=inp_path is None, key="rf_btn")
    if run:
        with st.spinner("Cleaning transcript with Azure GPT..."):
            ok, msg, outp = refine_segments_streamlit(inp_path, out_file_ref, lang=lang_ref, batch_size=batch_size)
        if ok:
            st.success(msg)
            try:
                with open(outp,"r",encoding="utf-8") as f: data = json.load(f)
                st.caption("Preview (first 5 lines):")
                st.json(data[:5] if isinstance(data,list) else data.get("segments",[])[:5])
                with open(outp,"rb") as f: st.download_button("Download refined JSON", f, file_name=Path(outp).name, key="rf_dl")
            except Exception as e:
                st.error(f"Could not preview: {e}")
        else:
            st.error(msg)

# --- Tab 5: Score Highlights ---
with tabs[4]:
    st.header("⭐ Score Highlights (Teacher-friendly)")
    colA, colB = st.columns(2)
    with colA:
        vid_mode = st.radio("Video source", ["Pick from downloads/", "Upload"], horizontal=True, key="sc_video_source")
        video_path = None
        if vid_mode == "Pick from downloads/":
            vids = [f for f in list_dir(APP_OUT) if f.lower().endswith((".mp4",".mkv",".webm",".mov",".m4v"))]
            pick_v = st.selectbox("Select video", ["-- choose --"] + vids, index=0, key="sc_video_pick")
            if pick_v != "-- choose --":
                video_path = str(Path(APP_OUT) / pick_v); st.caption(f"Selected video: `{video_path}`")
        else:
            upv = st.file_uploader("Upload video", type=["mp4","mkv","webm","mov","m4v"], key="sc_upload_video")
            if upv:
                vv = Path("uploads") / upv.name; vv.parent.mkdir(parents=True, exist_ok=True); vv.write_bytes(upv.read())
                video_path = str(vv); st.caption(f"Uploaded: `{video_path}`")
    with colB:
        tr_mode = st.radio("Transcript JSON", ["Pick from transcripts/", "Upload"], horizontal=True, key="sc_transcript_json")
        transcript_path = None
        if tr_mode == "Pick from transcripts/":
            js = [f for f in list_dir("outputs/transcripts") if f.lower().endswith(".json")]
            pick_j = st.selectbox("Select transcript JSON", ["-- choose --"] + js, index=0, key="sc_transcript_pick")
            if pick_j != "-- choose --":
                transcript_path = str(Path("outputs/transcripts") / pick_j); st.caption(f"Selected transcript: `{transcript_path}`")
        else:
            upj = st.file_uploader("Upload transcript JSON", type=["json"], key="sc_upload_transcript")
            if upj:
                jj = Path("uploads") / upj.name; jj.parent.mkdir(parents=True, exist_ok=True); jj.write_bytes(upj.read())
                transcript_path = str(jj); st.caption(f"Uploaded: `{transcript_path}`")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        use_azure = st.checkbox("Use Azure GPT scoring", value=False, key="sc_use_azure")
        min_score = st.slider("Min score threshold", 0.0, 1.0, 0.62, 0.01, key="sc_min_score")
    with col2:
        max_clips = st.number_input("Max clips (0 = no cap)", min_value=0, max_value=100, value=0, step=1, key="sc_max_clips")
        azure_weight = st.slider("Azure weight (0=base, 1=Azure)", 0.0, 1.0, 1.0, 0.05, key="sc_az_weight")
    with col3:
        min_len = st.slider("Min clip length (s)", 5, 60, int(MIN_LEN), 1, key="sc_min_len")
        max_len = st.slider("Max clip length (s)", min_len, 120, int(MAX_LEN), 1, key="sc_max_len")
        step    = st.slider("Window step (s)", 1, 30, int(STEP), 1, key="sc_step")
    scene_thr = st.slider("Scene-change sensitivity", 0.0, 1.0, float(SCENE_THRESHOLD), 0.05,
                          help="0 disables scene detection; higher = stricter", key="sc_scene_thr")
    out_path_hl = st.text_input("Output JSON", "outputs/highlights/highlights.json", key="sc_out_json")
    run = st.button("Score Highlights", type="primary",
                    disabled=(video_path is None or transcript_path is None), key="sc_btn")
    if run:
        with st.spinner("Scoring windows…"):
            ok, msg, outp, rows = score_highlights_streamlit(
                video_path=video_path, transcript_json_path=transcript_path, out_path=out_path_hl,
                use_azure=use_azure, min_score=float(min_score), max_clips=int(max_clips),
                azure_weight=float(azure_weight), min_len=float(min_len),
                max_len=float(max_len), step=float(step), scene_thr=float(scene_thr)
            )
        if ok:
            st.success(msg)
            if rows:
                st.caption("Top results")
                st.dataframe(
                    [{"start":r["start"], "end":r["end"], "score":r["score"],
                      "text": r["text"][:140]+"…" if len(r["text"])>140 else r["text"]} for r in rows],
                    use_container_width=True
                )
            try:
                with open(outp, "rb") as f: st.download_button("Download highlights JSON", f, file_name=Path(outp).name, key="sc_dl_highlights")
            except Exception:
                st.info(f"Saved at: `{outp}`")
        else:
            st.error(msg)

# --- Tab 6: Cut Clips ---
with tabs[5]:
    st.header("✂️ Cut Clips from Highlights")
    colA, colB = st.columns(2)
    with colA:
        vmode = st.radio("Video source", ["Pick from downloads/", "Upload"], horizontal=True, key="cc_video_source")
        video_path2 = None
        if vmode == "Pick from downloads/":
            vids = [f for f in list_dir(APP_OUT) if f.lower().endswith((".mp4",".mkv",".webm",".mov",".m4v"))]
            pick_v = st.selectbox("Select video", ["-- choose --"] + vids, index=0, key="cc_video_pick")
            if pick_v != "-- choose --":
                video_path2 = str(Path(APP_OUT) / pick_v); st.caption(f"Selected video: `{video_path2}`")
        else:
            upv = st.file_uploader("Upload video", type=["mp4","mkv","webm","mov","m4v"], key="cc_upload_video")
            if upv:
                vv = Path("uploads") / upv.name; vv.parent.mkdir(parents=True, exist_ok=True); vv.write_bytes(upv.read())
                video_path2 = str(vv); st.caption(f"Uploaded: `{video_path2}`")
    with colB:
        jmode = st.radio("Highlights JSON", ["Pick from outputs/highlights/", "Upload"], horizontal=True, key="cc_highlights_json")
        highlights_path = None
        if jmode == "Pick from outputs/highlights/":
            js = [f for f in list_dir("outputs/highlights") if f.lower().endswith(".json")]
            pick_j = st.selectbox("Select highlights JSON", ["-- choose --"] + js, index=0, key="cc_highlights_pick")
            if pick_j != "-- choose --":
                highlights_path = str(Path("outputs/highlights") / pick_j); st.caption(f"Selected: `{highlights_path}`")
        else:
            upj = st.file_uploader("Upload highlights JSON", type=["json"], key="cc_upload_highlights")
            if upj:
                jj = Path("uploads") / upj.name; jj.parent.mkdir(parents=True, exist_ok=True); jj.write_bytes(upj.read())
                highlights_path = str(jj); st.caption(f"Uploaded: `{highlights_path}`")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1: crf = st.slider("CRF (quality)", 14, 30, 18, 1, key="cc_crf")
    with col2: preset = st.selectbox("Preset", ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=4, key="cc_preset")
    with col3: audio_bitrate = st.selectbox("Audio bitrate", ["96k","128k","160k","192k","256k"], index=1, key="cc_aud_bitrate")
    with col4: accurate_seek = st.checkbox("Accurate seeking (slower, frame-accurate)", value=False, key="cc_acc_seek")
    out_dir = st.text_input("Output folder", "data/clips", key="cc_out_dir")
    run = st.button("Cut Clips", type="primary", disabled=(video_path2 is None or highlights_path is None), key="cc_btn")
    if run:
        with st.spinner("Cutting clips with ffmpeg…"):
            ok, msg, files = cut_clips_streamlit(video_path=video_path2, highlights_json_path=highlights_path,
                                                 out_dir=out_dir, crf=int(crf), preset=preset,
                                                 audio_bitrate=audio_bitrate, accurate_seek=accurate_seek)
        if ok:
            st.success(msg); st.caption("Generated clips:")
            for f in files: st.write("• ", f)
            try:
                zip_base = Path(out_dir).as_posix().rstrip("/").replace("/", "_")
                zip_path = shutil.make_archive(zip_base, "zip", out_dir)
                with open(zip_path, "rb") as f:
                    st.download_button("Download all clips (ZIP)", f, file_name=Path(zip_path).name, key="cc_zip")
            except Exception as e:
                st.info(f"Clips saved in `{out_dir}` (ZIP failed: {e})")
        else:
            st.error(msg)

# --- Tab 7: Slice SRT per Clip ---
with tabs[6]:
    st.header("🗂️ Slice SRT per Clip")
    colA, colB = st.columns(2)
    with colA:
        srt_mode = st.radio("Master SRT", ["Pick file", "Upload"], horizontal=True, key="sl_master_srt")
        master_srt = None
        if srt_mode == "Pick file":
            srt_pool = []
            for root in ["outputs/transcripts", "data", "downloads", "data/composite", "data/clips"]:
                if Path(root).exists():
                    for fn in list_dir(root):
                        if fn.lower().endswith(".srt"): srt_pool.append((root, fn))
            choices = ["-- choose --"] + [f"{r}/{f}" for r,f in srt_pool]
            picked = st.selectbox("Select SRT", choices, index=0, key="sl_srt_pick")
            if picked != "-- choose --":
                master_srt = picked; st.caption(f"Selected: `{master_srt}`")
        else:
            ups = st.file_uploader("Upload master SRT", type=["srt"], key="sl_upload_srt")
            if ups:
                savep = Path("uploads") / ups.name; savep.parent.mkdir(parents=True, exist_ok=True); savep.write_bytes(ups.read())
                master_srt = str(savep); st.caption(f"Uploaded: `{master_srt}`")
    with colB:
        jmode = st.radio("Highlights JSON", ["Pick file", "Upload"], horizontal=True, key="sl_highlights_json")
        highlights = None
        if jmode == "Pick file":
            pool = []
            for root in ["outputs/highlights", "outputs", "data", "uploads"]:
                if Path(root).exists():
                    for fn in list_dir(root):
                        if fn.lower().endswith(".json"): pool.append((root, fn))
            jchoices = ["-- choose --"] + [f"{r}/{f}" for r,f in pool]
            pick_j = st.selectbox("Select highlights JSON", jchoices, index=0, key="sl_highlights_pick")
            if pick_j != "-- choose --":
                highlights = pick_j; st.caption(f"Selected: `{highlights}`")
        else:
            upj = st.file_uploader("Upload highlights JSON", type=["json"], key="sl_upload_highlights")
            if upj:
                jp = Path("uploads") / upj.name; jp.parent.mkdir(parents=True, exist_ok=True); jp.write_bytes(upj.read())
                highlights = str(jp); st.caption(f"Uploaded: `{highlights}`")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1: pad = st.number_input("Pad seconds", min_value=0.0, max_value=10.0, value=0.0, step=0.5, key="sl_pad")
    with col2: max_sec = st.number_input("Max cue duration", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="sl_max_sec")
    with col3: max_chars = st.number_input("Max chars per line", min_value=10, max_value=80, value=30, step=1, key="sl_max_chars")
    with col4: max_lines = st.number_input("Max lines per cue", min_value=1, max_value=4, value=2, step=1, key="sl_max_lines")
    out_dir_srt = st.text_input("Output folder", "outputs/srt_per_clip", key="sl_out_dir")
    run = st.button("Slice SRT", type="primary", disabled=(master_srt is None or highlights is None), key="sl_btn")
    if run:
        with st.spinner("Slicing SRT into per-clip files…"):
            ok, msg, files = slice_srt_per_clip_streamlit(
                master_srt_path=master_srt, highlights_json_path=highlights,
                out_dir=out_dir_srt, pad=float(pad), max_sec=float(max_sec),
                max_chars=int(max_chars), max_lines=int(max_lines)
            )
        if ok:
            st.success(msg)
            show = files[:3] if files else []
            for fp in show:
                st.caption(fp)
                try:
                    st.code(Path(fp).read_text(encoding="utf-8")[:800] + ("..." if Path(fp).stat().st_size > 800 else ""), language="text")
                except Exception: pass
            try:
                zip_base = Path(out_dir_srt).as_posix().rstrip("/").replace("/", "_")
                zip_path = shutil.make_archive(zip_base, "zip", out_dir_srt)
                with open(zip_path, "rb") as f:
                    st.download_button("Download all SRTs (ZIP)", f, file_name=Path(zip_path).name, key="sl_zip")
            except Exception as e:
                st.info(f"SRTs saved in `{out_dir_srt}` (ZIP failed: {e})")
        else:
            st.error(msg)

# --- Tab 8: Add Background ---
with tabs[7]:
    st.header("🖼️ Add Background to Clips")
    in_dir_bg = st.text_input("Input clips folder", "data/clips", key="bg_in_dir")
    out_dir_bg = st.text_input("Output folder", "data/composite", key="bg_out_dir")
    bg_mode = st.radio("Orientation", ["portrait", "landscape"], horizontal=True, key="bg_orientation")
    fg_size = st.number_input(
        "Foreground size (optional)",
        min_value=0, max_value=2000, value=0,
        help="Height if portrait, Width if landscape. Leave 0 for default (1600/960).",
        key="bg_fg_size"
    )
    bg_file = st.text_input("Background image", "assets/backgrounds/bgcontentlabs.jpg", key="bg_file")
    run = st.button("Add Background", type="primary", key="bg_btn")
    if run:
        with st.spinner("Compositing clips…"):
            ok, msg, files = add_background_streamlit(
                in_dir=in_dir_bg,
                out_dir=out_dir_bg,
                bg_img=bg_file,
                mode=bg_mode,
                fg_size=fg_size if fg_size > 0 else None
            )
        if ok:
            st.success(msg)
            for f in files[:3]:
                st.video(f)
            try:
                zip_base = Path(out_dir_bg).as_posix().rstrip("/").replace("/", "_")
                zip_path = shutil.make_archive(zip_base, "zip", out_dir_bg)
                with open(zip_path, "rb") as f:
                    st.download_button("Download all composites (ZIP)", f, file_name=Path(zip_path).name, key="bg_zip")
            except Exception as e:
                st.info(f"Composites saved in `{out_dir_bg}` (ZIP failed: {e})")
        else:
            st.error(msg)

# --- Tab 9: Burn Subtitles ---
with tabs[8]:
    st.header("🔤 Burn Subtitles")
    composite_dir = st.text_input("Composite video folder", "data/composite", key="bs_composite_dir")
    srt_dir = st.text_input("SRT folder", "outputs/srt_per_clip", key="bs_srt_dir")
    out_dir_final = st.text_input("Output folder", "data/final", key="bs_out_dir")
    col1, col2, col3, col4 = st.columns(4)
    with col1: font = st.text_input("Font", "Arial", key="bs_font")
    with col2: fontsize = st.number_input("Font size", 12, 72, 20, key="bs_fontsize")
    with col3: margin_v = st.number_input("Margin V", 0, 100, 30, key="bs_marginv")
    with col4: outline = st.number_input("Outline thickness", 0, 10, 2, key="bs_outline")
    run = st.button("Burn Subtitles", type="primary", key="bs_btn")
    if run:
        with st.spinner("Burning subtitles..."):
            ok, msg, files = burn_subtitles_streamlit(
                composite_dir=composite_dir, srt_dir=srt_dir, out_dir=out_dir_final,
                font=font, fontsize=int(fontsize), margin_v=int(margin_v), outline=int(outline)
            )
        if ok:
            st.success(msg)
            for f in files[:3]: st.video(f)
            try:
                zip_base = Path(out_dir_final).as_posix().rstrip("/").replace("/", "_")
                zip_path = shutil.make_archive(zip_base, "zip", out_dir_final)
                with open(zip_path, "rb") as f:
                    st.download_button("Download all final videos (ZIP)", f, file_name=Path(zip_path).name, key="bs_zip")
            except Exception as e:
                st.info(f"Videos saved in `{out_dir_final}` (ZIP failed: {e})")
        else:
            st.error(msg)

# --- Tab 10: Final Output Browser ---
with tabs[9]:
    st.header("📦 Final Output Browser")
    cols = st.columns(3)
    roots = ["downloads", "outputs/transcripts", "outputs/highlights", "outputs/srt_per_clip", "data/clips", "data/composite", "data/final"]
    for i, root in enumerate(roots):
        with cols[i % 3]:
            st.subheader(root)
            files = list_dir(root)
            if not files:
                st.caption("— empty —")
                continue
            for f in files:
                p = Path(root) / f
                ext = p.suffix.lower()
                if ext in (".mp4",".mov",".m4v",".webm"):
                    st.video(str(p))
                elif ext in (".json",".srt",".txt"):
                    with st.expander(f):
                        try:
                            st.code(p.read_text(encoding="utf-8")[:2000], language="text")
                        except Exception:
                            st.write(f)
                else:
                    st.write(f)
