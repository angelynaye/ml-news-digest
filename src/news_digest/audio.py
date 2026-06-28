"""TTS for per-item audio + combined 'listen to all'.

Two providers, configurable:
  - polly: AWS Polly neural voices (requires polly:SynthesizeSpeech IAM permission).
  - say:   macOS built-in `say` command. Free, offline, decent quality with
           the Siri voices (~250-char/sec). Pipes through `afconvert` to MP3.

Output: data/audio/{item_id}.mp3 served by server.py at /audio/{item_id}.mp3.
"""
from __future__ import annotations
import logging
import os
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def _script_for_item(item: dict) -> str:
    parts = [f"{item['title']}, from {item['source']}."]
    if item.get("tldr"):
        parts.append(item["tldr"])
    if item.get("impact_bullets"):
        parts.append("Why it matters: " + " ; ".join(item["impact_bullets"][:3]))
    if item.get("bullets"):
        parts.append("Key points: " + " ; ".join(item["bullets"][:3]))
    return " ".join(parts).replace("&", "and")


def _combined_script(items: list[dict]) -> str:
    parts = ["Daily ML infra digest. "]
    for i, it in enumerate(items, 1):
        parts.append(f"Item {i}: {it['title']}, from {it['source']}.")
        if it.get("tldr"):
            parts.append(it["tldr"])
        if it.get("impact_bullets"):
            parts.append("Why it matters: " + " ; ".join(it["impact_bullets"][:2]))
    return " ".join(parts).replace("&", "and")


# ---------------- Polly ----------------

def _synth_polly(client, text: str, voice_id: str, out_path: Path) -> bool:
    ssml = f"<speak><prosody rate='medium'>{text}</prosody></speak>"
    try:
        resp = client.synthesize_speech(
            Text=ssml, TextType="ssml",
            OutputFormat="mp3", VoiceId=voice_id, Engine="neural",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp["AudioStream"].read())
        return True
    except Exception as e:
        log.warning("Polly synth failed for %s: %s", out_path.name, e)
        return False


# ---------------- macOS `say` ----------------

def _synth_say(text: str, voice: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write text to a temp file (avoids /dev/stdin quirks + argv length limits).
    txt_path = Path(tempfile.mkstemp(suffix=".txt")[1])
    aiff_path = Path(tempfile.mkstemp(suffix=".aiff")[1])
    try:
        txt_path.write_text(text)
        proc = subprocess.run(
            ["say", "-v", voice, "-o", str(aiff_path), "-f", str(txt_path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not aiff_path.exists() or aiff_path.stat().st_size == 0:
            log.warning("say failed (rc=%s): %s", proc.returncode, proc.stderr.strip())
            return False
        # Convert AIFF → MP3 via afconvert. -f mp4f -d aac produces AAC in an mp4
        # container which browsers play fine; we name it .mp3 for convenience.
        conv = subprocess.run(
            ["afconvert", str(aiff_path), str(out_path),
             "-f", "mp4f", "-d", "aac", "-b", "64000"],
            capture_output=True, text=True, timeout=120,
        )
        if conv.returncode != 0 or not out_path.exists():
            log.warning("afconvert failed (rc=%s): %s — keeping .aiff", conv.returncode, conv.stderr.strip())
            out_path.write_bytes(aiff_path.read_bytes())
        return True
    except Exception as e:
        log.warning("say pipeline failed: %s", e)
        return False
    finally:
        for p in (txt_path, aiff_path):
            try: p.unlink()
            except Exception: pass


# ---------------- Dispatch ----------------

def generate_audio(items: list[dict], audio_cfg: dict) -> dict:
    if not audio_cfg.get("enabled"):
        return {}
    provider = audio_cfg.get("provider", "polly").lower()
    audio_dir = Path(audio_cfg.get("output_dir", "data/audio"))
    results: dict[str, bool] = {}

    polly_client = None
    if provider == "polly":
        try:
            import boto3
            region = audio_cfg.get("region", os.environ.get("AWS_REGION", "us-east-1"))
            polly_client = boto3.client("polly", region_name=region)
        except Exception as e:
            log.warning("Polly client init failed: %s — falling back to `say`", e)
            provider = "say"

    polly_voice_en = audio_cfg.get("polly_voice", "Joanna")
    polly_voice_zh = audio_cfg.get("polly_voice_zh", "Zhiyu")
    say_voice_en   = audio_cfg.get("say_voice", "Samantha")
    say_voice_zh   = audio_cfg.get("say_voice_zh", "Tingting")

    def synth(text: str, out_path: Path, lang: str = "en") -> bool:
        polly_voice = polly_voice_zh if lang == "zh" else polly_voice_en
        say_voice   = say_voice_zh   if lang == "zh" else say_voice_en
        if provider == "polly":
            ok = _synth_polly(polly_client, text, polly_voice, out_path)
            if ok:
                return True
            log.info("Polly failed; falling back to `say` for %s", out_path.name)
            return _synth_say(text, say_voice, out_path)
        return _synth_say(text, say_voice, out_path)

    for it in items:
        out = audio_dir / f"{it['id']}.mp3"
        results[it["id"]] = synth(_script_for_item(it), out, lang=it.get("lang", "en"))
        if results[it["id"]]:
            it["audio_url"] = f"/audio/{it['id']}.mp3"
    # Combined audio: split into English and Chinese sub-tracks so neither voice
    # mangles the other's text.
    en_items = [it for it in items if it.get("lang") != "zh"]
    zh_items = [it for it in items if it.get("lang") == "zh"]
    if en_items:
        synth(_combined_script(en_items), audio_dir / "combined.mp3", lang="en")
        results["__combined__"] = True
    if zh_items:
        synth(_combined_script(zh_items), audio_dir / "combined_zh.mp3", lang="zh")
        results["__combined_zh__"] = True
    return results
