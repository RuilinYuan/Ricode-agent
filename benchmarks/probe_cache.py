"""探针：检查代理对 cache_control 与 max_tokens=1 的支持，打印原始响应。"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: F401  加载 .env
import httpx

base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
if not base.endswith("/v1"):
    base += "/v1"
key = os.environ["ANTHROPIC_API_KEY"]
model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

long_system = "你是助手。" + "测试文本。" * 300  # 足够长的稳定前缀

def probe(label, messages, max_tokens=16):
    r = httpx.post(f"{base}/chat/completions",
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": model, "messages": messages, "max_tokens": max_tokens},
                   timeout=120)
    print(f"\n=== {label} (HTTP {r.status_code}) ===")
    try:
        data = r.json()
        print("usage:", json.dumps(data.get("usage"), ensure_ascii=False))
        msg = (data.get("choices") or [{}])[0].get("message", {})
        print("content:", str(msg.get("content"))[:80])
    except Exception:
        print("非 JSON 响应:", r.text[:300])

probe("纯字符串 system",
      [{"role": "system", "content": long_system}, {"role": "user", "content": "ping"}])

sysmsg = {"role": "system", "content": [
    {"type": "text", "text": long_system, "cache_control": {"type": "ephemeral"}}]}
probe("带 cache_control 第1次(写)", [sysmsg, {"role": "user", "content": "ping"}])
probe("带 cache_control 第2次(应命中)", [sysmsg, {"role": "user", "content": "ping"}])
probe("max_tokens=1",
      [{"role": "system", "content": long_system}, {"role": "user", "content": "ping"}], 1)
