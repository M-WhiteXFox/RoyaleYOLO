import os
import re
import cloudscraper
from bs4 import BeautifulSoup

TXT_FILE   = "card.txt"

OUTPUT_DIR = "images"

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def normalize_url(resize_url):
    m = re.search(r"(https?://[^/]+)(?:/cdn-cgi/image/[^/]+)?(/static/.*)", resize_url)
    return (m.group(1) + m.group(2)) if m else resize_url

# 1. 创建 cloudscraper session（自动处理 Cloudflare 挑战）
session = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)

# 2. 先请求主页，拿到所有 clearance cookie
session.get("https://royaleapi.com/")

def get_image_urls(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return [img["src"] for img in soup.find_all("img") if img.get("src")]

def download_one(url):
    # 尝试两个 URL：缩放版 & 原图版
    candidates = [url, normalize_url(url)]
    filename = os.path.basename(candidates[-1].split("?")[0])
    save_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(save_path):
        print(f"✔ 跳过（已存在）: {filename}")
        return

    for u in candidates:
        try:
            r = session.get(u, timeout=15)
            r.raise_for_status()
            with open(save_path, "wb") as wf:
                wf.write(r.content)
            print(f"✔ 已保存: {filename} （{u}）")
            return
        except Exception as e:
            code = getattr(e.response, "status_code", None) if hasattr(e, 'response') else None
            print(f"  → {u} 下载失败 HTTP {code or ''}，{e}")
    print(f"✖ 最终失败: {filename}")

def main():
    ensure_dir(OUTPUT_DIR)
    urls = get_image_urls(TXT_FILE)
    print(f"共{len(urls)}张，开始下载…")
    for u in urls:
        download_one(u)

if __name__ == "__main__":
    main()
