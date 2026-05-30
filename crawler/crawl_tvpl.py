from playwright.sync_api import sync_playwright
import time
import os

#1. List documents
DOCUMENTS = [
    {
        "name": "BLLĐ_2019",
        "url": "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-333670.aspx",
        "type": "base"
    },

    {
        "name": "VBHN_125_2025",
        "url": "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Van-ban-hop-nhat-125-VBHN-VPQH-2025-Bo-luat-Lao-dong-so-45-2019-QH14-672381.aspx",
        "type": "current"
    },


]

def crawl_multiple_documents():
    print("START MULTI-CRAWL PIPELINE")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        page = context.new_page()

        for doc in DOCUMENTS:
            print(f"\n--- Crawling: {doc['name']} ---")

            try:
                page.goto(doc["url"], wait_until="domcontentloaded", timeout=60000)
                time.sleep(5)

                content = page.locator(".content1").first

                if content.is_visible():
                    raw_text = content.inner_text()

                    if len(raw_text) > 500:
                        #build folder path theo type
                        folder_path = os.path.join(base_dir, "..", "data", "raw", doc["type"])
                        
                        # create dict if not exist
                        os.makedirs(folder_path, exist_ok=True)

                        file_path = os.path.join(folder_path, f"{doc['name']}.txt")

                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(raw_text)

                        print(f"Saved: {file_path}")
                    else:
                        print("Content too short")

                else:
                    print("Cannot find content")

            except Exception as e:
                print(f"Error: {e}")

        browser.close()
        print("\n DONE ALL DOCUMENTS")

if __name__ == "__main__":
    crawl_multiple_documents()