import json
import logging
import re
from pathlib import Path
from bs4 import BeautifulSoup
import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def clean_html_to_markdown(html_content: str) -> str:
    """
    Cleans Vietnamese legal HTML and converts it to basic structured Markdown.
    Focuses on preserving 'Chương', 'Điều', 'Khoản' hierarchy.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Process paragraphs and divisions
    lines = []
    for tag in soup.find_all(['p', 'div', 'tr']):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
            
        # Hierarchy detection
        # Chương X
        if re.match(r"^(Chương)\s+[IVX\d]+", text, re.IGNORECASE):
            lines.append(f"\n## {text}\n")
        # Điều X
        elif re.match(r"^(Điều)\s+\d+", text, re.IGNORECASE):
            lines.append(f"\n### {text}\n")
        # Mục X
        elif re.match(r"^(Mục)\s+\d+", text, re.IGNORECASE):
            lines.append(f"\n#### {text}\n")
        else:
            lines.append(text)

    # Join and clean up whitespace
    content = "\n".join(lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()

def consolidate_hf(data_dir: Path, output_file: Path):
    """
    Merges metadata and content JSONL files.
    Standardizes everything to String IDs.
    """
    metadata_path = data_dir / "metadata" / "metadata_all.jsonl"
    content_path = data_dir / "content" / "content_all.jsonl"
    
    if not metadata_path.exists() or not content_path.exists():
        logger.error(f"Missing input files in {data_dir}")
        return

    # 1. Load Metadata into memory (indexed by ID)
    # Metadata is relatively small (~150k IDs), should fit in RAM.
    logger.info("Loading metadata into memory...")
    meta_map = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                meta_id = str(item.get("id"))
                meta_map[meta_id] = item
            except:
                continue
    logger.info(f"Loaded {len(meta_map):,} metadata records.")

    # 2. Stream Content and join
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Consolidating and cleaning content -> {output_file}")
    count_joined = 0
    count_orphans = 0
    
    with open(content_path, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm.tqdm(f_in, desc="Processing documents"):
            try:
                content_item = json.loads(line)
                doc_id = str(content_item.get("id"))
                
                # Fetch metadata
                meta = meta_map.get(doc_id, {})
                if meta:
                    count_joined += 1
                else:
                    count_orphans += 1
                
                # Clean HTML
                raw_html = content_item.get("content_html", "")
                markdown_text = clean_html_to_markdown(raw_html)
                
                # Create consolidated record
                refined = {
                    "doc_id": doc_id,
                    "title": meta.get("so_hieu", f"Document {doc_id}"),
                    "type": meta.get("loai_van_ban", "Unknown"),
                    "issue_date": meta.get("ngay_ban_hanh"),
                    "authority": meta.get("co_quan_ban_hanh"),
                    "status": meta.get("tinh_trang_hieu_luc"),
                    "content_markdown": markdown_text,
                    "raw_metadata": meta
                }
                
                f_out.write(json.dumps(refined, ensure_ascii=False) + "\n")
                
            except Exception as e:
                # logger.debug(f"Error processing row: {e}")
                continue

    logger.info("Consolidation complete.")
    logger.info(f"  Joined documents: {count_joined:,}")
    logger.info(f"  Orphan documents (no metadata): {count_orphans:,}")
    logger.info(f"  Total processed: {count_joined + count_orphans:,}")

if __name__ == "__main__":
    DATA_DIR = Path("data/huggingface")
    OUT_FILE = Path("data/processed/hf_processed.jsonl")
    consolidate_hf(DATA_DIR, OUT_FILE)
