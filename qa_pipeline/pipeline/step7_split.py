import json
import random
from pathlib import Path
from collections import defaultdict, Counter

def step7_split(input_path: str, output_dir: str) -> None:
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found. Run Step 6b first.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Loading data from: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"✅ Loaded {len(data)} samples for splitting. Target: 70/15/15 stratified.")
    
    # Separate by class for stratification
    label_to_samples = defaultdict(list)
    for sample in data:
        label = sample.get("routing_label", "none")
        label_to_samples[label].append(sample)
        
    train_data = []
    dev_data = []
    test_data = []
    
    # Cố định seed để đảm bảo quá trình chia dữ liệu lặp lại được (reproducible)
    random.seed(42) 
    
    for label, samples in label_to_samples.items():
        # Xáo trộn từng file trước khi chia
        random.shuffle(samples)
        
        n = len(samples)
        n_train = int(n * 0.70)
        n_dev = int(n * 0.15)
        
        train_data.extend(samples[:n_train])
        dev_data.extend(samples[n_train:n_train+n_dev])
        test_data.extend(samples[n_train+n_dev:])
        
    # Xáo trộn lại các file cuối cùng để nhãn không bị gom cục bộ
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    total = len(train_data) + len(dev_data) + len(test_data)
    
    print("\n📊 Statistical Distribution of Splits:")
    for name, split_data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
        counts = Counter(s.get("routing_label") for s in split_data)
        percent = (len(split_data) / total) * 100
        print(f"   [ {name} ] - {len(split_data)} samples ({percent:.1f}%)")
        for k, v in counts.items():
            print(f"     * {k:16s}: {v}")
            
    # Save files
    with (out_dir / "train.json").open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
        
    with (out_dir / "dev.json").open("w", encoding="utf-8") as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
        
    with (out_dir / "test.json").open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ All splits successfully saved to: {out_dir}/")

if __name__ == "__main__":
    # Để chắc chắn đọc đúng đường dẫn khi chạy từ thư mục Root
    ROOT_DIR = Path(__file__).parent.parent.parent
    INPUT_PATH = ROOT_DIR / "qa_pipeline/data/checkpoints/step6b_augmented.json"
    OUTPUT_DIR = ROOT_DIR / "qa_pipeline/data/final"
    
    step7_split(str(INPUT_PATH), str(OUTPUT_DIR))
