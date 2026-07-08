import os
import json
import glob
import math
import re

# KONFIGURASI
database_dir = 'asli'   # Folder berisi JSON referensi tanda tangan asli
test_dir = 'test'       # Folder berisi file JSON yang akan diuji
output_txt = 'test_results.txt' # File output hasil pengujian

GRID_SIZE = 15
SIMILARITY_THRESHOLD = 72.0

# 1. LOAD JSON
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# 2. EKSTRAK ID DARI NAMA FILE (GPDS FORMAT)
# Contoh: "c-001-17 (Copy).json" -> ID "001"
def get_id_from_filename(filename):
    base = os.path.basename(filename)
    # Mencari 3 digit angka pertama berurutan sebagai ID orang
    match = re.search(r'\d{3}', base)
    if match:
        return match.group(0)
    return "UNKNOWN"

# 3. AUTO-CROP (CARI BOUNDING BOX)
def get_bounding_box(img):
    h = len(img)
    w = len(img[0])
    
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    has_pixel = False
    
    for y in range(h):
        for x in range(w):
            if img[y][x] == 1:
                has_pixel = True
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                
    if not has_pixel:
        return 0, 0, 0, 0
    return min_x, min_y, max_x, max_y

# 4. EKSTRAKSI FITUR (GRID DENSITY DENGAN SQUARE PADDING)
def extract_grid_features(img, grid_size=5):
    min_x, min_y, max_x, max_y = get_bounding_box(img)
    
    if max_x == 0 and max_y == 0:
        return [0.0] * (grid_size * grid_size)
        
    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1
    
    # Bounding Box berbentuk Persegi (Square)
    # Mencegah distorsi tanda tangan saat dibagi grid
    max_dim = max(box_w, box_h)
    
    # Hitung pergeseran agar tanda tangan tetap di tengah persegi
    pad_x = (max_dim - box_w) // 2
    pad_y = (max_dim - box_h) // 2
    
    v_min_x = min_x - pad_x
    v_min_y = min_y - pad_y
    
    cell_size = max_dim / grid_size
    
    features = [0.0] * (grid_size * grid_size)
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if img[y][x] == 1:
                grid_x = int((x - v_min_x) / cell_size)
                grid_y = int((y - v_min_y) / cell_size)
                
                # Batasan aman (Clamping)
                if grid_x >= grid_size: grid_x = grid_size - 1
                if grid_y >= grid_size: grid_y = grid_size - 1
                if grid_x < 0: grid_x = 0
                if grid_y < 0: grid_y = 0
                
                idx = grid_y * grid_size + grid_x
                features[idx] += 1.0
                
    # Normalisasi rasio kepadatan
    cell_area = cell_size * cell_size
    if cell_area > 0:
        for i in range(len(features)):
            features[i] = features[i] / cell_area
            
    return features

# 5. PENGUKURAN KESAMAAN (COSINE SIMILARITY)
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1)
    norm_b = sum(b * b for b in vec2)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return (dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))) * 100.0

# 6. INFERENSI (VERIFIKASI 1-KE-1 DENGAN RATA-RATA)
def verify_signature(test_path, database_folder, threshold=85.0):
    test_id = get_id_from_filename(test_path)
    
    # Cari semua file referensi di database yang ID-nya sama dengan gambar test
    reference_files = [f for f in glob.glob(os.path.join(database_folder, '*.json')) 
                    if get_id_from_filename(f) == test_id]
    
    if not reference_files:
        return test_id, 0.0, "ERROR: Tidak ada data referensi untuk ID ini di database."
        
    test_img = load_json(test_path)
    test_features = extract_grid_features(test_img, GRID_SIZE)
    
    total_score = 0.0
    
    # Bandingkan gambar test dengan setiap gambar referensi (yang ID-nya sama)
    for ref_path in reference_files:
        ref_img = load_json(ref_path)
        ref_features = extract_grid_features(ref_img, GRID_SIZE)
        score = cosine_similarity(ref_features, test_features)
        total_score += score
        
    # Hitung rata-rata
    average_score = total_score / len(reference_files)
    
    decision = "ASLI" if average_score >= threshold else "PALSU"
    return test_id, average_score, decision

# =====================================================
# MAIN PROGRAM (BATCH PROCESSING)
# =====================================================
if __name__ == "__main__":
    print("Memulai proses pengujian... Hasil akan disimpan di", output_txt)
    
    # Otomatis memproses semua gambar JSON yang ada di folder 'test'
    test_files = glob.glob(os.path.join(test_dir, '*.json'))
    
    # Membuka file txt untuk menulis (mode 'w')
    with open(output_txt, 'w', encoding='utf-8') as f_out:
        if not test_files:
            msg = f"Tidak ada file .json di dalam folder '{test_dir}'."
            print(msg)
            f_out.write(msg + "\n")
        else:
            for test_path in test_files:
                file_name = os.path.basename(test_path)
                print(f"Sedang mengecek file: {file_name} ...")
                test_id, avg_score, decision = verify_signature(test_path, database_dir, SIMILARITY_THRESHOLD)
                
                # Menulis hasil ke dalam file txt
                if "ERROR" in decision:
                    f_out.write(f"{file_name}, {decision}\n")
                else:
                    f_out.write(f"{file_name}, {avg_score:.2f}%, {decision}\n")
                    
    print("Pengujian selesai. Silakan cek file", output_txt)