# ========================
# 1. Load Dataset Manual
# ========================
def load_dataset(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Ambil header kolom
    header = lines[0].strip().split(',')

    # Baca setiap baris data sebagai dictionary
    for line in lines[1:]:
        values = line.strip().split(',')
        row = {}
        for i in range(len(header)):
            key = header[i]
            if key == 'Outcome':
                row[key] = int(values[i])  # Target (0 atau 1)
            else:
                row[key] = float(values[i])  # Fitur numerik
        data.append(row)

    # Tampilkan informasi awal
    print("\nJumlah data:", len(data))
    print("Fitur:", header[:-1])
    print("Target: Outcome (0 = Tidak Diabetes, 1 = Diabetes)")
    print("\n5 data pertama:")
    for i in range(5):
        print(data[i])
    
    return data

# =======================================
# 2. Preprocessing: Median & Normalisasi
# =======================================
def preprocess_data(data):
    cols = list(data[0].keys())
    cols.remove('Outcome')  # Hilangkan kolom target

    # Hitung median untuk setiap fitur (kecuali 0)
    medians = {}
    for col in cols:
        non_zero = []
        for row in data:
            if row[col] != 0:
                non_zero.append(row[col])
        sorted_vals = sorted(non_zero)
        n = len(sorted_vals)
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        medians[col] = median

    # Ganti nilai 0 dengan median
    for row in data:
        for col in cols:
            if row[col] == 0:
                row[col] = medians[col]
    
    # Normalisasi min-max
    for col in cols:
        values = [row[col] for row in data]
        min_val = min(values)
        max_val = max(values)
        for row in data:
            if max_val != min_val:
                row[col] = (row[col] - min_val) / (max_val - min_val)
    
    return data

# ================================
# 3. Fungsi bantu dasar (tanpa lib)
# ================================
def unique(values):
    # Kembalikan list nilai unik
    uniq = []
    for val in values:
        if val not in uniq:
            uniq.append(val)
    return uniq

def count_labels(values):
    # Hitung frekuensi setiap label
    counter = {}
    for v in values:
        if v in counter:
            counter[v] += 1
        else:
            counter[v] = 1
    return counter

# ====================================================
# 4. Kelas Decision Tree dengan pendekatan ID3 (entropy)
# ====================================================
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # Gabungkan fitur dan label ke satu dataset
        dataset = []
        for i in range(len(X)):
            row = X[i].copy()
            row['Outcome'] = y[i]
            dataset.append(row)
        self.tree = self._build_tree(dataset, 0)

    def _build_tree(self, dataset, depth):
        y = [row['Outcome'] for row in dataset]

        # Kondisi berhenti: semua label sama, kedalaman maksimal, atau data terlalu sedikit
        if self._all_same(y) or depth >= self.max_depth or len(dataset) < self.min_samples_split:
            return {'leaf': True, 'value': self._most_common(y)}
        
        # Cari split terbaik
        best = self._best_split(dataset)
        if best is None:
            return {'leaf': True, 'value': self._most_common(y)}
        
        # Rekursi untuk subtree kiri dan kanan
        left = self._build_tree(best['left'], depth + 1)
        right = self._build_tree(best['right'], depth + 1)
        return {
            'leaf': False,
            'feature': best['feature'],
            'threshold': best['threshold'],
            'left': left,
            'right': right
        }

    def _all_same(self, items):
        # Cek apakah semua item identik
        for item in items:
            if item != items[0]:
                return False
        return True

    def _most_common(self, items):
        # Ambil label terbanyak
        counter = count_labels(items)
        max_count = -1
        most = None
        for k in counter:
            if counter[k] > max_count:
                max_count = counter[k]
                most = k
        return most

    def _best_split(self, dataset):
        best_gain = -1
        best_split = None
        features = list(dataset[0].keys())
        features.remove('Outcome')

        for feature in features:
            values = sorted(unique([row[feature] for row in dataset]))
            for i in range(1, len(values)):
                threshold = (values[i-1] + values[i]) / 2
                left = [r for r in dataset if r[feature] <= threshold]
                right = [r for r in dataset if r[feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue

                y_parent = [r['Outcome'] for r in dataset]
                y_left = [r['Outcome'] for r in left]
                y_right = [r['Outcome'] for r in right]
                gain = self._info_gain(y_parent, y_left, y_right)

                # Cetak proses evaluasi split
                # print("\nEvaluasi Split:")
                # print(f"Fitur: {feature}, Threshold: {threshold:.4f}")
                # print(f"Entropy Parent: {self._entropy(y_parent):.4f}")
                # print(f"Entropy Left: {self._entropy(y_left):.4f} (jumlah: {len(y_left)})")
                # print(f"Entropy Right: {self._entropy(y_right):.4f} (jumlah: {len(y_right)})")
                # print(f"Information Gain: {gain:.4f}")

                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}
        return best_split

    def _entropy(self, labels):
        # Hitung entropy secara manual
        counts = count_labels(labels)
        n = len(labels)
        entropy = 0
        for lbl in counts:
            p = counts[lbl] / n
            if p > 0:
                entropy -= p * self._log2(p)
        return entropy

    def _log2(self, x):
        # Hitung log2(x) tanpa library
        if x <= 0:
            return 0

        n = 0
        while x < 1:
            x *= 2
            n -= 1
        while x >= 2:
            x /= 2
            n += 1

        y = x - 1
        term = y
        result = y
        for i in range(2, 10):  # 8 iterasi Taylor
            term *= -y
            result += term / i

        return n + result

    def _info_gain(self, parent, left, right):
        # Rumus ID3: IG = Entropy(parent) - rata-rata entropy anak
        total = len(parent)
        w_left = len(left) / total
        w_right = len(right) / total
        return self._entropy(parent) - (w_left * self._entropy(left) + w_right * self._entropy(right))

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

# =============================
# 5. Manual Split Data Training/Test
# =============================
def split_data(data, test_ratio):
    X = []
    y = []
    for row in data:
        x = row.copy()
        y.append(x.pop('Outcome'))
        X.append(x)

    total = len(X)
    test_size = int(total * test_ratio)

    # Shuffle pseudo-random manual
    for i in range(total):
        j = (i * 17 + 13) % total
        X[i], X[j] = X[j], X[i]
        y[i], y[j] = y[j], y[i]

    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

# ==========================
# 6. Evaluasi Model Manual
# ==========================
def evaluate(y_true, y_pred):
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 1:
            fp += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    pres = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (pres * rec) / (pres + rec) if (pres + rec) > 0 else 0
    print("\nConfusion Matrix:")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Akurasi: {acc:.2f}")
    print(f"Presisi: {pres:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

# =======================
# 7. MAIN PROGRAM
# =======================
if __name__ == "__main__":
    print("Decision Tree - Prediksi Diabetes (Metode ID3 - Entropy + Information Gain)")
    data = load_dataset("diabetes.csv")
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, test_ratio=0.2)

    tree = DecisionTree(max_depth=4)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    evaluate(y_test, y_pred)

    print("\nContoh Prediksi:")
    for i in range(5):
        print(f"Data ke-{i+1}: Asli={y_test[i]}, Prediksi={y_pred[i]}")
