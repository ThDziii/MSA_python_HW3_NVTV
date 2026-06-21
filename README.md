# Heart Disease Streamlit Demo

Dự đoán bệnh tim bằng ML: train model trong notebook, predict qua web app Streamlit.

## Cấu trúc project

```
HW3/
├── train.ipynb          ← preprocess · normalize · train · save model
├── app.py               ← UI Streamlit (chỉ predict, không train)
├── Data_raw/            ← dữ liệu CSV
├── models/              ← model đã lưu (tạo sau khi chạy notebook)
└── requirements.txt
```

## Cài đặt (lần đầu)

```bash
cd HW3
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Nên dùng **Python 3.12** (tránh 3.14 — XGBoost hay lỗi trên Mac).

## Chạy project

**Bước 1:** Mở `train.ipynb` → chọn kernel Python 3.12 → **Run All**

→ Lưu model vào `models/`:
- `heart_disease_models.joblib`
- `best_model.joblib`
- `training_report.csv`

**Bước 2:** Chạy web app

```bash
source .venv/bin/activate
streamlit run app.py
```

Mở **http://localhost:8501** → nhập thông tin bệnh nhân → bấm **Predict**.

## Dữ liệu

Notebook dùng:

- `Data_raw/raw_train.csv` — train
- `Data_raw/raw_val.csv` — đánh giá accuracy

CSV cần các cột:

`age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal, target`

## Lưu ý

- App **không train lại** — chỉ load model từ `models/`
- Sửa data hoặc model → chạy lại `train.ipynb` (Run All) trước khi predict
# MSA_python_HW3_NVTV
