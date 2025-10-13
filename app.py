import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

# -------------------------------
# ส่วนที่ 1: สร้างและฝึกโมเดล
# -------------------------------
model_accuracy = 0.0
best_params = {}

try:
    df = pd.read_csv("winequality-red.csv")
    df["quality_label"] = df["quality"].apply(lambda v: 1 if v >= 7 else 0)

    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]

    FEATURE_COLUMNS = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, predictions)

    print("✅ โมเดลฝึกเสร็จเรียบร้อย")
    print(f"📊 ความแม่นยำ: {model_accuracy*100:.2f}%")
    print(f"Best Parameters: {best_params}")

except FileNotFoundError:
    print("❌ ไม่พบไฟล์ winequality-red.csv")
    FEATURE_COLUMNS = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "ph", "sulphates", "alcohol"
    ]
    model = None
    scaler = None

# -------------------------------
# ส่วนที่ 2: ฟังก์ชันทำนาย
# -------------------------------
def predict_quality(*features):
    if model is None or scaler is None:
        return "❌ ยังไม่ได้ฝึกโมเดล"
    try:
        features_dict = {col: val for col, val in zip(FEATURE_COLUMNS, features)}
        df_new = pd.DataFrame([features_dict])
        df_new = df_new[FEATURE_COLUMNS]
        scaled = scaler.transform(df_new)
        probs = model.predict_proba(scaled)[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        text = "🍷 ไวน์คุณภาพดี" if pred == 1 else "⚠️ ไวน์คุณภาพไม่ดี"
        return f"{text} (ความมั่นใจ {conf:.2%})"
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {e}"

# -------------------------------
# ส่วนที่ 3: หน้าเว็บ Gradio (ธีมไวน์เข้ม)
# -------------------------------
feature_translations = {
    "fixed acidity": "ความเข้มข้นของกรดคงที่",
    "volatile acidity": "ความเป็นกรดระเหย",
    "citric acid": "กรดซิตริก",
    "residual sugar": "น้ำตาลคงเหลือ",
    "chlorides": "คลอไรด์",
    "free sulfur dioxide": "ซัลเฟอร์ไดออกไซด์อิสระ",
    "total sulfur dioxide": "ซัลเฟอร์ไดออกไซด์ทั้งหมด",
    "density": "ความหนาแน่น",
    "ph": "ค่าความเป็นกรด-ด่าง",
    "sulphates": "ซัลเฟต",
    "alcohol": "ดีกรีแอลกอฮอล์"
}

sample_good = [7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0]
sample_bad = [6.0, 0.90, 0.00, 5.0, 0.10, 5.0, 15.0, 0.9970, 3.20, 0.40, 8.5]

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="rose", secondary_hue="rose")
) as demo:
    gr.HTML(
        """
        <div style='background-color:#1a001f;padding:20px;border-radius:15px;color:white;text-align:center'>
        <h1>🍷 Grandeur Wine Analyzer</h1>
        <p>ระบบวิเคราะห์คุณภาพไวน์</p>
        </div>
        """
    )

    gr.HTML(
        f"<div style='text-align:center;color:#d8a3ff;'><h3>📊 ความแม่นยำของโมเดล: {model_accuracy*100:.2f}%</h3></div>"
    )

    inputs_list = [
        gr.Number(
            label=f"{col.replace('_', ' ').title()} ({feature_translations.get(col, '')})",
            precision=3,
        )
        for col in FEATURE_COLUMNS
    ]

    # ตัวอย่าง
    gr.Examples(
        examples=[sample_good, sample_bad],
        inputs=inputs_list,
        label="📍 ตัวอย่างไวน์คุณภาพดี / ไวน์คุณภาพไม่ดี",
    )

    with gr.Row():
        predict_btn = gr.Button("🔮 ทำนายคุณภาพไวน์", variant="primary")
        clear_btn = gr.ClearButton(value="ล้างข้อมูล")

    output = gr.Textbox(
        label="ผลการทำนาย (Prediction Result)",
        interactive=False,
        lines=2,
        show_copy_button=True,
    )

    with gr.Accordion("📘 คำอธิบายค่าทางเคมีของไวน์ (Feature Descriptions)", open=False):
        gr.Markdown(
            """
            - **Fixed Acidity:** กรดหลักที่ไม่ระเหย ช่วยให้ไวน์มีรสสมดุล  
            - **Volatile Acidity:** หากค่าสูงจะทำให้ไวน์มีกลิ่นเหมือนน้ำส้มสายชู  
            - **Citric Acid:** เพิ่มความสดชื่นเล็กน้อยให้ไวน์  
            - **Residual Sugar:** น้ำตาลที่เหลือหลังการหมัก ยิ่งมากไวน์จะยิ่งหวาน  
            - **Chlorides:** ปริมาณเกลือในไวน์  
            - **Free/Total Sulfur Dioxide:** ป้องกันการบูดเสียและยับยั้งจุลินทรีย์  
            - **Density:** ความหนาแน่นของไวน์ (ใกล้เคียงน้ำ)  
            - **pH:** ค่าความเป็นกรด-ด่างของไวน์  
            - **Sulphates:** ช่วยเพิ่มความมั่นคงของไวน์  
            - **Alcohol:** ดีกรีแอลกอฮอล์ในไวน์  
            """
        )

    gr.HTML(
        "<p style='text-align:center;color:grey;font-size:0.8em;'>สร้างโดย Grandeur Wine AI — ใช้เทคนิค Random Forest (Decision Trees Ensemble)</p>"
    )

    predict_btn.click(fn=predict_quality, inputs=inputs_list, outputs=output)
    clear_btn.add(inputs_list + [output])

# -------------------------------
# ส่วนที่ 4: รันโปรแกรม
# -------------------------------
if __name__ == "__main__":
    demo.launch()
