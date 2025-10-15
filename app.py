import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import gradio as gr
import os

# -------------------------------
# ส่วนที่ 1: สร้างและฝึกโมเดล Decision Tree
# -------------------------------
model_accuracy = 0.0
best_params = {}
history_file = "wine_history.csv"

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
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(dt, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, predictions)

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
# ส่วนที่ 2: ฟังก์ชันทำนาย + บันทึกประวัติ
# -------------------------------
def predict_quality(*features):
    """ทำนายคุณภาพไวน์และบันทึกข้อมูล"""
    if model is None or scaler is None:
        return "❌ ยังไม่ได้ฝึกโมเดล", None

    try:
        features_dict = {col: val for col, val in zip(FEATURE_COLUMNS, features)}
        df_new = pd.DataFrame([features_dict])
        df_new = df_new[FEATURE_COLUMNS]
        scaled = scaler.transform(df_new)

        probs = model.predict_proba(scaled)[0]
        pred = np.argmax(probs)
        conf = probs[pred]

        # ข้อความผลลัพธ์ (แสดงทั้งสองความน่าจะเป็น)
        result_text = (
            f"🍷 ผลการทำนาย:\n\n"
            f"- ไวน์คุณภาพดี (Good): {probs[1]*100:.2f}%\n"
            f"- ไวน์คุณภาพไม่ดี (Bad): {probs[0]*100:.2f}%\n\n"
            f"📌 สรุปผล: {'✅ ไวน์คุณภาพดี' if pred == 1 else '⚠️ ไวน์คุณภาพไม่ดี'}"
        )

        text_label = "ไวน์คุณภาพดี" if pred == 1 else "ไวน์คุณภาพไม่ดี"

        # บันทึกประวัติ
        df_new["predicted_label"] = text_label
        df_new["prob_good"] = f"{probs[1]*100:.2f}%"
        df_new["prob_bad"] = f"{probs[0]*100:.2f}%"
        df_new["model_used"] = "Decision Tree"
        df_new["accuracy"] = f"{model_accuracy*100:.2f}%"

        if os.path.exists(history_file):
            df_old = pd.read_csv(history_file)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(history_file, index=False)

        return result_text, df_all.tail(10)

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {e}", None


def clear_inputs():
    """ล้างเฉพาะช่องกรอก โดยไม่ล้างประวัติ"""
    return [None] * len(FEATURE_COLUMNS)


# -------------------------------
# ส่วนที่ 3: หน้าเว็บ Gradio
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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose", secondary_hue="rose")) as app:
    gr.HTML(
        """
        <div style='background-color:#1a001f;padding:20px;border-radius:15px;color:white;text-align:center'>
        <h1>🍷 Grandeur Wine Analyzer</h1>
        <p>ระบบวิเคราะห์คุณภาพไวน์ด้วย Decision Tree</p>
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

    gr.Examples(
        examples=[sample_good, sample_bad],
        inputs=inputs_list,
        label="📍 ตัวอย่างไวน์คุณภาพดี / ไวน์คุณภาพไม่ดี",
    )

    with gr.Row():
        predict_btn = gr.Button("🔮 ทำนายคุณภาพไวน์", variant="primary")
        clear_btn = gr.Button("🧹 ล้างข้อมูลเฉพาะช่องกรอก")

    output_text = gr.Textbox(
        label="ผลการทำนาย (Prediction Result)",
        interactive=False,
        lines=6,
        show_copy_button=True,
    )

    history_output = gr.Dataframe(
        headers=FEATURE_COLUMNS + ["predicted_label", "prob_good", "prob_bad", "model_used", "accuracy"],
        label="📜 ประวัติการทำนายล่าสุด",
        interactive=False
    )

    predict_btn.click(
        fn=predict_quality,
        inputs=inputs_list,
        outputs=[output_text, history_output]
    )

    clear_btn.click(fn=clear_inputs, inputs=None, outputs=inputs_list)

    with gr.Accordion("📘 คำอธิบายค่าทางเคมีของไวน์", open=False):
        gr.Markdown(
            """
            - **Fixed Acidity:** กรดหลักที่ไม่ระเหย  
            - **Volatile Acidity:** หากค่าสูงจะทำให้ไวน์มีกลิ่นเหมือนน้ำส้มสายชู  
            - **Citric Acid:** เพิ่มความสดชื่นให้ไวน์  
            - **Residual Sugar:** น้ำตาลที่เหลือหลังการหมัก  
            - **Chlorides:** ปริมาณเกลือในไวน์  
            - **Free/Total Sulfur Dioxide:** ป้องกันการบูดเสีย  
            - **Density:** ความหนาแน่นของไวน์  
            - **pH:** ค่าความเป็นกรด-ด่าง  
            - **Sulphates:** เพิ่มความคงตัวของไวน์  
            - **Alcohol:** ดีกรีแอลกอฮอล์ในไวน์  
            """
        )

    gr.HTML(
        "<p style='text-align:center;color:grey;font-size:0.8em;'>สร้างโดย Grandeur Wine AI — ใช้เทคนิค Decision Tree</p>"
    )

# -------------------------------
# ส่วนที่ 4: รันโปรแกรม (รองรับ Render)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
