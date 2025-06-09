import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 加载预训练的随机森林模型
@st.cache_resource
def load_model():
    return joblib.load('rf.pkl')

# 烧伤类型映射
burn_type_mapping = {
    0: "空白 (无烧伤)",
    1: "浅二度烧伤",
    2: "深二度烧伤",
    3: "三度烧伤",
    4: "电击烧伤",
    5: "火焰烧伤"
}

# 创建SHAP力图
def create_shap_plot(model, input_data):
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(input_data)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    plt.tight_layout()
    
    # 将图表转换为图像对象
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    return buf

# 主应用
def main():
    st.title("烧伤类型识别系统")
    st.write("请输入以下生物标志物指标值，系统将预测烧伤类型:")
    
    # 创建输入表单
    with st.form("burn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            bg1 = st.number_input("BG1", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            il1b = st.number_input("IL-1β", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            bg2 = st.number_input("BG2", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            
        with col2:
            bg4 = st.number_input("BG4", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            cyclic_amp = st.number_input("Cyclic AMP", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        
        submitted = st.form_submit_button("预测烧伤类型")
    
    if submitted:
        # 准备输入数据
        input_data = pd.DataFrame([[bg1, il1b, bg2, bg4, cyclic_amp]], 
                                 columns=['BG1', 'IL-1β', 'BG2', 'BG4', 'Cyclic AMP'])
        
        # 加载模型
        model = load_model()
        
        # 进行预测
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # 显示结果
        st.subheader("预测结果")
        st.write(f"预测烧伤类型: **{burn_type_mapping[prediction]}**")
        
        # 显示概率
        st.write("\n各类别预测概率:")
        for i, prob in enumerate(probabilities):
            st.write(f"- {burn_type_mapping[i]}: {prob*100:.2f}%")
        
        # 显示SHAP力图
        st.subheader("特征重要性分析 (SHAP)")
        shap_plot = create_shap_plot(model, input_data)
        st.image(shap_plot, use_column_width=True)
        
        # 解释SHAP图
        st.write("""
        **SHAP图说明:**
        - 该图显示了每个特征对模型预测的贡献程度
        - 条形的长度表示特征的重要性
        - 正值表示该特征增加了预测为当前类别的概率
        - 负值表示该特征减少了预测为当前类别的概率
        """)

if __name__ == "__main__":
    main()