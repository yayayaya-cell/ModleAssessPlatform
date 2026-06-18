import streamlit as st
import pandas as pd
import polars as pl
from openai import OpenAI
from io import StringIO
from utils import *

# ---------- 页面配置 ----------
st.set_page_config(page_title="天猫复购模型评估平台", layout="wide")

# ---------- 自定义CSS ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf5 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1a2e 0%, #1a2f4e 50%, #234a73 100%);
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] * {
        color: #e0e6ef !important;
    }
    .sidebar-title {
        color: #ffd700 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 10px;
    }
    .nav-label {
        color: #64b5f6 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 8px 12px;
        margin-top: 20px;
        border-left: 3px solid #64b5f6;
    }
    div[role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        padding: 14px 18px !important;
        margin: 6px 0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        transition: 0.3s !important;
        cursor: pointer !important;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.15) !important;
        transform: translateX(5px) !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: rgba(100,181,246,0.25) !important;
        border-color: #ffd700 !important;
        color: #ffd700 !important;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    [data-testid="stMetric"] label {
        color: white !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
        border: 2px dashed #2a5298;
        padding: 1.5rem;
    }
    h1 { color: #0f1a2e; font-weight: 700; border-bottom: 3px solid #2a5298; padding-bottom: 10px; }
    h2 { color: #1a2f4e; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- DeepSeek 客户端 ----------
deepseek_client = OpenAI(
    api_key="sk-35522db36d8f465cb24142b1a81d2100",
    base_url="https://api.deepseek.com"
)

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown('<p class="sidebar-title">🧠 复购预测平台</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="nav-label">🧭 功能导航</p>', unsafe_allow_html=True)

    page = st.radio(
        "",
        ["📊  模型评估", "👥  用户数据可视化", "🤖  模型可视化", "🎓  学习助手"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.info("💡 将 CSV / PY 文件拖入对应页面即可分析")
    st.markdown("---")
    st.caption("© 2024 天猫复购模型评估 v2.0")

# ==================== 界面一：模型评估 ====================
if page == "📊  模型评估":
    st.header("📊 模型评估结果分析")
    st.write("上传真实标签文件与预测结果文件，自动对比 label 列并生成评估报告")

    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        st.markdown("### 📋 真实标签文件")
        true_file = st.file_uploader("上传包含真实标签的 CSV 文件", type=["csv"], key="true_file")
        if true_file is not None:
            df_true = load_eval_csv(true_file)
            if df_true.columns.duplicated().any():
                df_true = df_true.loc[:, ~df_true.columns.duplicated()]
            st.success(f"✅ 已上传 ({df_true.shape[0]} 行 × {df_true.shape[1]} 列)")
            st.dataframe(df_true.head(3), use_container_width=True)

    with col_upload2:
        st.markdown("### 📊 预测结果文件")
        pred_file = st.file_uploader("上传包含预测结果的 CSV 文件", type=["csv"], key="pred_file")
        if pred_file is not None:
            df_pred = load_eval_csv(pred_file)
            if df_pred.columns.duplicated().any():
                df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()]
            st.success(f"✅ 已上传 ({df_pred.shape[0]} 行 × {df_pred.shape[1]} 列)")
            st.dataframe(df_pred.head(3), use_container_width=True)

    if true_file is not None and pred_file is not None:
        st.markdown("---")
        st.subheader("🔗 选择 label 列")

        # 自动查找 label 列
        true_label_cols = [c for c in df_true.columns if 'label' in c.lower()]
        pred_label_cols = [c for c in df_pred.columns if 'label' in c.lower()]

        if not true_label_cols:
            true_label_cols = df_true.columns.tolist()
        if not pred_label_cols:
            pred_label_cols = df_pred.columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            true_label_col = st.selectbox("真实文件的 label 列", true_label_cols)
        with col2:
            pred_label_col = st.selectbox("预测文件的 label 列", pred_label_cols)
        with col3:
            prob_cols = [c for c in df_pred.columns if any(k in c.lower() for k in ['prob', 'score'])]
            prob_col = st.selectbox("预测概率列（可选）", ['无'] + prob_cols)

        if st.button("🚀 开始分析", type="primary", use_container_width=True):

            with st.spinner("正在处理数据..."):
                try:
                    # 提取列
                    y_true_raw = df_true[true_label_col]
                    y_pred_raw = df_pred[pred_label_col]
                    y_score_raw = df_pred[prob_col] if prob_col != '无' else None

                    # 对齐长度
                    min_len = min(len(y_true_raw), len(y_pred_raw))
                    if len(y_true_raw) != len(y_pred_raw):
                        st.warning(f"⚠️ 长度不一致，取较短 {min_len} 行")

                    y_true_raw = y_true_raw.iloc[:min_len]
                    y_pred_raw = y_pred_raw.iloc[:min_len]
                    if y_score_raw is not None:
                        y_score_raw = y_score_raw.iloc[:min_len]

                    # 转为数值
                    y_true = pd.to_numeric(y_true_raw, errors='coerce')
                    y_pred = pd.to_numeric(y_pred_raw, errors='coerce')
                    y_score = pd.to_numeric(y_score_raw, errors='coerce') if y_score_raw is not None else None

                    # 去NaN
                    mask = y_true.notna() & y_pred.notna()
                    if y_score is not None:
                        mask &= y_score.notna()
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    if y_score is not None:
                        y_score = y_score[mask]

                    # 转整数标签
                    y_true = y_true.round().astype(int)
                    y_pred = y_pred.round().astype(int)

                    st.info(f"📊 有效数据: {len(y_true)} 行 | 标签值: {sorted(y_true.unique())}")

                    if len(y_true) < 2:
                        st.error("❌ 有效数据不足")
                        st.stop()

                    if y_true.nunique() > 20:
                        st.error(f"❌ 标签类别过多({y_true.nunique()})，请确认选择的是分类标签列")
                        st.stop()

                    # 计算指标
                    metrics = calc_metrics(y_true, y_pred, y_score)

                except Exception as e:
                    st.error(f"❌ 数据处理出错: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())
                    st.stop()

            # 显示指标
            st.markdown("---")
            st.subheader("📈 核心评估指标")
            col_m = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                col_m[i].metric(label=k, value=f"{v:.4f}")

            # 统计
            correct = int((y_true == y_pred).sum())
            incorrect = int((y_true != y_pred).sum())
            total = correct + incorrect
            accuracy = correct / total * 100 if total > 0 else 0

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ 预测正确", correct)
            c2.metric("❌ 预测错误", incorrect)
            c3.metric("📊 正确率", f"{accuracy:.2f}%")

            # 可视化
            st.markdown("---")
            st.subheader("📊 可视化分析")
            tab1, tab2, tab3 = st.tabs(["混淆矩阵", "ROC 曲线", "PR 曲线"])

            with tab1:
                fig_cm = plot_confusion_matrix(y_true, y_pred)
                st.plotly_chart(fig_cm, use_container_width=True)
            with tab2:
                if y_score is not None and y_true.nunique() == 2:
                    fig_roc = plot_roc_curve(y_true, y_score)
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("💡 需要二分类且提供概率列")
            with tab3:
                if y_score is not None and y_true.nunique() == 2:
                    fig_pr = plot_pr_curve(y_true, y_score)
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info("💡 需要二分类且提供概率列")

            # AI 分析
            st.markdown("---")
            st.subheader("🤖 AI 智能分析")
            with st.spinner("AI 分析中..."):
                prompt = f"""分析模型表现：
指标：{str({k: round(v, 4) for k, v in metrics.items()})}
样本数：{total}，正确：{correct}，错误：{incorrect}，正确率：{accuracy:.2f}%
请简要评估，给出3条改进建议。用中文。"""
                try:
                    resp = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是ML评估专家，简洁回答。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=800,
                        timeout=30
                    )
                    st.markdown(resp.choices[0].message.content)
                except Exception as e:
                    st.warning(f"AI分析失败: {e}")

# ==================== 界面二：用户数据可视化 ====================
elif page == "👥  用户数据可视化":
    st.header("👥 用户数据可视化")
    st.write("上传用户特征 CSV，进行探索性分析和交互式图表")

    uploaded_file = st.file_uploader("拖拽用户数据 .csv 文件至此", type=["csv"])
    if uploaded_file is not None:
        sample_size = st.slider("采样行数（建议 5万~20万）", 10000, 200000, 50000, step=10000)
        try:
            df_polars = load_user_data(uploaded_file, sample_size=sample_size)
            st.success(f"成功加载 {df_polars.height} 行数据")
        except Exception as e:
            st.error(f"读取文件出错：{e}")
            st.stop()

        st.subheader("数据概览")
        st.write(f"**当前展示**: {df_polars.height} 行 × {df_polars.width} 列")
        st.write(f"**列名**: {df_polars.columns}")
        st.write("**数据类型**:")
        dtype_info = pd.DataFrame({
            '列名': df_polars.columns,
            '类型': [str(t) for t in df_polars.dtypes]
        }).set_index('列名')
        st.dataframe(dtype_info)

        st.subheader("数值列描述性统计")
        desc_df = get_basic_stats(df_polars)
        st.dataframe(desc_df.to_pandas(), use_container_width=True)

        st.subheader("交互式图表")
        c1, c2 = st.columns(2)
        with c1:
            chart_type = st.selectbox("图表类型", ["直方图", "箱线图", "散点图"])
        with c2:
            numeric_cols = [c for c in df_polars.columns if df_polars[c].dtype in [pl.Float64, pl.Int64]]
            if len(numeric_cols) == 0:
                st.warning("未检测到数值列")
            else:
                selected_col = st.selectbox("选择列", numeric_cols)
                if chart_type == "直方图":
                    fig = plot_histogram(df_polars, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "箱线图":
                    fig = plot_boxplot(df_polars, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    col_y = st.selectbox("Y 轴", numeric_cols, index=min(1, len(numeric_cols) - 1))
                    fig = plot_scatter(df_polars, selected_col, col_y)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== 界面三：模型可视化 ====================
elif page == "🤖  模型可视化":
    st.header("🤖 程序模型可视化与对比")
    st.write("上传多个 .py 文件（建议至少5个），自动解析模型并用 DeepSeek 生成评价")


    @st.cache_data(show_spinner=False)
    def get_ai_review(model_name: str) -> str:
        prompt = f"""请简要总结 {model_name} 的优缺点，用Markdown格式：
        **优点：**
        - 优点1
        - 优点2
        **缺点：**
        - 缺点1
        - 缺点2"""
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "你是ML专家。"}, {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=500
            )
            return resp.choices[0].message.content
        except:
            return "**生成失败**"


    uploaded_files = st.file_uploader("拖拽 .py 文件至此", type=["py"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) < 5:
            st.warning(f"当前仅 {len(uploaded_files)} 个文件，建议至少5个")
        st.success(f"已上传 {len(uploaded_files)} 个文件")

        all_model_names = set()
        all_parsed_results = []

        for uf in uploaded_files:
            code = StringIO(uf.getvalue().decode("utf-8")).read()
            try:
                parsed = parse_python_file(code)
                used = extract_used_models(code)
                for cls in parsed.get('model_classes', []):
                    all_model_names.add(cls['name'])
                for m in used:
                    all_model_names.add(m)
                all_parsed_results.append({'file_name': uf.name, 'parsed': parsed, 'used_models': used, 'code': code})
            except:
                st.error(f"{uf.name} 解析错误，已跳过")

        if all_model_names:
            st.success(f"识别到: {', '.join(sorted(all_model_names))}")

            model_file_map = {}
            for mn in all_model_names:
                model_file_map[mn] = [r['file_name'] for r in all_parsed_results
                                      if mn in set(
                        [c['name'] for c in r['parsed'].get('model_classes', [])] + r['used_models'])]

            st.subheader("模型来源统计")
            source_df = pd.DataFrame([
                {"模型": m, "出现次数": len(f), "来源文件": ", ".join(f)}
                for m, f in model_file_map.items()
            ]).sort_values("出现次数", ascending=False)
            st.dataframe(source_df, use_container_width=True)

            models_info = [get_model_info(n) for n in sorted(all_model_names)]

            st.subheader("各模型详情（AI 评价）")
            for info in models_info:
                st.markdown(f"### 🧠 {info['name']}")
                st.latex(info.get('formula', ''))
                st.write("**参数**:", ', '.join(info.get('params', [])))
                sf = model_file_map.get(info['name'], [])
                if sf:
                    st.caption(f"📁 {', '.join(sf)}")
                with st.spinner(f"生成 {info['name']} 评价..."):
                    review = get_ai_review(info['name']).replace('<br>', '\n')
                    st.markdown(review)
                st.markdown("---")

            if len(models_info) > 1:
                st.subheader("模型横向对比（AI 生成）")
                compare_mode = st.radio("对比方式", ["全部对比", "按类别对比"], horizontal=True)

                if compare_mode == "全部对比":
                    groups = [("所有模型", models_info)]
                else:
                    tree_kw = ['Tree', 'Forest', 'Boost', 'GBDT', 'LightGBM', 'XGBoost']
                    linear_kw = ['Logistic', 'Linear', 'SVM', 'SVC', 'Bayes']
                    tree_m, linear_m, other_m = [], [], []
                    for info in models_info:
                        name = info['name']
                        if any(k in name for k in tree_kw):
                            tree_m.append(info)
                        elif any(k in name for k in linear_kw):
                            linear_m.append(info)
                        else:
                            other_m.append(info)
                    groups = [(n, g) for n, g in [("树模型", tree_m), ("线性模型", linear_m), ("其他", other_m)] if
                              len(g) > 1]

                for gname, gmodels in groups:
                    if len(gmodels) > 1:
                        st.markdown(f"#### {gname}")
                        with st.spinner(f"生成 {gname} 对比..."):
                            prompt = f"请用Markdown表格比较: {', '.join([m['name'] for m in gmodels])}\n| 模型 | 优点 | 缺点 | 适用场景 |\n|------|------|------|---------|"
                            try:
                                resp = deepseek_client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[{"role": "system", "content": "用Markdown表格输出，不用HTML。"},
                                              {"role": "user", "content": prompt}],
                                    temperature=0.3, max_tokens=1500
                                )
                                txt = resp.choices[0].message.content
                                for tag in ['<br>', '<br/>', '<table>', '</table>', '<tr>', '</tr>', '<td>', '</td>',
                                            '<th>', '</th>', '<thead>', '</thead>', '<tbody>', '</tbody>']:
                                    txt = txt.replace(tag, '\n' if 'br' in tag else '')
                                st.markdown(txt)
                            except:
                                st.error("对比生成失败")
                        st.markdown("---")
        else:
            st.info("未识别到常见模型类。请确保代码包含 LogisticRegression、RandomForestClassifier 等实例化语句。")

# ==================== 界面四：学习助手 ====================
elif page == "🎓  学习助手":
    st.header("🎓 复购预测学习助手")
    st.write("基于 DeepSeek API，询问关于模型评估、特征工程、复购预测等问题")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是复购预测学习助手，有什么可以帮你的？"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "system", "content": "你是天猫复购预测专家，用中文简洁回答。"},
                                  *st.session_state.messages],
                        stream=True
                    )
                    full = ""
                    placeholder = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full += chunk.choices[0].delta.content
                            placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                except Exception as e:
                    st.error(f"API出错: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full})