import ast
import pandas as pd
import polars as pl
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List


# ---------- 界面一：模型评估 ----------
def load_eval_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def calc_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """计算分类评估指标"""
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # 转为整数标签
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)

    if len(np.unique(y_true)) > 50:
        raise ValueError("标签类别数 > 50")

    n_classes = len(np.unique(y_true))
    avg = 'binary' if n_classes == 2 else 'weighted'

    try:
        metrics = {
            '准确率': accuracy_score(y_true, y_pred),
            '精确率': precision_score(y_true, y_pred, average=avg, zero_division=0),
            '召回率': recall_score(y_true, y_pred, average=avg, zero_division=0),
            'F1分数': f1_score(y_true, y_pred, average=avg, zero_division=0),
        }
    except:
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
        y_pred_enc = le.transform(y_pred) if len(set(y_pred) - set(y_true)) == 0 else le.fit_transform(y_pred)
        n_classes = len(le.classes_)
        avg = 'binary' if n_classes == 2 else 'weighted'
        metrics = {
            '准确率': accuracy_score(y_true_enc, y_pred_enc),
            '精确率': precision_score(y_true_enc, y_pred_enc, average=avg, zero_division=0),
            '召回率': recall_score(y_true_enc, y_pred_enc, average=avg, zero_division=0),
            'F1分数': f1_score(y_true_enc, y_pred_enc, average=avg, zero_division=0),
        }

    if y_score is not None:
        y_score = np.array(y_score).astype(float)
        if len(y_score) == len(y_true) and n_classes == 2:
            y_score = y_score[mask]
            y_score = y_score[~np.isnan(y_score)]
            if len(y_score) == len(y_true):
                try:
                    metrics['AUC'] = roc_auc_score(y_true, y_score)
                except:
                    pass

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.round(np.array(y_true)).astype(int)
    y_pred = np.round(np.array(y_pred)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    fig = ff.create_annotated_heatmap(
        z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
        colorscale='Blues', showscale=True
    )
    fig.update_layout(title='混淆矩阵', xaxis_title='预测值', yaxis_title='真实值')
    return fig


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy', line=dict(color='darkorange'), name='ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='随机'))
    fig.update_layout(title='ROC 曲线', xaxis_title='假正率', yaxis_title='真正率')
    return fig


def plot_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', fill='tozeroy', name='PR'))
    fig.update_layout(title='Precision-Recall 曲线', xaxis_title='召回率', yaxis_title='精确率')
    return fig


# ---------- 界面二：用户数据可视化 ----------
def load_user_data(file, sample_size=50000) -> pl.DataFrame:
    return pl.read_csv(file, n_rows=sample_size)


def get_basic_stats(df: pl.DataFrame) -> pl.DataFrame:
    return df.describe()


def plot_histogram(df: pl.DataFrame, column: str):
    return px.histogram(df.to_pandas(), x=column, marginal='box', title=f'{column} 分布')


def plot_boxplot(df: pl.DataFrame, column: str):
    return px.box(df.to_pandas(), y=column, title=f'{column} 箱线图')


def plot_scatter(df: pl.DataFrame, col_x: str, col_y: str):
    return px.scatter(df.to_pandas(), x=col_x, y=col_y, title=f'{col_x} vs {col_y}', opacity=0.5, marginal_y='violin')


# ---------- 界面三：模型可视化 ----------
MODEL_KNOWLEDGE = {
    'LogisticRegression': {'name': '逻辑回归',
                           'formula': r'P(y=1|x) = \frac{1}{1+e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}',
                           'params': ['penalty', 'C', 'solver', 'max_iter']},
    'RandomForestClassifier': {'name': '随机森林', 'formula': r'集成多棵决策树，投票或平均得出结果',
                               'params': ['n_estimators', 'max_depth', 'min_samples_split']},
    'XGBClassifier': {'name': 'XGBoost', 'formula': r'梯度提升树：F_k(x) = F_{k-1}(x) + \eta \cdot h_k(x)',
                      'params': ['n_estimators', 'learning_rate', 'max_depth']},
    'LightGBM': {'name': 'LightGBM', 'formula': r'基于直方图的梯度提升树',
                 'params': ['n_estimators', 'num_leaves', 'learning_rate']},
    'LGBMClassifier': {'name': 'LightGBM', 'formula': r'基于直方图的梯度提升树',
                       'params': ['n_estimators', 'num_leaves', 'learning_rate']},
    'SVC': {'name': '支持向量机', 'formula': r'最大化间隔，约束 y_i(w·x_i+b)\ge1', 'params': ['C', 'kernel', 'gamma']},
    'DecisionTreeClassifier': {'name': '决策树', 'formula': r'基于特征分裂的树状结构',
                               'params': ['max_depth', 'min_samples_split']},
    'GradientBoostingClassifier': {'name': 'GBDT', 'formula': r'F_k(x)=F_{k-1}(x)+\nu h_k(x)',
                                   'params': ['n_estimators', 'learning_rate']},
    'KNeighborsClassifier': {'name': 'K近邻', 'formula': r'多数投票决定类别', 'params': ['n_neighbors', 'weights']},
    'MLPClassifier': {'name': '多层感知机', 'formula': r'多层神经网络', 'params': ['hidden_layer_sizes', 'activation']},
    'GaussianNB': {'name': '朴素贝叶斯', 'formula': r'P(x_i|y) 服从高斯分布', 'params': ['var_smoothing']},
}


def parse_python_file(file_content: str) -> Dict[str, Any]:
    tree = ast.parse(file_content)
    parsed = {'classes': [], 'functions': [], 'model_classes': []}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [base.id if isinstance(base, ast.Name) else None for base in node.bases]
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            class_info = {'name': node.name, 'bases': bases, 'methods': methods}
            parsed['classes'].append(class_info)
            if any(b in ['BaseEstimator', 'ClassifierMixin', 'nn.Module'] for b in bases if
                   b) or node.name in MODEL_KNOWLEDGE:
                parsed['model_classes'].append(class_info)
        elif isinstance(node, ast.FunctionDef):
            parsed['functions'].append({'name': node.name, 'args': [a.arg for a in node.args.args]})
    return parsed


def get_model_info(model_class_name: str) -> Dict:
    return MODEL_KNOWLEDGE.get(model_class_name, {'name': model_class_name, 'formula': '请参考代码', 'params': []})


def extract_used_models(code: str) -> List[str]:
    model_names = set()
    try:
        tree = ast.parse(code)
        known = set(MODEL_KNOWLEDGE.keys())
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[-1]
                    imports[name] = alias.name.split('.')[-1]
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                class_name = None
                if isinstance(func, ast.Name):
                    class_name = imports.get(func.id, func.id)
                elif isinstance(func, ast.Attribute):
                    class_name = func.attr
                if class_name and class_name in known:
                    model_names.add(class_name)
        return sorted(model_names)
    except SyntaxError:
        return []