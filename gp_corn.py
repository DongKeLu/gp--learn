"""
玉米基本面截面GP因子挖掘框架
============================================
数据源：
  - 基本面：国内玉米基本面日度数据_副本.xlsx（每行=一个日期截面，20+特征）
  - 价格  ：corn_en_no_oi.txt（玉米期货日线OHLCV）
任务：
  在每个日期截面，用GP搜索特征数学组合，预测玉米期货未来N日收益率
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import operator
import math
import copy
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 第一部分：数据加载与清洗
# =============================================================================

def load_and_clean_data(
    fundamental_path: str,
    futures_path: str,
    target_horizon: int = 5
) -> pd.DataFrame:
    """
    加载并清洗基本面+期货数据，计算未来收益目标变量。

    参数
    ----
    fundamental_path : str
        Excel文件路径
    futures_path : str
        期货价格CSV路径
    target_horizon : int
        预测周期（天），默认5日

    返回
    ----
    pd.DataFrame
        合并后的截面数据，index=日期，columns=特征+target
    """

    # ---------- 基本面数据 ----------
    df_raw = pd.read_excel(fundamental_path, header=None)

    # 双层表头：第0行=大类，第1行=子类
    row0 = df_raw.iloc[0].fillna('').astype(str)
    row1 = df_raw.iloc[1].fillna('').astype(str)

    # 构建列名：大类_子类 或 大类
    col_names = []
    for j in range(df_raw.shape[1]):
        c0 = row0.iloc[j].strip()
        c1 = row1.iloc[j].strip()
        if c0 and c1 and c0 != c1:
            col_names.append(f'{c0}_{c1}')
        elif c0:
            col_names.append(c0)
        else:
            col_names.append(f'col_{j}')

    df_fund = df_raw.iloc[2:].copy()
    df_fund.columns = col_names
    df_fund = df_fund.rename(columns={'日期': 'date'})
    df_fund['date'] = pd.to_datetime(df_fund['date'], errors='coerce')
    df_fund = df_fund.dropna(subset=['date'])
    df_fund = df_fund.set_index('date').sort_index()

    # 处理 '#N/A' 和 '#REF!' 等字符串
    for col in df_fund.columns:
        df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')

    # 前向填充 + 线性插值补缺失值
    df_fund = df_fund.ffill().interpolate(method='linear')

    # ---------- 期货数据 ----------
    df_fut = pd.read_csv(futures_path)
    df_fut['date'] = pd.to_datetime(df_fut['date'])
    df_fut = df_fut.set_index('date').sort_index()

    # 计算未来N日收益率（shift前移，用未来价格计算）
    df_fut['target'] = df_fut['close'].shift(-target_horizon) / df_fut['close'] - 1

    # ---------- 合并 ----------
    df = df_fund.join(df_fut[['target', 'close', 'volume']], how='inner')
    df = df.dropna(subset=['target'])
    df = df.dropna(how='all', axis=1)

    print(f'截面数量: {len(df)} 天')
    print(f'日期范围: {df.index[0].date()} ~ {df.index[-1].date()}')
    print(f'特征数量: {len(df.columns) - 3} 个（已剔除 date / target / close / volume）')

    return df


# =============================================================================
# 第二部分：特征池定义
# =============================================================================

def build_feature_pool() -> List[Tuple[str, str]]:
    """
    定义候选特征池：(列名前缀, 中文描述)
    在GP中会组合这些特征。
    """
    return [
        ('北港价格',               '北港现货价格'),
        ('珠三角价格',             '珠三角现货价格'),
        ('北港价格_珠三角价格',    '南北价差（北港-珠三角）'),
        ('仓单',                  '期货仓单量'),
        ('北港到货量',            '北港到货量'),
        ('深加工收购价_东北深加工均价', '东北深加工均价'),
        ('深加工收购价_华北深加工均价', '华北深加工均价'),
        ('深加工收购价_东北深加工均价_深加工收购价_华北深加工均价', '东北-华北价差'),
        ('东北深加工收购量_总计',  '东北深加工收购总量'),
        ('山东到车辆_总量',       '山东到车辆总量'),
        ('小麦替代成本优势',       '小麦替代成本优势'),
        # ---- 价格动量（人工构造）----
        ('北港价格_ret5',         '北港5日收益'),
        ('珠三角价格_ret5',        '珠三角5日收益'),
        # ---- 仓单变化 ----
        ('仓单_change5',          '仓单5日变化'),
    ]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始特征基础上，计算30+个派生特征，
    覆盖：价差类、动量类、变化率类、比率类、滚动统计类。
    """
    df = df.copy()

    # ---- 1. 价差类特征 ----
    # 南北现货价差
    if '北港价格' in df.columns and '珠三角价格' in df.columns:
        df['南北价差'] = df['珠三角价格'] - df['北港价格']
        df['南北价差_pct'] = df['南北价差'] / df['北港价格']

    # 东北-华北深加工价差（均价）
    price_cols = [c for c in df.columns if '均价' in c]
    if len(price_cols) >= 2:
        df['东北华北深加工价差'] = df[price_cols[0]] - df[price_cols[-1]]

    # 仓单与到货量比值（库存压力代理）
    if '仓单' in df.columns and '北港到货量' in df.columns:
        df['仓单_到货量比'] = df['仓单'] / (df['北港到货量'] + 1)

    # ---- 2. 动量类特征（多周期） ----
    price_bases = ['北港价格', '珠三角价格']
    for col in price_bases:
        if col not in df.columns:
            continue
        for period in [3, 5, 10, 20]:
            # 收益率（动量）
            df[f'{col}_ret{period}'] = df[col].pct_change(period).replace([np.inf, -np.inf], np.nan)
            # 涨跌方向（0/1离散特征，GP可与连续特征组合）
            df[f'{col}_up{period}'] = (df[col].diff(period) > 0).astype(float)

    # ---- 3. 变化率/差分类特征 ----
    if '仓单' in df.columns:
        for period in [5, 10, 20]:
            df[f'仓单_change{period}'] = df['仓单'].diff(period)
            df[f'仓单_pct{period}'] = df['仓单'].pct_change(period).replace([np.inf, -np.inf], np.nan)

    if '北港到货量' in df.columns:
        for period in [5, 10]:
            df[f'北港到货量_change{period}'] = df['北港到货量'].diff(period)
            df[f'北港到货量_pct{period}'] = df['北港到货量'].pct_change(period).replace([np.inf, -np.inf], np.nan)

    # ---- 4. 比率/标准化类特征 ----
    # 仓单库存分位数（相对历史位置）
    if '仓单' in df.columns:
        df['仓单_zscore20'] = (df['仓单'] - df['仓单'].rolling(20, min_periods=5).mean()) / \
                              (df['仓单'].rolling(20, min_periods=5).std() + 1e-8)

    if '北港到货量' in df.columns:
        df['到货量_zscore20'] = (df['北港到货量'] - df['北港到货量'].rolling(20, min_periods=5).mean()) / \
                                (df['北港到货量'].rolling(20, min_periods=5).std() + 1e-8)

    # 仓单/到货量比值分位数
    if '仓单_到货量比' in df.columns:
        df['仓单到货比_zscore20'] = (df['仓单_到货量比'] - df['仓单_到货量比'].rolling(20, min_periods=5).mean()) / \
                                     (df['仓单_到货量比'].rolling(20, min_periods=5).std() + 1e-8)

    # ---- 5. 滚动统计类特征 ----
    # 价格均线乖离率
    if '北港价格' in df.columns:
        df['北港价格_MA5'] = df['北港价格'].rolling(5, min_periods=3).mean()
        df['北港价格_MA10'] = df['北港价格'].rolling(10, min_periods=5).mean()
        df['北港价格_MA20'] = df['北港价格'].rolling(20, min_periods=10).mean()
        df['北港价格_MA5_dev'] = (df['北港价格'] - df['北港价格_MA5']) / df['北港价格_MA5']
        df['北港价格_MA20_dev'] = (df['北港价格'] - df['北港价格_MA20']) / df['北港价格_MA20']
        df['北港价格_MA5_MA20_cross'] = (df['北港价格_MA5'] - df['北港价格_MA20']) / df['北港价格_MA20']

    if '珠三角价格' in df.columns:
        df['珠三角价格_MA5'] = df['珠三角价格'].rolling(5, min_periods=3).mean()
        df['珠三角价格_MA20'] = df['珠三角价格'].rolling(20, min_periods=10).mean()
        df['珠三角价格_MA20_dev'] = (df['珠三角价格'] - df['珠三角价格_MA20']) / df['珠三角价格_MA20']

    # 仓单滚动均值（库存趋势）
    if '仓单' in df.columns:
        df['仓单_MA5'] = df['仓单'].rolling(5, min_periods=3).mean()
        df['仓单_MA20'] = df['仓单'].rolling(20, min_periods=10).mean()
        df['仓单_MA5_MA20_cross'] = df['仓单_MA5'] - df['仓单_MA20']

    # 成交量/到货量滚动均值（供需热度）
    if '北港到货量' in df.columns:
        df['到货量_MA5'] = df['北港到货量'].rolling(5, min_periods=3).mean()
        df['到货量_MA20'] = df['北港到货量'].rolling(20, min_periods=10).mean()
        df['到货量_MA5_MA20_cross'] = df['到货量_MA5'] - df['到货量_MA20']

    # ---- 6. 复合派生特征 ----
    # 小麦替代成本优势的动量
    if '小麦替代成本优势' in df.columns:
        df['小麦替代成本_ret5'] = df['小麦替代成本优势'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        df['小麦替代成本_change5'] = df['小麦替代成本优势'].diff(5)
        df['小麦替代成本_zscore10'] = (df['小麦替代成本优势'] - df['小麦替代成本优势'].rolling(10, min_periods=5).mean()) / \
                                       (df['小麦替代成本优势'].rolling(10, min_periods=5).std() + 1e-8)

    # 价格比值特征（相对强弱）
    if '北港价格' in df.columns and '珠三角价格' in df.columns:
        df['南北价比'] = df['珠三角价格'] / (df['北港价格'] + 1)

    # ---- 7. 山东到车辆特征（如果列存在） ----
    vehicle_cols = [c for c in df.columns if '山东到车辆' in c and c != '山东到车辆']
    if vehicle_cols:
        total_col = vehicle_cols[0]
        df['车辆总量_MA5'] = df[total_col].rolling(5, min_periods=3).mean()
        df['车辆总量_MA20'] = df[total_col].rolling(20, min_periods=10).mean()
        df['车辆总量_change5'] = df[total_col].diff(5)
        df['车辆总量_pct5'] = df[total_col].pct_change(5).replace([np.inf, -np.inf], np.nan)

    # ---- 8. 东北深加工收购量特征（如果列存在）----
    proc_cols = [c for c in df.columns if '东北深加工收购量' in c and c != '东北深加工收购量']
    if proc_cols:
        total_col = proc_cols[0]
        df['东北收购量_MA5'] = df[total_col].rolling(5, min_periods=3).mean()
        df['东北收购量_change5'] = df[total_col].diff(5)

    # 最终统一补缺失值
    df = df.ffill().interpolate(method='linear')

    # 清理极端值（clip到±5σ）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0 and std < 1e10:
            df[col] = df[col].clip(lower=mean - 5 * std, upper=mean + 5 * std)

    print(f'派生特征计算完毕，共 {len(df.columns)} 列')

    return df


# =============================================================================
# 第三部分：GP核心 —— 染色体（树）表示
# =============================================================================

@dataclass
class GpConfig:
    """GP超参数"""
    pop_size: int = 200          # 种群规模
    generations: int = 100        # 迭代代数
    crossover_rate: float = 0.7   # 交叉概率
    mutation_rate: float = 0.2    # 变异概率
    elite_rate: float = 0.1       # 精英保留比例
    max_depth: int = 4            # 树最大深度
    min_leaf_samples: int = 30    # 叶节点最小样本
    tournament_size: int = 5      # 锦标赛选择大小


class GpNode:
    """GP树节点"""
    def __init__(self, op=None, left=None, right=None, const_val=None, feature=None):
        self.op = op          # 操作符（如 add, sub, div, mul, gt, lt...）
        self.left = left      # 左子节点
        self.right = right    # 右子节点
        self.const_val = const_val  # 常量叶节点的值
        self.feature = feature       # 特征叶节点的列名
        self.value = None    # 缓存评估结果

    def is_leaf(self):
        return self.op is None and (self.const_val is not None or self.feature is not None)


class GpTree:
    """GP树（染色体）"""
    def __init__(self, config: GpConfig, feature_names: List[str]):
        self.config = config
        self.feature_names = feature_names
        self.root = None
        self.fitness = None
        self.ic = None        # IC（与收益的秩相关）
        self.ic_mean = None   # 滚动IC均值（跨截面）
        self.formula = None   # 可读公式字符串

    # ---- 树构建方法 ----

    @staticmethod
    def _random_tree(depth: int, config: GpConfig, feature_names: List[str],
                     force_leaf_p=0.3) -> GpNode:
        """随机生成一棵树（生长法）"""
        if depth >= config.max_depth or random.random() < force_leaf_p:
            # 叶节点
            if random.random() < 0.5:
                # 特征节点
                feat = random.choice(feature_names)
                return GpNode(feature=feat)
            else:
                # 常量节点（-2 ~ 2 的随机数）
                val = random.uniform(-2, 2)
                return GpNode(const_val=val)
        else:
            # 内部节点
            op = random.choice(OPERATORS)
            left = GpTree._random_tree(depth + 1, config, feature_names, force_leaf_p * 0.8)
            right = GpTree._random_tree(depth + 1, config, feature_names, force_leaf_p * 0.8)
            return GpNode(op=op, left=left, right=right)

    def random_init(self):
        """随机初始化整棵树"""
        self.root = self._random_tree(0, self.config, self.feature_names)

    # ---- 树评估 ----

    def eval_node(self, node: GpNode, X: pd.DataFrame) -> np.ndarray:
        """递归评估树节点，返回np数组"""
        if node.is_leaf():
            if node.feature:
                vals = X[node.feature].values.astype(float)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                return vals
            else:
                return np.full(len(X), node.const_val)

        left_vals = self.eval_node(node.left, X)
        right_vals = self.eval_node(node.right, X)

        # 安全除法：避免除以0
        safe_right = np.where(np.abs(right_vals) < 1e-10, 1e-10, right_vals)

        op_map = {
            'add':  left_vals + right_vals,
            'sub':  left_vals - right_vals,
            'mul':  left_vals * right_vals,
            'div':  np.where(np.abs(safe_right) > 1e-10, left_vals / safe_right, 0.0),
            'abs':  np.abs(left_vals),
            'sqrt': np.sign(left_vals) * np.sqrt(np.abs(left_vals)),
            'log':  np.sign(left_vals) * np.log1p(np.abs(left_vals)),
            'neg':  -left_vals,
            'max2': np.maximum(left_vals, right_vals),
            'min2': np.minimum(left_vals, right_vals),
        }

        return op_map.get(node.op, left_vals)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """对截面X计算信号值"""
        vals = self.eval_node(self.root, X)
        # 标准化（z-score）防止量纲差异
        std = np.nanstd(vals)
        if std < 1e-10:
            return np.zeros_like(vals)
        return (vals - np.nanmean(vals)) / std

    # ---- 适应度函数 ----

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        截面GP适应度：在所有日期截面上计算IC均值
        df.index = date, df.columns 包含特征和 'target'
        """
        signals = self.predict(df)
        targets = df['target'].values

        # 移除 NaN
        mask = ~(np.isnan(signals) | np.isnan(targets))
        if mask.sum() < self.config.min_leaf_samples:
            self.fitness = -999.0
            self.ic = -999.0
            self.ic_mean = -999.0
            return self.fitness

        # 计算 IC（Spearman秩相关）
        ic, _ = spearman_ic(signals[mask], targets[mask])
        self.ic = ic

        # 滚动IC均值（每个截面单独算IC，再平均）
        ic_list = []
        window = 60  # 60天滚动窗口
        dates = df.index
        values = pd.DataFrame({'signal': signals, 'target': targets}, index=dates)

        for i in range(window, len(values)):
            win_df = values.iloc[i-window:i]
            win_mask = ~(win_df['signal'].isna() | win_df['target'].isna())
            if win_mask.sum() > 10:
                s = win_df.loc[win_mask, 'signal'].values
                t = win_df.loc[win_mask, 'target'].values
                c, _ = spearman_ic(s, t)
                if not np.isnan(c):
                    ic_list.append(c)

        self.ic_mean = np.mean(ic_list) if ic_list else 0.0
        self.fitness = self.ic_mean

        # 更新公式
        self.formula = self._build_formula(self.root)

        return self.fitness

    def _build_formula(self, node: GpNode) -> str:
        """把树还原为可读公式字符串"""
        if node.is_leaf():
            if node.feature:
                return f'[{node.feature}]'
            else:
                return f'{node.const_val:.4f}'
        op_sym = OP_SYMBOLS.get(node.op, node.op)
        left_str = self._build_formula(node.left)
        right_str = self._build_formula(node.right)
        return f'({left_str} {op_sym} {right_str})'

    # ---- 遗传操作 ----

    def crossover(self, other: 'GpTree') -> Tuple['GpTree', 'GpTree']:
        """单点交叉"""
        tree1, tree2 = copy.deepcopy(self), copy.deepcopy(other)
        n1, n2 = self._random_internal_node(tree1.root), self._random_internal_node(tree2.root)
        if n1 and n2:
            n1.op, n2.op = n2.op, n1.op
            n1.left, n2.left = n2.left, n1.left
            n1.right, n2.right = n2.right, n1.right
        return tree1, tree2

    def mutate(self):
        """子树变异"""
        node = self._random_node(self.root)
        if node:
            new_subtree = self._random_tree(0, self.config, self.feature_names)
            if node.op:
                node.left = new_subtree.left
                node.right = new_subtree.right
            else:
                node.feature = new_subtree.feature
                node.const_val = new_subtree.const_val

    @staticmethod
    def _random_node(node: GpNode):
        nodes = [node]
        stack = [node]
        while stack:
            n = stack.pop()
            if n.left:
                nodes.append(n.left)
                stack.append(n.left)
            if n.right:
                nodes.append(n.right)
                stack.append(n.right)
        return random.choice(nodes) if nodes else None

    @staticmethod
    def _random_internal_node(node: GpNode):
        nodes = []
        if node.op:
            nodes.append(node)
            if node.left:
                nodes += GpTree._collect_internal(node.left)
            if node.right:
                nodes += GpTree._collect_internal(node.right)
        return random.choice(nodes) if nodes else None

    @staticmethod
    def _collect_internal(node: GpNode):
        nodes = []
        if node.op:
            nodes.append(node)
            if node.left:
                nodes += GpTree._collect_internal(node.left)
            if node.right:
                nodes += GpTree._collect_internal(node.right)
        return nodes


# =============================================================================
# 第四部分：GP操作符与辅助函数
# =============================================================================

OPERATORS = [
    'add', 'sub', 'mul', 'div',
    'abs', 'sqrt', 'log',
    'max2', 'min2',
]

OP_SYMBOLS = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
    'abs': 'abs', 'sqrt': 'sqrt', 'log': 'log',
    'max2': 'max', 'min2': 'min',
}


def spearman_ic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """计算Spearman秩相关系数（IC）"""
    n = len(x)
    if n < 3:
        return 0.0, 0.0
    rank_x = pd.Series(x).rank().values
    rank_y = pd.Series(y).rank().values
    ic = np.corrcoef(rank_x, rank_y)[0, 1]
    return ic, ic / np.sqrt(n)


def tournament_selection(population: List[GpTree], k: int) -> GpTree:
    """锦标赛选择"""
    selected = random.sample(population, min(k, len(population)))
    return max(selected, key=lambda t: t.fitness)


# =============================================================================
# 第五部分：GP主循环
# =============================================================================

def run_gp(df: pd.DataFrame,
           feature_cols: List[str],
           config: Optional[GpConfig] = None,
           seed: int = 42) -> 'GpTree':
    """
    运行GP算法，返回最优个体。
    """
    if config is None:
        config = GpConfig()
    random.seed(seed)
    np.random.seed(seed)

    # 只保留有数据的列
    available_features = [f for f in feature_cols if f in df.columns]
    if not available_features:
        raise ValueError(f'没有找到有效特征。可用列: {df.columns.tolist()}')

    print(f'\n========== GP 开始 ==========')
    print(f'种群规模: {config.pop_size}, 代数: {config.generations}')
    print(f'有效特征数: {len(available_features)}')
    print(f'截面样本数: {len(df)}')
    print(f'================================')

    # ---- 初始化种群 ----
    population = []
    for _ in range(config.pop_size):
        tree = GpTree(config, available_features)
        tree.random_init()
        population.append(tree)

    # ---- 评估初始种群 ----
    for tree in population:
        tree.evaluate(df)

    population.sort(key=lambda t: t.fitness, reverse=True)
    best = copy.deepcopy(population[0])

    # ---- 进化主循环 ----
    for gen in range(config.generations):
        new_pop = []

        # 精英保留
        elite_count = int(config.pop_size * config.elite_rate)
        new_pop.extend(copy.deepcopy(population[:elite_count]))

        # 生成剩余个体
        while len(new_pop) < config.pop_size:
            r = random.random()
            if r < config.crossover_rate:
                p1 = tournament_selection(population, config.tournament_size)
                p2 = tournament_selection(population, config.tournament_size)
                c1, c2 = p1.crossover(p2)
                new_pop.extend([c1, c2])
            elif r < config.crossover_rate + config.mutation_rate:
                p = tournament_selection(population, config.tournament_size)
                child = copy.deepcopy(p)
                child.mutate()
                new_pop.append(child)
            else:
                p = tournament_selection(population, config.tournament_size)
                new_pop.append(copy.deepcopy(p))

        # 截断超出规模
        new_pop = new_pop[:config.pop_size]

        # 评估
        for tree in new_pop:
            tree.evaluate(df)

        population = new_pop
        population.sort(key=lambda t: t.fitness, reverse=True)

        if population[0].fitness > best.fitness:
            best = copy.deepcopy(population[0])

        if gen % 10 == 0 or gen == config.generations - 1:
            print(f'  Gen {gen:3d} | Best IC Mean: {best.ic_mean:.4f} | IC: {best.ic:.4f} | '
                  f'Fitness: {best.fitness:.4f}')

    print(f'\n========== GP 完成 ==========')
    print(f'最优公式: {best.formula}')
    print(f'IC Mean : {best.ic_mean:.4f}')
    print(f'IC      : {best.ic:.4f}')
    print(f'================================')

    return best


# =============================================================================
# 第六部分：样本外验证
# =============================================================================

def out_of_sample_test(best_tree: GpTree,
                       df: pd.DataFrame,
                       train_ratio: float = 0.7,
                       n_splits: int = 5) -> dict:
    """
    滚动时间序列交叉验证，评估GP公式的样本外稳定性。
    """
    n = len(df)
    train_size = int(n * train_ratio)
    step = (n - train_size) // n_splits

    results = []
    for i in range(n_splits):
        train_end = train_size + i * step
        test_start = train_end
        test_end = min(test_start + step, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        # 在训练集上评估
        ic_train, _ = evaluate_formula(best_tree, train_df)
        # 在测试集上评估
        ic_test, _ = evaluate_formula(best_tree, test_df)

        results.append({
            'fold': i + 1,
            'train_ic': ic_train,
            'test_ic': ic_test,
            'train_size': len(train_df),
            'test_size': len(test_df),
        })
        print(f'  Fold {i+1} | Train IC: {ic_train:.4f} | Test IC: {ic_test:.4f}')

    ic_tests = [r['test_ic'] for r in results]
    print(f'\n样本外 IC 均值: {np.mean(ic_tests):.4f}, 标准差: {np.std(ic_tests):.4f}')
    print(f'胜率（正IC比例）: {sum(1 for x in ic_tests if x > 0) / len(ic_tests):.2%}')

    return {
        'results': results,
        'oos_ic_mean': np.mean(ic_tests),
        'oos_ic_std': np.std(ic_tests),
        'win_rate': sum(1 for x in ic_tests if x > 0) / len(ic_tests),
    }


def evaluate_formula(tree: GpTree, df: pd.DataFrame) -> Tuple[float, float]:
    """对给定数据集评估公式的IC"""
    signals = tree.predict(df)
    targets = df['target'].values
    mask = ~(np.isnan(signals) | np.isnan(targets))
    if mask.sum() < 10:
        return 0.0, 0.0
    ic, _ = spearman_ic(signals[mask], targets[mask])
    return ic, np.nanmean(signals[mask])


# =============================================================================
# 第七部分：主程序入口
# =============================================================================

def main():
    DATA_DIR = '/Users/dongkelu/Desktop/种子测试'

    fundamental_path = f'{DATA_DIR}/国内玉米基本面日度数据_副本.xlsx'
    futures_path = f'{DATA_DIR}/corn_en_no_oi.txt'

    # 1. 加载数据
    df = load_and_clean_data(fundamental_path, futures_path, target_horizon=5)

    # 2. 计算派生特征
    df = compute_derived_features(df)

    # 4. 定义候选特征池（原始 + 全部派生特征）
    # GP会自动从这些列名中选取
    feature_cols = [
        # ---- 原始特征 ----
        '北港价格', '珠三角价格', '仓单', '北港到货量', '小麦替代成本优势',
        # ---- 价差类 ----
        '南北价差', '南北价差_pct', '南北价比',
        '东北华北深加工价差', '仓单_到货量比',
        # ---- 动量类（多周期）----
        '北港价格_ret3', '北港价格_ret5', '北港价格_ret10', '北港价格_ret20',
        '珠三角价格_ret3', '珠三角价格_ret5', '珠三角价格_ret10', '珠三角价格_ret20',
        '北港价格_up3', '北港价格_up5', '北港价格_up10', '北港价格_up20',
        '珠三角价格_up3', '珠三角价格_up5', '珠三角价格_up10', '珠三角价格_up20',
        # ---- 变化率/差分类 ----
        '仓单_change5', '仓单_change10', '仓单_change20',
        '仓单_pct5', '仓单_pct10', '仓单_pct20',
        '北港到货量_change5', '北港到货量_change10',
        '北港到货量_pct5', '北港到货量_pct10',
        # ---- 分位数/标准化类 ----
        '仓单_zscore20', '到货量_zscore20', '仓单到货比_zscore20',
        # ---- 均线乖离率 ----
        '北港价格_MA5_dev', '北港价格_MA20_dev', '北港价格_MA5_MA20_cross',
        '珠三角价格_MA20_dev',
        '仓单_MA5_MA20_cross',
        '到货量_MA5_MA20_cross',
        # ---- 小麦替代类 ----
        '小麦替代成本_ret5', '小麦替代成本_change5', '小麦替代成本_zscore10',
        # ---- 山东到车辆类 ----
        '车辆总量_MA5', '车辆总量_MA20', '车辆总量_change5', '车辆总量_pct5',
        # ---- 东北收购量类 ----
        '东北收购量_MA5', '东北收购量_change5',
    ]

    # 只保留实际存在于df中的列（避免列名不匹配报错）
    feature_cols = [f for f in feature_cols if f in df.columns]

    print(f'\n使用特征 ({len(feature_cols)} 个):')
    for i in range(0, len(feature_cols), 4):
        row = feature_cols[i:i+4]
        print('  ' + '  |  '.join(f'{f:<25}' for f in row))

    # 4. 配置GP
    config = GpConfig(
        pop_size=200,
        generations=100,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elite_rate=0.1,
        max_depth=4,
        min_leaf_samples=30,
        tournament_size=5,
    )

    # 5. 运行GP
    best_tree = run_gp(df, feature_cols, config=config, seed=42)

    # 6. 样本外验证
    print('\n========== 样本外验证（5折滚动） ==========')
    oos_results = out_of_sample_test(best_tree, df, train_ratio=0.7, n_splits=5)

    # 7. 输出Top公式
    print('\n========== 最优因子公式 ==========')
    print(f'公式: {best_tree.formula}')
    print(f'IC Mean: {best_tree.ic_mean:.4f}')
    print(f'样本外 IC: {oos_results["oos_ic_mean"]:.4f} ± {oos_results["oos_ic_std"]:.4f}')
    print(f'胜率: {oos_results["win_rate"]:.2%}')


if __name__ == '__main__':
    main()
