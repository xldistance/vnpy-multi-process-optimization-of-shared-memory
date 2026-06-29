import random
import traceback
import pickle
import struct
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from gc import collect
from itertools import chain, product
from multiprocessing import cpu_count
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union,Callable
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import scipy.stats as scs
import seaborn as sns

from pandas import DataFrame, to_datetime
from pyecharts import options as opts
from pyecharts.charts import (
    Bar,    # 柱状图
    Line,   # 折线图
    Page,   # 多图同表
)
from pyecharts.globals import CurrentConfig
from pyecharts.commons.utils import JsCode
from vnpy.app.portfolio_strategy.base import (
    EngineType,
)
from vnpy.app.portfolio_strategy.template import StrategyTemplate
from vnpy.event import EventEngine
from vnpy.trader.constant import Direction, Offset, OrderType, Status,Exchange
from vnpy.trader.database import database_manager
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import ContractData, OrderData, TickData, BarData, TradeData, PositionData, Interval,STOPORDER_PREFIX,StopOrder,StopOrderStatus
from vnpy.trader.utility import (
    TZ_INFO,
    GetFilePath,
    extract_vt_symbol,
    load_h5,
    round_to,
    save_json,
    save_redis_data,
    load_redis_data,
)
import technical_indicators as tech
from colorama import init, Fore
# matplotlib 美化样式：bmh，ggplot
plt.style.use("ggplot")
# Set seaborn style
sns.set_style("whitegrid")
# 初始化 colorama
init(autoreset=True)
@dataclass
class PositionDetail:
    active_orders: dict  # 活动委托单
    long_pos: float      # 多头仓位
    short_pos: float     # 空头仓位
    pos: float           # 净持仓

INTERVAL_DELTA_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}
TARGET_MAP = {
    "sortino_value":"rgr_value",
    "rgr_value":"sortino_value",
}
PARENT_PATH = Path(__file__).parent  # 获取当前运行程序父路径
# 优先离线使用pyecharts
JS_HOST = str(PARENT_PATH.parent.parent / "pyecharts-assets" / "assets" / "v6") + "/"
if Path(JS_HOST).exists():
    CurrentConfig.ONLINE_HOST = JS_HOST
# 涨跌颜色设置
long_color = "#00c853"
long_color2 = "#15a451"
short_color = "#ff1744"
short_color2 = "#DB1E44"
purple_color = "#DB46F2"
purple_color2 = "#A433B6"
purple_js_color = JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#DB46F2'},
                            {offset: 1, color: '#A433B6'}
                        ])
                    """)
# js颜色，涨为long_color，跌为short_color
js_color = JsCode("""
                        function(params) {
                            if (params.value >= 0) {
                                return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                    {offset: 0, color: '#00c853'},
                                    {offset: 1, color: '#15a451'}
                                ]);
                            } else {
                                return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                    {offset: 0, color: '#ff1744'},
                                    {offset: 1, color: '#DB1E44'}
                                ]);
                            }
                        }
                    """)
# 定义格式化函数和富文本样式
formatter = JsCode("""
function(params) {
    if (params.value >= 0) {
        return '{green|' + params.value + '}';
    } else {
        return '{red|' + params.value + '}';
    }
}
""")
# 设置图形样式选项，包括颜色
# color：使用js_color函数动态设置颜色
itemstyle_opts = opts.ItemStyleOpts(
    color=js_color
)
# 配置工具箱选项，包括数据缩放功能
toolbox_opts = opts.ToolboxOpts(
    is_show=True,
    pos_left=None,
    pos_right="0px",
    pos_bottom="0px",
    feature=opts.ToolBoxFeatureOpts(
        data_zoom=opts.ToolBoxFeatureDataZoomOpts(
            xaxis_index=None,
            yaxis_index=None,
        )
    ),
)
# 设置数据缩放选项，定义缩放范围
datazoom_opts = [
                    opts.DataZoomOpts(range_start=0, range_end=100, type_="slider"),
                    opts.DataZoomOpts(range_start=0, range_end=100, type_="inside")
                ]
# 设置鼠标悬停高亮
emphasis_opts = opts.EmphasisOpts(
    focus="self",
    # 悬停文字样式
    label_opts=opts.LabelOpts(
        position='top',
        formatter=formatter,
        rich={
            'red': {'color': short_color},
            'green': {'color': long_color}
        },
        font_family="方正韵动中黑简体",
        font_size=18
    ),
    # 鼠标悬停放大图形
    itemstyle_opts=opts.ItemStyleOpts(
        color=js_color,
        border_color=long_color,
        border_width=2,
    )
)
# 隐藏数据点的标签
label_opts = opts.LabelOpts(is_show=False)
# 增强的字体样式配置
enhanced_title_opts = opts.TextStyleOpts(
    font_family="方正韵动中黑简体", 
    font_size=14,
)

# ECharts v6 在没有显式 title.left/textAlign 时，多行标题的默认锚点行为和 v5 不一致，
# 会导致 account_info_binance.py 等脚本生成的标题看起来没有贴在图表最左侧。
# account_info_binance.py 会复用本模块导入后的 pyecharts.options，因此在这里统一修正
# TitleOpts 的默认值，可同时影响回测图表和账户信息图表。
DEFAULT_TITLE_POS_LEFT = "0px"
DEFAULT_TITLE_TEXT_ALIGN = "left"


def _patch_pyecharts_title_opts() -> None:
    """让未显式指定位置的 pyecharts 标题默认左对齐，兼容 ECharts v6。"""
    if getattr(opts.TitleOpts, "_vnpy_left_title_patch", False):
        return

    original_init = opts.TitleOpts.__init__

    def init_with_left_title(self, *args, **kwargs):
        if kwargs.get("pos_left") is None and kwargs.get("pos_right") is None:
            kwargs["pos_left"] = DEFAULT_TITLE_POS_LEFT
        if kwargs.get("text_align") in (None, "auto"):
            kwargs["text_align"] = DEFAULT_TITLE_TEXT_ALIGN
        original_init(self, *args, **kwargs)

    opts.TitleOpts.__init__ = init_with_left_title
    opts.TitleOpts._vnpy_left_title_patch = True


_patch_pyecharts_title_opts()

enhanced_tooltip_opts = opts.TooltipOpts(
    trigger="item",
    axis_pointer_type="cross",
    background_color="rgba(50, 50, 50, 0.9)",
    border_width=1,
    border_color="#777",
    textstyle_opts=opts.TextStyleOpts(color="#fff",font_family="方正韵动中黑简体",  font_size=14),
    # 设置提示框的位置，使其悬停在屏幕底部
    position=JsCode(
        """
    function (point, params, dom, rect, size) {
        return [
        size.viewSize[0] / 2 - dom.clientWidth / 2, 
        size.viewSize[1] - dom.clientHeight - 10
        ];
    }
    """
    )
)

enhanced_legend_opts = opts.LegendOpts(
    pos_top="5%",
    pos_left="center",
    orient="horizontal",
    textstyle_opts=enhanced_title_opts
)
axislabel_opts=opts.LabelOpts(rotate=0, font_family="方正韵动中黑简体", font_size=12)
@lru_cache(maxsize=1024)
def _cached_extract_vt_symbol(vt_symbol: str) -> Tuple[str, Exchange, str]:
    """缓存版本的vt_symbol解析"""
    return extract_vt_symbol(vt_symbol)
# NumPy 结构化数组类型定义 - BarData (无 turnover)
BAR_DTYPE = np.dtype([
    ('datetime_ns', 'i8'),
    ('open_price', 'f8'),
    ('high_price', 'f8'),
    ('low_price', 'f8'),
    ('close_price', 'f8'),
    ('volume', 'f8'),
    ('open_interest', 'f8'),
    ('symbol_idx', 'i4'),
])

# NumPy 结构化数组类型定义 - TickData (无 turnover)
TICK_DTYPE = np.dtype([
    ('datetime_ns', 'i8'),
    ('last_price', 'f8'),
    ('volume', 'f8'),
    ('open_interest', 'f8'),
    ('bid_price_1', 'f8'),
    ('bid_price_2', 'f8'),
    ('bid_price_3', 'f8'),
    ('bid_price_4', 'f8'),
    ('bid_price_5', 'f8'),
    ('ask_price_1', 'f8'),
    ('ask_price_2', 'f8'),
    ('ask_price_3', 'f8'),
    ('ask_price_4', 'f8'),
    ('ask_price_5', 'f8'),
    ('bid_volume_1', 'f8'),
    ('bid_volume_2', 'f8'),
    ('bid_volume_3', 'f8'),
    ('bid_volume_4', 'f8'),
    ('bid_volume_5', 'f8'),
    ('ask_volume_1', 'f8'),
    ('ask_volume_2', 'f8'),
    ('ask_volume_3', 'f8'),
    ('ask_volume_4', 'f8'),
    ('ask_volume_5', 'f8'),
    ('symbol_idx', 'i4'),
])

INDEX_DTYPE = np.dtype([
    ('dt_idx', 'i4'),
    ('sym_idx', 'i4'),
    ('data_idx', 'i4'),
])


@lru_cache(maxsize=131072)
def _datetime_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)

@lru_cache(maxsize=131072)
def _ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000)

def get_risk_weights(strategy_type:str):
    """
    根据策略类型返回权重
    """
    weights = {
        # 高频/日内：更关注日常波动
        "high_freq": {"downside": 0.6, "maxdd": 0.2, "cvar": 0.2},
        
        # 趋势跟踪：容忍日常波动，严控极端回撤
        "trend": {"downside": 0.3, "maxdd": 0.5, "cvar": 0.2},
        
        # 套利/中性：追求稳定，均衡考虑
        "arbitrage": {"downside": 0.4, "maxdd": 0.3, "cvar": 0.3},
        
        # 均衡：默认
        "balanced": {"downside": 0.4, "maxdd": 0.35, "cvar": 0.25},
    }
    return weights.get(strategy_type, weights["balanced"])

def calc_rgr(cagr_value:float, stability_return:float, annual_downside_risk:float, 
                max_drawdown_percent:float, return_skew:float, return_kurt:float, c_var:float,strategy_type:str = "trend"):
    # log处理：边际效用递减，防止过度追求极端高收益
    if cagr_value > 0:
        gain = np.log(1 + cagr_value)
    else:
        gain = -np.log(1 - cagr_value)
    # 偏度：右偏(正)奖励，左偏(负)惩罚
    skew_factor = 1 + 0.1 * np.tanh(return_skew)
    # 峰度：>3肥尾惩罚，<3薄尾奖励
    kurt_factor = 1 / (1 + 0.05 * max(return_kurt - 3, 0))
    # 综合风险：加权组合（权重可调）
    downside_risk = max(annual_downside_risk, 1e-6)
    max_dd = abs(max_drawdown_percent) / 100.0
    cvar_risk = abs(c_var) if c_var != 0 else max_dd * 0.5
    weights = get_risk_weights(strategy_type)
    combined_risk = weights["downside"] * downside_risk + weights["maxdd"] * max_dd + weights["cvar"] * cvar_risk
    
    return (gain * stability_return * skew_factor * kurt_factor) / combined_risk

class FastBarView:
    """优化后的BarData视图 - 构造时缓存所有值"""
    __slots__ = ['_symbol', '_exchange', '_gateway', '_vt_symbol', '_interval', '_dt',
                 '_close', '_open', '_high', '_low', '_volume', '_oi']
    
    def __init__(self, arr, idx, dt, symbol, exchange, gateway, vt_symbol, interval):
        self._symbol = symbol
        self._exchange = exchange
        self._gateway = gateway
        self._vt_symbol = vt_symbol
        self._interval = interval
        self._dt = dt
        row = arr[idx]
        self._open = float(row['open_price'])
        self._high = float(row['high_price'])
        self._low = float(row['low_price'])
        self._close = float(row['close_price'])
        self._volume = float(row['volume'])
        self._oi = float(row['open_interest'])
    
    @property
    def symbol(self): return self._symbol
    @property
    def exchange(self): return self._exchange
    @property
    def gateway_name(self): return self._gateway
    @property
    def vt_symbol(self): return self._vt_symbol
    @property
    def datetime(self): return self._dt
    @property
    def interval(self): return self._interval
    @property
    def open_price(self): return self._open
    @property
    def high_price(self): return self._high
    @property
    def low_price(self): return self._low
    @property
    def close_price(self): return self._close
    @property
    def volume(self): return self._volume
    @property
    def open_interest(self): return self._oi


class FastTickView:
    """优化后的TickData视图 - 构造时缓存所有值"""
    __slots__ = ['_symbol', '_exchange', '_gateway', '_vt_symbol', '_dt',
                 '_last_price', '_volume', '_oi',
                 '_bid_price_1', '_bid_price_2', '_bid_price_3', '_bid_price_4', '_bid_price_5',
                 '_ask_price_1', '_ask_price_2', '_ask_price_3', '_ask_price_4', '_ask_price_5',
                 '_bid_volume_1', '_bid_volume_2', '_bid_volume_3', '_bid_volume_4', '_bid_volume_5',
                 '_ask_volume_1', '_ask_volume_2', '_ask_volume_3', '_ask_volume_4', '_ask_volume_5']
    
    def __init__(self, arr, idx, dt, symbol, exchange, gateway, vt_symbol):
        self._symbol = symbol
        self._exchange = exchange
        self._gateway = gateway
        self._vt_symbol = vt_symbol
        self._dt = dt
        row = arr[idx]
        self._last_price = float(row['last_price'])
        self._volume = float(row['volume'])
        self._oi = float(row['open_interest'])
        self._bid_price_1 = float(row['bid_price_1'])
        self._bid_price_2 = float(row['bid_price_2'])
        self._bid_price_3 = float(row['bid_price_3'])
        self._bid_price_4 = float(row['bid_price_4'])
        self._bid_price_5 = float(row['bid_price_5'])
        self._ask_price_1 = float(row['ask_price_1'])
        self._ask_price_2 = float(row['ask_price_2'])
        self._ask_price_3 = float(row['ask_price_3'])
        self._ask_price_4 = float(row['ask_price_4'])
        self._ask_price_5 = float(row['ask_price_5'])
        self._bid_volume_1 = float(row['bid_volume_1'])
        self._bid_volume_2 = float(row['bid_volume_2'])
        self._bid_volume_3 = float(row['bid_volume_3'])
        self._bid_volume_4 = float(row['bid_volume_4'])
        self._bid_volume_5 = float(row['bid_volume_5'])
        self._ask_volume_1 = float(row['ask_volume_1'])
        self._ask_volume_2 = float(row['ask_volume_2'])
        self._ask_volume_3 = float(row['ask_volume_3'])
        self._ask_volume_4 = float(row['ask_volume_4'])
        self._ask_volume_5 = float(row['ask_volume_5'])
    
    @property
    def symbol(self): return self._symbol
    @property
    def exchange(self): return self._exchange
    @property
    def gateway_name(self): return self._gateway
    @property
    def vt_symbol(self): return self._vt_symbol
    @property
    def datetime(self): return self._dt
    @property
    def last_price(self): return self._last_price
    @property
    def volume(self): return self._volume
    @property
    def open_interest(self): return self._oi
    @property
    def bid_price_1(self): return self._bid_price_1
    @property
    def bid_price_2(self): return self._bid_price_2
    @property
    def bid_price_3(self): return self._bid_price_3
    @property
    def bid_price_4(self): return self._bid_price_4
    @property
    def bid_price_5(self): return self._bid_price_5
    @property
    def ask_price_1(self): return self._ask_price_1
    @property
    def ask_price_2(self): return self._ask_price_2
    @property
    def ask_price_3(self): return self._ask_price_3
    @property
    def ask_price_4(self): return self._ask_price_4
    @property
    def ask_price_5(self): return self._ask_price_5
    @property
    def bid_volume_1(self): return self._bid_volume_1
    @property
    def bid_volume_2(self): return self._bid_volume_2
    @property
    def bid_volume_3(self): return self._bid_volume_3
    @property
    def bid_volume_4(self): return self._bid_volume_4
    @property
    def bid_volume_5(self): return self._bid_volume_5
    @property
    def ask_volume_1(self): return self._ask_volume_1
    @property
    def ask_volume_2(self): return self._ask_volume_2
    @property
    def ask_volume_3(self): return self._ask_volume_3
    @property
    def ask_volume_4(self): return self._ask_volume_4
    @property
    def ask_volume_5(self): return self._ask_volume_5


class SharedHistoryDataAccessor:
    """共享历史数据的零拷贝访问器 - 修复版"""
    
    __slots__ = ['_data_array', '_symbols', '_symbol_info_cache', '_is_bar',
                 '_interval', '_dt_to_symbols', '_shm', '_last_dt_ns']
    
    def __init__(self, data_array:np.ndarray, dt_to_symbols:Dict[int,Dict[str,int]], symbols:List[str], symbol_info:Dict[str,Tuple[str,Exchange,str]], is_bar=True, interval=None):
        self._data_array = data_array
        self._symbols = symbols
        self._is_bar = is_bar
        self._interval = interval
        self._shm = None
        self._dt_to_symbols = dt_to_symbols
        self._symbol_info_cache = {
            vt_symbol: (info[0], info[1], info[2], vt_symbol)
            for vt_symbol, info in symbol_info.items()
        }
        self._last_dt_ns = None
    
    def get(self, dt, default=None):
        """O(1)时间复杂度的数据查找 - 修复版"""
        dt_ns = int(dt.timestamp() * 1_000_000_000)
        
        symbol_data = self._dt_to_symbols.get(dt_ns)
        if symbol_data is None:
            return default
        
        # 【关键修复】每次返回新字典，不复用缓存
        result = {}
        
        data_array = self._data_array
        symbol_info_cache = self._symbol_info_cache
        
        if self._is_bar:
            interval = self._interval
            for vt_symbol, data_idx in symbol_data.items():
                info = symbol_info_cache[vt_symbol]
                view = FastBarView.__new__(FastBarView)
                view._symbol = info[0]
                view._exchange = info[1]
                view._gateway = info[2]
                view._vt_symbol = info[3]
                view._interval = interval
                view._dt = dt
                row = data_array[data_idx]
                view._open = float(row['open_price'])
                view._high = float(row['high_price'])
                view._low = float(row['low_price'])
                view._close = float(row['close_price'])
                view._volume = float(row['volume'])
                view._oi = float(row['open_interest'])
                result[vt_symbol] = view
        else:
            for vt_symbol, data_idx in symbol_data.items():
                info = symbol_info_cache[vt_symbol]
                view = FastTickView.__new__(FastTickView)
                view._symbol = info[0]
                view._exchange = info[1]
                view._gateway = info[2]
                view._vt_symbol = info[3]
                view._dt = dt
                row = data_array[data_idx]
                view._last_price = float(row['last_price'])
                view._volume = float(row['volume'])
                view._oi = float(row['open_interest'])
                view._bid_price_1 = float(row['bid_price_1'])
                view._bid_price_2 = float(row['bid_price_2'])
                view._bid_price_3 = float(row['bid_price_3'])
                view._bid_price_4 = float(row['bid_price_4'])
                view._bid_price_5 = float(row['bid_price_5'])
                view._ask_price_1 = float(row['ask_price_1'])
                view._ask_price_2 = float(row['ask_price_2'])
                view._ask_price_3 = float(row['ask_price_3'])
                view._ask_price_4 = float(row['ask_price_4'])
                view._ask_price_5 = float(row['ask_price_5'])
                view._bid_volume_1 = float(row['bid_volume_1'])
                view._bid_volume_2 = float(row['bid_volume_2'])
                view._bid_volume_3 = float(row['bid_volume_3'])
                view._bid_volume_4 = float(row['bid_volume_4'])
                view._bid_volume_5 = float(row['bid_volume_5'])
                view._ask_volume_1 = float(row['ask_volume_1'])
                view._ask_volume_2 = float(row['ask_volume_2'])
                view._ask_volume_3 = float(row['ask_volume_3'])
                view._ask_volume_4 = float(row['ask_volume_4'])
                view._ask_volume_5 = float(row['ask_volume_5'])
                result[vt_symbol] = view
        
        self._last_dt_ns = dt_ns
        return result if result else default


class SharedMemoryManager:
    """零拷贝共享内存管理器"""
    
    HEADER_FORMAT = 'qqqibi'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    def __init__(self):
        self.shm_main = None
        self.shm_main_name = ""
    
    def create_shared_memory(self, dts, history_data):
        if not dts or not history_data:
            raise ValueError("dts 和 history_data 不能为空")
        
        first_data = None
        for dt in dts:
            first_data = history_data.get(dt, {})
            if first_data:
                break
        
        first_item = next(iter(first_data.values()))
        is_bar = isinstance(first_item, BarData)
        dtype = BAR_DTYPE if is_bar else TICK_DTYPE
        
        all_symbols = set()
        for bars_dict in history_data.values():
            all_symbols.update(bars_dict.keys())
        symbols = sorted(all_symbols)
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        
        symbol_info = {}

        for vt_symbol in symbols:
            symbol_info[vt_symbol] = _cached_extract_vt_symbol(vt_symbol)
            
        # 优化：预估数据量并预分配数组
        total_data_count = sum(len(bars_dict) for bars_dict in history_data.values())
        data_array = np.empty(total_data_count, dtype=dtype)
        index_list = np.empty(total_data_count, dtype=INDEX_DTYPE)
        
        sorted_dts = sorted(dts)
        dt_to_idx = {dt: i for i, dt in enumerate(sorted_dts)}
        
        data_idx = 0
        for dt in sorted_dts:
            dt_idx = dt_to_idx[dt]
            bars_dict = history_data.get(dt, {})
            
            for vt_symbol, data in bars_dict.items():
                sym_idx = symbol_to_idx[vt_symbol]
                index_list[data_idx] = (dt_idx, sym_idx, data_idx)
                
                if is_bar:
                    data_array[data_idx] = (
                        _datetime_to_ns(data.datetime),
                        data.open_price, data.high_price, data.low_price, data.close_price,
                        data.volume, getattr(data, 'open_interest', 0.0), sym_idx,
                    )
                else:
                    data_array[data_idx] = (
                        _datetime_to_ns(data.datetime),
                        data.last_price, data.volume, getattr(data, 'open_interest', 0.0),
                        getattr(data, 'bid_price_1', 0.0), getattr(data, 'bid_price_2', 0.0),
                        getattr(data, 'bid_price_3', 0.0), getattr(data, 'bid_price_4', 0.0),
                        getattr(data, 'bid_price_5', 0.0),
                        getattr(data, 'ask_price_1', 0.0), getattr(data, 'ask_price_2', 0.0),
                        getattr(data, 'ask_price_3', 0.0), getattr(data, 'ask_price_4', 0.0),
                        getattr(data, 'ask_price_5', 0.0),
                        getattr(data, 'bid_volume_1', 0.0), getattr(data, 'bid_volume_2', 0.0),
                        getattr(data, 'bid_volume_3', 0.0), getattr(data, 'bid_volume_4', 0.0),
                        getattr(data, 'bid_volume_5', 0.0),
                        getattr(data, 'ask_volume_1', 0.0), getattr(data, 'ask_volume_2', 0.0),
                        getattr(data, 'ask_volume_3', 0.0), getattr(data, 'ask_volume_4', 0.0),
                        getattr(data, 'ask_volume_5', 0.0), sym_idx,
                    )
                data_idx += 1
        
        # 截断到实际使用的大小
        data_array = data_array[:data_idx]
        index_array = index_list[:data_idx]
        
        dts_array = np.array([_datetime_to_ns(dt) for dt in sorted_dts], dtype=np.int64)
        
        # 使用更快的序列化：msgpack或直接二进制编码（如果metadata结构固定）
        metadata = {'symbols': symbols, 'symbol_info': symbol_info,
                'interval': first_item.interval if is_bar else None}
        metadata_bytes = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
        
        total_size = (self.HEADER_SIZE + data_array.nbytes + dts_array.nbytes +
                    index_array.nbytes + len(metadata_bytes))
        self.shm_main = SharedMemory(create=True, size=total_size)
        
        header = struct.pack(self.HEADER_FORMAT, len(data_array), len(sorted_dts),
                            len(index_array), len(symbols), is_bar, len(metadata_bytes))
        self.shm_main.buf[:self.HEADER_SIZE] = header
        
        offset = self.HEADER_SIZE
        self.shm_main.buf[offset:offset + data_array.nbytes] = data_array.tobytes()
        offset += data_array.nbytes
        self.shm_main.buf[offset:offset + dts_array.nbytes] = dts_array.tobytes()
        offset += dts_array.nbytes
        self.shm_main.buf[offset:offset + index_array.nbytes] = index_array.tobytes()
        offset += index_array.nbytes
        self.shm_main.buf[offset:offset + len(metadata_bytes)] = metadata_bytes
        
        self.shm_main_name = self.shm_main.name
        return self.shm_main_name    


    @staticmethod
    def read_from_shared_memory(shm_name):
        """从共享内存读取数据 - 进一步优化版：使用字典推导式"""
        shm = SharedMemory(name=shm_name)
        
        header = struct.unpack(SharedMemoryManager.HEADER_FORMAT,
                            bytes(shm.buf[:SharedMemoryManager.HEADER_SIZE]))
        num_data, num_dts, num_index, num_symbols, is_bar, metadata_size = header
        is_bar = bool(is_bar)
        dtype = BAR_DTYPE if is_bar else TICK_DTYPE
        
        offset = SharedMemoryManager.HEADER_SIZE
        data_array = np.ndarray((num_data,), dtype=dtype, buffer=shm.buf, offset=offset)
        offset += num_data * dtype.itemsize
        
        dts_array = np.ndarray((num_dts,), dtype=np.int64, buffer=shm.buf, offset=offset)
        offset += num_dts * 8
        
        index_array = np.ndarray((num_index,), dtype=INDEX_DTYPE, buffer=shm.buf, offset=offset)
        offset += num_index * INDEX_DTYPE.itemsize
        
        metadata = pickle.loads(bytes(shm.buf[offset:offset + metadata_size]))
        symbols = metadata['symbols']
        
        # 优化：直接使用numpy数组操作，避免类型转换开销
        dt_indices = index_array['dt_idx']
        sym_indices = index_array['sym_idx']
        data_indices = index_array['data_idx']
        
        # 使用argsort按dt_ns分组，然后构建字典
        dt_ns_values = dts_array[dt_indices]
        
        # 预分配字典
        dt_to_symbols = {}
        
        # 使用numpy的unique获取分组边界
        unique_dt_ns, first_indices, counts = np.unique(
            dt_ns_values, return_index=True, return_counts=True
        )
        
        # 按dt_ns排序索引
        sort_order = np.argsort(dt_ns_values)
        sorted_sym_indices = sym_indices[sort_order]
        sorted_data_indices = data_indices[sort_order]
        
        # 批量构建字典
        pos = 0
        for i, dt_ns in enumerate(unique_dt_ns):
            count = counts[i]
            inner_dict = {}
            for j in range(count):
                idx = pos + j
                inner_dict[symbols[sorted_sym_indices[idx]]] = int(sorted_data_indices[idx])
            dt_to_symbols[int(dt_ns)] = inner_dict
            pos += count
        
        dts = [_ns_to_datetime(int(ns)) for ns in dts_array]
        
        accessor = SharedHistoryDataAccessor(
            data_array, dt_to_symbols, symbols, metadata['symbol_info'],
            is_bar, metadata.get('interval', Interval.MINUTE)
        )
        accessor._shm = shm
        return dts, accessor


    def cleanup(self):
        if self.shm_main:
            try:
                self.shm_main.close()
                self.shm_main.unlink()
            except:
                pass
            self.shm_main = None
    
    def __del__(self):
        self.cleanup()

# 全局变量
_CACHED_DTS = None
_CACHED_HISTORY_DATA = None
_CACHED_SHM_NAME = ""


def init_worker(shm_name):
    global _CACHED_SHM_NAME
    _CACHED_SHM_NAME = shm_name


def get_shared_data():
    global _CACHED_DTS, _CACHED_HISTORY_DATA, _CACHED_SHM_NAME
    if _CACHED_DTS is None:
        _CACHED_DTS, _CACHED_HISTORY_DATA = SharedMemoryManager.read_from_shared_memory(_CACHED_SHM_NAME)
    return _CACHED_DTS, _CACHED_HISTORY_DATA

# ============================================================================================
# 优化设置类
# ============================================================================================
class OptimizationSetting:
    """
    回测优化设置
    """
    # ----------------------------------------------------------------------------------------------------
    def __init__(self):
        self.params = {}
        self.target_name = ""
    # ----------------------------------------------------------------------------------------------------
    def add_parameter(self, name: str, start: Union[float, int], end: Union[float, int], step: Union[float, int]):
        """
        添加优化参数。此函数用于添加一个参数到优化过程中，该参数将从start开始，以step的步长递增，直到end。参数值可以是整数或浮点数，取决于start和step的类型。
        
        参数:
        - name (str): 参数名称，用于标识参数。
        - start (Union[float, int]): 参数值的起始点。
        - end (Union[float, int]): 参数值的终止点。
        - step (Union[float, int]): 参数值的步长，即每次增加的值。
        """
        name = name.strip()  # 去除空格
        if not end and not step:
            self.params[name] = [start]
            return
        assert start < end, "参数优化起始点必须小于终止点"
        assert step > 0, "参数优化步进必须大于0"
        # 确定参数值的类型
        value_type = float
        if isinstance(start, int) and isinstance(step, int):
            value_type = int
        # 使用Decimal进行精确的浮点数计算
        decimal_start = Decimal(str(start))
        decimal_end = Decimal(str(end))
        decimal_step = Decimal(str(step))
        # 生成参数值列表
        value_list = [value_type(value) for value in self.params_range(decimal_start, decimal_end, decimal_step)]

        self.params[name] = value_list
    # ----------------------------------------------------------------------------------------------------
    def params_range(self, start: Decimal, end: Decimal, step: Decimal):
        """
        生成一个包含起始至结束值范围内的迭代器。
        :param start: 起始值。
        :param end: 结束值。
        :param step: 步长。
        :return: 包含数值的迭代器。
        """
        while start <= end:
            yield start
            start += step
    # ----------------------------------------------------------------------------------------------------
    def set_target(self, target_name: str):
        """
        设置优化目标
        """
        self.target_name = target_name
    # ----------------------------------------------------------------------------------------------------
    def generate_setting(self):
        """
        生成所有参数组合的设置列表。
        :return: 包含所有参数组合的设置列表。
        """
        # 获取参数的键和值
        keys = self.params.keys()
        values = self.params.values()
        # 生成所有可能的参数值组合
        products = product(*values)
        # 将每个参数组合转换为字典格式并添加到设置列表中
        settings = [dict(zip(keys, p)) for p in products]

        return settings
# ----------------------------------------------------------------------------------------------------
MAIN_ENGINE = MainEngine(EventEngine())
class BacktestingEngine:
    """
    回测引擎
    """
    engine_type = EngineType.BACKTESTING
    get_file_path = GetFilePath()
    # ----------------------------------------------------------------------------------------------------
    def __init__(self):
        """ """
        self.vt_symbols: List[str] = []
        self.start: datetime = None
        self.end: datetime = None
        self.strategy_class = None
        self.rates: Dict[str, float] = {}
        self.slippages: Dict[str, float] = {}
        self.sizes: Dict[str, float] = {}
        self.price_ticks: Dict[str, float] = {}

        self.capital: float = 1_000_000.0

        self.strategy: StrategyTemplate = None
        self.bars: Dict[str, BarData] = {}
        self.ticks: Dict[str, TickData] = {}
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0

        self.stop_order_count = 0
        self.stop_orders: Dict[str, StopOrder] = {}
        self.active_stop_orders: Dict[str, StopOrder] = {}

        self.limit_order_count = 0
        self.limit_orders: Dict[str, OrderData] = {}
        self.active_limit_orders: Dict[str, OrderData] = {}

        self.trade_count = 0
        self.trades = {}
        self.daily_results: Dict[datetime.date, ContractDailyResult] = {}
        self.daily_df = None
        # 带单交易状态
        self.copy_trade_status = False
        # 净值指标
        self.net_value = 1  # 复利净值初始值为1
        self.net_values = []
        # 持仓盈亏初始化
        self.long_avg_cost: Dict[str, float] = defaultdict(float)  # 多头持仓均价
        self.short_avg_cost: Dict[str, float] = defaultdict(float)  # 空头持仓均价
        self.long_pos: Dict[str, float] = defaultdict(float)  # 多头仓位
        self.short_pos: Dict[str, float] = defaultdict(float)  # 空头仓位
        self.long_profit: Dict[str, float] = defaultdict(float)  # 多头盈亏
        self.short_profit: Dict[str, float] = defaultdict(float)  # 空头盈亏
        self.long_profit_total: float = 0.0      # 总多头盈亏
        self.short_profit_total: float = 0.0     # 总空头盈亏
        self.realized_pnl = 0.0  # 初始化已实现盈亏
        self.long_count: Dict[str, int] = defaultdict(int)    # 多头开仓计数
        self.short_count: Dict[str, int] = defaultdict(int)    # 多头开仓计数
        self.trading_days = 365
        # 7日年化波动率监测阈值百分比
        self.volatility_threshold = 55
        # 【新增】优化后的数据结构
        self.dts: List[datetime] = []
        # 改用按时间索引的嵌套字典: Dict[datetime, Dict[vt_symbol, BarData/TickData]]
        self.history_data: Dict[datetime, Dict[str, Union[TickData, BarData]]] = {}
        
        # 【新增】共享内存管理器
        self.shm_manager: Optional[SharedMemoryManager] = None
    # ----------------------------------------------------------------------------------------------------
    def get_engine_type(self):
        """
        获取strategy_engine引擎类型(LIVE,BACKTESTING)
        """
        return self.engine_type
    # ----------------------------------------------------------------------------------------------------
    def get_copy_trade_status(self):
        """
        获取带单交易状态
        """
        return self.copy_trade_status
    # ----------------------------------------------------------------------------------------------------
    def set_capital(self, capital):
        """
        设置初始资金
        """
        self.capital = capital
    # ----------------------------------------------------------------------------------------------------
    def clear_data(self) -> None:
        """
        重置回测变量
        """
        # 订单相关数据重置
        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()
        
        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()
        
        # 交易数据重置
        self.trade_count = 0
        self.trades.clear()
        
        # 每日结果数据重置
        self.daily_results.clear()
        self.daily_df = None
        
        # 财务指标重置
        self.net_value = 1
        self.net_values.clear()
        
        # 仓位统计重置
        self.long_avg_cost.clear()
        self.short_avg_cost.clear()
        
        self.long_pos.clear()
        self.short_pos.clear()
        
        self.long_count.clear()
        self.short_count.clear()
        
        self.long_profit.clear()
        self.short_profit.clear()
        
        self.long_profit_total = 0.0
        self.short_profit_total = 0.0
        self.realized_pnl = 0.0
        
        # 基础数据重置
        self.strategy = None
        self.bars.clear()
        self.ticks.clear()
        self.datetime = None
    # ----------------------------------------------------------------------------------------------------
    def set_parameters(
        self,
        vt_symbols: List[str],
        interval: Interval,
        start: datetime,
        rates: Dict[str, float],
        slippages: Dict[str, float],
        sizes: Dict[str, float],
        price_ticks: Dict[str, float],
        capital: float = 0.0,  # 总资金
        trading_days: int = 365,  # 年交易日
        end: datetime = None,
    ) -> None:
        """
        设置回测参数
        """
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.rates = rates
        self.slippages = slippages
        self.sizes = sizes
        self.price_ticks = price_ticks

        self.start = start
        self.end = end
        self.capital = capital

        self.trading_days = trading_days
    # ----------------------------------------------------------------------------------------------------
    def add_strategy(self, strategy_class: type, setting: dict) -> None:
        """
        添加策略类
        """
        self.strategy_class = strategy_class
        self.strategy = strategy_class(self, strategy_class.__name__, self.vt_symbols, setting)
        self.strategy.interval = self.interval
        self.strategy.sizes = self.sizes
        self.strategy.price_ticks = self.price_ticks
        self.strategy.capital = self.capital
        self.strategy.balance = self.capital
        self.strategy.long_pos = self.long_pos
        self.strategy.short_pos = self.short_pos
        self.strategy.long_profit = self.long_profit
        self.strategy.short_profit = self.short_profit
        self.strategy.long_profit_total = self.long_profit_total
        self.strategy.short_profit_total = self.short_profit_total
        if hasattr(self.strategy, "on_us_rates"):
            # 读取美联储利率数据
            self.us_rates: DataFrame = load_h5("us_rates")["data"]
            self.us_rates = self.us_rates[(self.us_rates["日期"] >= self.start.date()) & (self.us_rates["日期"] <= self.end.date())].copy()
        if setting:
            unactive_param = [loss_param for loss_param in list(setting.keys()) if loss_param not in self.strategy.parameters]
            assert not unactive_param, f"不在策略参数列表内的回测参数:{unactive_param}"
    # ----------------------------------------------------------------------------------------------------
    def load_data(self, concurrent: bool = True) -> None:
        """
        【优化】载入历史数据，使用按时间索引的数据结构
        """
        self.output("开始加载历史数据")
        if not self.end or self.end > datetime.now():
            self.end = datetime.now()
        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # redis缓存key
        cache_key = f"{sorted(self.vt_symbols)}_{self.start.date()}_{self.end.date()}"
        dts_file = f"{cache_key}_history_dts"
        data_file = f"{cache_key}_history_data"
        
        cached_dts = load_redis_data(dts_file)
        cached_data = load_redis_data(data_file)
        
        if cached_dts and cached_data:
            self.dts = cached_dts
            self.history_data = cached_data
            self.output(f"所有标的:{self.vt_symbols}，REDIS数据已加载")
            return

        # 【优化】直接使用嵌套字典结构收集数据
        history_data: Dict[datetime, Dict[str, Union[TickData, BarData]]] = defaultdict(dict)
        dts_set: Set[datetime] = set()

        if concurrent:
            max_workers = min(len(self.vt_symbols), cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self.generate_bar, vt_symbol)
                    for vt_symbol in self.vt_symbols
                ]
                for future in as_completed(futures):
                    result_dts, result_data = future.result()
                    dts_set.update(result_dts)
                    # 【优化】直接合并嵌套字典，无需二次转换
                    for dt, symbol_data in result_data.items():
                        history_data[dt].update(symbol_data)
        else:
            for vt_symbol in self.vt_symbols:
                result_dts, result_data = self.generate_bar(vt_symbol)
                dts_set.update(result_dts)
                # 【优化】直接合并嵌套字典
                for dt, symbol_data in result_data.items():
                    history_data[dt].update(symbol_data)

        # 时间排序
        self.dts = sorted(dts_set)
        self.history_data = history_data
        # 缓存到redis
        save_redis_data(dts_file, self.dts)
        save_redis_data(data_file, self.history_data)
        self.output(f"所有标的:{self.vt_symbols}历史数据加载完成")

    def generate_bar(self, vt_symbol: str):
        """
        1.载入历史数据
        2.多进程并行函数无法赋值到类全局变量
        【优化】直接返回嵌套字典结构 Dict[datetime, Dict[vt_symbol, BarData]]
        """
        # 【优化】直接使用嵌套字典结构，避免元组键和二次转换
        history_data: Dict[datetime, Dict[str, Union[BarData, TickData]]] = defaultdict(dict)
        dts: Set[datetime] = set()
        
        symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
        data_count = 0
        
        if self.interval == Interval.MINUTE:
            data = load_bar_data(vt_symbol, self.interval, self.start, self.end)
        else:
            data = load_tick_data(vt_symbol, self.start, self.end)
        
        for raw_data in data:
            raw_data.vt_symbol = raw_data.vt_symbol.replace("/DATABASE", f"/{gateway_name}")
            raw_data.gateway_name = gateway_name
            dt = raw_data.datetime
            dts.add(dt)
            # 【优化】直接使用嵌套字典
            history_data[dt][vt_symbol] = raw_data
            data_count += 1
        
        if not data_count:
            self.output(f"标的：{vt_symbol}，载入起止时间：{self.start}-{self.end}历史数据为空")
        self.output(f"标的：{vt_symbol}，历史数据加载完成，数据量：{data_count}")
        
        # 【优化】直接返回嵌套字典，无需转换
        return dts, history_data
    # ----------------------------------------------------------------------------------------------------
    def get_contract_detail(self, vt_symbol) -> Optional[ContractData]:
        """
        查询size(合约每手乘数)，price_tick(最小价格变动价位)，margin_ratio(保证金比率)等参数
        """
        contracts = MAIN_ENGINE.load_contracts()
        return contracts.get(vt_symbol, None)
    # ----------------------------------------------------------------------------------------------------
    def get_contract_tick(self, vt_symbol) -> Optional[TickData]:
        """
        获取合约tick最新价(last_price),买卖价量(bid_price_1,ask_price_1,bid_volume_1,ask_volume_1)等参数
        """
        return self.ticks.get(vt_symbol, None)
    # ----------------------------------------------------------------------------------------------------
    def get_position_detail(self, vt_symbol):
        """
        查询指定合约的持仓详情
        
        Args:
            vt_symbol: 合约代码
            
        Returns:
            OrderedDict: 包含以下字段的持仓详情
                - active_orders: 该合约的活跃订单字典 {vt_orderid: order}
                - long_pos: 多头持仓量
                - short_pos: 空头持仓量
                - pos: 净持仓 (多头 - 空头)
        """
        # 筛选指定合约的活跃订单
        active_orders = {
            vt_orderid: order
            for vt_orderid, order in self.active_limit_orders.items()
            if order.vt_symbol == vt_symbol
        }
        
        # 获取持仓数据
        long_pos = self.long_pos[vt_symbol]
        short_pos = self.short_pos[vt_symbol]
        
        # 计算净持仓
        pos = long_pos - short_pos
        
        # 创建 PositionDetail 实例
        pos_detail = PositionDetail(active_orders, long_pos, short_pos, pos)
        
        return pos_detail
    # ----------------------------------------------------------------------------------------------------
    def get_order(self, vt_orderid: str) -> Optional[OrderData]:
        """
        通过vt_orderid获取委托单
        """
        return self.limit_orders.get(vt_orderid, None)
    # ----------------------------------------------------------------------------------------------------
    def run_backtesting(self, trading_stock: bool = False) -> None:
        """运行策略回测 - 终极性能优化版"""
        strategy = self.strategy
        strategy.on_init()

        # 本地缓存所有频繁访问的变量
        dts = self.dts
        history_data = self.history_data
        vt_symbols = self.vt_symbols
        days = self.days
        is_minute = self.interval == Interval.MINUTE
        bars = self.bars
        ticks = self.ticks
        n_dts = len(dts)
        
        # 【终极优化】数据访问方式
        use_accessor = isinstance(history_data, SharedHistoryDataAccessor)
        if use_accessor:
            accessor_get = history_data.get
            history_data_list = None  # 不预构建，实时获取
        else:
            history_data_get = history_data.get
            history_data_list = [history_data_get(dt) for dt in dts]
        
        # 预缓存策略方法
        strategy_on_bars = strategy.on_bars
        strategy_on_tick = strategy.on_tick
        strategy_pos = strategy.pos
        strategy_update_order = strategy.update_order
        strategy_update_trade = strategy.update_trade
        strategy_update_position = strategy.update_position
        strategy_on_stop_order = strategy.on_stop_order
        
        # 预缓存订单相关
        active_limit_orders = self.active_limit_orders
        active_stop_orders = self.active_stop_orders
        sizes = self.sizes
        slippages = self.slippages
        rates = self.rates
        trades = self.trades
        limit_orders = self.limit_orders
        
        # 预缓存日结算相关
        daily_results = self.daily_results
        
        has_on_us_rates = hasattr(strategy, "on_us_rates")
        if has_on_us_rates:
            us_rates = self.us_rates
            
        # 预缓存仓位更新相关
        long_pos = self.long_pos
        short_pos = self.short_pos
        long_avg_cost = self.long_avg_cost
        short_avg_cost = self.short_avg_cost
        long_profit = self.long_profit
        short_profit = self.short_profit
        long_count = self.long_count
        short_count = self.short_count
        capital = self.capital

        # 累计已实现盈亏（按方向）
        long_realized_pnl_total = [0.0]
        short_realized_pnl_total = [0.0]
        
        # 使用列表存储计数器
        trade_count = [self.trade_count]
        limit_order_count = [self.limit_order_count]
        stop_order_count = [self.stop_order_count]
        realized_pnl = [self.realized_pnl]
        
        strategy_long_profit = long_profit
        strategy_short_profit = short_profit
        unrealized_pnl_total = [sum(long_profit.values()) + sum(short_profit.values())]
        
        # 【优化】内联 balance 更新，减少函数调用
        def update_balance_and_profit(vt_symbol, trade_price, size):
            old_long_pnl = strategy_long_profit.get(vt_symbol, 0)
            old_short_pnl = strategy_short_profit.get(vt_symbol, 0)
            
            new_long_pnl = 0
            new_short_pnl = 0
            lp = long_pos[vt_symbol]
            sp = short_pos[vt_symbol]
            if lp > 0:
                new_long_pnl = (trade_price - long_avg_cost[vt_symbol]) * lp * size
            if sp > 0:
                new_short_pnl = (short_avg_cost[vt_symbol] - trade_price) * sp * size
            
            strategy_long_profit[vt_symbol] = new_long_pnl
            strategy_short_profit[vt_symbol] = new_short_pnl
            unrealized_pnl_total[0] += (new_long_pnl - old_long_pnl) + (new_short_pnl - old_short_pnl)
            strategy.balance = capital + realized_pnl[0] + unrealized_pnl_total[0]
        def update_push_position(vt_symbol, trade_price):
            """更新并推送持仓信息"""
            size = sizes[vt_symbol]
            symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
            
            # 推送多头持仓
            lp = long_pos[vt_symbol]
            if lp > 0:
                long_pnl = (trade_price - long_avg_cost[vt_symbol]) * lp * size
                pos = PositionData(
                    symbol=symbol,
                    exchange=exchange,
                    direction=Direction.LONG,
                    volume=lp,
                    price=long_avg_cost[vt_symbol],
                    pnl=long_pnl,
                    gateway_name=gateway_name,
                )
                strategy_update_position(pos)
            
            # 推送空头持仓
            sp = short_pos[vt_symbol]
            if sp > 0:
                short_pnl = (short_avg_cost[vt_symbol] - trade_price) * sp * size
                pos = PositionData(
                    symbol=symbol,
                    exchange=exchange,
                    direction=Direction.SHORT,
                    volume=sp,
                    price=short_avg_cost[vt_symbol],
                    pnl=short_pnl,
                    gateway_name=gateway_name,
                )
                strategy_update_position(pos)
            
            # 如果仓位为0，也需要推送一次空仓位
            if lp == 0:
                pos = PositionData(
                    symbol=symbol,
                    exchange=exchange,
                    direction=Direction.LONG,
                    volume=0,
                    price=0,
                    pnl=0,
                    gateway_name=gateway_name,
                )
                strategy_update_position(pos)
            
            if sp == 0:
                pos = PositionData(
                    symbol=symbol,
                    exchange=exchange,
                    direction=Direction.SHORT,
                    volume=0,
                    price=0,
                    pnl=0,
                    gateway_name=gateway_name,
                )
                strategy_update_position(pos)
        # 预计算常量
        DIRECTION_LONG = Direction.LONG
        DIRECTION_SHORT = Direction.SHORT
        OFFSET_OPEN = Offset.OPEN
        STATUS_SUBMITTING = Status.SUBMITTING
        STATUS_NOTTRADED = Status.NOTTRADED
        STATUS_ALLTRADED = Status.ALLTRADED
        STOPORDER_TRIGGERED = StopOrderStatus.TRIGGERED
        
        # 移除停牌股票
        if trading_stock:
            last_dt = dts[-1]
            now = datetime.now(TZ_INFO)
            bars_at_last = accessor_get(last_dt) if use_accessor else history_data_list[-1]
            for vt_symbol in vt_symbols[:]:
                data = bars_at_last.get(vt_symbol)
                if data and (now - data.datetime).days > 180:
                    vt_symbols.remove(vt_symbol)

        # ========== 内联订单撮合 ==========
        def cross_limit_orders_fast():
            if not active_limit_orders:
                return
            
            to_remove = []
            market_data = bars if is_minute else ticks
            
            for vt_orderid, order in list(active_limit_orders.items()):
                data = market_data.get(order.vt_symbol)
                if data is None:
                    continue
                
                order_dir = order.direction
                order_price = order.price
                
                if is_minute:
                    if order_dir == DIRECTION_LONG:
                        cross_price = data.low_price
                        best_price = data.open_price
                        if not (order_price >= cross_price and cross_price < 9999999):
                            if order.status == STATUS_SUBMITTING:
                                order.status = STATUS_NOTTRADED
                                strategy_update_order(order)
                            continue
                        trade_price = min(order_price, best_price)
                        pos_change = order.volume
                    else:
                        cross_price = data.high_price
                        best_price = data.open_price
                        if not (order_price <= cross_price and cross_price < 9999999):
                            if order.status == STATUS_SUBMITTING:
                                order.status = STATUS_NOTTRADED
                                strategy_update_order(order)
                            continue
                        trade_price = max(order_price, best_price)
                        pos_change = -order.volume
                else:
                    if order_dir == DIRECTION_LONG:
                        cross_price = data.ask_price_1
                        if not (order_price >= cross_price and cross_price < 9999999):
                            if order.status == STATUS_SUBMITTING:
                                order.status = STATUS_NOTTRADED
                                strategy_update_order(order)
                            continue
                        trade_price = min(order_price, cross_price)
                        pos_change = order.volume
                    else:
                        cross_price = data.bid_price_1
                        if not (order_price <= cross_price and cross_price < 9999999):
                            if order.status == STATUS_SUBMITTING:
                                order.status = STATUS_NOTTRADED
                                strategy_update_order(order)
                            continue
                        trade_price = max(order_price, cross_price)
                        pos_change = -order.volume
                
                order.traded = order.volume
                order.status = STATUS_ALLTRADED
                strategy_update_order(order)
                to_remove.append(vt_orderid)
                
                trade_count[0] += 1
                vt_symbol = order.vt_symbol
                
                trade = TradeData(
                    symbol=order.symbol, exchange=order.exchange,
                    orderid=order.orderid, tradeid=str(trade_count[0]),
                    direction=order_dir, offset=order.offset,
                    price=trade_price, volume=order.volume,
                    datetime=current_dt, gateway_name=order.gateway_name,
                )
                
                vol = trade.volume
                size = sizes[vt_symbol]
                
                if trade.offset == OFFSET_OPEN:
                    if order_dir == DIRECTION_LONG:
                        long_count[vt_symbol] += 1
                        old_cost = long_avg_cost[vt_symbol] * long_pos[vt_symbol] * size
                        new_cost = trade_price * vol * size
                        long_pos[vt_symbol] += vol
                        if long_pos[vt_symbol] > 0:
                            long_avg_cost[vt_symbol] = (old_cost + new_cost) / (long_pos[vt_symbol] * size)
                    else:
                        short_count[vt_symbol] += 1
                        old_cost = short_avg_cost[vt_symbol] * short_pos[vt_symbol] * size
                        new_cost = trade_price * vol * size
                        short_pos[vt_symbol] += vol
                        if short_pos[vt_symbol] > 0:
                            short_avg_cost[vt_symbol] = (old_cost + new_cost) / (short_pos[vt_symbol] * size)
                else:
                    if order_dir == DIRECTION_LONG:
                        # 买入平空：只有当有空头持仓时才计算盈亏
                        actual_close_vol = min(vol, short_pos[vt_symbol])
                        if actual_close_vol > 0:
                            slip = actual_close_vol * size * slippages[vt_symbol]
                            comm = actual_close_vol * trade_price * size * rates[vt_symbol]
                            pnl = (short_avg_cost[vt_symbol] - trade_price) * actual_close_vol * size - slip - comm
                            realized_pnl[0] += pnl
                            short_realized_pnl_total[0] += pnl
                            short_pos[vt_symbol] -= actual_close_vol
                            if short_pos[vt_symbol] == 0:
                                short_avg_cost[vt_symbol] = 0
                    else:
                        # 卖出平多：只有当有多头持仓时才计算盈亏
                        actual_close_vol = min(vol, long_pos[vt_symbol])
                        if actual_close_vol > 0:
                            slip = actual_close_vol * size * slippages[vt_symbol]
                            comm = actual_close_vol * trade_price * size * rates[vt_symbol]
                            pnl = (trade_price - long_avg_cost[vt_symbol]) * actual_close_vol * size - slip - comm
                            realized_pnl[0] += pnl
                            long_realized_pnl_total[0] += pnl
                            long_pos[vt_symbol] -= actual_close_vol
                            if long_pos[vt_symbol] == 0:
                                long_avg_cost[vt_symbol] = 0
                
                update_balance_and_profit(vt_symbol, trade_price, size)
                strategy_update_trade(trade)
                trades[trade.vt_tradeid] = trade
                strategy_pos[vt_symbol] += pos_change
                update_push_position(vt_symbol, trade_price)
            for vt_orderid in to_remove:
                active_limit_orders.pop(vt_orderid,None)

        def cross_stop_orders_fast():
            if not active_stop_orders:
                return
            
            to_remove = []
            market_data = bars if is_minute else ticks
            
            for stop_orderid, stop_order in list(active_stop_orders.items()):
                vt_symbol = stop_order.vt_symbol
                data = market_data.get(vt_symbol)
                if data is None:
                    continue
                
                stop_dir = stop_order.direction
                stop_price = stop_order.price
                
                if is_minute:
                    if stop_dir == DIRECTION_LONG:
                        if stop_order.reverse:
                            if data.low_price > stop_price:
                                continue
                            trade_price = min(stop_price, data.open_price)
                        else:
                            if data.high_price < stop_price:
                                continue
                            trade_price = max(stop_price, data.open_price)
                        pos_change = stop_order.volume
                    else:
                        if stop_order.reverse:
                            if data.high_price < stop_price:
                                continue
                            trade_price = max(stop_price, data.open_price)
                        else:
                            if data.low_price > stop_price:
                                continue
                            trade_price = min(stop_price, data.open_price)
                        pos_change = -stop_order.volume
                else:
                    last_price = data.last_price
                    if stop_dir == DIRECTION_LONG:
                        if stop_order.reverse:
                            if last_price > stop_price:
                                continue
                            trade_price = min(stop_price,last_price)
                        else:
                            if last_price < stop_price:
                                continue
                            trade_price = max(stop_price, last_price)
                        pos_change = stop_order.volume
                    else:
                        if stop_order.reverse:
                            if last_price < stop_price:
                                continue
                            trade_price = max(stop_price,last_price)
                        else:
                            if last_price > stop_price:
                                continue
                            trade_price = min(stop_price, last_price)
                        pos_change = -stop_order.volume
                
                symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
                
                limit_order_count[0] += 1
                order = OrderData(
                    symbol=symbol, exchange=exchange,
                    orderid=str(limit_order_count[0]),
                    direction=stop_dir, offset=stop_order.offset,
                    price=stop_price, volume=stop_order.volume,
                    traded=stop_order.volume, status=STATUS_ALLTRADED,
                    datetime=current_dt, gateway_name=gateway_name,
                )
                limit_orders[order.vt_orderid] = order
                
                trade_count[0] += 1
                trade = TradeData(
                    symbol=symbol, exchange=exchange,
                    orderid=order.orderid, tradeid=str(trade_count[0]),
                    direction=stop_dir, offset=stop_order.offset,
                    price=trade_price, volume=stop_order.volume,
                    datetime=current_dt, gateway_name=gateway_name,
                )
                trades[trade.vt_tradeid] = trade
                
                vol = trade.volume
                size = sizes[vt_symbol]
                
                if trade.offset == OFFSET_OPEN:
                    if stop_dir == DIRECTION_LONG:
                        long_count[vt_symbol] += 1
                        old_cost = long_avg_cost[vt_symbol] * long_pos[vt_symbol] * size
                        new_cost = trade_price * vol * size
                        long_pos[vt_symbol] += vol
                        if long_pos[vt_symbol] > 0:
                            long_avg_cost[vt_symbol] = (old_cost + new_cost) / (long_pos[vt_symbol] * size)
                    else:
                        short_count[vt_symbol] += 1
                        old_cost = short_avg_cost[vt_symbol] * short_pos[vt_symbol] * size
                        new_cost = trade_price * vol * size
                        short_pos[vt_symbol] += vol
                        if short_pos[vt_symbol] > 0:
                            short_avg_cost[vt_symbol] = (old_cost + new_cost) / (short_pos[vt_symbol] * size)
                else:
                    if stop_dir == DIRECTION_LONG:
                        # 买入平空：只有当有空头持仓时才计算盈亏
                        actual_close_vol = min(vol, short_pos[vt_symbol])
                        if actual_close_vol > 0:
                            slip = actual_close_vol * size * slippages[vt_symbol]
                            comm = actual_close_vol * trade_price * size * rates[vt_symbol]
                            pnl = (short_avg_cost[vt_symbol] - trade_price) * actual_close_vol * size - slip - comm
                            realized_pnl[0] += pnl
                            short_realized_pnl_total[0] += pnl
                            short_pos[vt_symbol] -= actual_close_vol
                            if short_pos[vt_symbol] == 0:
                                short_avg_cost[vt_symbol] = 0
                    else:
                        # 卖出平多：只有当有多头持仓时才计算盈亏
                        actual_close_vol = min(vol, long_pos[vt_symbol])
                        if actual_close_vol > 0:
                            slip = actual_close_vol * size * slippages[vt_symbol]
                            comm = actual_close_vol * trade_price * size * rates[vt_symbol]
                            pnl = (trade_price - long_avg_cost[vt_symbol]) * actual_close_vol * size - slip - comm
                            realized_pnl[0] += pnl
                            long_realized_pnl_total[0] += pnl
                            long_pos[vt_symbol] -= actual_close_vol
                            if long_pos[vt_symbol] == 0:
                                long_avg_cost[vt_symbol] = 0
                
                update_balance_and_profit(vt_symbol, trade_price, size)
                stop_order.status = STOPORDER_TRIGGERED
                to_remove.append(stop_orderid)
                
                strategy_on_stop_order(stop_order)
                strategy_update_order(order)
                strategy_update_trade(trade)
                strategy_pos[vt_symbol] += pos_change
                update_push_position(vt_symbol, trade_price)

            for stop_orderid in to_remove:
                active_stop_orders.pop(stop_orderid, None)

        # ========== 主回测循环 - 无分支判断版本 ==========
        try:
            if is_minute:
                # ===== Bar数据初始化阶段 =====
                day_count = 0
                ix = 0
                prev_date = None
                
                for ix in range(n_dts):
                    dt = dts[ix]
                    cur_date = dt.date()
                    if prev_date is not None and cur_date != prev_date:
                        day_count += 1
                        if day_count >= days:
                            break
                    prev_date = cur_date
                    
                    current_dt = dt
                    self.datetime = dt
                    bars_at_dt = accessor_get(dts[ix]) if use_accessor else history_data_list[ix]
                    if bars_at_dt:
                        bars.update(bars_at_dt)
                    
                    cross_limit_orders_fast()
                    cross_stop_orders_fast()
                    strategy_on_bars(bars)

                strategy.inited = True
                self.output("策略初始化完成")
                strategy.on_start()
                strategy.trading = True
                self.output("开始回放历史数据")

                # ===== Bar数据主回放阶段 =====
                prev_date = None
                last_us_rates_date = None
                
                for i in range(ix, n_dts):
                    dt = dts[i]
                    cur_date = dt.date()
                    current_dt = dt
                    self.datetime = dt
                    
                    if prev_date is not None and cur_date != prev_date:
                        if has_on_us_rates and cur_date != last_us_rates_date:
                            now_rates = us_rates[us_rates["日期"] <= cur_date]
                            if not now_rates.empty:
                                strategy.on_us_rates(now_rates.iloc[-1:])
                                last_us_rates_date = cur_date
                    prev_date = cur_date
                    
                    bars_at_dt = accessor_get(dts[i]) if use_accessor else history_data_list[i]
                    if bars_at_dt:
                        bars.update(bars_at_dt)
                    
                    cross_limit_orders_fast()
                    cross_stop_orders_fast()
                    strategy_on_bars(bars)
                    
                    if bars:
                        daily_result = daily_results.get(cur_date)
                        if daily_result is None:
                            close_prices = {vt: bar.close_price for vt, bar in bars.items()}
                            daily_results[cur_date] = PortfolioDailyResult(cur_date, close_prices)
                        else:
                            daily_result.update_close_prices(
                                {vt: bar.close_price for vt, bar in bars.items()}
                            )
            else:
                # ===== Tick数据初始化阶段 =====
                day_count = 0
                ix = 0
                prev_date = None
                
                for ix in range(n_dts):
                    dt = dts[ix]
                    cur_date = dt.date()
                    if prev_date is not None and cur_date != prev_date:
                        day_count += 1
                        if day_count >= days:
                            break
                    prev_date = cur_date
                    
                    current_dt = dt
                    self.datetime = dt
                    ticks_at_dt = accessor_get(dts[ix]) if use_accessor else history_data_list[ix]
                    if ticks_at_dt:
                        ticks.update(ticks_at_dt)
                    
                    cross_limit_orders_fast()
                    cross_stop_orders_fast()
                    
                    for tick in ticks.values():
                        strategy_on_tick(tick)

                strategy.inited = True
                self.output("策略初始化完成")
                strategy.on_start()
                strategy.trading = True
                self.output("开始回放历史数据")

                # ===== Tick数据主回放阶段 =====
                for i in range(ix, n_dts):
                    dt = dts[i]
                    cur_date = dt.date()
                    current_dt = dt
                    self.datetime = dt
                    
                    ticks_at_dt = accessor_get(dts[i]) if use_accessor else history_data_list[i]
                    if ticks_at_dt:
                        ticks.update(ticks_at_dt)
                    
                    cross_limit_orders_fast()
                    cross_stop_orders_fast()
                    
                    for tick in ticks.values():
                        strategy_on_tick(tick)
                    
                    if ticks:
                        daily_result = daily_results.get(cur_date)
                        if daily_result is None:
                            close_prices = {vt: tick.last_price for vt, tick in ticks.items()}
                            daily_results[cur_date] = PortfolioDailyResult(cur_date, close_prices)
                        else:
                            daily_result.update_close_prices(
                                {vt: tick.last_price for vt, tick in ticks.items()}
                            )
                                
        except Exception:
            self.output("触发异常，回测终止")
            self.output(traceback.format_exc())
            return
        finally:
            self.trade_count = trade_count[0]
            self.limit_order_count = limit_order_count[0]
            self.stop_order_count = stop_order_count[0]
            self.realized_pnl = realized_pnl[0]
            
            strategy.long_pos = long_pos
            strategy.short_pos = short_pos
            strategy.long_profit = strategy_long_profit
            strategy.short_profit = strategy_short_profit
            
            # 使用最后一个bar的收盘价重新计算浮动盈亏
            unrealized_long = 0.0
            unrealized_short = 0.0
            
            last_prices = {}
            if is_minute:
                for vt_symbol, bar in bars.items():
                    if bar:
                        last_prices[vt_symbol] = bar.close_price
            else:
                for vt_symbol, tick in ticks.items():
                    if tick:
                        last_prices[vt_symbol] = tick.last_price
            
            for vt_symbol in vt_symbols:
                last_price = last_prices.get(vt_symbol, 0)
                size = sizes.get(vt_symbol, 1)
                
                lp = long_pos.get(vt_symbol, 0)
                sp = short_pos.get(vt_symbol, 0)
                
                if lp > 0 and last_price > 0:
                    unrealized_long += (last_price - long_avg_cost.get(vt_symbol, 0)) * lp * size
                if sp > 0 and last_price > 0:
                    unrealized_short += (short_avg_cost.get(vt_symbol, 0) - last_price) * sp * size
            
            for vt_symbol in vt_symbols:
                last_price = last_prices.get(vt_symbol, 0)
                size = sizes.get(vt_symbol, 1)
                lp = long_pos.get(vt_symbol, 0)
                sp = short_pos.get(vt_symbol, 0)
                
                if lp > 0 and last_price > 0:
                    strategy_long_profit[vt_symbol] = (last_price - long_avg_cost.get(vt_symbol, 0)) * lp * size
                else:
                    strategy_long_profit[vt_symbol] = 0
                    
                if sp > 0 and last_price > 0:
                    strategy_short_profit[vt_symbol] = (short_avg_cost.get(vt_symbol, 0) - last_price) * sp * size
                else:
                    strategy_short_profit[vt_symbol] = 0

            long_profit_total = long_realized_pnl_total[0] + unrealized_long
            short_profit_total = short_realized_pnl_total[0] + unrealized_short
            
            self.long_profit_total = long_profit_total
            self.short_profit_total = short_profit_total
            strategy.long_profit_total = long_profit_total
            strategy.short_profit_total = short_profit_total
            strategy.balance = capital + realized_pnl[0] + unrealized_long + unrealized_short

        self.output("历史数据回放结束")    
    # ----------------------------------------------------------------------------------------------------
    def calculate_result(self) -> DataFrame:
        """计算回测结果 - 优化版"""
        self.output("开始计算逐日盯市盈亏")
        if not self.trades:
            self.output("成交记录为空，无法计算")
            return None
        
        # 批量添加交易数据
        daily_results = self.daily_results
        for trade in self.trades.values():
            trade_date = trade.datetime.date()
            daily_result = daily_results.get(trade_date)
            if daily_result:
                daily_result.add_trade(trade)

        # 计算日回测结果
        pre_closes = {}
        start_poses = {}
        sizes = self.sizes
        rates = self.rates
        slippages = self.slippages

        for daily_result in daily_results.values():
            daily_result.calculate_pnl(pre_closes, start_poses, sizes, rates, slippages)
            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        # ============ 计算7日年化波动率 ============
        sorted_dates = sorted(daily_results.keys())
        n_days = len(sorted_dates)
        
        if n_days >= 2:
            all_symbols = set()
            for dr in daily_results.values():
                all_symbols.update(dr.close_prices.keys())
            all_symbols = sorted(all_symbols)
            n_symbols = len(all_symbols)
            
            if n_symbols > 0:
                symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}
                
                close_matrix = np.full((n_days, n_symbols), np.nan, dtype=np.float64)
                for day_idx, d in enumerate(sorted_dates):
                    cp = daily_results[d].close_prices
                    for vt_symbol, price in cp.items():
                        if price > 0:
                            close_matrix[day_idx, symbol_to_idx[vt_symbol]] = price
                
                # 向量化前值填充
                for col in range(n_symbols):
                    mask = np.isnan(close_matrix[:, col])
                    if mask.any() and not mask.all():
                        idx = np.where(~mask, np.arange(n_days), 0)
                        np.maximum.accumulate(idx, out=idx)
                        close_matrix[:, col] = close_matrix[idx, col]
                
                prev_close = close_matrix[:-1]
                curr_close = close_matrix[1:]
                with np.errstate(divide='ignore', invalid='ignore'):
                    returns_matrix = np.where(
                        (prev_close > 0) & np.isfinite(prev_close) & np.isfinite(curr_close),
                        (curr_close - prev_close) / prev_close,
                        np.nan
                    )
                
                window = 6
                annualization_factor = np.sqrt(self.trading_days) * 100
                n_returns = returns_matrix.shape[0]
                
                if n_returns >= window:
                    shape = (n_returns - window + 1, window, n_symbols)
                    strides = (returns_matrix.strides[0], returns_matrix.strides[0], returns_matrix.strides[1])
                    windowed = np.lib.stride_tricks.as_strided(returns_matrix, shape=shape, strides=strides)
                    
                    with np.errstate(all='ignore'):
                        rolling_std = np.nanstd(windowed, axis=1, ddof=1)
                        rolling_annual_vol = rolling_std * annualization_factor
                        avg_vol = np.nanmean(rolling_annual_vol, axis=1)
                    
                    for i in range(len(avg_vol)):
                        date_idx = i + window
                        if date_idx < n_days and np.isfinite(avg_vol[i]):
                            daily_results[sorted_dates[date_idx]].total_avg_volatility = float(avg_vol[i])
                
                for day_idx in range(2, min(window, n_days)):
                    window_returns = returns_matrix[:day_idx]
                    with np.errstate(all='ignore'):
                        stds = np.nanstd(window_returns, axis=0, ddof=1)
                        annual_vols = stds * annualization_factor
                        valid_mask = np.isfinite(annual_vols)
                        if np.any(valid_mask):
                            daily_results[sorted_dates[day_idx]].total_avg_volatility = float(np.nanmean(annual_vols[valid_mask]))

        # 使用numpy数组直接构建，避免Python列表
        n = len(daily_results)
        dates = np.empty(n, dtype='datetime64[D]')
        trade_counts = np.zeros(n, dtype=np.int32)
        turnovers = np.zeros(n, dtype=np.float64)
        commissions = np.zeros(n, dtype=np.float64)
        slippages_arr = np.zeros(n, dtype=np.float64)
        total_pnls = np.zeros(n, dtype=np.float64)
        net_pnls = np.zeros(n, dtype=np.float64)
        volatilities = np.zeros(n, dtype=np.float64)
        close_sums = np.zeros(n, dtype=np.float64)

        for i, daily_result in enumerate(daily_results.values()):
            dates[i] = daily_result.date
            trade_counts[i] = daily_result.trade_count
            turnovers[i] = daily_result.turnover
            commissions[i] = daily_result.commission
            slippages_arr[i] = daily_result.slippage
            total_pnls[i] = daily_result.total_pnl
            net_pnls[i] = daily_result.net_pnl
            volatilities[i] = daily_result.total_avg_volatility
            close_sums[i] = sum(daily_result.close_prices.values())

        self.daily_df = DataFrame({
            "date": dates,
            "trade_count": trade_counts,
            "turnover": turnovers,
            "commission": commissions,
            "slippage": slippages_arr,
            "total_pnl": total_pnls,
            "net_pnl": net_pnls,
            "total_avg_volatility": volatilities,
            "close_prices": close_sums,
        }).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df
    # ----------------------------------------------------------------------------------------------------
    def statistics_status(self, array: np.ndarray):
        """
        返回array均值，标准差，偏度，峰度
        """
        stats = scs.describe(array)
        return stats[2], np.sqrt(stats[3]), stats[4], stats[5]
    # ----------------------------------------------------------------------------------------------------
    def calculate_statistics(self, df: DataFrame = None, strategy_name="", output_statistics=True, show_volatility: bool = False) -> Dict[str, Union[float, int]]:
        """
        计算回测统计结果并生成相关图表。
        """
        self.output("开始计算策略统计指标")
        if hasattr(self.strategy, "strategy_name"):
            strategy_name = self.strategy.strategy_name
        formatted_trades: List[Dict, Any] = []
        
        if df is None:
            self.output("策略统计指标返回默认0.0值")
            return defaultdict(float)
        
        # 计算账户资金相关的时间序列数据
        df["bh_balance"] = df["close_prices"] / df["close_prices"].iloc[0] * self.capital
        df["balance"] = df["net_pnl"].cumsum() + self.capital
        df["return"] = (np.log(df["balance"]) - np.log(df["balance"].shift(1))).fillna(0)
        df["highlevel"] = df["balance"].rolling(min_periods=1, window=len(df), center=False).max()
        df["drawdown"] = df["balance"] - df["highlevel"]
        df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100
        return_array = df["return"].values
        
        # 计算各项指标
        buyhold_returns = (np.log(df["bh_balance"]) - np.log(df["bh_balance"].shift(1))).fillna(0).values
        d_ratio_value, d_ratio_first_value, d_ratio_last_value = tech.compute_d_ratio(
            return_bh=buyhold_returns, return_pred=return_array, trading_days=self.trading_days
        )
        
        # 统计指标计算
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = len(df)
        profit_days = len(df[df["net_pnl"] > 0])
        loss_days = len(df[df["net_pnl"] < 0])
        end_balance = df["balance"].iloc[-1]
        max_drawdown = df["drawdown"].min()
        max_drawdown_percent = df["ddpercent"].min()
        
        # 计算最长回撤持续时间
        max_drawdown_end = df["drawdown"].idxmin()
        longest_drawdown_duration = 0
        longest_drawdown_start = ""
        longest_drawdown_end = ""

        if isinstance(max_drawdown_end, date):
            # 找出所有回撤期间（回撤值 < 0）
            in_drawdown = df["drawdown"] < 0
            
            if in_drawdown.any():
                # 标记回撤区间的变化点
                drawdown_changes = in_drawdown.astype(int).diff().fillna(0)
                
                # 找到所有回撤开始和结束点
                start_indices = df.index[drawdown_changes == 1].tolist()
                end_indices = df.index[drawdown_changes == -1].tolist()
                
                # 如果第一个数据点就在回撤中
                if in_drawdown.iloc[0]:
                    start_indices.insert(0, df.index[0])
                
                # 如果最后一个数据点仍在回撤中
                if in_drawdown.iloc[-1]:
                    end_indices.append(df.index[-1])
                
                # 计算每个回撤期间的持续天数
                for start, end in zip(start_indices, end_indices):
                    duration = (end - start).days
                    if duration > longest_drawdown_duration:
                        longest_drawdown_duration = duration
                        longest_drawdown_start = start
                        longest_drawdown_end = end
        
        total_trade_pnl = df["total_pnl"].sum()
        total_net_pnl = df["net_pnl"].sum()
        daily_net_pnl = total_net_pnl / total_days
        total_commission = df["commission"].sum()
        daily_commission = total_commission / total_days
        total_slippage = df["slippage"].sum()
        daily_slippage = total_slippage / total_days
        total_turnover = df["turnover"].sum()
        daily_turnover = total_turnover / total_days
        total_trade_count = df["trade_count"].sum()
        daily_trade_count = total_trade_count / total_days
        
        long_count_total = sum(self.long_count.values())
        short_count_total = sum(self.short_count.values())
        ls_open_ratio = long_count_total / short_count_total if short_count_total else 0
        ls_profit_ratio = self.long_profit_total / self.short_profit_total if self.short_profit_total else 0
        
        total_return = (end_balance / self.capital - 1) * 100
        annual_return = total_return / total_days * self.trading_days
        return_mean, return_std, return_skew, return_kurt = self.statistics_status(return_array)
        
        sortino_value = tech.sortino_ratio(return_array)
        omega_value = tech.omega_ratio(return_array)
        information_value = tech.information_ratio(return_array)
        annual_volatility_value = tech.annual_volatility(return_array)
        cagr_value = tech.cagr(return_array)
        annual_downside_risk = tech.downside_risk(return_array)
        c_var = tech.conditional_value_at_risk(return_array, 0.01)
        var = tech.value_at_risk(return_array)
        fat_tail_risk = abs(c_var) / abs(var) if var else 0
        calmar_value = tech.calmar_ratio(return_array)
        stability_return = tech.stability_of_timeseries(return_array)
        tail_value = tech.tail_ratio(return_array)
        sharpe_value = tech.sharpe_ratio(return_array)
        
        balances = df["balance"].to_numpy()
        ratios = balances / balances[0] * self.net_value
        self.net_values = np.round(ratios, 3).tolist()
        self.net_values[0] = self.net_value
        self.net_value = self.net_values[-1]
        
        rgr_value = tech.calc_rgr(cagr_value, stability_return, annual_downside_risk, 
                                max_drawdown_percent, return_skew, return_kurt, c_var, 
                                strategy_type="trend")
        cost_benefit_ratio = abs((total_slippage + total_commission) / total_trade_pnl)
        
        if output_statistics:
            self.output("-" * 70)
            self.output(f"策略名称：{strategy_name}，交易合约数量：{len(self.vt_symbols)}，交易标的列表：{self.vt_symbols}")
            self.output(f"首个交易日：\t{start_date}，最后交易日：\t{end_date}，总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}，亏损交易日：\t{loss_days}")
            self.output(f"起始资金：\t{self.capital:.3f}，结束资金：\t{end_balance:.3f}")
            self.output(f"总盈亏：\t{total_net_pnl:.3f}，年化收益率：\t{annual_return:.3f}%")
            self.output(f"总收益率：\t{total_return:.3f}%，复利净值：\t{self.net_values[-1]:.3f}")
            self.output(Fore.RED + f"最大回撤资金: \t{max_drawdown:.3f}，最大回撤率: \t{max_drawdown_percent:.3f}%，最长回撤日期:\t{longest_drawdown_start}至{longest_drawdown_end}，最长回撤天数: \t{longest_drawdown_duration}")
            self.output(f"总手续费：\t{total_commission:.3f}，总滑点：\t{total_slippage:.3f}")
            self.output(f"总成交金额：\t{total_turnover:.3f}，总成交笔数：\t{total_trade_count}")
            self.output(f"多空开仓比：{ls_open_ratio:.3f}，总多头开仓次数：{long_count_total}，总空头开仓次数：{short_count_total}，多空盈亏比：{ls_profit_ratio:.3f}，总多头盈亏：{self.long_profit_total:.3f}，总空头盈亏：{self.short_profit_total:.3f}")
            self.output(f"日均盈亏：\t{daily_net_pnl:.3f}，日均手续费：\t{daily_commission:.3f}，日均滑点：\t{daily_slippage:.3f}，日均成交金额：\t{daily_turnover:.3f}，日均成交笔数：\t{daily_trade_count:.3f}")
            self.output(f"日均收益率：\t{return_mean*100:.3f}%，收益率标准差：\t{return_std*100:.3f}%，收益率偏度：\t{return_skew:.3f}，收益率峰度：\t{return_kurt:.3f}")
            self.output(Fore.CYAN + f"sortino_value：\t{sortino_value:.3f}，rgr_value：\t{rgr_value:.3f}")
            self.output(Fore.CYAN + f"d_ratio_value：\t{d_ratio_value:.3f}，d_ratio_first_value：\t{d_ratio_first_value:.3f}，d_ratio_last_value：\t{d_ratio_last_value:.3f}")
            self.output(Fore.RED + f"年化下行风险率：\t{annual_downside_risk:.3f}")
            self.output(Fore.RED + f"c_var：\t{c_var:.3f}，var：\t{var:.3f}，fat_tail_risk：\t{fat_tail_risk:.3f}")
            self.output(f"sharpe_value：\t{sharpe_value:.3f}")
            self.output(f"calmar_value：\t{calmar_value:.3f}")
            self.output(f"omega_value：\t{omega_value:.3f}")
            self.output(f"信息比率：\t{information_value:.3f}")
            self.output(f"年化波动率：\t{annual_volatility_value:.3f}")
            self.output(f"年化复合增长率：\t{cagr_value:.3f}")
            self.output(f"收益稳定率：\t{stability_return:.3f}")
            self.output(f"尾部比率：\t{tail_value:.3f}")
            self.output(f"成本收益比率：\t{cost_benefit_ratio:.3f}")
            
            # 保存结果
            result_path: Path = self.get_file_path.backtesting_path(strategy_name, "portfolio_strategy")
            df.to_csv(result_path, encoding="utf_8_sig")
            
            # 保存交易明细
            for trade in self.get_all_trades():
                trade_dict = trade.__dict__
                trade_dict["exchange"] = trade_dict["exchange"].value
                trade_dict["direction"] = trade_dict["direction"].value
                trade_dict["offset"] = trade_dict["offset"].value
                formatted_trades.append(trade_dict)
            DataFrame(formatted_trades).to_csv(
                str(result_path).replace("_backtesting_", "_trades_"), 
                encoding="utf_8_sig"
            )
            
            # ==================== PyEcharts 优化美化版图表 ====================
            
            # 图表1: 账户资金（对数坐标）
            balance_list = df["balance"].round(3).tolist()
            min_balance = min(balance_list)
            max_balance = max(balance_list)
            
            y_ticks = []
            current_tick = self.capital if self.capital else 5e5 * len(self.vt_symbols)
            while current_tick <= max_balance:
                y_ticks.append(current_tick)
                current_tick *= 2
            
            bar_1 = Bar()
            bar_1.add_xaxis(df["balance"].index.tolist())
            bar_1.add_yaxis(
                f"策略:{strategy_name}\n结束资金：{end_balance:.3f}\n起止时间：{df['balance'].index[0]}至{df['balance'].index[-1]}",
                balance_list,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#00c853'},
                            {offset: 1, color: '#15a451'}
                        ])
                    """),
                    border_radius=4,
                ),
                label_opts=label_opts,
                emphasis_opts=emphasis_opts
            )
            bar_1.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"总收益率：{total_return:.3f}%，sortino：{sortino_value:.3f}，d_ratio_value：{d_ratio_value:.3f}，d_ratio_first_value：{d_ratio_first_value:.3f}，d_ratio_last_value：{d_ratio_last_value:.3f}\n账户资金，年化收益率：{annual_return:.3f}%，rgr_value：{rgr_value:.3f}",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"起始资金：{self.capital:.2f} → 结束资金：{end_balance:.2f}",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                yaxis_opts=opts.AxisOpts(
                    type_="log",
                    is_show=True,
                    axislabel_opts=opts.LabelOpts(formatter="{value}"),
                    splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(type_="dashed", opacity=0.5)),
                    min_=min_balance,
                    max_=max_balance
                ),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=axislabel_opts
                )
            )
            bar_1.set_series_opts(
                markline_opts=opts.MarkLineOpts(
                    data=[{"yAxis": tick, "label": {"formatter": f"{tick:.0f}"}} for tick in y_ticks],
                    linestyle_opts=opts.LineStyleOpts(type_="solid", color=purple_js_color, width=1, opacity=0.5)
                )
            )
            
            # 图表2: 日盈亏
            day_profit_count = sum(df["net_pnl"] > 0)
            day_loss_count = sum(df["net_pnl"] < 0)
            day_win_rate = day_profit_count / (day_profit_count + day_loss_count) if (day_profit_count + day_loss_count) else 0
            
            bar_2 = Bar()
            bar_2.add_xaxis(df["net_pnl"].index.tolist())
            bar_2.add_yaxis(
                f"总日盈利次数：{day_profit_count}，亏损次数：{day_loss_count}，日胜率：{day_win_rate:.3f}",
                df["net_pnl"].round(3).tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=js_color,
                    border_radius=4
                ),
                label_opts=label_opts,
            )
            
            if show_volatility:
                total_avg_volatility = df["total_avg_volatility"].round(3).values
                greater = np.sum(total_avg_volatility > self.volatility_threshold)
                high_ratio = greater / len(total_avg_volatility)
                avg_volatility = np.mean(total_avg_volatility)
                max_volatility = np.max(total_avg_volatility)
                min_volatility = np.min(total_avg_volatility)
                # 添加波动率折线
                line = Line()
                line.add_xaxis(df["total_avg_volatility"].index.tolist())
                line.add_yaxis(
                    series_name=f"平均7日年化波动率：{avg_volatility:.3f}%，超过{self.volatility_threshold}%日数：{greater}，占比：{high_ratio*100:.3f}%，最高波动率：{max_volatility:.3f}%，最低波动率：{min_volatility:.3f}%",
                    y_axis=list(total_avg_volatility),
                    yaxis_index=1,
                    is_smooth=True,
                    symbol="circle",
                    symbol_size=6,
                    linestyle_opts=opts.LineStyleOpts(width=2, color=purple_color),
                    itemstyle_opts=opts.ItemStyleOpts(color=purple_js_color,border_radius=4),
                    label_opts=label_opts,
                )
                
                bar_2.extend_axis(
                    yaxis=opts.AxisOpts(
                        type_="value",
                        name="波动率(%)",
                        name_location="middle",
                        name_gap=50,
                        position="right",
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(color=purple_color)
                        ),
                        axislabel_opts=opts.LabelOpts(color=purple_color)
                    )
                )
                bar_2.overlap(line)
            
            bar_2.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"最大日盈利：{df['net_pnl'].max():.3f}\n最大日亏损：{df['net_pnl'].min():.3f}",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"日均盈亏：{daily_net_pnl:.3f}",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=axislabel_opts
                )
            )
            
            # 图表3: 周盈亏
            iso_cal = df.index.isocalendar()
            year_week_keys = iso_cal.year.astype(str) + '-' + iso_cal.week.astype(str).str.zfill(2)
            weekly_pnl = df["net_pnl"].groupby(year_week_keys).sum().round(3)
            week_dates_idx = to_datetime(weekly_pnl.index + '-1', format='%G-%V-%u')
            week_dates = week_dates_idx.date.tolist()
            week_pnl_list = weekly_pnl.tolist()
            
            if len(weekly_pnl):
                max_week_pnl = weekly_pnl.max()
                min_week_pnl = min(weekly_pnl.min(), 0)
                week_profit_count = (weekly_pnl > 0).sum()
                week_loss_count = (weekly_pnl < 0).sum()
            else:
                max_week_pnl = min_week_pnl = 0
                week_profit_count = week_loss_count = 0
            
            week_win_rate = week_profit_count / len(weekly_pnl) if len(weekly_pnl) else 0
            
            bar_3 = Bar()
            bar_3.add_xaxis(week_dates)
            bar_3.add_yaxis(
                f"总周盈利次数：{week_profit_count}，周亏损次数：{week_loss_count}，周胜率：{week_win_rate * 100:.3f}%",
                week_pnl_list,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=js_color,
                    border_radius=4
                ),
                label_opts=label_opts,
            )
            bar_3.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"最大周盈利：{max_week_pnl:.3f}\n最大周亏损：{min_week_pnl:.3f}",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle="周度盈亏统计",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 图表4: 月盈亏
            datetime_index = to_datetime(df.index)
            monthly_pnl = df["net_pnl"].groupby(datetime_index.to_period('M')).sum().round(3)
            month_dates = monthly_pnl.index.to_timestamp().date.tolist()
            month_pnl_list = monthly_pnl.tolist()
            
            if len(monthly_pnl):
                max_month_pnl = monthly_pnl.max()
                min_month_pnl = min(monthly_pnl.min(), 0)
                month_profit_count = (monthly_pnl > 0).sum()
                month_loss_count = (monthly_pnl < 0).sum()
            else:
                max_month_pnl = min_month_pnl = 0
                month_profit_count = month_loss_count = 0
            
            month_win_rate = month_profit_count / (month_profit_count + month_loss_count) if (month_profit_count + month_loss_count) else 0
            
            bar_4 = Bar()
            bar_4.add_xaxis(month_dates)
            bar_4.add_yaxis(
                f"总月盈利次数：{month_profit_count}，月亏损次数：{month_loss_count}，月胜率：{month_win_rate * 100:.3f}%\n近一年月平均盈亏：{np.mean(month_pnl_list[-12:]):.3f}",
                month_pnl_list,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=js_color,
                    border_radius=4
                ),
                label_opts=label_opts,
            )
            
            if show_volatility:
                df["volatility_above"] = df["total_avg_volatility"] > self.volatility_threshold
                month_volatility_above = df.groupby(datetime_index.to_period('M'))["volatility_above"].sum()
                month_volatility_list = month_volatility_above.tolist()
                # 添加折线
                line = Line()
                line.add_xaxis(month_dates)
                line.add_yaxis(
                    series_name=f"月7日年化波动率超{self.volatility_threshold}%日数，最高：{max(month_volatility_list)}日，最低：{min(month_volatility_list)}日",
                    y_axis=month_volatility_list,
                    yaxis_index=1,
                    is_smooth=True,
                    symbol="circle",
                    symbol_size=6,
                    linestyle_opts=opts.LineStyleOpts(width=2, color=purple_color),
                    itemstyle_opts=opts.ItemStyleOpts(color=purple_js_color,border_radius=4),
                    label_opts=label_opts,
                )
                
                bar_4.extend_axis(
                    yaxis=opts.AxisOpts(
                        type_="value",
                        name="日数",
                        name_location="middle",
                        name_gap=50,
                        position="right",
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(color=purple_color)
                        ),
                        axislabel_opts=opts.LabelOpts(color=purple_color)
                    )
                )
                bar_4.overlap(line)
            
            bar_4.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"最大月盈利：{max_month_pnl:.3f}\n最大月亏损：{min_month_pnl:.3f}",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle="月度盈亏统计",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 图表5: 回撤资金
            bar_5 = Bar()
            bar_5.add_xaxis(df["drawdown"].index.tolist())
            bar_5.add_yaxis(
                f"最大回撤资金：{max_drawdown:.3f}\n最长回撤日期: \t{longest_drawdown_start}至{longest_drawdown_end}，最长回撤天数: \t{longest_drawdown_duration}",
                df["drawdown"].round(3).tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#ff1744'},
                            {offset: 1, color: '#DB1E44'}
                        ])
                    """),
                    border_radius=4,
                ),
                label_opts=label_opts,
                emphasis_opts=emphasis_opts
            )
            bar_5.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="回撤资金",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"最长回撤持续天数：{longest_drawdown_duration}天",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 图表6: 成交记录（如果有交易）
            trade_datetimes = [trade["datetime"] for trade in formatted_trades]
            trade_prices = [trade["price"] for trade in formatted_trades]
            
            if trade_datetimes:
                # 构建更详细的标记点数据
                trades_opts_data = [
                    opts.MarkPointItem(
                        name=(
                            f"订单ID:{trade['orderid']},\n"
                            f"标的:{trade['symbol']},\n"
                            f"时间:{trade['datetime']},\n"
                            f"方向:{trade['direction']},{trade['offset']},\n"
                            f"价格:{trade['price']},\n"
                            f"成交量:{trade['volume']:.3f},\n"
                        ),
                        itemstyle_opts=opts.ItemStyleOpts(
                            color=long_color if trade["direction"] == "多" else short_color
                        ),
                        coord=[trade["datetime"], trade["price"] * random.randrange(1000, 1010) / 1000],
                        value=f"{trade['direction']}{trade['offset']}",
                    )
                    for trade in formatted_trades
                ]

                
                bar_6 = Line()
                bar_6.add_xaxis(trade_datetimes)
                bar_6.add_yaxis(
                    f"总交易日：{total_days}，总成交笔数：{len(formatted_trades)}，日均成交笔数：{daily_trade_count:.3f}\n多空开仓比：{ls_open_ratio:.3f}，总多头开仓次数：{long_count_total}，总空头开仓次数：{short_count_total}，多空盈亏比：{ls_profit_ratio:.3f}，总多头盈亏：{self.long_profit_total:.3f}，总空头盈亏：{self.short_profit_total:.3f}",
                    trade_prices,
                    is_smooth=True,
                    symbol="circle",
                    symbol_size=6,
                    itemstyle_opts=opts.ItemStyleOpts(color=short_color),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    label_opts=label_opts,
                )
                bar_6.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"交易时间：{trade_datetimes[0].date()}至{trade_datetimes[-1].date()}\n成交价格",
                        title_textstyle_opts=enhanced_title_opts,
                        subtitle=f"成交记录数：{len(formatted_trades)}",
                        subtitle_textstyle_opts=enhanced_title_opts
                    ),
                    toolbox_opts=toolbox_opts,
                    datazoom_opts=datazoom_opts,
                    tooltip_opts=enhanced_tooltip_opts,
                    legend_opts=enhanced_legend_opts,
                    xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
                )
                bar_6.set_series_opts(
                    markpoint_opts=opts.MarkPointOpts(
                        data=trades_opts_data,
                        # 标记圆形："circle""，方形："rect""， 圆角方形："roundRect""，三角形："triangle""，菱形："diamond""，水滴："pin""，箭头："arrow"
                        symbol="pin",
                        symbol_size=50,
                        label_opts=opts.LabelOpts(
                            color="#FFFFFF",
                            font_family="方正韵动中黑简体",
                            font_size=14
                        )
                    )
                )
            
            # 图表7: 复利净值
            bar_7 = Bar()
            bar_7.add_xaxis(df["balance"].index.tolist())
            bar_7.add_yaxis(
                f"复利净值最高点：{max(self.net_values)}\t复利净值最低点：{min(self.net_values)}",
                self.net_values,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#00c853'},
                            {offset: 1, color: '#15a451'}
                        ])
                    """),
                    border_radius=4,
                ),
                label_opts=label_opts,
                emphasis_opts=emphasis_opts
            )
            bar_7.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="复利净值",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"初始净值：{self.net_values[0]:.3f} → 最终净值：{self.net_values[-1]:.3f}",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 图表8: 回撤百分比
            bar_8 = Bar()
            bar_8.add_xaxis(df["ddpercent"].index.tolist())
            bar_8.add_yaxis(
                f"回撤百分比\n最大回撤率：{max_drawdown_percent:.3f}%",
                round(df["ddpercent"], 3).tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#ff1744'},
                            {offset: 1, color: '#DB1E44'}
                        ])
                    """),
                    border_radius=4,
                ),
                label_opts=label_opts,
                emphasis_opts=emphasis_opts
            )
            bar_8.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="回撤百分比",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"最大回撤：{max_drawdown_percent:.3f}%",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 图表9: 盈亏分布直方图
            hist, bin_edges = np.histogram(df["net_pnl"], bins=50)
            bar_9 = Bar()
            bar_9.add_xaxis([round(x, 3) for x in bin_edges[1:].tolist()])
            bar_9.add_yaxis(
                "盈亏分布直方图",
                hist.tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#536dfe'},
                            {offset: 1, color: '#3d5afe'}
                        ])
                    """)
                ),
                label_opts=label_opts,
            )
            bar_9.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="频数",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle="盈亏分布统计（50个区间）",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(
                    name="盈亏区间",
                    axislabel_opts=axislabel_opts
                ),
                yaxis_opts=opts.AxisOpts(name="频数")
            )
            
            # 图表10: 手续费
            bar_10 = Bar()
            bar_10.add_xaxis(df["commission"].index.tolist())
            bar_10.add_yaxis(
                f"日均手续费：{df['commission'].mean():.3f}\n日最高手续费：{df['commission'].max():.3f}",
                round(df["commission"], 3).tolist(),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            {offset: 0, color: '#ffa726'},
                            {offset: 1, color: '#ff9800'}
                        ])
                    """)
                ),
                label_opts=label_opts,
            )
            bar_10.set_global_opts(
                title_opts=opts.TitleOpts(
                    title="手续费",
                    title_textstyle_opts=enhanced_title_opts,
                    subtitle=f"总手续费：{total_commission:.3f}",
                    subtitle_textstyle_opts=enhanced_title_opts
                ),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
                xaxis_opts=opts.AxisOpts(axislabel_opts=axislabel_opts)
            )
            
            # 组装页面
            page = Page(layout=Page.SimplePageLayout)
            add_bars = [bar_1, bar_2, bar_3, bar_4, bar_5] + ([bar_6] if trade_datetimes else []) + [bar_7, bar_8, bar_9, bar_10]
            for bar in add_bars:
                bar.width = "100%"
                page.add(bar)
            
            page.render(str(result_path).replace(".csv", ".html"))
            
            # 删除超过1天的旧文件
            file_path = PARENT_PATH / "backtesting_result"
            for file_path in chain(file_path.glob("*.html"), file_path.glob("*.csv")):
                file_name = file_path.name
                try:
                    file_date = datetime.strptime(file_name.split("_")[0], "%Y-%m-%d %H-%M-%S")
                    if (datetime.now() - file_date).days > 1:
                        file_path.unlink()
                except ValueError:
                    continue
        
        # 返回统计结果
        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "longest_drawdown_duration": longest_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "return_mean": return_mean,
            "return_std": return_std,
            "return_skew": return_skew,
            "return_kurt": return_kurt,
            "sharpe_value": sharpe_value,
            "calmar_value": calmar_value,
            "sortino_value": sortino_value,
            "d_ratio_value": d_ratio_value,
            "d_ratio_first_value": d_ratio_first_value,
            "d_ratio_last_value": d_ratio_last_value,
            "omega_value": omega_value,
            "information_value": information_value,
            "annual_volatility_value": annual_volatility_value,
            "cagr_value": cagr_value,
            "rgr_value": rgr_value,
            "annual_downside_risk": annual_downside_risk,
            "c_var": c_var,
            "var": var,
            "stability_return": stability_return,
            "tail_value": tail_value,
            "cost_benefit_ratio": cost_benefit_ratio,
        }
        
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)
        
        self.output("策略统计指标计算完成")
        return statistics


    # ----------------------------------------------------------------------------------------------------
    def show_chart(self, df: DataFrame = None):
        """
        使用matplotlib绘制财务指标图表。
        df: 包含财务指标数据的DataFrame。
        """
        # ========== 输入验证 ==========
        if df is None:
            return None

        if not isinstance(df, DataFrame):
            raise TypeError(f"期望 DataFrame 类型，实际收到 {type(df).__name__}")

        required_columns = {'balance', 'drawdown', 'ddpercent', 'net_pnl'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"DataFrame 缺少必需列: {missing_columns}")

        # ========== 数据预处理（性能优化：一次性提取，避免重复索引） ==========
        n_points = len(df)
        x_range = np.arange(n_points)  # 预计算 x 轴范围

        # 预提取为 numpy 数组，减少 pandas 开销
        data_cache: Dict[str, np.ndarray] = {
            'balance': df['balance'].values,
            'drawdown': df['drawdown'].values,
            'ddpercent': df['ddpercent'].values,
            'net_pnl': df['net_pnl'].values,
        }

        # ========== 图表配置 ==========
        plots_config = [
            {"title": "Balance", "data": "balance", "plot_type": "line"},
            {"title": "Drawdown", "data": "drawdown", "plot_type": "fill"},
            {"title": "Drawdown %", "data": "ddpercent", "plot_type": "fill"},
            {"title": "Daily PnL", "data": "net_pnl", "plot_type": "bar"},
            {"title": "PnL Distribution", "data": "net_pnl", "plot_type": "hist"},
        ]

        # ========== 绘图函数映射（策略模式，提高可扩展性） ==========
        def _plot_line(ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
            """绘制折线图"""
            ax.plot(x, y, linewidth=1.2, color='#1f77b4')
            ax.legend([kwargs.get('label', 'Value')], loc='upper left')

        def _plot_fill(ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
            """绘制填充面积图（适用于回撤展示）"""
            ax.fill_between(x, y, alpha=0.6, color='#d62728')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        def _plot_bar(ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
            """
            绘制条形图（带自适应降采样）

            性能优化：当数据点超过阈值时，自动聚合降采样
            避免大量条形导致渲染缓慢
            """
            max_bars = 500  # 条形图最大数量阈值

            if len(y) > max_bars:
                # 降采样：按组聚合求和
                step = len(y) // max_bars + 1
                y_sampled = np.array([
                    y[i:i + step].sum() for i in range(0, len(y), step)
                ])
                x_sampled = np.arange(len(y_sampled))
            else:
                x_sampled, y_sampled = x, y

            # 根据正负值设置颜色
            colors = np.where(y_sampled >= 0, '#2ecc71', '#e74c3c')
            ax.bar(x_sampled, y_sampled, color=colors, width=0.8, edgecolor='none')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xticks([])  # 隐藏 x 轴刻度

        def _plot_hist(ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
            """绘制直方图（盈亏分布）"""
            ax.hist(y, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
            ax.axvline(x=np.mean(y), color='green', linestyle='--', linewidth=1, label=f'Mean: {np.mean(y):.2f}')
            ax.legend(loc='upper right', fontsize=8)

        # 绘图函数映射表
        plot_functions: Dict[str, Callable] = {
            'line': _plot_line,
            'fill': _plot_fill,
            'bar': _plot_bar,
            'hist': _plot_hist,
        }

        # ========== 创建图表（性能优化：一次性创建所有子图） ==========
        fig, axes = plt.subplots(
            nrows=len(plots_config),
            ncols=1,
            figsize=(12, 16),
            dpi=100,
            constrained_layout=True  # 替代 tight_layout()，性能更好
        )

        # ========== 绑定渲染 ==========
        for ax, config in zip(axes, plots_config):
            # 获取预缓存的数据
            y_data = data_cache[config['data']]

            # 设置子图标题
            ax.set_title(config['title'], fontsize=11, fontweight='bold', loc='left')

            # 调用对应的绘图函数
            plot_func = plot_functions.get(config['plot_type'])
            if plot_func:
                plot_func(ax, x_range, y_data, label=config['data'])

            # 统一样式：添加网格
            ax.grid(True, alpha=0.3, linestyle='--')

        # ========== 显示图表 ==========
        plt.show()
        return fig
    # ----------------------------------------------------------------------------------------------------
    def run_optimization(self, optimization_setting: OptimizationSetting, max_workers: int = 0, target_reverse=True, trading_stock: bool = False):
        """
        多进程优化函数，使用共享内存在进程间共享历史数据，减少内存占用。

        参数:
            optimization_setting (OptimizationSetting): 包含优化参数组合和目标名称的对象。
            max_workers (int): 并行处理的最大工作进程数，默认为0时自动设置为CPU核心数。
            target_reverse (bool): 指示是否反转优化目标，默认为True。
            trading_stock (bool): 是否按股票停牌规则过滤标的，需与单次回测保持一致。

        返回:
            List[Tuple[Dict,float,Dict]]: 包含每个优化参数组合及其对应性能结果的列表。
        """
        # Get optimization setting and target
        settings = optimization_setting.generate_setting()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("优化参数组合为空，请检查")
            return

        if not target_name:
            self.output(f"优化目标：{target_name}未设置，请检查")
            return

        # 确保数据已加载
        if not self.dts or not self.history_data:
            self.load_data()

        # 创建零拷贝共享内存管理器
        self.shm_manager = SharedMemoryManager()
        shm_name = self.shm_manager.create_shared_memory(self.dts, self.history_data)
        self.output(f"零拷贝共享内存创建成功: {shm_name}")

        # 设置运行进程数量
        if not max_workers:
            max_workers = cpu_count()

        pool = None
        try:
            # 使用 initializer 传递共享内存名称
            pool = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker,
                initargs=(shm_name,)  # 只需要一个共享内存名称
            )

            results = []
            for setting in settings:
                result = pool.submit(
                    optimize_with_shared_memory,
                    *(
                        target_name,
                        self.strategy_class,
                        setting,
                        self.vt_symbols,
                        self.start,
                        self.end,
                        self.rates,
                        self.slippages,
                        self.sizes,
                        self.price_ticks,
                        self.capital,
                        self.interval,
                        self.days,
                        trading_stock,
                    ),
                )
                results.append(result)

            # 对优化结果排序并打印
            result_values: List[Tuple[str, float, Dict]] = sorted(
                (res.result() for res in results), # 过滤统计结果异常数据
                key=lambda x: x[1],
                reverse=target_reverse
            )
            # 第二个优化目标
            target_name_2 = TARGET_MAP[target_name]
            params, target_1_values, target_2_values = [], [], []
            optimization_results = []
            for value in result_values:
                param, target_value, statistics = value
                target_1_value = round(target_value, 3)
                params.append(param)
                target_1_values.append(target_1_value)

                target_2_value = round(statistics[target_name_2], 3)
                target_2_values.append(target_2_value)
                optimization_result = (param, {target_name: target_1_value}, {target_name_2: target_2_value})

                optimization_results.append(optimization_result)
                msg = f"优化结果：{optimization_result}"
                self.output(msg)

            save_json("optimization_result.json", optimization_results)
            # 优化结果输出到pyecharts
            page = Page(layout=Page.SimplePageLayout)
            bar_1 = Bar()
            bar_1.add_xaxis(params)
            bar_1.add_yaxis(f"{self.strategy.__class__.__name__}\n\n{target_name}优化分布图",
                            target_1_values,
                            color=short_color,
                            itemstyle_opts=opts.ItemStyleOpts(color=js_color,border_radius=4),
                            emphasis_opts=emphasis_opts,
                            )  # 主标题
            # 添加波动率折线
            line = Line()
            line.add_xaxis(params)
            line.add_yaxis(
                series_name=target_name_2,
                y_axis=target_2_values,
                yaxis_index=1,
                is_smooth=True,
                symbol="circle",
                symbol_size=6,
                linestyle_opts=opts.LineStyleOpts(width=2, color=purple_color),
                itemstyle_opts=opts.ItemStyleOpts(color=purple_js_color,border_radius=4),
                label_opts=label_opts
            )

            bar_1.extend_axis(
                yaxis=opts.AxisOpts(
                    type_="value",
                    name=target_name_2,
                    name_location="middle",
                    name_gap=50,
                    position="right",
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(color=purple_color)
                    ),
                    axislabel_opts=opts.LabelOpts(color=purple_color)
                )
            )
            bar_1.overlap(line)
            bar_1.set_global_opts(
                opts.TitleOpts(title=f"{target_name}", title_textstyle_opts=enhanced_title_opts,),
                toolbox_opts=toolbox_opts,
                datazoom_opts=datazoom_opts,
                tooltip_opts=enhanced_tooltip_opts,
                legend_opts=enhanced_legend_opts,
            )
            bar_1.set_series_opts(label_opts=label_opts)  # 系列配置项
            bar_1.width = "100%"
            page.add(bar_1)
            opt_path = str(self.get_file_path.opt_path(f"multiprocess_{self.strategy.strategy_name}", "portfolio_strategy"))
            page.render(opt_path)

            # 删除保存超1天的优化html文件
            file_path = PARENT_PATH / "optimization_result"
            for file_path in file_path.glob("*.html"):  # 搜索父路径下所有指定文件
                file_name = file_path.name  # 获取文件名
                file_date = datetime.strptime(file_name.split("_")[0], "%Y-%m-%d %H-%M-%S")
                if (datetime.now() - file_date).days > 1:
                    file_path.unlink()  # 移除文件

            return result_values
        finally:
            # 确保清理共享内存和进程池
            if pool:
                pool.shutdown(wait=True)

            # 清理共享内存
            if self.shm_manager:
                self.shm_manager.cleanup()
                self.output("共享内存已清理")
    # ----------------------------------------------------------------------------------------------------
    def load_ticks(self, strategy: StrategyTemplate, days: int) -> None:
        """
        初始化标的数据days:天数
        """
        self.days = days
    # ----------------------------------------------------------------------------------------------------
    def load_bars(self, strategy: StrategyTemplate, days: int, interval: Interval) -> None:
        """
        初始化标的数据days:天数
        """
        self.days = days
    # ----------------------------------------------------------------------------------------------------
    def send_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        line: bool,
        reverse: bool,
        lock: bool,
        order_type: OrderType,
        reference: str,
    ):
        """
        发送委托单
        """
        # 价格取整到最小变动
        price = round_to(price, self.price_ticks[vt_symbol])
        # 过滤非正常下单价格与委托量
        if not price or not volume:
            return []
        # 平仓时仓位为0直接返回
        if offset == Offset.CLOSE:
            if not self.strategy.pos[vt_symbol]:
                return
        if stop:
            vt_orderid = self.send_stop_order(self, vt_symbol, direction, offset, price, volume, reverse, OrderType.STOP)
        else:
            vt_orderid = self.send_limit_order(self, vt_symbol, direction, offset, price, volume, OrderType.LIMIT)
        return [vt_orderid]
    # ----------------------------------------------------------------------------------------------------
    def send_stop_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        reverse: bool,
        order_type: OrderType,
    ):
        """
        发送本地停止单
        """
        self.stop_order_count += 1
        stop_order = StopOrder(
            vt_symbol=vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            reverse=reverse,
            order_type=order_type,
            stop_orderid=f"{STOPORDER_PREFIX}_{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )
        # 停止单委托时间
        if not stop_order.order_datetime:
            stop_order.order_datetime = self.datetime
        self.strategy.on_stop_order(stop_order)
        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order
        return stop_order.stop_orderid
    # ----------------------------------------------------------------------------------------------------
    def send_limit_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        order_type: OrderType,
    ):
        """
        发送限价单
        """
        self.limit_order_count += 1
        symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
        order = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            type=order_type,
            traded=volume,
            status=Status.NOTTRADED,
            gateway_name=gateway_name,
        )
        order.datetime = self.datetime
        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid
    # ----------------------------------------------------------------------------------------------------
    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str):
        """
        用vt_orderid撤销委托单
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)
    # ----------------------------------------------------------------------------------------------------
    def cancel_all(self, strategy: StrategyTemplate):
        """
        撤销所有活动限价单和停止单，清空策略维护的限价单和停止单委托字典
        """
        vt_orderids = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids = list(self.active_stop_orders.keys())
        for stop_orderid in stop_orderids:
            self.cancel_stop_order(strategy, stop_orderid)
        # 清空策略维护的限价委托单
        if hasattr(strategy, "long_vt_orderids"):
            raw_vt_orderids: List[List[str]] = [
                *strategy.long_vt_orderids.values(),
                *strategy.short_vt_orderids.values(),
                *strategy.sell_vt_orderids.values(),
                *strategy.cover_vt_orderids.values(),
            ]
            vt_orderids: List[str] = [vt_orderid for vt_orderids in raw_vt_orderids for vt_orderid in vt_orderids]
            # 取消未触发限价委托单
            for vt_orderid in vt_orderids:
                self.cancel_order(strategy, vt_orderid)
            # 无需重置策略vt_orderids字典会在策略update_order里面移除撤单的vt_orderid

        # 清空策略维护的停止单stop_orderids
        if hasattr(strategy, "long_stop_orderids"):
            raw_stop_orderids: List[List[str]] = [
                *strategy.long_stop_orderids.values(),
                *strategy.short_stop_orderids.values(),
                *strategy.sell_stop_orderids.values(),
                *strategy.cover_stop_orderids.values(),
            ]
            stop_orderids: List[str] = [stop_orderid for stop_orderids in raw_stop_orderids for stop_orderid in stop_orderids]
            # 取消未触发停止单
            for stop_orderid in stop_orderids:
                self.cancel_order(strategy, stop_orderid)
            # 无需重置策略stop_orderids字典会策略在on_stop_order移除撤单的stop_orderid
    # ----------------------------------------------------------------------------------------------------
    def cancel_stop_order(self, strategy: StrategyTemplate, stop_orderid: str):
        """
        用stop_orderid撤销停止单
        """
        if stop_orderid not in self.active_stop_orders:
            return
        stop_order = self.active_stop_orders.pop(stop_orderid)
        stop_order.status = StopOrderStatus.CANCELLED

        self.strategy.on_stop_order(stop_order)
    # ----------------------------------------------------------------------------------------------------
    def cancel_limit_order(self, strategy: StrategyTemplate, vt_orderid: str):
        """
        用vt_orderid撤销限价单
        """
        if vt_orderid not in self.active_limit_orders:
            return
        order = self.active_limit_orders.pop(vt_orderid)
        order.status = Status.CANCELLED
        self.strategy.update_order(order)
    # ----------------------------------------------------------------------------------------------------
    def write_log(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """
        写入日志
        """
        self.output(msg)
    # ----------------------------------------------------------------------------------------------------
    def send_email(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """
        发送邮件
        """
        pass
    # ----------------------------------------------------------------------------------------------------
    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """
        保存策略变量到本地
        """
        pass
    # ----------------------------------------------------------------------------------------------------
    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """
        推送策略到事件引擎
        """
        pass
    # ----------------------------------------------------------------------------------------------------
    def get_random_int(self,start,end):
        """
        获取随机整数
        """
        return MAIN_ENGINE.get_random_int(start,end)
    # ----------------------------------------------------------------------------------------------------
    def output(self, msg: str) -> None:
        """
        回测引擎输出日志信息
        """
        print(f"{datetime.now()}\t{msg}")
    # ----------------------------------------------------------------------------------------------------
    def get_all_trades(self) -> List[TradeData]:
        """
        获取所有成交记录
        """
        return list(self.trades.values())
    # ----------------------------------------------------------------------------------------------------
    def get_all_orders(self) -> List[OrderData]:
        """
        获取所有限价委托单
        """
        return list(self.limit_orders.values())
    # ----------------------------------------------------------------------------------------------------
    def get_all_daily_results(self) -> List["PortfolioDailyResult"]:
        """
        获取PortfolioDailyResult统计
        """
        return list(self.daily_results.values())
# ----------------------------------------------------------------------------------------------------
class ContractDailyResult:
    __slots__ = ['date', 'close_price', 'pre_close', 'trades', 'trade_count',
                'start_pos', 'end_pos', 'turnover', 'commission', 'slippage',
                'trading_pnl', 'holding_pnl', 'total_pnl', 'net_pnl']

    def __init__(self, result_date, close_price):
        self.date = result_date
        self.close_price = close_price
        self.pre_close = 0.0
        self.trades = []
        self.trade_count = 0
        self.start_pos = 0.0
        self.end_pos = 0.0
        self.turnover = 0.0
        self.commission = 0.0
        self.slippage = 0.0
        self.trading_pnl = 0.0
        self.holding_pnl = 0.0
        self.total_pnl = 0.0
        self.net_pnl = 0.0

    def calculate_pnl(self, pre_close, start_pos, size, rate, slippage):
        self.pre_close = pre_close if pre_close else 1.0
        self.start_pos = start_pos
        self.end_pos = start_pos
        
        close_price = self.close_price
        self.holding_pnl = start_pos * (close_price - self.pre_close) * size
        self.trade_count = len(self.trades)
        
        LONG = Direction.LONG
        for trade in self.trades:
            vol = trade.volume
            price = trade.price
            pos_change = vol if trade.direction == LONG else -vol
            self.end_pos += pos_change
            turnover = vol * size * price
            self.trading_pnl += pos_change * (close_price - price) * size
            self.slippage += vol * size * slippage
            self.turnover += turnover
            self.commission += turnover * rate

        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage
# ----------------------------------------------------------------------------------------------------
class PortfolioDailyResult:
    """优化版日结算结果"""
    __slots__ = ['date', 'close_prices', 'last_close_prices', 'pre_closes',
                'start_poses', 'end_poses', 'contract_results', 'trade_count',
                'turnover', 'commission', 'slippage', 'trading_pnl', 'holding_pnl',
                'total_pnl', 'net_pnl', 'total_avg_volatility', 'returns',
                'annualized_volatility']

    def __init__(self, result_date, close_prices):
        self.date = result_date
        self.close_prices = close_prices
        self.last_close_prices = {}
        self.pre_closes = {}
        self.start_poses = {}
        self.end_poses = {}
        self.returns = defaultdict(list)
        self.annualized_volatility = defaultdict(float)
        self.total_avg_volatility = 0.0
        
        self.contract_results = {
            vt: ContractDailyResult(result_date, price) 
            for vt, price in close_prices.items()
        }
        
        self.trade_count = 0
        self.turnover = 0.0
        self.commission = 0.0
        self.slippage = 0.0
        self.trading_pnl = 0.0
        self.holding_pnl = 0.0
        self.total_pnl = 0.0
        self.net_pnl = 0.0

    def add_trade(self, trade):
        contract_result = self.contract_results.get(trade.vt_symbol)
        if contract_result:
            contract_result.trades.append(trade)

    def calculate_pnl(self, pre_closes, start_poses, sizes, rates, slippages):
        self.pre_closes = pre_closes
        
        for vt_symbol, cr in self.contract_results.items():
            cr.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol], rates[vt_symbol], slippages[vt_symbol]
            )
            self.trade_count += cr.trade_count
            self.turnover += cr.turnover
            self.commission += cr.commission
            self.slippage += cr.slippage
            self.trading_pnl += cr.trading_pnl
            self.holding_pnl += cr.holding_pnl
            self.total_pnl += cr.total_pnl
            self.net_pnl += cr.net_pnl
            self.end_poses[vt_symbol] = cr.end_pos

    def update_close_prices(self, close_prices):
        self.close_prices = close_prices
        for vt_symbol, close_price in close_prices.items():
            cr = self.contract_results.get(vt_symbol)
            if cr:
                cr.close_price = close_price
        self.last_close_prices = dict(close_prices)
# ----------------------------------------------------------------------------------------------------
def optimize_with_shared_memory(
    target_name: str,
    strategy_class,
    setting: dict,
    vt_symbols: list,
    start: datetime,
    end: datetime,
    rates: dict,
    slippages: dict,
    sizes: dict,
    price_ticks: dict,
    capital: float,
    interval,
    days: int = 0,
    trading_stock: bool = False,
):
    """使用零拷贝共享内存的多进程优化函数 - 优化版"""
    dts, history_accessor = get_shared_data()
    
    engine = BacktestingEngine()
    engine.set_parameters(vt_symbols=vt_symbols, start=start, end=end, rates=rates,
                            slippages=slippages, sizes=sizes, price_ticks=price_ticks,
                            capital=capital, interval=interval)
    engine.add_strategy(strategy_class, setting)
    engine.dts = dts
    engine.history_data = history_accessor
    # 关键修复：优化子进程必须继承父引擎的预热天数，否则指标初始化窗口与单次回测不一致
    engine.days = days

    engine.run_backtesting(trading_stock=trading_stock)
    daily_df = engine.calculate_result()

    statistics = engine.calculate_statistics(daily_df, output_statistics=False)
    target_value = statistics[target_name]
    
    # 优化：显式清理大对象
    engine.dts = None
    engine.history_data = None
    engine.clear_data()
    
    return (str(setting), target_value, statistics)
# ----------------------------------------------------------------------------------------------------
def load_bar_data(vt_symbol: str, interval: Interval, start: datetime, end: datetime) -> List[BarData]:
    """
    读取bar数据
    """
    symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
    bar_data = database_manager.load_bar_data(symbol, exchange, interval, start, end)
    return bar_data
# ----------------------------------------------------------------------------------------------------
def load_tick_data(vt_symbol: str, start: datetime, end: datetime) -> List[TickData]:
    """
    读取tick数据
    """
    symbol, exchange, gateway_name = _cached_extract_vt_symbol(vt_symbol)
    tick_data = database_manager.load_tick_data(symbol, exchange, start, end)
    return tick_data
