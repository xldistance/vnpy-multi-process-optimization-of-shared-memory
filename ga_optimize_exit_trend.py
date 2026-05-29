from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Union, Dict
from decimal import Decimal
import random
import numpy as np
import gc
import warnings
import pickle
import json5 as json
warnings.simplefilter('ignore', category=RuntimeWarning)

from deap import algorithms, base, creator, tools
from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
from pyecharts.charts import Bar, Page
from colorama import init, Fore
init(autoreset=True)

from vnpy.app.portfolio_strategy.backtesting import (
    BacktestingEngine,
    Interval,
    SharedMemoryManager,
    init_worker,
    get_shared_data,
    js_color,
    label_opts,
    itemstyle_opts,
    emphasis_opts,
    enhanced_title_opts,
    toolbox_opts,
    datazoom_opts,
    enhanced_tooltip_opts,
    enhanced_legend_opts,
)
from vnpy.app.portfolio_strategy.strategies.binances_exit_trend_strategy import (
    BinancesExitTrendStrategy
)
from vnpy.trader.utility import GetFilePath, get_price_ticks,save_json

STRATEGY = BinancesExitTrendStrategy

########################################################################
# 参数信息
param_info = [
    {"name": "buffer_size", "start": 20, "end": 80, "step": 1},

]
# 优化参数与优化目标变量名称
params_list = [p["name"] for p in param_info]
targets = ["rgr_value", "sortino_value"]

########################################################################
# 全局变量 - 子进程中使用
_GA_CONFIG = None

def _init_ga_worker(shm_name: str, config_bytes: bytes):
    """扩展的 worker 初始化函数，同时传递共享内存名和配置"""
    global _GA_CONFIG
    init_worker(shm_name)  # 调用原有的初始化函数
    _GA_CONFIG = pickle.loads(config_bytes)  # 反序列化配置

########################################################################
def build_discrete_values(info: dict) -> List[Union[int, float]]:
    """
    根据单个参数的 (start, end, step) 生成一个离散值列表。
    例如:
        start=0.1, end=0.4, step=0.1  --> [0.1, 0.2, 0.3]
        start=1,   end=5,   step=1    --> [1, 2, 3, 4]
    """
    # 交易周期使用时间切片参数
    if info["name"] == "trade_window":
        # 分钟周期
        #return [10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48, 60, 72, 80, 90, 96, 120, 144, 160, 180, 240]
        #return [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48, 60, 72, 80, 90, 96, 120, 144, 160, 180, 240, 288, 360, 480, 720, 1440]
        # 小时周期
        return [1, 2, 3, 4, 6, 8, 12, 24]
    start = Decimal(str(info["start"]))
    end = Decimal(str(info["end"]))
    step = Decimal(str(info["step"]))
    values = []
    val = start
    while val <= end + step / Decimal("0.9999999"):
        if (float(val) % 1 == 0) and (float(step) % 1 == 0):
            values.append(int(val))
        else:
            values.append(float(val))
        val += step
    return values

all_discrete_values = [build_discrete_values(item) for item in param_info]

########################################################################
def parameter_generate() -> List[Union[int, float]]:
    """
    从每个参数可选的离散值集合中，随机选出一个值，构成最终的 22 维参数向量。
    """
    return [random.choice(vals) for vals in all_discrete_values]

########################################################################
def object_func_shared(strategy_avg: List[Union[int, float]]):
    """使用共享内存的优化目标函数"""
    global _GA_CONFIG
    
    # 从共享内存获取数据
    dts, history_accessor = get_shared_data()
    config = _GA_CONFIG
    
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbols=config['vt_symbols'],
        start=config['start'],
        end=config['end'],
        rates=config['rates'],
        slippages=config['slippages'],
        sizes=config['sizes'],
        price_ticks=config['price_ticks'],
        capital=config['capital'],
        interval=config['interval'],
    )
    
    setting = dict(zip(params_list, strategy_avg))
    engine.add_strategy(config['strategy_class'], setting)
    
    # 直接使用共享内存数据，无需重新加载
    engine.dts = dts
    engine.history_data = history_accessor
    
    engine.run_backtesting()
    daily_df = engine.calculate_result()
    statistics = engine.calculate_statistics(daily_df, output_statistics=False)
    
    target_1 = round(statistics[targets[0]], 3)
    target_2 = round(statistics[targets[1]], 3)
    
    engine.clear_data()
    gc.collect()
    # 如果任一目标值等于0说明回测异常，所有目标都返回0
    if target_1 == 0 or target_2 == 0:
        return 0,0
    return target_1, target_2

########################################################################
def mutate_param(individual, mutpb=0.3):
    """
    对个体进行变异，让变异后的取值可达到原 all_discrete_values[i]["end"] 最大值的 2 倍。
    即从 [param_info[i]["start"], param_info[i]["end"]] 扩展到 2 * param_info[i]["end"]。
    """
    for i in range(len(individual)):
        # 以 mutpb 的概率决定是否对第 i 个基因(参数)进行变异
        if random.random() < mutpb:
            # 取出该参数的原始离散值集合、最大值、以及对应 (start, end, step)
            discrete_vals = all_discrete_values[i]
            info = param_info[i]
            # 特殊处理 trade_window,只从预定义列表中选择
            if info["name"] == "trade_window":
                individual[i] = random.choice(all_discrete_values[i])
                continue
            # 计算扩展后新的上界(2 倍)
            extended_max = 2 * info["end"]
            # 如果 param_info[i]["end"] 本身就是 float，需要保持步长也为 float
            step_decimal = Decimal(str(info["step"]))
            start_decimal = Decimal(str(info["end"])) + step_decimal
            end_decimal = Decimal(str(extended_max))
            # 先将原有离散值拷贝到扩展集合中
            extended_vals = set(discrete_vals)
            # 在 [end+step, 2*end] 区间内继续按同样步长生成值
            val = start_decimal
            while val <= end_decimal + step_decimal / Decimal("1000000"):
                # 根据 step 和当前 val 判断用 int 还是 float
                if (float(val) % 1 == 0) and (float(step_decimal) % 1 == 0):
                    extended_vals.add(int(val))
                else:
                    extended_vals.add(float(val))
                val += step_decimal
            # 排序之后再随机挑选
            individual[i] = random.choice(sorted(list(extended_vals)))
    return (individual,)

########################################################################
# 定义多目标：最大化1，最小化-1
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

########################################################################
# 自定义的进化过程（包含精英策略及收敛检测）函数
def eaMuPlusLambdaWithConvergence(
    population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
    stats=None, halloffame=None, verbose=True, converge_rounds=3
):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + (stats.fields if stats else [])
    # 初始种群评估
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    # 收敛检测辅助计数
    converge_count = 0
    # 迭代进化
    for gen in range(1, ngen + 1):
        # 产生子代(包含交叉与变异)
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        # 评估子代
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 与上一代合并，然后选出下一代
        population = toolbox.select(population + offspring, mu)

        if halloffame is not None:
            halloffame.update(population)
        # 记录统计量
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        # 收敛检测
        fitness_arr = np.asarray([ind.fitness.values for ind in population])
        # 计算范围
        range_vals = fitness_arr.max(axis=0) - fitness_arr.min(axis=0)
        # 自定义绝对和/或相对阈值
        abs_tol, rel_tol = 1e-4, 1e-3
        range_rel = range_vals / (np.abs(fitness_arr.mean(axis=0)) + 1e-12)
        # 判断是否收敛
        if np.all((range_vals < abs_tol) | (range_rel < rel_tol)):
            converge_count += 1
        else:
            converge_count = 0

        if converge_count >= converge_rounds:
            if verbose:
                print(f"===> 连续 {converge_rounds} 代满足收敛阈值，Gen {gen} 提前终止进化")
            break

    return population, logbook

########################################################################
def optimize():
    """执行遗传算法优化（使用共享内存）"""
    # 回测参数配置
    vt_symbols = [
        "BTCUSDT_BINANCES/BINANCES",
        "ETHUSDT_BINANCES/BINANCES",
    ]
    price_ticks = {vt: get_price_ticks()[vt] * 3 for vt in vt_symbols}
    # 手续费率
    rates = {vt: 4 / 10000 for vt in vt_symbols}
    # 滑点
    slippages = {vt: price_ticks[vt] * 2 for vt in vt_symbols}
    # 合约杠杆
    sizes = {vt: 3 for vt in vt_symbols}
    
    start = datetime(2024, 1, 1)
    end = datetime(2027, 1, 1)
    capital = float(len(vt_symbols) * 5e5)
    interval = Interval.MINUTE

    # 配置字典（将传递给子进程）
    config = {
        'vt_symbols': vt_symbols,
        'start': start,
        'end': end,
        'rates': rates,
        'slippages': slippages,
        'sizes': sizes,
        'price_ticks': price_ticks,
        'capital': capital,
        'interval': interval,
        'strategy_class': STRATEGY,
    }
    config_bytes = pickle.dumps(config)

    # 创建引擎并加载数据
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbols=vt_symbols, start=start, end=end,
        rates=rates, slippages=slippages, sizes=sizes,
        price_ticks=price_ticks, capital=capital, interval=interval
    )
    engine.load_data()

    # 创建共享内存
    shm_manager = SharedMemoryManager()
    shm_name = shm_manager.create_shared_memory(engine.dts, engine.history_data)
    print(f"零拷贝共享内存创建成功: {shm_name}")

    pool = None
    try:
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, parameter_generate)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.7)
        toolbox.register("mutate", mutate_param, mutpb=0.3)
        toolbox.register("evaluate", object_func_shared)
        toolbox.register("select", tools.selNSGA2)

        # 使用共享内存初始化进程池
        pool = ProcessPoolExecutor(
            max_workers=32,
            initializer=_init_ga_worker,
            initargs=(shm_name, config_bytes)
        )
        toolbox.register("map", pool.map)

        # 遗传算法参数
        # 根据参数数量动态调整
        n_params = len(param_info)
        pop_size = max(100, n_params * 5)   # 初始种群大小
        mu = int(pop_size * 0.8)            # 每一代选出的个体数
        lamb = int(pop_size * 1.2)      # 每一代产生的子代数

        cxpb, mutpb, n_gen = 0.7, 0.3, 40   # 交叉概率，变异概率，进化轮数
        # 初始化种群
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        # 统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        # 使用自定义带收敛检测的算法
        pop, logbook = eaMuPlusLambdaWithConvergence(
            population=pop, toolbox=toolbox, mu=mu, lambda_=lamb,
            cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
            stats=stats, halloffame=hof, verbose=True, converge_rounds=3
        )

        return pop, hof

    finally:
        if pool:
            pool.shutdown(wait=True)
        shm_manager.cleanup()
        print("共享内存已清理")

########################################################################
if __name__ == "__main__":
    strategy_name = STRATEGY.__name__.replace("Strategy", "")
    pop, hof = optimize()

    # 过滤重复的最优个体
    # 从最后一代种群中选取前 50 名最优的个体（基于多目标适应度）
    best_individuals = tools.selBest(pop, 50)
    unique_individuals = [ind for ind in best_individuals]


    optimize_params = []
    index_results = []
    target_1_results = []
    target_2_results = []

    for index, individual in enumerate(unique_individuals):
        params_dict = dict(zip(params_list, individual))
        fitness_values = dict(zip(targets, individual.fitness.values))
        print(
            Fore.CYAN + f"最优参数{index}: {params_dict}, "
            + f"目标值({targets[0]}, {targets[1]}) = {fitness_values}"
        )
        optimize_params.append(str(params_dict))
        index_results.append(f"{index}->{params_dict}")
        target_1_results.append(individual.fitness.values[0])
        target_2_results.append(individual.fitness.values[1])
    # 保存优化参数到json
    save_json(f"ga_optimize_params_{strategy_name}.json",optimize_params)
    # 绘图保存
    get_file_path = GetFilePath()
    opt_path = str(
        get_file_path.opt_path(f"ga_{strategy_name}", "portfolio_strategy")
    ).replace("cta_strategy", "portfolio_strategy")

    bar_1 = Bar()
    bar_1.add_xaxis(index_results)
    bar_1.add_yaxis(
        f"{strategy_name}\n\n{targets[0]}优化分布图",
        target_1_results, color=js_color,
        label_opts=label_opts,
        itemstyle_opts=itemstyle_opts,
        emphasis_opts=emphasis_opts,
    )
    bar_1.set_global_opts(
        opts.TitleOpts(title=targets[0], title_textstyle_opts=enhanced_title_opts),
        toolbox_opts=toolbox_opts,
        datazoom_opts=datazoom_opts,
        tooltip_opts=enhanced_tooltip_opts,
        legend_opts=enhanced_legend_opts,
    )

    bar_2 = Bar()
    bar_2.add_xaxis(index_results)
    bar_2.add_yaxis(
        f"{targets[1]}优化分布图",
        target_2_results, color=js_color,
        label_opts=label_opts,
        itemstyle_opts=itemstyle_opts,
        emphasis_opts=emphasis_opts,
    )
    bar_2.set_global_opts(
        opts.TitleOpts(title=targets[1], title_textstyle_opts=enhanced_title_opts),
        toolbox_opts=toolbox_opts,
        datazoom_opts=datazoom_opts,
        tooltip_opts=enhanced_tooltip_opts,
        legend_opts=enhanced_legend_opts,
    )

    page = Page(layout=Page.SimplePageLayout)
    for bar in [bar_1, bar_2]:
        bar.width = "100%"
        page.add(bar)
    page.render(opt_path)
    input("按任意键退出")