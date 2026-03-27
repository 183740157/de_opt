from constants.algo_const import TEST_FUNCS
from constants.algo_const import TRAIN_FUNCS


def get_global_optimum(func_instance):
    """
    获取函数的全局最优解和最优值
    Args:
        func_instance: 函数类的实例

    Returns:
        tuple: (x_global, f_global)
            x_global: 全局最优解
            f_global: 全局最优值
    """

    x_global = func_instance.x_global  # 全局最优解
    f_global = func_instance.evaluate(x_global)
    # f_global = func_instance.f_global  # 全局最优值

    return x_global, f_global

if __name__ == '__main__':
    # 遍历所有函数
    for func in TRAIN_FUNCS:
        # 创建函数实例，维度为10
        func_instance = func(ndim=30)

        # 获取全局最优解和最优值
        x_global, f_global = get_global_optimum(func_instance)

        # 打印结果
        # print(f"函数 {func.__name__} 的全局最优解: {x_global}")
        # print(f"函数 {func.__name__} 的全局最优值: {f_global}")
        print(f"{f_global}")
