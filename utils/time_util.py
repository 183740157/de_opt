#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：de_opt
@File    ：time_util.py
@IDE     ：PyCharm
@Author  ：wubc
@Date    ：2024/10/17 09:54
描述      ：
"""

from datetime import datetime


def get_now_time(format):
    # 获取当前时间
    now = datetime.now()

    # 格式化时间
    formatted_time = now.strftime(format)

    return formatted_time


def get_now_second():
    # 获取当前时间
    second = datetime.now().timestamp()
    return round(second)


if __name__ == "__main__":
    starttime = get_now_second()
    print(starttime)
    print(get_now_second() - starttime)
