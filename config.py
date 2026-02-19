#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : config.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 定义场景.
@Version : 1.0
"""

DEFAULT_SCENES = {
    "hr": {
        "name": "人力资源政策",
        "keywords": [
            "请假", "年假", "入职", "离职", "合同", "社保",
            # 统计/枚举类触发词
            "HR文档", "人事文件", "人事制度", "人力资源文档",
        ],
        "path": "./data/hr"
    },
    "it": {
        "name": "IT支持",
        "keywords": [
            "电脑", "网络", "账号", "打印机", "软件", "VPN",
            # 统计/枚举类触发词
            "IT文档", "IT手册", "IT文件", "技术文档",
        ],
        "path": "./data/it"
    },
    "finance": {
        "name": "财务报销",
        "keywords": [
            "报销", "发票", "差旅", "付款", "预算", "费用",
            # 统计/枚举类触发词
            "财务文件", "财务制度", "报销文件", "报销制度", "财务文档",
        ],
        "path": "./data/finance"
    }
}

# 兼容旧代码引用，后续动态配置请通过 config_store.get_scenes() 获取。
SCENES = DEFAULT_SCENES
