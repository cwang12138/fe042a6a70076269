"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 22:39
# @Author  : Chen Wang
# @Site    : 
# @File    : logger.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import logging
import os


def get_logger(root, name=None, is_debug_in_screen=True, log_path='run.log'):
    """

    :param root: the log file's root path, str
    :param name: the logger's name, str
    :param is_debug_in_screen: whether to show DEBUG in screen, bool
    :return: logging object
    """
    # when is_debug_in_screen is True, show DEBUG and INFO in screen
    # when is_debug_in_screen is False, show DEBUG in file and info in both screen & file
    # INFO will always be in screen
    # DEBUG will always be in file

    # create a logger
    logger = logging.getLogger(name)

    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    class MyFormatter(logging.Formatter):
        def __init__(self, fmt='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M'):
            super().__init__(fmt=fmt, datefmt=datefmt)

        def format(self, record):
            spliter = ' '
            record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
            record.args = tuple()       # Clear args to avoid duplicate formatting
            return super().format(record)

    # define the formate
    formatter = MyFormatter()

    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if is_debug_in_screen:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create a handler for write log to file
    logfile = os.path.join(root, log_path)
    if not os.path.exists(root):
        os.makedirs(root)
    print('Creat Log File in: ', logfile)
    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # add Handler to logger
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
