import os
import sys
import pydata_sphinx_theme

# Добавь путь к папке с кодом
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',  # авто документации из docstrings
    'sphinx.ext.viewcode',  # ссылки на исходный код
    'sphinx.ext.napoleon',  # поддержка Google и NumPy стиля документации
    'sphinx.ext.todo',  # поддержка TODO
    'sphinx.ext.coverage',  # проверяет покрытие документации
    'sphinx.ext.ifconfig',  # условные директивы в документации
]

todo_include_todos = True  # показывать TODO в готовой документации

# Настройки темы
html_theme = "pydata_sphinx_theme"

