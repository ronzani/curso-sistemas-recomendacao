import numpy as np
from rich.console import Console
from rich.table import Table


class FancyMatrix:
    """Classe que representa uma matriz com índices de linhas e colunas nomeados."""

    def __init__(self, num_lines: int, num_columns: int):
        self.line_index = {}
        self.column_index = {}
        self.data = np.zeros((num_lines, num_columns), dtype=np.float16)

    def __setitem__(self, key: tuple, value: np.float16):
        i, j = key
        if isinstance(key, tuple):
            line, column = key
            if line not in self.line_index:
                self.line_index[line] = len(self.line_index.keys())
            if column not in self.column_index:
                self.column_index[column] = len(self.column_index.keys())

            i = self.line_index[line]
            j = self.column_index[column]
            self.data[i, j] = value
        else:
            raise KeyError("Key must be a tuple")

    def print(self, title="Table"):
        table = Table(title=title)
        table.add_column("", justify="center", style="cyan", no_wrap=True)
        for col in self.column_index.keys():
            table.add_column(col, justify="center")
        for line in self.line_index.keys():
            line_index = self.line_index[line]
            #line name + the float values
            row = [line]+["-" if i==0 else format(i, ".2f")   for i in self.data[line_index]]
            #row_str = list(map(lambda x: '-' if x==0 else str(x), row))
            table.add_row(*row)
        console = Console()
        console.print(table)

    def __getitem__(self, key):
        """ Retorna um elemento ou uma linha representando.

        Args:
          key: int ou tuple de inteiros (i, j)
        """
        if isinstance(key, tuple):
            line, column = key
            if line not in self.line_index:
                raise KeyError(f"The line key {line} has not been found in the index.")
            if column not in self.column_index:
                raise KeyError(f"The column key {column} has not been found in the index.")
            i = self.line_index[line]
            j = self.column_index[column]
            return self.data[i,j]
        line = key
        if line not in self.line_index:
            raise KeyError(f"The line key {line} has not been found in the index.")
        i = self.line_index[line]
        return self.data[i]

    def toStr(self):
        return ','.join(self.line_index.keys())+'\n'+str(self.data)

    def __str__(self) -> str:
        return self.toStr()

    def __repr__(self) -> str:
        table = Table(title="")
        table.add_column("", justify="center", style="cyan", no_wrap=True)
        for col in self.column_index.keys():
            table.add_column(col, justify="center")
        for line in self.line_index.keys():
            line_index = self.line_index[line]
            # line name + the float values
            row = [line]+["-" if i == 0 else format(i, ".2f") for i in self.data[line_index]]
            # row_str = list(map(lambda x: '-' if x==0 else str(x), row))
            table.add_row(*row)
        console = Console()
        with console.capture() as capture:
            console.print(table)

        return capture.get()
