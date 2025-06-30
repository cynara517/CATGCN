# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from typing import List, Dict

from sympy.parsing.latex import parse_latex
from sympy import Symbol, Basic, Add, Mul, Pow

@dataclass
class VariableNode:
    name: str
    influences: List[str]
    operation: str  # add, multiply, divide, pow, negative, leaf

    def __str__(self, level=0):
        ret = "\t" * level + f"Variable: {self.name}\n"
        ret += "\t" * level + f"Operation: {self.operation}\n"
        if self.influences:
            ret += "\t" * level + "Influences:\n"
            for var in self.influences:
                ret += "\t" * (level + 1) + f"-> {var}\n"
        return ret

class LatexGraphParser:
    def __init__(self):
        self.med_counter = 1
        self.nodes: List[VariableNode] = []
        self.cache: Dict[Basic, str] = {}

    def new_med_name(self) -> str:
        name = f"med_{self.med_counter}"
        self.med_counter += 1
        return name

    def parse_equations(self, latex_str: str) -> List[VariableNode]:
        """
        Parse multiline LaTeX input and extract equations.
        """
        lines = latex_str.splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r'\\begin{aligned}|\\end{aligned}', '', line)
            line = line.replace('&', '')
            line = line.replace(r'\left', '').replace(r'\right', '')
            if line.endswith(r'\\'):
                line = line[:-2].strip()
            if not line:
                continue
            self.process_one_equation(line)
        return self.nodes

    def process_one_equation(self, eq: str):
        # Match equations of the form: \frac{dX}{dt} = RHS
        m = re.match(r'.*?\\frac\s*{\s*d\s*([A-Za-z_][A-Za-z0-9_/]*)\s*}{\s*d\s*t\s*}\s*=\s*(.*)$', eq)
        if not m:
            return
        var_name = m.group(1).strip()
        rhs_latex = m.group(2).strip()
        target_dot = f"{var_name}_dot"

        # Preprocess LaTeX RHS
        rhs_latex = re.sub(r'\\frac\s*{\s*d\s*V\s*}{\s*d\s*t\s*}', "Vdot", rhs_latex)
        rhs_latex = rhs_latex.replace(r'\times', '*').replace('×', '*')
        rhs_latex = rhs_latex.replace(r'\left', '').replace(r'\right', '')

        print(f"--- DEBUG RHS for {target_dot}: ---")
        print(rhs_latex)
        print("-----------------------------\n")

        try:
            expr = parse_latex(rhs_latex)
        except Exception as e:
            print(f"Warning: failed to parse LaTeX line:\n  {rhs_latex}\n  Error: {e}")
            return

        root_med = self._traverse(expr)
        final_node = VariableNode(
            name=target_dot,
            influences=[root_med] if root_med else [],
            operation="sum"
        )
        self.nodes.append(final_node)

    def _traverse(self, expr: Basic) -> str:
        if expr in self.cache:
            return self.cache[expr]

        if isinstance(expr, Symbol) or expr.is_Number:
            name = str(expr)
            self.cache[expr] = name
            return name

        if isinstance(expr, Add):
            child_names = [self._traverse(arg) for arg in expr.args]
            med_name = self.new_med_name()
            self.nodes.append(VariableNode(med_name, child_names, "add"))
            self.cache[expr] = med_name
            return med_name

        if isinstance(expr, Mul):
            power_args = [arg for arg in expr.args if isinstance(arg, Pow) and arg.args[1] == -1]
            if power_args:
                denom_pow = power_args[0]
                numerators = [arg for arg in expr.args if arg is not denom_pow]
                num_names = [self._traverse(num) for num in numerators]
                numerator_rep = num_names[0] if len(num_names) == 1 else self._add_med_node(num_names, "multiply")
                denom_rep = self._traverse(denom_pow.args[0])
                med_name = self.new_med_name()
                self.nodes.append(VariableNode(med_name, [numerator_rep, denom_rep], "divide"))
                self.cache[expr] = med_name
                return med_name

            child_names = [self._traverse(arg) for arg in expr.args]
            if len(child_names) > 1:
                med_name = self.new_med_name()
                self.nodes.append(VariableNode(med_name, child_names, "multiply"))
                self.cache[expr] = med_name
                return med_name
            else:
                return child_names[0]

        if isinstance(expr, Pow):
            base, exponent = expr.args
            base_rep = self._traverse(base)
            exp_rep = self._traverse(exponent)
            med_name = self.new_med_name()
            self.nodes.append(VariableNode(med_name, [base_rep, exp_rep], "pow"))
            self.cache[expr] = med_name
            return med_name

        if expr.is_Negative:
            child = -expr
            child_rep = self._traverse(child)
            med_name = self.new_med_name()
            self.nodes.append(VariableNode(med_name, [child_rep], "negative"))
            self.cache[expr] = med_name
            return med_name

        s = str(expr)
        self.cache[expr] = s
        return s

    def _add_med_node(self, children: List[str], op: str) -> str:
        name = self.new_med_name()
        self.nodes.append(VariableNode(name, children, op))
        return name

if __name__ == "__main__":
    with open('latex.txt', 'r', encoding='utf-8') as f:
        latex_content = f.read()

    parser = LatexGraphParser()
    all_nodes = parser.parse_equations(latex_content)

    with open('variable_relationships.txt', 'w', encoding='utf-8') as outf:
        outf.write("Variable Relationships with Intermediate Calculations:\n\n")
        for node in all_nodes:
            outf.write(node.__str__())
            outf.write("\n")

    print("Variable Relationships with Intermediate Calculations:")
    for node in all_nodes:
        print(node)
