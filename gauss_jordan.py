"""
Trabalho Computacional 1 - Resolucao de Sistemas Lineares pelo Metodo de Gauss-Jordan
=======================================================================================

Este script resolve sistemas lineares Ax = b usando o Metodo de Gauss-Jordan,
exibindo a matriz aumentada inicial, cada operacao elementar sobre as linhas,
a forma escalonada reduzida final, a classificacao do sistema e a solucao.

Como usar:
    python3 gauss_jordan.py

O programa pede o numero de equacoes (m) e de incognitas (n), depois solicita
a matriz de coeficientes (A) e o vetor de resultados (b) separadamente -
mantendo a diferenca clara entre "matriz de coeficientes" e "matriz resultante"
exigida no enunciado.

Tambem ha uma versao didatica com exemplos prontos (--demo) e a possibilidade
de usar este modulo como biblioteca (importando a funcao gauss_jordan).
"""

from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple, Optional
import sys

Matrix = List[List[float]]

EPS = 1e-9


# ---------------------------------------------------------------------------
# NUCLEO DO ALGORITMO
# ---------------------------------------------------------------------------

class Step:
    """Representa uma etapa do processo de eliminacao, para o historico."""

    def __init__(self, description: str, matrix: Matrix, changed_rows: Optional[List[int]] = None):
        self.description = description
        self.matrix = [row[:] for row in matrix]
        self.changed_rows = changed_rows or []


def quase_zero(x: float, eps: float = EPS) -> bool:
    return abs(x) < eps


def montar_matriz_aumentada(coef: Matrix, rhs: Matrix) -> Matrix:
    """Junta a matriz de coeficientes (A) com a matriz/coluna de resultados (b)
    formando a matriz aumentada [A|b]. Mantidas como estruturas separadas na
    entrada para deixar explicita a diferenca entre coeficientes e resultado."""
    return [coef[i] + rhs[i] for i in range(len(coef))]


def gauss_jordan(coef: Matrix, rhs: Matrix) -> Tuple[List[Step], Matrix, int, List[int], str]:
    """
    Executa a eliminacao de Gauss-Jordan com pivotamento parcial.

    Parametros:
        coef: matriz de coeficientes (m x n)
        rhs:  matriz/coluna de termos independentes (m x 1)

    Retorna:
        steps:          lista de Step com o historico completo
        matriz_final:   forma escalonada reduzida (m x (n+1))
        rank:           posto da matriz de coeficientes
        pivot_cols:     indices das colunas que receberam pivo
        classificacao:  'DETERMINADO' | 'INDETERMINADO' | 'IMPOSSIVEL'
    """
    m = len(coef)
    n = len(coef[0])

    A = montar_matriz_aumentada(coef, rhs)
    steps: List[Step] = [Step("Matriz aumentada inicial [A | b]", A)]

    pivot_row = 0
    pivot_cols: List[int] = []

    for col in range(n):
        if pivot_row >= m:
            break

        # pivotamento parcial: maior valor absoluto na coluna, a partir de pivot_row
        max_row = pivot_row
        max_val = abs(A[pivot_row][col])
        for r in range(pivot_row + 1, m):
            if abs(A[r][col]) > max_val:
                max_val = abs(A[r][col])
                max_row = r

        if quase_zero(max_val):
            continue  # coluna toda nula abaixo: variavel livre, sem pivo aqui

        if max_row != pivot_row:
            A[pivot_row], A[max_row] = A[max_row], A[pivot_row]
            steps.append(Step(
                f"Troca de linhas: L{pivot_row + 1} <-> L{max_row + 1} "
                f"(pivotamento parcial - maior modulo na coluna {col + 1})",
                A, [pivot_row, max_row]
            ))

        pivot_val = A[pivot_row][col]
        if not quase_zero(pivot_val - 1):
            A[pivot_row] = [v / pivot_val for v in A[pivot_row]]
            steps.append(Step(
                f"L{pivot_row + 1} -> L{pivot_row + 1} / ({formatar(pivot_val)})   "
                f"[normaliza o pivo da coluna {col + 1} para 1]",
                A, [pivot_row]
            ))

        for r in range(m):
            if r == pivot_row:
                continue
            factor = A[r][col]
            if not quase_zero(factor):
                A[r] = [A[r][c] - factor * A[pivot_row][c] for c in range(n + 1)]
                steps.append(Step(
                    f"L{r + 1} -> L{r + 1} - ({formatar(factor)}) * L{pivot_row + 1}   "
                    f"[elimina coluna {col + 1} na linha {r + 1}]",
                    A, [r]
                ))

        pivot_cols.append(col)
        pivot_row += 1

    rank = pivot_row

    inconsistente = any(not quase_zero(A[r][n]) for r in range(rank, m))

    if inconsistente:
        classificacao = "IMPOSSIVEL"
    elif rank == n:
        classificacao = "DETERMINADO"
    else:
        classificacao = "INDETERMINADO"

    steps.append(Step("Forma escalonada reduzida obtida", A))

    return steps, A, rank, pivot_cols, classificacao


# ---------------------------------------------------------------------------
# FORMATACAO E EXIBICAO
# ---------------------------------------------------------------------------

def formatar(x: float) -> str:
    if quase_zero(x):
        return "0"
    if quase_zero(x - round(x)):
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def imprimir_matriz(matriz: Matrix, n_coef: int, linhas_destacadas: Optional[List[int]] = None) -> None:
    """Imprime a matriz aumentada com uma barra vertical separando
    coeficientes do vetor de resultados, e marca com '*' as linhas alteradas."""
    linhas_destacadas = linhas_destacadas or []
    larguras = []
    for j in range(len(matriz[0])):
        larguras.append(max(len(formatar(linha[j])) for linha in matriz))

    for i, linha in enumerate(matriz):
        marcador = "> " if i in linhas_destacadas else "  "
        partes = []
        for j, valor in enumerate(linha):
            texto = formatar(valor).rjust(larguras[j])
            partes.append(texto)
            if j == n_coef - 1:
                partes.append("|")
        print(f"  {marcador}[ {'  '.join(partes)} ]")
    print()


def imprimir_passo(idx: int, total: int, step: Step, n_coef: int) -> None:
    print(f"--- Passo {idx}/{total} ---")
    print(f"  {step.description}")
    imprimir_matriz(step.matrix, n_coef, step.changed_rows)


def imprimir_classificacao(classificacao: str) -> None:
    textos = {
        "DETERMINADO": (
            "SISTEMA POSSIVEL E DETERMINADO (SPD)",
            "O posto da matriz de coeficientes e igual ao numero de incognitas.\n"
            "  Existe exatamente uma solucao."
        ),
        "INDETERMINADO": (
            "SISTEMA POSSIVEL E INDETERMINADO (SPI)",
            "O posto e menor que o numero de incognitas.\n"
            "  Existem infinitas solucoes, parametrizadas pelas variaveis livres."
        ),
        "IMPOSSIVEL": (
            "SISTEMA IMPOSSIVEL (SI)",
            "Uma linha da forma escalonada equivale a 0 = k, com k != 0.\n"
            "  O sistema nao admite solucao."
        ),
    }
    titulo, descricao = textos[classificacao]
    print("=" * 70)
    print(f"  CLASSIFICACAO: {titulo}")
    print(f"  {descricao}")
    print("=" * 70)
    print()


def imprimir_solucao(matriz_final: Matrix, n: int, pivot_cols: List[int], classificacao: str) -> None:
    print("SOLUCAO:")
    if classificacao == "IMPOSSIVEL":
        print("  Nao ha solucao para este sistema.\n")
        return

    pivot_set = set(pivot_cols)
    for j in range(n):
        if j in pivot_set:
            row = pivot_cols.index(j)
            expr = formatar(matriz_final[row][n])
            for k in range(n):
                if k not in pivot_set and not quase_zero(matriz_final[row][k]):
                    coef_k = -matriz_final[row][k]
                    sinal = "+" if coef_k >= 0 else "-"
                    expr += f" {sinal} {formatar(abs(coef_k))}*x{k + 1}"
            print(f"  x{j + 1} = {expr}")
        else:
            print(f"  x{j + 1} = t{j + 1}  (variavel livre)")
    print()


# ---------------------------------------------------------------------------
# ENTRADA DE DADOS (mantendo coeficientes e resultado como estruturas distintas)
# ---------------------------------------------------------------------------

def ler_inteiro(mensagem: str, minimo: int = 1, maximo: int = 10) -> int:
    while True:
        try:
            valor = int(input(mensagem).strip())
            if minimo <= valor <= maximo:
                return valor
            print(f"  Digite um valor entre {minimo} e {maximo}.")
        except ValueError:
            print("  Digite um numero inteiro valido.")


def ler_float(mensagem: str) -> float:
    while True:
        try:
            texto = input(mensagem).strip().replace(",", ".")
            return float(texto)
        except ValueError:
            print("  Digite um numero valido (use ponto ou virgula decimal).")


def ler_matriz_coeficientes(m: int, n: int) -> Matrix:
    print(f"\nDigite a MATRIZ DE COEFICIENTES (A), {m} linha(s) x {n} coluna(s).")
    print("Para cada linha, informe os coeficientes separados por espaco.")
    coef: Matrix = []
    for i in range(m):
        while True:
            entrada = input(f"  Linha {i + 1} (ex.: 2 1 -1): ").strip()
            valores = entrada.replace(",", ".").split()
            if len(valores) != n:
                print(f"    Informe exatamente {n} valor(es).")
                continue
            try:
                coef.append([float(v) for v in valores])
                break
            except ValueError:
                print("    Valores invalidos, tente novamente.")
    return coef


def ler_vetor_resultado(m: int) -> Matrix:
    print(f"\nDigite a MATRIZ/VETOR DE RESULTADOS (b), {m} valor(es) - um por equacao.")
    rhs: Matrix = []
    for i in range(m):
        valor = ler_float(f"  b{i + 1} (resultado da equacao {i + 1}): ")
        rhs.append([valor])
    return rhs


# ---------------------------------------------------------------------------
# EXEMPLOS PRONTOS (modo demonstracao)
# ---------------------------------------------------------------------------

EXEMPLOS = {
    "1": ("Possivel e Determinado (3x3)",
          [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [[8], [-11], [-3]]),
    "2": ("Possivel e Indeterminado (2x3)",
          [[1, 2, 3], [2, 4, 6]], [[6], [12]]),
    "3": ("Impossivel (2x2)",
          [[1, 1], [1, 1]], [[2], [5]]),
}


def executar(coef: Matrix, rhs: Matrix) -> None:
    n = len(coef[0])

    print("\n" + "=" * 70)
    print("  MATRIZ DE COEFICIENTES (A)            MATRIZ DE RESULTADOS (b)")
    for i in range(len(coef)):
        linha_a = "  ".join(formatar(v).rjust(6) for v in coef[i])
        print(f"  [ {linha_a} ]                         [ {formatar(rhs[i][0]).rjust(6)} ]")
    print("=" * 70)

    steps, matriz_final, rank, pivot_cols, classificacao = gauss_jordan(coef, rhs)

    print(f"\nHISTORICO COMPLETO DO PROCESSO DE ELIMINACAO ({len(steps)} passos)\n")
    for idx, step in enumerate(steps, start=1):
        imprimir_passo(idx, len(steps), step, n)

    imprimir_classificacao(classificacao)
    imprimir_solucao(matriz_final, n, pivot_cols, classificacao)


def modo_interativo() -> None:
    print("=" * 70)
    print("  TRABALHO COMPUTACIONAL 1 - METODO DE GAUSS-JORDAN")
    print("=" * 70)
    m = ler_inteiro("\nNumero de equacoes (m): ", 1, 10)
    n = ler_inteiro("Numero de incognitas (n): ", 1, 10)
    coef = ler_matriz_coeficientes(m, n)
    rhs = ler_vetor_resultado(m)
    executar(coef, rhs)


def modo_demo() -> None:
    print("=" * 70)
    print("  MODO DEMONSTRACAO - exemplos para os tres tipos de sistema")
    print("=" * 70)
    for chave, (nome, coef, rhs) in EXEMPLOS.items():
        print(f"\n\n########## EXEMPLO {chave}: {nome} ##########")
        executar(coef, rhs)


def main() -> None:
    if "--demo" in sys.argv:
        modo_demo()
    else:
        modo_interativo()


if __name__ == "__main__":
    main()
