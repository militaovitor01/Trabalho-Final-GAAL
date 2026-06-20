# Trabalho Computacional 1 — Resolução de Sistemas Lineares pelo Método de Gauss-Jordan

Este trabalho contém **duas implementações** do mesmo algoritmo, atendendo a todos os
requisitos do enunciado:

## 1. `index.html` — Interface gráfica (recomendado para apresentação)

Aplicativo web autocontido (HTML + CSS + JavaScript, sem dependências externas).
**Basta abrir o arquivo `index.html` em qualquer navegador** — não precisa instalar nada.

**Como usar:**
1. Defina o número de equações (m) e incógnitas (n) e clique em "Gerar grade de coeficientes".
2. Preencha a grade: a coluna à esquerda da barra `|` é a **matriz de coeficientes (A)**;
   a coluna à direita (em destaque) é a **matriz/vetor de resultados (b)** — propositalmente
   separadas visualmente, conforme exigido.
3. Clique em **Resolver sistema**.
4. Use as setas "‹ ›" para navegar passo a passo por cada operação elementar, com a
   matriz sendo destacada a cada alteração, ou clique em "Ver lista completa de operações"
   para o histórico inteiro de uma vez.
5. Três exemplos prontos (um de cada classificação) estão disponíveis na lateral.

A interface identifica automaticamente o tipo do sistema — **Possível e Determinado**,
**Possível e Indeterminado** ou **Impossível** — e exibe a solução final (incluindo a
expressão das variáveis livres no caso indeterminado).

## 2. `gauss_jordan.py` — Versão em Python (linha de comando)

Implementação em Python puro (sem bibliotecas externas) do mesmo algoritmo, para quem
preferir/precisar de uma versão em Python conforme o enunciado sugere.

**Como executar:**
```bash
python3 gauss_jordan.py
```
O programa pede o número de equações e incógnitas, depois solicita a matriz de
coeficientes e o vetor de resultados **separadamente** (dois passos de entrada distintos),
e imprime no terminal a matriz inicial, cada operação elementar, a forma escalonada
reduzida, a classificação do sistema e a solução.

**Modo demonstração** (roda automaticamente os três exemplos — determinado, indeterminado
e impossível — sem precisar digitar nada):
```bash
python3 gauss_jordan.py --demo
```

## Sobre o algoritmo

Ambas as versões implementam Gauss-Jordan com **pivotamento parcial** (escolhe, em cada
coluna, a linha com maior valor absoluto para ser o pivô, reduzindo erros de
arredondamento) e tolerância numérica de `1e-9` para tratar valores que deveriam ser
zero mas carregam erro de ponto flutuante.

Classificação do sistema, comparando o posto (rank) da matriz de coeficientes com o
número de incógnitas (n) e verificando inconsistências (linha do tipo `0 = k`, `k≠0`):

| Condição | Classificação |
|---|---|
| posto = n e sistema consistente | Possível e Determinado (solução única) |
| posto < n e sistema consistente | Possível e Indeterminado (infinitas soluções) |
| linha inconsistente (`0 = k`, k≠0) | Impossível (nenhuma solução) |
