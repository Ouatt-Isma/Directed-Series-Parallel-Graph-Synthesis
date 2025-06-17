# DSPG Transformer for Trust Network Analysis

## Overview
This repository implements an optimized algorithm for transforming non-DSPG (Directed Series-Parallel Graph) trust networks into DSPGs, which are used for reasoning about trust with Subjective Logic operators (fusion and discounting). The method ensures the preservation of as much information as possible and reduces unnecessary edge removal, thereby improving trust propagation in uncertain environments.

## Problem Statement
Traditional DSPG transformation methods prioritize uncertainty reduction, but they often result in a loss of useful information. This repository addresses these issues by introducing a new class of subgraphs, **Parallel Non-intersecting Path Subnetworks (PNPS)**, and redefines the criteria for transforming non-DSPG networks. The optimized approach ensures the preservation of valuable trust information while maintaining DSPG compliance.

## Key Features
- **Parallel Non-intersecting Path Subnetworks (PNPS):** A new concept to improve the DSPG transformation process.
- **Optimized DSPG Transformation Algorithm:** A low-level algorithm that ensures the retention of relevant edges.
- **Improved DSPG Checker:** Refined criteria to validate DSPGs without removing unnecessary edges.

## Installation
To run the code, simply clone the repository and make sure you have a LaTeX editor to compile the algorithm documentation.

```bash
git clone https://github.com/Ouatt-Isma/Directed-Series-Parallel-Graph-Synthesis.git
cd Directed-Series-Parallel-Graph-Synthesis
```



## Algorithm: Synthesize a DSPG from a non-DSPG

You can download and view the full algortihm here: [Full DSPG Synthesis Algorithm](./content/dspgsynthalg.pdf).

## Algorithm: DSPG Local Checker

You can download and view the DSPG Local Checker Algortihm here: [DSPG Local Algorithm](./content/dspgcheckeralg.pdf).

## Theorem and Proof Following the ones in the paper

You can download and view the full proof here: [Theorems and Proofs](./content/thmsproofs.pdf).

