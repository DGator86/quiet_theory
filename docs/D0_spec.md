\# D0 Spec — Unitary Qudit Substrate (QuIET)



\## 0. Purpose

Define D0 as a mathematically explicit substrate that can later generate D1–D3 by constraint/limit processes.



D0 is:

\- pregeometric (no metric, no spatial adjacency)

\- pretemporal (no time parameter)

\- unitary (global pure state)

\- maximally entangled in a precise sense (to be defined)



\## 1. Primitive Objects

\### 1.1 Nodes (qudits)

Let there be N nodes (finite or countably infinite).  

Each node i carries a local Hilbert space H\_i ≅ C^{d\_i}.  

\- d\_i may be constant (d) or drawn from a small set (e.g., {2,3}) if mixed-qudit types are allowed.



Define the total Hilbert space:

H\_total = ⊗\_{i=1..N} H\_i



\### 1.2 Global state

D0 is specified by a single normalized pure state:

|Ψ⟩ ∈ H\_total,   ⟨Ψ|Ψ⟩ = 1



There is no Hamiltonian and no time evolution in D0.



\## 2. What “Maximally Entangled” Means in D0

We require an entanglement condition that replaces geometry.



\### 2.1 Candidate condition (perfect tensors / AME)

For fixed N and fixed local dimension d, define:

|Ψ⟩ is AME(N, d) if every reduced density matrix on any subset of size ≤ floor(N/2) is maximally mixed.



Equivalently:

For any subset A with |A| = k ≤ floor(N/2),

ρ\_A = Tr\_{A^c}(|Ψ⟩⟨Ψ|) = I\_{d^k} / d^k



This gives uniform entanglement across partitions.



\### 2.2 Why AME is a good D0 boundary condition

\- No preferred partition: symmetry between subsystems

\- Supplies “connectivity without distance”

\- Provides a clean mathematical target for “D0 as unitary + maximal entanglement”



\## 3. Entanglement Graph as Derived Structure (No geometry yet)

Define mutual information between node i and j:

I(i:j) = S(ρ\_i) + S(ρ\_j) − S(ρ\_{ij})



This yields a complete weighted graph on nodes:

G\_MI = (V, w\_ij = I(i:j))



D0 has no spatial edges; “edges” are derived from correlations.



\## 4. D0 Invariants / Observables (Allowed measurements)

D0 has no measurements internally.

But as theorists we can compute invariants from |Ψ⟩ such as:

\- subsystem entropies S(ρ\_A)

\- MI graph spectra

\- tensor rank / perfect-tensor properties

\- stabilizer structure (if restricted class)



\## 5. D0 → D1 Trigger (placeholder)

D1 begins when the global unitary constraint is broken by an entropic defect.

D0 must therefore admit a notion of “defect” or “constraint violation”:

\- localized reduction of entanglement?

\- introduction of a non-unitary channel?

\- loss of perfect-tensor condition?



(Leave empty until D0 is finished.)



\## 6. Minimal Finite Example

We verified AME(6,2) in code (6 qubits, perfect tensor).

This is our first concrete D0 toy universe:

\- N = 6, d = 2

\- |Ψ⟩ = |Ξ\_{6,2}⟩ (AME state)

\- For any k ≤ 3, ρ\_A = I / 2^k



Next: generalize to d=3 and to scalable constructions.



