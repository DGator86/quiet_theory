\# D0 Model v1 — Mathematical Definition (QuIET)

\# D0 Model v1 — Mathematical Definition (QuIET)



\## 0. Purpose

Define D0 as a precise quantum-information object with:

\- no primitive time

\- no primitive geometry

\- maximal multipartite entanglement as the default “vacuum”

\- derived notions of connectivity from correlations



This is the object we will later “perturb” to define D1.



---



\## 1) Degrees of Freedom

Let V be an index set of sites (qudits).



Toy finite model:

\- |V| = N (small N for simulation)

\- Local dimension d\_i = d (uniform) unless stated otherwise



Hilbert space:

H := ⊗\_{i∈V} H\_i ,  with  H\_i ≅ C^{d\_i}.



---



\## 2) D0 State

D0 is a normalized pure state |Ψ\_D0⟩ ∈ H.



Constraints:

1\) Purity: ρ := |Ψ\_D0⟩⟨Ψ\_D0|, Tr ρ = 1

2\) No time parameter: there is no preferred t; “evolution” is not defined in D0.

3\) No geometry: there is no adjacency or distance primitive on V.



---



\## 3) Maximal Entanglement Postulate (Finite Toy)

For uniform d and N sites:



Definition (AME):

|Ψ⟩ is absolutely maximally entangled AME(N, d) if for all A ⊂ V with |A| ≤ floor(N/2):

ρ\_A := Tr\_{A^c}(ρ) = I\_{d^{|A|}} / d^{|A|}.



Interpretation:

Every small subsystem looks maximally mixed; all information resides in global correlations.



Note:

For qubits (d=2), AME(6,2) exists and is the canonical toy reference.



---



\## 4) Derived Correlation Structure (Pregeometry)

Define von Neumann entropy:

S(A) = -Tr(ρ\_A log ρ\_A).



Define mutual information between sites i and j:

I(i:j) = S({i}) + S({j}) - S({i,j}).



Define a weighted complete graph:

w\_ij := I(i:j).



This graph is a \*derived\* object (a summary of correlations), not an input geometry.



In AME(N,d) toy models, pairwise mutual information can be uniform/featureless; therefore

pregeometry may require higher-order correlation measures (tripartite / multipartite).



---



\## 5) Observables Allowed in D0

Allowed: any operator O acting on H with expectation ⟨O⟩ = ⟨Ψ|O|Ψ⟩.

Not allowed: any operator that presupposes space, time, locality, or measurement collapse.



---



\## 6) “D0 Vacuum” vs “Defects” (Bridge to D1)

Define the D0 vacuum as the AME-like maximally entangled state class.



Define a defect as a controlled violation:

\- Still pure globally

\- But some subsystem A has ρ\_A ≠ I / d^{|A|}



We will define D1 as the dynamics (rule) for defect nucleation + propagation, with entropy acting as the clock.



---



\## 7) Concrete Test Target (Code)

We operationalize D0(vacuum) in code by:

\- constructing an explicit AME(6,2) state vector |Ψ⟩

\- verifying: ∀ A with |A| ≤ 3, ρ\_A = I/2^{|A|} (within tolerance)



Your `ame62\_check.py` is the current validation harness for this.



\## 0. Purpose

Define D0 as a precise quantum-information object with:

\- no primitive time

\- no primitive geometry

\- maximal multipartite entanglement as the default “vacuum”

\- derived notions of connectivity from correlations



This is the object we will later “perturb” to define D1.



---



\## 1) Degrees of Freedom

Let V be an index set of sites (qudits).



Toy finite model:

\- |V| = N (small N for simulation)

\- Local dimension d\_i = d (uniform) unless stated otherwise



Hilbert space:

H := ⊗\_{i∈V} H\_i ,  with  H\_i ≅ C^{d\_i}.



---



\## 2) D0 State

D0 is a normalized pure state |Ψ\_D0⟩ ∈ H.



Constraints:

1\) Purity: ρ := |Ψ\_D0⟩⟨Ψ\_D0|, Tr ρ = 1

2\) No time parameter: there is no preferred t; “evolution” is not defined in D0.

3\) No geometry: there is no adjacency or distance primitive on V.



---



\## 3) Maximal Entanglement Postulate (Finite Toy)

For uniform d and N sites:



Definition (AME):

|Ψ⟩ is absolutely maximally entangled AME(N, d) if for all A ⊂ V with |A| ≤ floor(N/2):

ρ\_A := Tr\_{A^c}(ρ) = I\_{d^{|A|}} / d^{|A|}.



Interpretation:

Every small subsystem looks maximally mixed; all information resides in global correlations.



Note:

For qubits (d=2), AME(6,2) exists and is the canonical toy reference.



---



\## 4) Derived Correlation Structure (Pregeometry)

Define von Neumann entropy:

S(A) = -Tr(ρ\_A log ρ\_A).



Define mutual information between sites i and j:

I(i:j) = S({i}) + S({j}) - S({i,j}).



Define a weighted complete graph:

w\_ij := I(i:j).



This graph is a \*derived\* object (a summary of correlations), not an input geometry.



In AME(N,d) toy models, pairwise mutual information can be uniform/featureless; therefore

pregeometry may require higher-order correlation measures (tripartite / multipartite).



---



\## 5) Observables Allowed in D0

Allowed: any operator O acting on H with expectation ⟨O⟩ = ⟨Ψ|O|Ψ⟩.

Not allowed: any operator that presupposes space, time, locality, or measurement collapse.



---



\## 6) “D0 Vacuum” vs “Defects” (Bridge to D1)

Define the D0 vacuum as the AME-like maximally entangled state class.



Define a defect as a controlled violation:

\- Still pure globally

\- But some subsystem A has ρ\_A ≠ I / d^{|A|}



We will define D1 as the dynamics (rule) for defect nucleation + propagation, with entropy acting as the clock.



---



\## 7) Concrete Test Target (Code)

We operationalize D0(vacuum) in code by:

\- constructing an explicit AME(6,2) state vector |Ψ⟩

\- verifying: ∀ A with |A| ≤ 3, ρ\_A = I/2^{|A|} (within tolerance)



Your `ame62\_check.py` is the current validation harness for this.



