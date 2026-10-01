# Research log

## Origin material (chats/, not committed)

A conversation that begins with "are machines machines?", takes Levin's self-sorting arrays as its opening case, develops the "machine gaze" (seeing an entity through the variables that make it usable in a larger system), reads capitalism, Taylorism and data systems as practices that make that gaze true, proposes machinehood as a task-relative quotient and a "performative modeling relation" extending Rosen, introduces "machinic capture", lists a constellation of thinkers, and proposes three computational experiments (multiplicity of observables, reduced-order control failing outside its regime, an organisation optimized on primary throughput). The archive's citations are unverified leads.

## What was kept, changed and added

- Kept: machinehood as a quotient relative to an observation; the error of treating the quotient as the system; machinization (the conversation's "machinic capture") as intervention that makes the system fit the quotient; the three computational experiments; the asymmetry of adjustment between model and system.
- Made precise: the quotient is defined by closure (fibre condition, lumpability, commutation of Rosen's diagram), with regimes of validity; Kalman's realization result names what an input-output description cannot see; the closure defect is a conditional mutual information.
- Added: a reproduction of the self-sorting arrays from the authors' published rules and code, compared with their reported numbers, and a closure test of the sortedness observable; Balas's spillover instability as Proposition 2 with a block-triangular proof that one-sided spillover is harmless; a renewal formula for the organisation model and a proof that the hidden repertoire carries zero information at zero maintenance; the conserved quantity in the model-system loop.
- Corrected or qualified: the conversation presents algotype clustering as an unexpected competency; the original authors' own analysis attributes it to differences in algorithm efficiency, and the paper reports that mechanism. The conversation attributes to Heidegger the question whether humans become standing-reserve; the paper says only that humans are challenged forth into the ordering. Ashby's law is cited by his section number (S.11/7), since the scanned edition's page numbering was ambiguous. The Marx passage is cited by chapter and section, since the available scan reproduces the 1887 text with later pagination.
- Dropped: the information-bottleneck formulation (mentioned in the conversation; not needed for the results), von Uexküll and Gibson, the French and Chinese recaps.

## Findings from the sources

- [T1] Zhang, Goldstein and Levin (arXiv 2401.05375 text and the Adaptive Behavior abstract): cell-view bubble, insertion and selection sort; frozen cells; delayed gratification; chimeric arrays; aggregation metric; mean swaps for mixtures (1740.9, 2476.02, 1534.77); peak aggregation 0.72, 0.65, 0.69, 0.62; controls 0.61, 0.65, 0.57; efficiency hypothesis supported by duplicate-value experiments. Code at github.com/Zhangtaining/cell_research (cell classes and aggregation analysis read).
- [T1] Kalman 1963 abstract: input/output relations determine only the completely controllable and completely observable part.
- [T1] Balas 1978 abstract: control and observation spillover combined can destabilize a controller designed on finitely many modes.
- [T1] Conant and Ashby 1970 abstract: a maximally successful and simple regulator must be isomorphic with the system regulated.
- [T1] Wimsatt 1997 abstract: aggregativity conditions are met differently for different decompositions.
- [T1] Bongard and Levin 2023 abstract: living systems perform multiple functions in the same place at the same time; observer-dependent, pragmatic view.
- [T1] Espeland and Stevens 1998 abstract: commensuration as a mode of power.
- [T1] Beer 1984 abstract: viability as capability of independent existence.
- [T1] Arora, Magee and Munn 2024 abstract: "machinic gaze" introduced for AI reconfiguration of heritage representation.
- [T1] Public-domain texts read on archive.org scans: Taylor 1911 (Harper & Brothers; pp. 7, 36), Whitehead 1925 (Macmillan; pp. 74–75, 85–86), Marx 1887 translation (ch. XV, sec. 4), Ashby 1956 (S.11/7).
- [T2] Editions and translators of books without DOIs checked in Open Library (Foucault 1977, Heidegger 1977, Mumford 1967, Scott 1998, Braverman 1974, Deleuze 1995, Canguilhem 2008, Rosen 1991, Kemeny and Snell 1960, Ashby 1956) and the University of Minnesota Press page (Simondon 2017, trans. Malaspina and Rogove).

## Computations

All numbers come from `simulation/analyses.py` (run id `model`): the sorting reproduction (random sequential updates replace the original threads), the beam with five modes and a two-mode LQG design, the organisation model (renewal formula checked against event-level simulation), and the model-system loop.
