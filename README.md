# The Machine Gaze

Machinehood as Closure Under an Observation.

Whether something is a machine is usually treated as a fact about its parts. We treat it as a fact about a system together with an observation: a system is a machine under an observation when its dynamics close on the observed variables, so that the next observed state is fixed by the current one and the inputs. The condition is exact lumpability and the commutation of Rosen's modeling relation; it makes machinehood relative to the observation without making it arbitrary, and it names what a machine description leaves out. Three computations show what that remainder can contain. Self-sorting arrays with cell-level algorithms, reproduced after Zhang, Goldstein and Levin, all compute the same function, yet arrays that mix two algorithms cluster by algorithm while they sort, reaching aggregation of 0.59 to 0.65 where label-matched controls stay at 0.49, and at the same observed sortedness the remaining work differs 11-fold between algorithms. A controller designed on two modes of a beam is stable on its own model and destabilizes the beam unless the unmodelled modes have a damping ratio above 0.0093. In an organisation whose throughput is observed and whose repertoire for absorbing disturbances is not, ascent on observed throughput removes all maintenance and cuts long-run output to 40% of the optimum, at the policy under which the system closes exactly under its observation. We call the use of a closed quotient as if it were the system the machine gaze, and interventions that make a system close under the gaze's observation machinization.

## Build

```bash
uv run build.py          # -> paper/PAPER.pdf  (vendored canonical recipe)
```

Requires `pandoc` and `xelatex` on PATH. From the workspace you can also run `papers build the-machine-gaze`.

## Simulation

```bash
cd simulation && uv run run_all.py   # -> output/results.json and output/figures/*.png, 18 checks
```

Part of [piatra-papers](https://github.com/piatra-institute). See the workspace docs for the research and writing pipelines.
