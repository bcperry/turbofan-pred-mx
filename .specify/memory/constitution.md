<!--
Sync Impact Report
Version change: unratified template -> 1.0.0 (initial adoption)
Modified principles: five placeholder principles replaced with:
	I. NASA Data Provenance and Integrity
	II. Leakage-Free Remaining Useful Life Modeling
	III. Python and uv Only
	IV. Azure Machine Learning Reproducibility
	V. Tested, Measurable, and Responsible Results
Added sections: Project Scope and Operational Constraints; Development Workflow and Quality Gates;
	concrete Governance rules.
Removed sections: none; placeholder sections populated.
Templates and guidance:
	updated: .specify/templates/plan-template.md
	updated: .specify/templates/spec-template.md
	updated: .specify/templates/tasks-template.md
	updated: .github/copilot-instructions.md
	not applicable: .specify/templates/commands/*.md (directory absent)
	not applicable: README.md and docs/quickstart.md (absent)
Follow-up TODOs: none. Workspace identifiers, compute sizing, and quantitative model targets
	are feature-planning decisions, not deferred constitution placeholders.
-->
# Turbofan Predictive Maintenance Constitution

## Core Principles

### I. NASA Data Provenance and Integrity

The project MUST use the NASA Ames PCoE C-MAPSS turbofan degradation dataset as its
initial research source and retain the supplied documentation and citation. Data ingestion
MUST record the download URL, retrieval date, archive checksum, and selected subset identifiers.
Raw files MUST remain immutable; derived data MUST retain traceability to their source and
transformation configuration. Parsers MUST validate documented columns, engine identifiers,
cycle ordering, missing values, and ground-truth alignment. FD001 through FD004 MUST remain
distinguishable; their operating conditions and fault modes MUST be verified against the
dataset documentation before modeling. These controls prevent silent data corruption and
unsupported comparisons across distinct simulation scenarios.

### II. Leakage-Free Remaining Useful Life Modeling

The initial prediction target MUST be remaining useful life (RUL) in operating cycles.
Training labels MUST be derived from run-to-failure training trajectories; evaluation of
truncated test trajectories MUST use their supplied RUL ground truth. Any RUL cap or alternative
target definition MUST be documented and versioned. Training and validation partitions MUST
be disjoint by engine within each subset, using the pair of subset and engine identifier as
the identity. Preprocessing MUST be fitted only on training engines. Features and sequence
windows MUST use only information available at prediction time and MUST NOT cross engine
boundaries. Test labels MUST NOT influence feature selection, preprocessing, or hyperparameter
tuning. These rules prevent optimistic results caused by temporal or engine-level leakage.

### III. Python and uv Only

Modeling, data processing, training, and evaluation code MUST use Python. The project MUST
declare its supported Python version and dependencies in `pyproject.toml` and commit `uv.lock`.
Dependencies MUST be added with `uv add` (or `uv add --dev` for development dependencies);
environments MUST be restored with `uv sync --locked`, and project commands MUST run through
`uv run`. Direct `pip`, `python -m pip`, and `uv pip` commands are prohibited everywhere,
including notebooks, scripts, CI, documentation examples, and Azure ML environment builds.
Azure ML training environments MUST install uv without pip and synchronize the committed
lockfile; the selected uv version and Python runtime MUST be recorded. This single dependency
workflow prevents local and remote environment drift.

### IV. Azure Machine Learning Reproducibility

Azure Machine Learning (Azure ML) MUST be the target execution platform for training and
evaluation. Production training logic MUST live in reusable Python modules and executable
entry points, not depend on hidden notebook state. Notebooks MAY support exploration and
job submission. Jobs MUST be defined in version-controlled configuration or submission code
using supported Azure ML v2 interfaces. Each run MUST record the source revision and dirty
state, versioned data assets, environment identity, Python and uv versions, lockfile identity,
random seeds, split definition, parameters, and compute configuration. Runs MUST log metrics,
evaluation reports, fitted preprocessing, and model artifacts through MLflow or Azure ML
tracking. A documented submission flow MUST reproduce training in the chosen workspace;
local execution alone does not satisfy the training milestone.

### V. Tested, Measurable, and Responsible Results

Changes to ingestion, labels, splitting, features, training, or evaluation MUST include
automated tests for the affected behavior. Tests MUST cover RUL label calculations, test
ground-truth alignment, engine-disjoint splits, train-only preprocessing, and inference
input/output contracts as those capabilities are implemented. Experiments MUST establish
a simple baseline before adding model complexity. Evaluation MUST report RMSE and MAE in
cycles and a documented asymmetric NASA/PHM prognostic score, including its formula and
aggregation. Predictions and metrics MUST be reported per subset, with any combined results
explicitly justified. Model acceptance thresholds and comparison protocols MUST be defined
before final test evaluation. Claims MUST distinguish simulation performance from real-engine
readiness; outputs MUST NOT be represented as certified maintenance or flight-safety advice.

## Project Scope and Operational Constraints

- The first end-to-end deliverable MUST acquire and validate the dataset, prepare features
	and RUL targets, train a baseline in an Azure ML workspace, evaluate held-out engines,
	and retain the model and reports as traceable run artifacts.
- Initial scope MAY select one documented subset; adding other subsets MUST preserve
	their separate identities and evaluation protocols. Real-time serving, operational
	maintenance integration, and production certification are outside the initial milestone.
- The dataset source is
	<https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>.
	The supplied citation MUST be preserved: A. Saxena and K. Goebel (2008).
	"Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository,
	NASA Ames Research Center, Moffett Field, CA. Access and redistribution terms MUST be
	checked against the downloaded documentation; public download does not imply a license.
- Credentials MUST NOT be committed or logged. Workspace, subscription, resource group,
	data paths, and compute names MUST be configurable. Azure access MUST use managed identity
	or an approved developer identity with least-privilege access.
- Cloud runs MUST specify bounded compute and job duration, configure idle scale-down or
	auto-shutdown where supported, and document expected cost before submission. Resource
	creation and paid training runs MUST require explicit user authorization.
- Raw archives, extracted datasets, credentials, and generated model binaries MUST remain
	outside Git; versioned Azure ML assets and run metadata MUST provide their provenance.

## Development Workflow and Quality Gates

1. Specifications MUST state the target subset, RUL definition, user outcome, measurable
	 acceptance criteria, and any required Azure access or resource assumptions.
2. Plans MUST pass a Constitution Check before implementation and after design, covering
	 provenance, leakage prevention, Python/uv, Azure ML reproducibility, tests, and cloud cost.
3. Implementation MUST begin with a small locally testable slice. Contributors MUST run
	 focused tests through `uv run` and restore dependencies with `uv sync --locked` in CI.
4. Before a full experiment, a bounded Azure ML smoke job MUST validate data access,
	 uv-based environment setup, entry-point execution, metric logging, and artifact output.
5. Completion of the initial training milestone MUST include a successful Azure ML job ID,
	 its configuration, baseline comparison, held-out metrics, and retained model artifacts.
	 Documentation MUST provide reproducible setup, submission, result inspection, and
	 compute cleanup commands. Unrun cloud checks MUST be reported as pending, not passed.

## Governance

This constitution governs repository specifications, plans, tasks, implementation, and
reviews. Changes MUST be proposed with a rationale, impact assessment, and updates to
affected templates and guidance. The repository owner MUST approve amendments and any
temporary exception; exceptions MUST identify their scope, risk, owner, and expiry.

Versioning MUST follow semantic versioning: MAJOR for incompatible principle removals or
redefinitions, MINOR for new principles or materially expanded obligations, and PATCH for
clarifications without changed obligations. Each amendment MUST update the version, last
amended date, and Sync Impact Report; the original ratification date MUST remain unchanged.

Every implementation plan and pull request MUST assess compliance with the applicable
principles and include validation evidence or explicitly identified pending checks.
The current feature plan supplies implementation detail; `.github/copilot-instructions.md`
provides always-on contributor guidance and MUST remain consistent with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-10-16 | **Last Amended**: 2026-10-16
