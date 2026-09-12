# Archived: Scenario Simulator

This was Streamlit dashboard page `5_Scenario_Simulator.py`, removed from
`src/dashboard/pages/` on the `feature/removing-simulator` branch. It's kept
here (with the utils and templates that only it used) for reference rather
than deleted; full history is preserved via `git log --follow` on each file.

Contents:
- `5_Scenario_Simulator.py` — the Streamlit page (paths updated to be self-contained here)
- `predict_scenario.py` — ran the v3 model ensemble against a user-entered scenario
- `f1_constructors_rank_classifier.py`, `f1_dataset.py` — dashboard-local copies of the v3 model/dataset classes
- `scenario_template_data.csv`, `input_data_template.csv` — supporting data templates

To bring it back: move `5_Scenario_Simulator.py` into `src/dashboard/pages/`,
the two `.py` utils into `src/dashboard/pages/utils/`, the two `.csv` files
into `src/dashboard/templates/`, and change the import back to
`from pages.utils.predict_scenario import run_scenario`.
