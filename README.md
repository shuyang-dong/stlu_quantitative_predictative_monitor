# Logic-based Quantitative Predictive Monitoring and Control for Safe Human-Machine Interaction Under Uncertainty (AAAI2025)

## Overview

This repository is organized around two use cases — a medical case and a driving case — each of which demonstrates how logic-based quantitative monitoring and control can be applied in sequential decision-making under uncertainty. The structure follows the experiment pipeline described in Section 5 of the paper, including data generation, model training, evaluation, and statistical analysis. All experiments are reproducible with the provided scripts, datasets, and trained models.


## Table of Contents

* [Reproduction Instructions](#reproduction-instructions)
* [Medical Case Study](#medical-case-study)
* [Driving Case Study](#driving-case-study)
* [Statistical Tests and Figures](#statistical-tests-and-figures)

## Reproduction Instructions

To reproduce the results:

* Follow Steps 1–4 for Section 5.1 results
* Follow Step 5 after 1–4 for Section 5.2
* Follow Steps 6–7 after 1–5 for Section 5.3
* Step 8: Statistical tests
* Step 9: Figure drawing

Each step corresponds to a folder and code file. There’s no need to change hyperparameters unless otherwise noted.

---

The medical case study uses a modified Type-1 Diabetes simulator to evaluate the performance of logic-informed LSTM models for safe blood glucose regulation. The pipeline includes patient data simulation, LSTM training under different dropout loss strategies, evaluation via pre-alert and closed-loop metrics, and statistical comparison between baseline and proposed controllers.

## [Medical Case Study](#medical-case-study)

### Step 1: Patient Trace Generation

**Objective:** Generate simulated patient traces using the original simglucose simulator for model training.

1. **Install simulator**

   * Unzip and install from `simglucose-master-original.zip`: (Author and original page: https://github.com/jxx123/simglucose)

     ```bash
     cd '{path_simglucose}/simglucose-master-original'
     python setup.py install
     ```

2. **Run simulation script**

   * File: `generate_patient_trace_data_with_original_simglucose_simulator_controller.py`
   * Set paths at line 28 and 29 for saving output.

3. **Simulation parameters (set at line 30–39):**

   ```python
   # Start date: 2022-06-30
   # Simulation duration: 720 hours
   # Start time: 0:00
   # Random scenario seed: 15
   # CGM sensor: Dexcom
   # Sensor noise seed: 10
   # Insulin pump: Insulet
   # Controller: Basal-Bolus Controller
   ```

4. **Output**

   * Folder: `results/patient_trace_data.zip`
   * Contains traces for 30 patients (10 children, 10 adolescents, 10 adults)
   * Used for LSTM model training, testing, and validation in later steps

### Step 2: LSTM Training with Lqt

**Objective:** Train LSTM models with proposed loss function Lqt for different patient populations (child, adolescent, adult).

1. **Choose training script:**

   * `lstm_medical_data_with_dropout_adolescent.py`
   * `lstm_medical_data_with_dropout_adult.py`
   * `lstm_medical_data_with_dropout_child.py`

2. **Set required paths in the script:**
   * Paths to STLU monitor, patient data, output model/results, and normalization files

3. **Run the script** to train the LSTM model and evaluate performance.

4. **Output**

   * Trained models for each patient type (saved in `trained model.zip`)
   * Prediction results (dropout selection results, segment predictions)
   * `all_mean_value.csv` and `all_std_value.csv` (used in later steps for normalization recovery)

Note: Files with individual segment predictions are not included due to size limits but will be generated automatically after this step.\`

### Step 3: Choose Dropout with Lsat and Lacc

**Objective:** Select optimal dropout type and rate using two baseline loss functions: Lsat and Lacc.

#### a) Choose Dropout with Lsat

1. **Choose script based on patient group:**

   * `lstm_medical_data_choose_dropout_lsat_adolescent.py`
   * `lstm_medical_data_choose_dropout_lsat_adult.py`
   * `lstm_medical_data_choose_dropout_lsat_child.py`

2. **Set required paths in script:**

   * Paths to STLU monitor, patient data, trained models, and normalization files

3. **Run the script** to compute loss values for different dropout configurations.

4. **Output**

   * Folder: `results/choose_dropout_lsat_results.zip`
   * Contains dropout selection results for each patient type
   * Will be used in later prediction steps

#### b) Choose Dropout with Lacc

1. **Choose script based on patient group:**

   * `lstm_medical_data_choose_dropout_lacc_adolescent.py`
   * `lstm_medical_data_choose_dropout_lacc_adult.py`
   * `lstm_medical_data_choose_dropout_lacc_child.py`

2. **Set required paths in script:**

   * Paths to STLU monitor, patient data, trained models, and normalization files
  
3. **Run the script** to evaluate dropout configurations under Lacc loss.

4. **Output**

   * Folder: `results/choose_dropout_lacc_results.zip`
   * Dropout results to be used in Step 4 and beyond\`

### Step 4: Predict Test Set with Chosen Dropout

**Objective:** Evaluate the trained LSTM models on a test set using the dropout types and rates chosen in Steps 2 and 3.

#### a) Predict with Dropout Parameters from Lqt

1. **Choose script based on patient group:**

   * `lstm_medical_predict_testing_set_lqt_adolescent.py`
   * `lstm_medical_predict_testing_set_lqt_adult.py`
   * `lstm_medical_predict_testing_set_lqt_child.py`

2. **Set required paths in script:**

   * Paths to STLU monitor, patient data, trained models, normalization files, and output folder
   * Set dropout type/rate manually based on Step 2 results

3. **Run the script** to generate accuracy, F1 scores, and per-segment predictions

#### b) Predict with Dropout Parameters from Lsat

* Follow the same procedure as (a)
* Set `dropout_type_dict` and `dropout_rate_dict` based on lowest Lsat loss (Step 3a)

#### c) Predict with Dropout Parameters from Lacc

* Follow the same procedure as (a)
* Set dropout values according to lowest Lacc loss (Step 3b)

4. **Output:**

   * `metrics_results_*.csv`: Overall accuracy and F1 score for each patient type
   * `segment_results_*.csv`: Per-segment prediction file for each patient type
   * `results_*.csv`: Combined evaluation results

Note: Segment results files are not included due to size limits but will be regenerated. These outputs support Table 1 in the paper.

### Step 5: Pre-alert Time Evaluation

**Objective:** Evaluate pre-alert time based on prediction results from Step 4.

1. **Run the script**

   * File: `calculate_prediction_metrics_and_pre_alert_time.py`

2. **Set required paths in script:**

   * Paths to STLU monitor, prediction results from Step 4, and output folder
     
3. **Run the script** to compute pre-alert time and related prediction metrics.

4. **Output**

   * Results are saved in the `results` folder
   * Generated files will be used in subsequent closed-loop evaluation and statistical testing

### Step 6: Closed-loop Simulation

**Objective:** Evaluate the performance of the proposed controller and baseline in a closed-loop simulation setting for all patient types.

#### a) Install and Configure Modified simglucose

1. **Install modified simulator** from `simglucose_modified.zip`

   * Set the path for STLU in `sim_engine.py` at line 17 within `simglucose_modified/simglucose/simulation`

#### b) Run Baseline Simulations (No LSTM Controller)

1. **Script:** `medical_case_pipeline_new_controller_structure_no_lstm.py`
2. **Set the following paths:**

   * Set paths to STLU, trained models, normalization files, and simulation outputs
 
3. **Run baseline simulations:**

   * Scripts: `no_lstm_adolescent.py`, `no_lstm_adult.py`, `no_lstm_child.py`

#### c) Extract Meal Time and Amount

1. **Script:** `get_meal_time_amount_each_patient.py`
2. **Set paths:**

   * Input: baseline simulation results
     
3. **Run script** to generate meal info files

#### d) Run Simulations with Proposed Controller

1. **Script:** `medical_case_pipeline_new_controller_structure_eq_2.py`

2. **Set the following paths:**

   * Set paths to STLU, trained models, normalization files, meal files, and output folder
   * Confirm dropout types/rates are consistent with Step 4a

3. **Run proposed simulations:**

   * Scripts: `batchRun_adolescent_1.py`, `batchRun_adult_1.py`, `batchRun_child_1.py`

4. **Output**

   * Baseline results saved in: `results/no_lstm`
   * Proposed results saved in: `results/adult`, `results/adolescent`, `results/child`

### Step 7: Closed-loop Evaluation

**Objective:** Evaluate the performance of baseline and proposed controllers using simulation results.

1. **Run the script:**

   * File: `calculate_medical_case_pipeline_controller_metrics.py`

2. **Set required paths in script:**

   * Paths to simulation outputs and results directory

3. **Run the script** to compute evaluation metrics such as Time-in-Range (TIR), number of hazards, etc.

4. **Output**

   * All results are saved in the `results` folder
   * Metrics used for Table 3 in the paper)

### Step 8: Statistical Tests

**Objective:** Conduct statistical significance testing on pre-alert time, Time-in-Range (TIR), and number of hazards.

1. **Run the script:**

   * File: `statistical_test.py`

2. **Set required paths in the script:**

   * Paths to pre-alert results from Step 5, simulation results from Step 6, and output folders
   
3. **Run the script** to perform t-tests or other appropriate statistical comparisons

5. **Output**

   * Results are saved in the `results` folder and used to support quantitative claims in the paper

### Step 9: Figure Drawing

**Objective:** Reproduce figures in the paper comparing performance under different dropout settings and controller types.

1. **Run the script:**

   * File: `figure_drawing.py`

2. **Set required paths in the script:**

   * Paths to simulation traces, controller metrics, and dropout evaluation files

3. **Run the script to generate Figures 4–8:**

   * Lines 174–178: Load controller metrics results from Step 6 output

5. **Run the script** to generate all the required figures

6. **Output**

   * Figures are saved in the specified output directory and correspond to Figures 4–8 in the paper

---

The driving case study uses the SafeBench simulator to evaluate predictive monitoring of vehicle behaviors under different control policies, including logic-informed LSTM controllers.

## [Driving Case Study](#driving-case-study)

### Step 1: Vehicle Trace Generation

**Objective:** Generate simulated vehicle traces using the SafeBench simulator for model training.

1. **Install and configure SafeBench**

   * Install from: [https://github.com/trust-ai/SafeBench](https://github.com/trust-ai/SafeBench)
   * Replace folders `safebench` and `script` with the provided versions in `SafeBench_1`

2. **Run simulation script**

   * Navigate to `safebench/predictive_monitor_trace`
   * Run: `run_multiple_times.sh`
   * Set parameters:

     * `i` in range `[1, 15]` (random seed)
     * `--if_new_controller 0`
     * Update file paths as needed in the script

3. **Output**

   * Folder: `results/vehicle_trace_data.zip`
   * Contains 2 examples for each behavior type (e.g., cautious, aggressive)
   * These traces are used for model training and evaluation in subsequent steps\`

### Step 2: LSTM Training with Lqt

**Objective:** Train LSTM models using the proposed loss function Lqt for different vehicle behavior types.

1. **Choose training script based on behavior type:**

   * `lstm_driving_data_with_dropout_Lqt_behavior_0.py` (e.g., cautious)
   * `lstm_driving_data_with_dropout_Lqt_behavior_2.py` (e.g., aggressive)

2. **Set required paths in the script:**

   * Path to load vehicle trace data (from Step 1)
   * Path to STLU monitor package (from `stlu_monitor.zip`)
   * Path to save trained models
   * Path to save `all_mean_value.csv` and `all_std_value.csv` for normalization recovery
   * Path to save output metric results

3. **Run the script** to train models and evaluate dropout configurations.

4. **Output**

   * Trained models for each behavior type (saved in `trained model.zip`)
   * Dropout result files for each configuration
   * `all_mean_value.csv` and `all_std_value.csv` for standardization

Note: Segment-level prediction files are not included in the repo due to size limits but will be generated automatically after training.

### Step 3: Choose Dropout with Lsat and Lacc

**Objective:** Evaluate dropout types and rates using two baseline loss functions: Lsat and Lacc.

#### a) Choose Dropout with Lsat

1. **Choose script based on behavior type:**

   * `lstm_driving_data_choose_dropout_lsat_behavior_0.py`
   * `lstm_driving_data_choose_dropout_lsat_behavior_2.py`

2. **Set required paths in script:**

   * Path to STLU monitor package
   * Path to vehicle trace data from Step 1
   * Path to trained models from Step 2
   * Path to `all_mean_value.csv` and `all_std_value.csv` from Step 2
   * Path to save output results

3. **Run the script** to compute dropout performance under Lsat loss.

4. **Output**

   * Folder: `select_dropout_with_lossfunc_LSAT`
   * Dropout results to be used in Step 4 and beyond

#### b) Choose Dropout with Lacc

1. **Choose script based on behavior type:**

   * `lstm_driving_data_choose_dropout_lacc_behavior_0.py`
   * `lstm_driving_data_choose_dropout_lacc_behavior_2.py`

2. **Set required paths in script:**

   * Same as Lsat step above, updated for Lacc-specific paths and file names

3. **Run the script** to evaluate dropout settings under Lacc loss.

4. **Output**

   * Folder: `select_dropout_with_lossfunc_LACC`
   * Dropout results to be used in Step 4 and beyond\`

### Step 4: Predict Test Set with Chosen Dropout

**Objective:** Evaluate trained LSTM models on the test set using dropout types and rates selected from Lqt, Lsat, or Lacc.

1. **Choose prediction script based on behavior type and dropout selection:**

   * `lstm_driving_data_predict_testing_set_Lqt_behavior_0.py`
   * `lstm_driving_data_predict_testing_set_Lqt_behavior_2.py`
   * (Or equivalent scripts for Lsat and Lacc based evaluations)

2. **Set required paths in the script:**

   * Path to STLU monitor package
   * Path to vehicle trace data (from Step 1)
   * Path to trained LSTM models (from Step 2)
   * Path to normalization files (`all_mean_value.csv`, `all_std_value.csv` from Step 2)
   * Path to save output results
   * Confirm dropout parameters (type/rate) from `dropout_choice_result_*.csv` in Step 2 or 3

3. **Run the script** to compute prediction metrics such as accuracy and F1, and generate segment-level prediction outputs.

4. **Output**

   * `metrics_results_*.csv`: Test accuracy and F1 score for each behavior type
   * `segment_results_*.csv`: Prediction results on each test segment
   * These files support Table 1 results in the paper

### Step 5: Pre-alert Time Evaluation

**Objective:** Compute pre-alert time metrics from prediction results generated in Step 4.

1. **Run the script:**

   * File: `calculate_prediction_metrics_and_pre_alert_time.py`

2. **Set required paths in the script:**

   * Path to STLU monitor package
   * Path to segment result files (generated from Step 4, e.g., `segment_results_*.csv`)
   * Path to save output metrics and intermediate files

3. **Run the script** to evaluate pre-alert effectiveness for each behavior type.

4. **Output:**

   * Result files saved in the `results` folder
   * These outputs will be used in Step 6 (simulation comparison) and Step 7 (closed-loop evaluation)

### Step 6: Closed-loop Simulation

**Objective:** Run and compare closed-loop simulations using baseline and proposed controllers for different driving behaviors.

1. **Navigate to simulation script directory:**

   * `safebench/predictive_monitor_trace`

2. **Run baseline controller simulations:**

   * Command:

     ```bash
     ./run_multiple_times.sh --if_new_controller 1 --control_type no_lstm
     ```
   * Set `i` in range `[0, 3]` for different seeds

3. **Run proposed controller simulations:**

   * Command:

     ```bash
     ./run_multiple_times.sh --if_new_controller 1 --control_type lstm_with_monitor
     ```
   * Set `i` in range `[0, 3]` for different seeds

4. **Output:**

   * Baseline results saved in: `results/no_lstm`
   * Proposed controller results saved in: `results/lstm_with_monitor`
   * These outputs will be used in Step 7 and Step 8\`

### Step 7: Closed-loop Evaluation

**Objective:** Evaluate the closed-loop performance of both baseline and proposed controllers based on driving behavior simulations.

1. **Run the script:**

   * File: `metric_calcualtion_close_loop_driving_case.py`

2. **Set required paths in the script:**

   * Path to simulation result folder containing `no_lstm` and `lstm_with_monitor` subfolders (from Step 6)
   * Path to save computed evaluation metrics

3. **Run the script** to compute metrics such as number of hazards, time in safe range, or other task-specific evaluation results.

4. **Output:**

   * Metrics saved in the specified output folder
   * These results correspond to Table 3 in the paper

### Step 8: Figure Drawing

**Objective:** Generate driving-related figures as shown in the paper.

1. **Run the script:**

   * File: `figure_drawing_driving_case.py`

2. **Set required paths in the script:**

   * Paths to simulation outputs from previous steps (e.g., `results/no_lstm`, `results/lstm_with_monitor`)
   * Paths to controller metrics results used in closed-loop evaluation
   * Example filenames can be found in the script comments (e.g., `controller_metrics_result_behavior_*.csv`, `segment_results_*.csv`)

3. **Run the script** to generate visualizations for different dropout settings, controller comparisons, and behavioral metrics.

4. **Output:**

   * Figures will be saved in the designated output directory
   * These correspond to the driving case study visualizations presented in the paper

---
