function build_baseline_model()
% BUILD_BASELINE_MODEL Programmatically generates district_baseline_100k.slx
%
% ARCHITECTURAL DESIGNATION:
% district_baseline_100k.slx is a system-level architectural signal-flow model.
% It does NOT implement an individual-patient M/G/5 entity queue.
%
% All M/G/5 numerical statistics come from:
%   matlab/simulation/run_baseline_analysis.m
%
% Simulation assumptions are scenario parameters and must not be interpreted
% as measured clinical prevalence, clinical workflow observations, or deployment guarantees.

% Ensure scenario_config is on path
sim_scripts_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(sim_scripts_dir, '../../matlab/simulation'));

cfg = scenario_config();

model_name = 'district_baseline_100k';

if bdIsLoaded(model_name)
    close_system(model_name, 0);
end

new_system(model_name);
open_system(model_name);

set_param(model_name, 'SolverType', 'Variable-step');
set_param(model_name, 'Solver', 'VariableStepDiscrete');
set_param(model_name, 'StopTime', num2str(cfg.assumptions.total_annual_operating_seconds, '%.0f'));

% 1. 50-PHC AGGREGATE ARRIVAL STREAM (Derived directly from cfg)
aggregate_rate_val = cfg.assumptions.arrival_rate_per_phc_per_sec * cfg.assumptions.num_phcs;
add_block('simulink/Sources/Constant', [model_name, '/AGGREGATE_ARRIVAL_STREAM']);
set_param([model_name, '/AGGREGATE_ARRIVAL_STREAM'], ...
          'Value', num2str(aggregate_rate_val, '%.8f'), ...
          'Position', [40, 95, 110, 125]);

% 2. IMAGE ACQUISITION STAGE (Derived from cfg)
add_block('simulink/Continuous/Transport Delay', [model_name, '/ACQUISITION_STAGE']);
set_param([model_name, '/ACQUISITION_STAGE'], ...
          'DelayTime', num2str(cfg.assumptions.acquisition_mean_sec, '%.1f'), ...
          'Position', [150, 95, 200, 125]);

% 3. DETERMINISTIC IQA GATE PASS PROBABILITY (Derived from cfg)
iqa_pass_prob = 1 - (cfg.assumptions.iqa_initial_fail_prob * cfg.assumptions.iqa_recapture_fail_prob);
add_block('simulink/Math Operations/Gain', [model_name, '/IQA_GATE_PASS_SPLIT']);
set_param([model_name, '/IQA_GATE_PASS_SPLIT'], ...
          'Gain', num2str(iqa_pass_prob, '%.4f'), ...
          'Position', [240, 95, 280, 125]);

% 4. TELEMETRY NETWORK UPLINK TRANSFER (Derived from cfg)
uplink_delay_val = cfg.assumptions.image_payload_bits / cfg.assumptions.bandwidth_nominal_bps;
add_block('simulink/Continuous/Transport Delay', [model_name, '/TELEMETRY_UPLINK_DELAY']);
set_param([model_name, '/TELEMETRY_UPLINK_DELAY'], ...
          'DelayTime', num2str(uplink_delay_val, '%.4f'), ...
          'Position', [320, 95, 370, 125]);

% 5. E007 AI COMPUTE LATENCY (Derived from cfg.measured.e007_forward_pass_sec)
add_block('simulink/Continuous/Transport Delay', [model_name, '/E007_AI_COMPUTE_DELAY']);
set_param([model_name, '/E007_AI_COMPUTE_DELAY'], ...
          'DelayTime', num2str(cfg.measured.e007_forward_pass_sec, '%.6f'), ...
          'Position', [410, 95, 460, 125]);

% 6. REFERABLE TRIAGE BRANCHING (Derived from cfg.assumptions.prevalence_referable_dr)
add_block('simulink/Math Operations/Gain', [model_name, '/REFERABLE_TRIAGE_SPLIT']);
set_param([model_name, '/REFERABLE_TRIAGE_SPLIT'], ...
          'Gain', num2str(cfg.assumptions.prevalence_referable_dr, '%.4f'), ...
          'Position', [500, 95, 540, 125]);

% 7. DISTRICT SPECIALIST WORKLOAD ACCUMULATOR (Architectural Accumulator)
add_block('simulink/Discrete/Discrete-Time Integrator', [model_name, '/SPECIALIST_WORKLOAD_ACCUMULATOR']);
set_param([model_name, '/SPECIALIST_WORKLOAD_ACCUMULATOR'], 'gainval', '1.0', ...
          'LowerSaturationLimit', '0', 'LimitOutput', 'on', 'Position', [580, 90, 630, 130]);

% 8. COMPLETED SCREENINGS TELEMETRY SINK
add_block('simulink/Sinks/Scope', [model_name, '/COMPLETED_TELEMETRY_SCOPE']);
set_param([model_name, '/COMPLETED_TELEMETRY_SCOPE'], 'Position', [680, 95, 710, 125]);

add_line(model_name, 'AGGREGATE_ARRIVAL_STREAM/1', 'ACQUISITION_STAGE/1');
add_line(model_name, 'ACQUISITION_STAGE/1', 'IQA_GATE_PASS_SPLIT/1');
add_line(model_name, 'IQA_GATE_PASS_SPLIT/1', 'TELEMETRY_UPLINK_DELAY/1');
add_line(model_name, 'TELEMETRY_UPLINK_DELAY/1', 'E007_AI_COMPUTE_DELAY/1');
add_line(model_name, 'E007_AI_COMPUTE_DELAY/1', 'REFERABLE_TRIAGE_SPLIT/1');
add_line(model_name, 'REFERABLE_TRIAGE_SPLIT/1', 'SPECIALIST_WORKLOAD_ACCUMULATOR/1');
add_line(model_name, 'SPECIALIST_WORKLOAD_ACCUMULATOR/1', 'COMPLETED_TELEMETRY_SCOPE/1');

output_dir = fullfile(sim_scripts_dir, '..', 'scenarios');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, [model_name, '.slx']);
save_system(model_name, output_path);
close_system(model_name);
end