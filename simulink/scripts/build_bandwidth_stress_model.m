function build_bandwidth_stress_model()
% BUILD_BANDWIDTH_STRESS_MODEL Programmatically generates bandwidth_stress_test.slx
%
% ARCHITECTURAL DESIGNATION:
% Continuous-rate fluid approximation of store-and-forward edge buffering;
% not an entity-level packet simulation.
%
% Parameters derived from scenario_config():
%   buffer_rate = incoming_rate - transmitted_rate
%   where transmitted_rate = 0 during outage, and <= bandwidth after restoration.
%   Enforces a strict 64 GB physical ceiling.
%
% Simulation assumptions are scenario parameters and must not be interpreted
% as measured clinical prevalence, clinical workflow observations, or deployment guarantees.

sim_scripts_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(sim_scripts_dir, '../../matlab/simulation'));

cfg = scenario_config();

model_name = 'bandwidth_stress_test';

if bdIsLoaded(model_name)
    close_system(model_name, 0);
end

new_system(model_name);
open_system(model_name);

set_param(model_name, 'SolverType', 'Variable-step');
set_param(model_name, 'StopTime', '28800'); % 8-hour operational shift (seconds)

% 1. INCOMING DATA RATE (Derived from cfg.assumptions.incoming_bit_rate_per_phc_bps)
add_block('simulink/Sources/Constant', [model_name, '/INCOMING_DATA_RATE']);
set_param([model_name, '/INCOMING_DATA_RATE'], ...
          'Value', num2str(cfg.assumptions.incoming_bit_rate_per_phc_bps, '%.4f'), ...
          'Position', [40, 75, 100, 105]);

% 2. ACTIVE OUTAGE GENERATOR (Derived from cfg)
outage_start_val = cfg.assumptions.outage_start_sec;
outage_restore_val = outage_start_val + cfg.assumptions.simulated_outage_duration_sec;

add_block('simulink/Sources/Step', [model_name, '/OUTAGE_START_STEP']);
set_param([model_name, '/OUTAGE_START_STEP'], ...
          'Time', num2str(outage_start_val, '%.0f'), ...
          'After', '1', 'Position', [40, 150, 80, 180]);

add_block('simulink/Sources/Step', [model_name, '/OUTAGE_RESTORE_STEP']);
set_param([model_name, '/OUTAGE_RESTORE_STEP'], ...
          'Time', num2str(outage_restore_val, '%.0f'), ...
          'After', '1', 'Position', [40, 210, 80, 240]);

add_block('simulink/Math Operations/Subtract', [model_name, '/OUTAGE_WINDOW_LOGIC']);
set_param([model_name, '/OUTAGE_WINDOW_LOGIC'], 'Position', [130, 170, 160, 200]);

add_block('simulink/Logic and Bit Operations/Logical Operator', [model_name, '/NETWORK_ONLINE_FLAG']);
set_param([model_name, '/NETWORK_ONLINE_FLAG'], 'Operator', 'NOT', 'Position', [200, 170, 230, 200]);

% 3. CHANNEL BANDWIDTH (Derived from cfg.assumptions.bandwidth_nominal_bps)
add_block('simulink/Sources/Constant', [model_name, '/CHANNEL_BANDWIDTH']);
set_param([model_name, '/CHANNEL_BANDWIDTH'], ...
          'Value', num2str(cfg.assumptions.bandwidth_nominal_bps, '%.0f'), ...
          'Position', [200, 230, 260, 260]);

add_block('simulink/Math Operations/Product', [model_name, '/TRANSMITTED_RATE']);
set_param([model_name, '/TRANSMITTED_RATE'], 'Position', [300, 190, 330, 225]);

% 4. NET STORAGE INFLOW = Influx Rate - Transmitted Rate
add_block('simulink/Math Operations/Subtract', [model_name, '/NET_BUFFER_RATE']);
set_param([model_name, '/NET_BUFFER_RATE'], 'Position', [370, 80, 400, 115]);

% 5. 64 GB BOUNDED STORE-AND-FORWARD BUFFER (Derived from cfg)
storage_ceiling_bits = cfg.assumptions.edge_storage_capacity_bytes * 8;
add_block('simulink/Continuous/Integrator', [model_name, '/STORE_AND_FORWARD_BUFFER']);
set_param([model_name, '/STORE_AND_FORWARD_BUFFER'], ...
          'LimitOutput', 'on', ...
          'UpperSaturationLimit', num2str(storage_ceiling_bits, '%.10e'), ...
          'LowerSaturationLimit', '0', ...
          'Position', [450, 80, 490, 120]);

% 6. BUFFER OCCUPANCY SCOPE
add_block('simulink/Sinks/Scope', [model_name, '/BUFFER_OCCUPANCY_SCOPE']);
set_param([model_name, '/BUFFER_OCCUPANCY_SCOPE'], 'Position', [550, 85, 580, 115]);

add_line(model_name, 'OUTAGE_START_STEP/1', 'OUTAGE_WINDOW_LOGIC/1');
add_line(model_name, 'OUTAGE_RESTORE_STEP/1', 'OUTAGE_WINDOW_LOGIC/2');
add_line(model_name, 'OUTAGE_WINDOW_LOGIC/1', 'NETWORK_ONLINE_FLAG/1');
add_line(model_name, 'NETWORK_ONLINE_FLAG/1', 'TRANSMITTED_RATE/1');
add_line(model_name, 'CHANNEL_BANDWIDTH/1', 'TRANSMITTED_RATE/2');

add_line(model_name, 'INCOMING_DATA_RATE/1', 'NET_BUFFER_RATE/1');
add_line(model_name, 'TRANSMITTED_RATE/1', 'NET_BUFFER_RATE/2');
add_line(model_name, 'NET_BUFFER_RATE/1', 'STORE_AND_FORWARD_BUFFER/1');
add_line(model_name, 'STORE_AND_FORWARD_BUFFER/1', 'BUFFER_OCCUPANCY_SCOPE/1');

output_dir = fullfile(sim_scripts_dir, '..', 'scenarios');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, [model_name, '.slx']);
save_system(model_name, output_path);
close_system(model_name);
end