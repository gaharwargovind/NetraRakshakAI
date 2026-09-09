function cfg = scenario_config()
% SCENARIO_CONFIG Master parameter ledger for NetraRakshakAI systems simulation.
%
% CRITICAL SCIENTIFIC & ENGINEERING BOUNDARY:
% This configuration strictly segregates MEASURED engineering metrics obtained
% from local benchmark runs on the production pipeline from SCENARIO ASSUMPTIONS
% utilized for regional health systems queueing analysis.
%
% Simulation assumptions are scenario parameters and must not be interpreted
% as measured clinical prevalence, clinical workflow observations, or deployment guarantees.
%
% Target Deployment Scale:
%   50 Primary Health Centres (PHCs) -> 1 District Tele-Ophthalmology Centre
%   250-day equivalent operational horizon comprising 7,200,000 clinic operating seconds.
%   (Overnight and non-operating clinic intervals are compressed from the queueing clock
%   because the model evaluates active operational capacity).

%% =========================================================================
% SECTION 1: MEASURED & IMPLEMENTED PROJECT VALUES
% =========================================================================
cfg.measured = struct();

% Pure PyTorch forward pass latency (EfficientNet-B0, batch size 1, Apple M4 MPS)
% Measured Project Value (used for AI compute stage to avoid double-counting)
cfg.measured.e007_forward_pass_ms = 10.2;
cfg.measured.e007_forward_pass_sec = cfg.measured.e007_forward_pass_ms / 1000;

% Total E015 single-image end-to-end pipeline benchmark on Apple M4 (MPS device).
% Measured Project Value (includes IQA, preprocessing, E007, Grad-CAM, PDF/JSON, disk I/O)
cfg.measured.e015_total_pipeline_ms = 466.0;
cfg.measured.e015_total_pipeline_sec = cfg.measured.e015_total_pipeline_ms / 1000;

% Calibrated temperature scaling parameter
% Derived project parameter — E010 development calibration
cfg.measured.frozen_temperature = 0.7785;

% Deterministic optical quality checks
% Implemented Project Component (defined in src/inference/quality_gate.py)
cfg.measured.e013_metric_count = 8;


%% =========================================================================
% SECTION 2: SCENARIO ASSUMPTIONS (Operational & Queueing Estimates)
% =========================================================================
% NOTE: The following parameters are modeling assumptions for regional queueing
% analysis. They are NOT measured clinical facts or validated field trial data.
cfg.assumptions = struct();

% --- Network Hierarchy & Operational Horizon ---
cfg.assumptions.num_phcs = 50;
cfg.assumptions.num_district_centres = 1;
cfg.assumptions.annual_operating_days = 250;
cfg.assumptions.daily_operating_hours = 8;

% 250-day equivalent operational horizon comprising 7,200,000 clinic operating seconds
cfg.assumptions.total_annual_operating_seconds = ...
    cfg.assumptions.annual_operating_days * cfg.assumptions.daily_operating_hours * 3600;

% Macro annual encounter target
cfg.assumptions.target_annual_encounters = 100000;

% Arrival Process: 50 independent statistically identical homogeneous Poisson arrival streams
% Mean arrival rate per PHC during operational clinic hours (arrivals/second):
cfg.assumptions.arrival_rate_per_phc_per_sec = ...
    cfg.assumptions.target_annual_encounters / ...
    (cfg.assumptions.num_phcs * cfg.assumptions.total_annual_operating_seconds);

% Aggregate district arrival rate across all 50 clinics
cfg.assumptions.district_aggregate_arrival_rate_per_sec = ...
    cfg.assumptions.arrival_rate_per_phc_per_sec * cfg.assumptions.num_phcs;

% --- PHC Workflow & Image Acquisition ---
cfg.assumptions.acquisition_mean_sec = 180.0;
cfg.assumptions.acquisition_std_sec = 45.0;

% Fundus image payload size (dual-eye, 45-degree field, compressed)
cfg.assumptions.image_payload_bytes = 2.5 * 1024 * 1024; % 2.5 MB
cfg.assumptions.image_payload_bits = cfg.assumptions.image_payload_bytes * 8; % 20,971,520 bits

% Dynamically derived incoming telemetry bit rate per rural PHC
cfg.assumptions.incoming_bit_rate_per_phc_bps = ...
    cfg.assumptions.arrival_rate_per_phc_per_sec * cfg.assumptions.image_payload_bits;

% --- Deterministic E013 IQA Gate & Recapture Dynamics ---
cfg.assumptions.iqa_initial_fail_prob = 0.10;
cfg.assumptions.recapture_delay_sec = 90.0;
cfg.assumptions.iqa_recapture_fail_prob = 0.15;

% Scenario assumption: persistent ungradable cases are transmitted for remote clinical adjudication
cfg.assumptions.route_persistent_iqa_to_specialist = true;

% --- AI Compute Execution Latency Mode ---
cfg.assumptions.ai_compute_mode = 'pure_forward_pass';
if strcmp(cfg.assumptions.ai_compute_mode, 'pure_forward_pass')
    cfg.assumptions.ai_compute_latency_sec = cfg.measured.e007_forward_pass_sec;
else
    cfg.assumptions.ai_compute_latency_sec = cfg.measured.e015_total_pipeline_sec;
end

% --- Clinical Referral Prevalence ---
cfg.assumptions.prevalence_referable_dr = 0.15;

% --- Telemetry Uplink Channels ---
cfg.assumptions.bandwidth_nominal_bps = 1.0 * 1024 * 1024; % 1.0 Mbps
cfg.assumptions.bandwidth_sweep_bps = [ ...
    50   * 1024, ...
    128  * 1024, ...
    256  * 1024, ...
    512  * 1024, ...
    1024 * 1024, ...
    2048 * 1024  ...
];

% --- Edge Store-and-Forward Buffer Parameters ---
cfg.assumptions.edge_storage_capacity_bytes = 64 * 1024^3; % 64 GB physical ceiling
cfg.assumptions.outage_start_sec = 7200;                  % 2 hours into operational shift
cfg.assumptions.simulated_outage_duration_sec = 4 * 3600; % 4-hour blackout

% --- District Review Centre Multi-Server Capacity ---
cfg.assumptions.baseline_specialist_count = 5;
cfg.assumptions.specialist_review_mean_sec = 240.0;
cfg.assumptions.specialist_review_std_sec = 60.0;
end

