function results = run_baseline_analysis(custom_cfg)
% RUN_BASELINE_ANALYSIS Simulates a 250-day equivalent operational horizon
% comprising 7,200,000 clinic operating seconds across 50 rural PHCs.
%
% Overnight and non-operating intervals are compressed from the queueing clock
% because the model evaluates active operational capacity.
%
% Simulation assumptions are scenario parameters and must not be interpreted
% as measured clinical prevalence, clinical workflow observations, or deployment guarantees.

if nargin < 1 || isempty(custom_cfg)
    cfg = scenario_config();
else
    cfg = custom_cfg;
end

rng(42, 'twister');

total_horizon_sec = cfg.assumptions.total_annual_operating_seconds; % 7,200,000 s
num_phcs = cfg.assumptions.num_phcs;
lambda_phc = cfg.assumptions.arrival_rate_per_phc_per_sec;

% 1. Generate Arrivals for 50 independent statistically identical homogeneous Poisson arrival streams
arrival_records = [];
for p = 1:num_phcs
    current_time = 0;
    phc_arrivals = [];
    while true
        inter_arr = exprnd(1 / lambda_phc);
        current_time = current_time + inter_arr;
        if current_time > total_horizon_sec
            break;
        end
        phc_arrivals = [phc_arrivals; current_time, p]; %#ok<AGROW>
    end
    arrival_records = [arrival_records; phc_arrivals]; %#ok<AGROW>
end

arrival_records = sortrows(arrival_records, 1);
total_generated_arrivals = size(arrival_records, 1);
arrival_timestamps = arrival_records(:, 1);

% 2. Image Acquisition Duration
mu_acq = log(cfg.assumptions.acquisition_mean_sec^2 / ...
    sqrt(cfg.assumptions.acquisition_std_sec^2 + cfg.assumptions.acquisition_mean_sec^2));
sigma_acq = sqrt(log(1 + (cfg.assumptions.acquisition_std_sec^2 / cfg.assumptions.acquisition_mean_sec^2)));
acq_durations = lognrnd(mu_acq, sigma_acq, total_generated_arrivals, 1);

% 3. Deterministic E013 Quality Gate & Recapture Logic
initial_iqa_fails = rand(total_generated_arrivals, 1) < cfg.assumptions.iqa_initial_fail_prob;

recaptures_performed = 0;
persistent_iqa_failures = 0;
iqa_passed_mask = true(total_generated_arrivals, 1);
recapture_delay_applied = zeros(total_generated_arrivals, 1);

for i = 1:total_generated_arrivals
    if initial_iqa_fails(i)
        recaptures_performed = recaptures_performed + 1;
        recapture_delay_applied(i) = cfg.assumptions.recapture_delay_sec;
        
        if rand() < cfg.assumptions.iqa_recapture_fail_prob
            persistent_iqa_failures = persistent_iqa_failures + 1;
            iqa_passed_mask(i) = false;
        end
    end
end

ai_processed_count = sum(iqa_passed_mask);

% 4. Network Transmission Latency
tx_latency_sec = cfg.assumptions.image_payload_bits / cfg.assumptions.bandwidth_nominal_bps;

% 5. AI Processing Execution Latency
% E007 forward-pass latency (~10.2 ms) used for AI compute to avoid double-counting.
ai_latency_sec = zeros(total_generated_arrivals, 1);
ai_latency_sec(iqa_passed_mask) = cfg.assumptions.ai_compute_latency_sec;

% 6. Clinical Referral Branching
is_referable = false(total_generated_arrivals, 1);
routes_to_specialist = false(total_generated_arrivals, 1);

for i = 1:total_generated_arrivals
    if iqa_passed_mask(i)
        if rand() < cfg.assumptions.prevalence_referable_dr
            is_referable(i) = true;
            routes_to_specialist(i) = true;
        end
    else
        % Scenario assumption: persistent ungradable cases transmitted for remote clinical adjudication
        if cfg.assumptions.route_persistent_iqa_to_specialist
            routes_to_specialist(i) = true;
        end
    end
end

% 7. Centralized Specialist Review Queue (c = 5 Tele-Ophthalmologists)
c_specialists = cfg.assumptions.baseline_specialist_count;
specialist_ready_times = zeros(c_specialists, 1);

specialist_indices = find(routes_to_specialist);
num_specialist_cases = length(specialist_indices);

case_ready_timestamps = arrival_timestamps(specialist_indices) + ...
                        acq_durations(specialist_indices) + ...
                        recapture_delay_applied(specialist_indices) + ...
                        tx_latency_sec + ...
                        ai_latency_sec(specialist_indices);

[queue_arrival_times, sort_order] = sort(case_ready_timestamps);

queue_wait_durations = zeros(num_specialist_cases, 1);
review_service_durations = zeros(num_specialist_cases, 1);

mu_rev = log(cfg.assumptions.specialist_review_mean_sec^2 / ...
    sqrt(cfg.assumptions.specialist_review_std_sec^2 + cfg.assumptions.specialist_review_mean_sec^2));
sigma_rev = sqrt(log(1 + (cfg.assumptions.specialist_review_std_sec^2 / cfg.assumptions.specialist_review_mean_sec^2)));

for j = 1:num_specialist_cases
    t_ready = queue_arrival_times(j);
    [earliest_free_time, server_idx] = min(specialist_ready_times);
    
    start_time = max(t_ready, earliest_free_time);
    wait_time = start_time - t_ready;
    
    service_duration = lognrnd(mu_rev, sigma_rev);
    finish_time = start_time + service_duration;
    
    specialist_ready_times(server_idx) = finish_time;
    queue_wait_durations(j) = wait_time;
    review_service_durations(j) = service_duration;
end

results = struct();
results.target_annual_encounters = cfg.assumptions.target_annual_encounters;
results.total_generated_arrivals = total_generated_arrivals;
results.initial_iqa_failures = sum(initial_iqa_fails);
results.recaptures_performed = recaptures_performed;
results.persistent_iqa_failures = persistent_iqa_failures;
results.ai_processed_cases = ai_processed_count;
results.referable_cases_detected = sum(is_referable);
results.specialist_reviews_routed = num_specialist_cases;
results.completed_reviews = num_specialist_cases;

results.mean_specialist_wait_min = mean(queue_wait_durations) / 60;
results.p95_specialist_wait_min = prctile(queue_wait_durations, 95) / 60;
results.max_specialist_wait_min = max(queue_wait_durations) / 60;

total_specialist_capacity_sec = c_specialists * total_horizon_sec;
results.specialist_utilization_pct = ...
    (sum(review_service_durations) / total_specialist_capacity_sec) * 100;
results.daily_throughput = total_generated_arrivals / cfg.assumptions.annual_operating_days;
end