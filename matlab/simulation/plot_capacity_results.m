function plot_capacity_results(baseline_res, stress_res)
% PLOT_CAPACITY_RESULTS Generates systems capacity charts.
%
% Simulation assumptions are scenario parameters and must not be interpreted
% as measured clinical prevalence, clinical workflow observations, or deployment guarantees.

cfg = scenario_config();

if nargin < 2
    baseline_res = run_baseline_analysis(cfg);
    stress_res = run_bandwidth_analysis(cfg);
end

figure('Name', 'NetraRakshakAI Regional Queueing & Capacity Analysis', ...
       'Position', [100, 100, 1100, 750], 'Color', 'w');

% Subplot 1: Uplink Transmission Latency vs Bandwidth
subplot(2, 2, 1);
semilogx(stress_res.bandwidths_kbps, stress_res.uplink_delay_sec, 'b-o', 'LineWidth', 2);
grid on;
xlabel('Telemetry Uplink Bandwidth (kbps)');
ylabel('Payload Upload Delay (seconds)');
title('A: Edge-to-District Telemetry Latency');
xlim([40, 2500]);

% Subplot 2: Post-Outage Store-and-Forward Buffer Drain Time
subplot(2, 2, 2);
valid_drain = ~isinf(stress_res.drain_time_minutes);
plot(stress_res.bandwidths_kbps(valid_drain), stress_res.drain_time_minutes(valid_drain), 'r-s', 'LineWidth', 2);
grid on;
xlabel('Restored Uplink Bandwidth (kbps)');
ylabel('Buffer Drain Duration (minutes)');
title('B: Edge Buffer Clearance (4-hr Outage)');

% Subplot 3: Specialist Staffing Sensitivity (M/G/c Projection)
% Review duration derived strictly from cfg.assumptions.specialist_review_mean_sec
subplot(2, 2, 3);
staff_vector = 1:10;
total_service_sec = baseline_res.specialist_reviews_routed * cfg.assumptions.specialist_review_mean_sec;
total_annual_capacity_per_staff = cfg.assumptions.total_annual_operating_seconds;
util_curve = min(100, (total_service_sec ./ (staff_vector .* total_annual_capacity_per_staff)) * 100);

bar(staff_vector, util_curve, 'FaceColor', [0.2 0.5 0.6]);
grid on;
xlabel('Specialist Staffing Level (c)');
ylabel('Server Utilization (%)');
title('C: Central Review Room Staff Utilization');

% Subplot 4: Clinical Triage Encounter Breakdown
subplot(2, 2, 4);
triage_data = [
    baseline_res.ai_processed_cases - baseline_res.referable_cases_detected;
    baseline_res.referable_cases_detected;
    baseline_res.persistent_iqa_failures
];
triage_labels = {'Routine Non-Referable', 'Referable Detected', 'Persistent IQA Hold'};
pie(triage_data, triage_labels);
title('D: Regional Encounter Triage Split');
end