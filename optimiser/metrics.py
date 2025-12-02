from prometheus_client import Counter, Gauge, start_http_server

optimiser_runs_total = Counter("optimiser_runs_total", "Number of optimiser executions")
optimiser_last_fitness = Gauge("optimiser_last_fitness", "Best fitness from the most recent run")
optimiser_last_duration = Gauge("optimiser_last_duration_seconds", "Runtime of the most recent optimiser run")

def record_run(fitness, duration):
    optimiser_runs_total.inc()
    optimiser_last_fitness.set(fitness)
    optimiser_last_duration.set(duration)
