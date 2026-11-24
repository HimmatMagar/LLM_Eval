RAG Evaluation: Operational Metrics
This document summarizes practical considerations for measuring
latency, cost, and reliability in a RAG application.
1. Latency Measurement
1. Prefer latency distributions over averages. Measure P50, P95,
   and P99 latency, not just mean latency.
2. Measure component-level latency. Do not measure only end-to-end
   latency.
3. Measure TTFT separately. Track time to first token independently
   from total response time.
4. Watch for cold starts. Identify delays caused by initialization
   or inactive services.
5. Record context size and token counts. Track these alongside
   latency.
6. Distinguish latency from throughput. Latency measures the time
   taken by a request; throughput measures how many requests can be
   processed over a period.
7. Repeat runs. External APIs can be noisy, so repeated
   measurements provide a more reliable picture.
8. Track failures separately from latency. Do not treat failed
   requests as ordinary slow requests.
9. Set latency budgets. Define budgets at both the system and
   component levels.
10. Use representative and segmented workloads. Test different
    request types and workload patterns.
2. Cost Measurement
1. Measure cost per query.
2. Measure input and output costs separately.
3. Measure cost as a distribution. Examine how costs vary across
   requests rather than relying only on an average.
4. Segment cost by query type.
5. Set a cost budget. Define an acceptable cost per query and/or
   for the overall system.
3. Reliability Measurement
1. Measure the overall success and failure rates.
2. Categorize failures. Avoid using only one generic error rate.
3. Measure reliability under load separately. A pipeline may be
   highly reliable in a single-user offline test but start failing when
   concurrency rises.
4. Use enough samples. Reliability failures can be rare. Running 20
   requests and getting 20 successes does not prove the system is 100%
   reliable.
5. Use representative requests. Include different kinds of queries.

Practical Evaluation Checklist
Use this checklist when evaluating your RAG application:
- [ ] Measure P50, P95, and P99 latency.
- [ ] Measure latency for individual components and the end-to-end
  pipeline.
- [ ] Track time to first token (TTFT).
- [ ] Identify cold-start delays.
- [ ] Record context size and input/output token counts.
- [ ] Measure throughput separately from latency.
- [ ] Repeat tests to account for external API variability.
- [ ] Record and categorize failures.
- [ ] Define system-level and component-level latency budgets.
- [ ] Measure cost per query, including input and output costs.
- [ ] Compare cost across query types and set a cost budget.
- [ ] Test reliability with sufficient samples and under different
  concurrency levels.
- [ ] Include representative query types in the evaluation dataset.