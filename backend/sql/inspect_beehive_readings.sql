-- Run in pgAdmin/Supabase SQL editor before connecting the application.

SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'beehive_readings'
ORDER BY ordinal_position;

SELECT
    device_id,
    COUNT(*) AS reading_count,
    MIN(recorded_at) AS first_reading,
    MAX(recorded_at) AS latest_reading,
    EXTRACT(EPOCH FROM (NOW() - MAX(recorded_at))) / 60.0 AS minutes_since_latest
FROM public.beehive_readings
GROUP BY device_id
ORDER BY device_id;

SELECT
    device_id,
    DATE_TRUNC('hour', recorded_at AT TIME ZONE 'Asia/Colombo') AS sri_lanka_hour,
    COUNT(*) AS readings_in_hour,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_weight) AS median_weight,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY internal_temp) AS median_internal_temp,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY internal_humidity) AS median_internal_humidity,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY internal_co2) AS median_internal_co2
FROM public.beehive_readings
WHERE recorded_at >= NOW() - INTERVAL '14 days'
GROUP BY device_id, DATE_TRUNC('hour', recorded_at AT TIME ZONE 'Asia/Colombo')
ORDER BY device_id, sri_lanka_hour DESC;
