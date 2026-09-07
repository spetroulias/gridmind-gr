CREATE TABLE IF NOT EXISTS system_load (
    date DATE NOT NULL,
    period SMALLINT NOT NULL,

    net_load_mwh DOUBLE PRECISION NOT NULL,
    crete_flow_mwh DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (date, period),

    CONSTRAINT valid_system_load_period
        CHECK (period BETWEEN 1 AND 25)
);


CREATE TABLE IF NOT EXISTS res_production (
    date DATE NOT NULL,
    period SMALLINT NOT NULL,

    res_mwh DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (date, period),

    CONSTRAINT valid_res_period
        CHECK (period BETWEEN 1 AND 25)
);


CREATE TABLE IF NOT EXISTS generation_actual (
    date DATE NOT NULL,
    period SMALLINT NOT NULL,
    unit_name TEXT NOT NULL,
    technology TEXT NOT NULL,
    production_mwh DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        date,
        period,
        unit_name
    ),

    CONSTRAINT valid_generation_period
        CHECK (period BETWEEN 1 AND 25),

    CONSTRAINT valid_generation_technology
        CHECK (
            technology IN (
                'lignite',
                'petroleum',
                'natural_gas',
                'hydro',
                'res'
            )
        )
);


CREATE TABLE IF NOT EXISTS load_forecasts (
    timestamp TIMESTAMP PRIMARY KEY,
    forecast_mwh DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS res_forecasts (
    timestamp TIMESTAMP PRIMARY KEY,
    forecast_mwh DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);