"""
toto_db.py — DuckDB schema manager for TOTO predictions & results
Path on EC2: ~/statlot-649/statlot/toto/toto_db.py
"""
import duckdb
import os

DB_PATH = os.path.expanduser("~/statlot-649/statlot_toto.duckdb")

def get_conn():
    return duckdb.connect(DB_PATH)

def init_schema():
    con = get_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS toto_draws (
            draw_number     INTEGER PRIMARY KEY,
            draw_date       DATE NOT NULL,
            day_of_week     VARCHAR,
            n1 INTEGER, n2 INTEGER, n3 INTEGER,
            n4 INTEGER, n5 INTEGER, n6 INTEGER,
            additional      INTEGER,
            inserted_at     TIMESTAMP DEFAULT now()
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS toto_predictions (
            id              VARCHAR PRIMARY KEY,   -- draw_number + '_' + generated_ts
            draw_number     INTEGER,               -- the draw this predicts (NULL if unknown yet)
            draw_date       DATE,
            generated_at    TIMESTAMP DEFAULT now(),

            -- Mandatory tickets (JSON arrays)
            sys6_t1         JSON,   -- [n1,n2,n3,n4,n5,n6]
            sys6_t2         JSON,
            sys6_t3         JSON,
            sys7_t1         JSON,   -- [n1..n7]

            -- Optional system entries (NULL if not generated)
            sys8_t1         JSON,
            sys9_t1         JSON,
            sys10_t1        JSON,
            sys11_t1        JSON,
            sys12_t1        JSON,

            -- Bonus ticket
            bonus_t6        JSON,

            -- Additional number prediction
            additional_picks JSON,  -- [n1,n2,n3,n4,n5]

            -- Costs (SGD)
            cost_mandatory  DECIMAL(8,2),  -- 3*$1 + $7 = $10
            cost_with_sys8  DECIMAL(8,2),  -- + $28
            cost_with_sys12 DECIMAL(8,2),  -- + $924

            -- Model metadata
            engine_weights  JSON,
            draws_trained   INTEGER,
            backtest_3plus  VARCHAR,
            backtest_lift   VARCHAR,
            notes           VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS toto_results (
            id              VARCHAR PRIMARY KEY,   -- prediction_id + '_result'
            prediction_id   VARCHAR,
            draw_number     INTEGER,
            draw_date       DATE,
            checked_at      TIMESTAMP DEFAULT now(),

            -- Actual draw
            actual_n1 INTEGER, actual_n2 INTEGER, actual_n3 INTEGER,
            actual_n4 INTEGER, actual_n5 INTEGER, actual_n6 INTEGER,
            actual_additional INTEGER,

            -- Prize check per ticket (JSON: {group, matches, prize_sgd, combinations_checked})
            sys6_t1_result  JSON,
            sys6_t2_result  JSON,
            sys6_t3_result  JSON,
            sys7_t1_result  JSON,
            sys8_t1_result  JSON,   -- NULL if not bought
            sys9_t1_result  JSON,
            sys10_t1_result JSON,
            sys11_t1_result JSON,
            sys12_t1_result JSON,
            bonus_t6_result JSON,

            -- Summary
            best_group      INTEGER,   -- lowest group number won (1=best)
            total_prize_mandatory  DECIMAL(10,2),  -- prize from 3 Sys6 + 1 Sys7
            total_prize_full       DECIMAL(10,2),  -- if all sys entries bought
            total_cost_mandatory   DECIMAL(8,2),
            any_win                BOOLEAN DEFAULT FALSE,
            notes                  VARCHAR
        )
    """)

    # Weekly summary view
    con.execute("""
        CREATE VIEW IF NOT EXISTS v_weekly_summary AS
        SELECT
            p.draw_number,
            p.draw_date,
            p.generated_at,
            r.actual_n1, r.actual_n2, r.actual_n3,
            r.actual_n4, r.actual_n5, r.actual_n6,
            r.actual_additional,
            r.best_group,
            r.total_prize_mandatory,
            r.total_prize_full,
            r.any_win,
            p.sys6_t1, p.sys6_t2, p.sys6_t3, p.sys7_t1,
            p.sys8_t1
        FROM toto_predictions p
        LEFT JOIN toto_results r ON r.prediction_id = p.id
        ORDER BY p.draw_date DESC
    """)

    con.close()
    print(f"Schema initialised at {DB_PATH}")

if __name__ == "__main__":
    init_schema()
