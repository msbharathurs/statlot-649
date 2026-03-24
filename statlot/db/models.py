"""SQLAlchemy ORM models for StatLot 649."""
from sqlalchemy import Column, Integer, Float, String, Text
from db.database import Base


class DrawRecord(Base):
    __tablename__ = "draws"
    id = Column(Integer, primary_key=True, autoincrement=True)
    draw_number = Column(Integer, unique=True, nullable=False, index=True)
    draw_date   = Column(String(20), nullable=True)
    n1 = Column(Integer); n2 = Column(Integer); n3 = Column(Integer)
    n4 = Column(Integer); n5 = Column(Integer); n6 = Column(Integer)
    additional  = Column(Integer, nullable=True)

    def to_dict(self):
        return {
            "draw_number": self.draw_number,
            "draw_date": self.draw_date,
            "n1": self.n1, "n2": self.n2, "n3": self.n3,
            "n4": self.n4, "n5": self.n5, "n6": self.n6,
            "additional": self.additional,
            "nums": sorted([self.n1,self.n2,self.n3,self.n4,self.n5,self.n6]),
        }


class FeatureRecord(Base):
    __tablename__ = "features"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    draw_number  = Column(Integer, unique=True, nullable=False, index=True)
    features_json = Column(Text, nullable=False)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_date       = Column(String(30))
    model_type     = Column(String(20))
    pool_size      = Column(Integer)
    total_tested   = Column(Integer)
    avg_match      = Column(Float)
    rand_avg       = Column(Float)
    lift_pct       = Column(Float)
    three_plus_rate = Column(Float)
    four_plus_rate  = Column(Float)
    five_plus_count = Column(Integer)
    five_plus_rate  = Column(Float)
    result_json    = Column(Text)

    def to_summary(self):
        return {
            "id": self.id, "run_date": self.run_date,
            "model_type": self.model_type, "pool_size": self.pool_size,
            "total_tested": self.total_tested,
            "avg_match": self.avg_match, "rand_avg": self.rand_avg,
            "lift_pct": self.lift_pct,
            "three_plus_rate": self.three_plus_rate,
            "four_plus_rate": self.four_plus_rate,
            "five_plus_count": self.five_plus_count,
            "five_plus_rate": self.five_plus_rate,
        }


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    generated_date = Column(String(30))
    pool_size      = Column(Integer)
    model_type     = Column(String(20))
    draw_count     = Column(Integer)
    combos_json    = Column(Text)
    verified       = Column(Integer, default=0)
    actual_draw    = Column(String(50), nullable=True)
    best_match     = Column(Integer, nullable=True)

    def to_summary(self):
        return {
            "id": self.id, "generated_date": self.generated_date,
            "pool_size": self.pool_size, "model_type": self.model_type,
            "draw_count": self.draw_count, "verified": self.verified,
            "actual_draw": self.actual_draw, "best_match": self.best_match,
        }
