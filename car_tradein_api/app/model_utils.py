import pandas as pd
from fastai.tabular.all import load_learner
from datetime import datetime
from rapidfuzz import process, fuzz

class CarPriceModel:
    """Wraps FastAI learner + all fuzzy-matching / cleaning logic."""
    COND_BUCKETS      = ["Excellent","good","average","belowAverage","rough","great"]
    LINE_BUCKETS      = ["Mid","Economy","High","Exotic"]
    DRIVETRAIN_BUCKETS= ["FWD","AWD","RWD"]
    TRANS_BUCKETS     = ["Automatic","CVT","Manual","DualClutch","Other_Trans"]

    def __init__(self, model_path: str, data_path: str):
        self.learn = load_learner(model_path)
        df = pd.read_csv(data_path)

        self.valid_makes = df.make.unique().tolist()
        self.models_by_make = {
            m: df.loc[df.make == m, 'model'].unique().tolist()
            for m in self.valid_makes
        }
        self.trims_by_m_m = {
            (m, md): df.loc[(df.make == m) & (df.model == md), 'trim'].unique().tolist()
            for m in self.models_by_make for md in self.models_by_make[m]
        }

    # ---------- helpers ----------
    @staticmethod
    def _fuzzy(val, choices, thresh=0.7, default=None):
        # Exact match fast-path
        if val in choices:
            return val

        # RapidFuzz returns (match, score, idx)
        best = process.extractOne(val, choices, scorer=fuzz.token_sort_ratio)
        if best:
            best_match, best_score, _ = best
            if best_score / 100 >= thresh:
                return best_match

        return default


    # ---------- public API ----------
    def preprocess(self, raw: dict):
        """Return a single-row DataFrame ready for FastAI or raise a str error."""
        age = datetime.now().year - raw["year"]

        make  = self._fuzzy(raw["make"],  self.valid_makes, 0.7)
        if make is None:
            raise ValueError(f"Unknown make '{raw['make']}'")

        model = self._fuzzy(raw["model"], self.models_by_make[make], 0.6)
        if model is None:
            raise ValueError(f"Unknown model '{raw['model']}' for make '{make}'")

        trim  = self._fuzzy(raw["trim"],  self.trims_by_m_m.get((make, model), []),
                            0.5, "Other")

        data = {
            "age": age,
            "mileage": raw["mileage"],
            "make": make,
            "model": model,
            "trim": trim,
            "interior":  self._fuzzy(raw["interior"],  self.COND_BUCKETS, 0.7, "average"),
            "exterior":  self._fuzzy(raw["exterior"],  self.COND_BUCKETS, 0.7, "average"),
            "mechanical":self._fuzzy(raw["mechanical"],self.COND_BUCKETS, 0.7, "average"),
            "line":       self._fuzzy(raw["line"],       self.LINE_BUCKETS, 0.7, "Economy"),
            "drivetrain": self._fuzzy(raw["drivetrain"], self.DRIVETRAIN_BUCKETS,0.7,"AWD"),
            "transmission":self._fuzzy(raw["transmission"],self.TRANS_BUCKETS,0.7,"Automatic"),
        }
        return pd.DataFrame([data])

    def predict(self, payload: dict) -> float:
        df = self.preprocess(payload)
        dl = self.learn.dls.test_dl(df)
        pred, _ = self.learn.get_preds(dl=dl)
        return float(pred[0])
