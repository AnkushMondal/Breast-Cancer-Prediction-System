"""Model factory for required classifiers."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_models(random_state: int) -> dict[str, object]:
    """Instantiate all required machine learning models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=random_state,
            # FIXED: Removed multi_class="auto" as it's now deprecated/removed.
            # Modern scikit-learn chooses the best strategy automatically.
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            random_state=random_state,
        ),
    }
    return models