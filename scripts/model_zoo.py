"""
The model zoo: ~50 ML classifiers spanning common production staples to
rare/experimental techniques pulled from smaller GitHub/PyPI packages,
trained on the identical preprocessed split every other model in a given
dataset sees.

Each spec is a dict:
  key            short filename-safe id
  name           display name for the dashboard
  family         coarse category used for CVD-safe chart coloring (there are
                 too many models to give each its own hue, so the dashboard
                 colors by family instead)
  description    plain-language one-liner for the "model guide" card
  build          fn(n_classes, n_features, seed) -> unfitted sklearn-style estimator
  subsample_cap  optional int; if the training set exceeds this, a fixed-seed
                 random subsample is used for models that don't scale
                 (kernel SVMs, Gaussian Processes, graph-based label spreading,
                 anything that internally resamples per base estimator)
"""
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.semi_supervised import LabelSpreading
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

MODERATE_CAP = 40000
HEAVY_CAP = 3000
# libsvm's OvO solver, especially with class_weight="balanced" on imbalanced
# data, can converge extremely slowly (minutes, not seconds) well before
# MODERATE_CAP -- kernel SVC variants get their own tighter cap plus a hard
# iteration ceiling so a single slow-converging model can't stall the run.
SVM_CAP = 12000
# label spreading builds a full n x n similarity graph in memory -- at
# MODERATE_CAP (40k) that's a 40000x40000 float matrix (~12.8GB), so it needs
# its own much smaller cap.
GRAPH_CAP = 5000

BALANCED = dict(class_weight="balanced")


def _n_hidden_forest(seed):
    return dict(random_state=seed)


def build_registry(seed=42):
    specs = []

    def add(key, name, family, description, build, subsample_cap=None):
        specs.append(dict(key=key, name=name, family=family,
                           description=description, build=build,
                           subsample_cap=subsample_cap))

    # ---------------- Linear models ----------------
    add("logreg", "Logistic Regression", "linear",
        "Baseline linear classifier: a softmax over a weighted sum of features.",
        lambda nc, nf, s: LogisticRegression(max_iter=1000, **BALANCED, random_state=s))

    add("logreg_elasticnet", "Logistic Regression (ElasticNet)", "linear",
        "Logistic regression with combined L1+L2 regularization for sparser, more robust weights.",
        lambda nc, nf, s: LogisticRegression(max_iter=2000, penalty="elasticnet",
                                              solver="saga", l1_ratio=0.5, **BALANCED, random_state=s))

    add("ridge", "Ridge Classifier", "linear",
        "Least-squares classification with L2-regularized weights (no probabilistic loss).",
        lambda nc, nf, s: RidgeClassifier(**BALANCED, random_state=s))

    add("sgd_log", "SGD (Log Loss)", "linear",
        "Logistic regression trained via online stochastic gradient descent instead of a batch solver.",
        lambda nc, nf, s: SGDClassifier(loss="log_loss", **BALANCED, random_state=s))

    add("sgd_hinge", "SGD (Hinge / Linear SVM)", "linear",
        "Linear support vector machine trained via online stochastic gradient descent.",
        lambda nc, nf, s: SGDClassifier(loss="hinge", **BALANCED, random_state=s))

    add("sgd_modified_huber", "SGD (Modified Huber)", "linear",
        "Linear classifier with a loss that's robust to outliers while still yielding probability estimates.",
        lambda nc, nf, s: SGDClassifier(loss="modified_huber", **BALANCED, random_state=s))

    add("perceptron", "Perceptron", "linear",
        "The original 1958 linear classifier: mistake-driven weight updates, no regularization by default.",
        lambda nc, nf, s: Perceptron(**BALANCED, random_state=s))

    add("passive_aggressive", "Passive-Aggressive Classifier", "linear",
        "Online linear classifier that stays passive on correct predictions and aggressively corrects on mistakes.",
        lambda nc, nf, s: PassiveAggressiveClassifier(**BALANCED, random_state=s))

    add("lda", "Linear Discriminant Analysis", "linear",
        "Generative linear classifier assuming shared-covariance Gaussian class distributions.",
        lambda nc, nf, s: LinearDiscriminantAnalysis())

    # ---------------- Naive Bayes / generative ----------------
    add("gnb", "Gaussian Naive Bayes", "naive_bayes",
        "Assumes each feature is independently Gaussian within a class -- a fast, strong baseline floor.",
        lambda nc, nf, s: GaussianNB())

    add("bnb", "Bernoulli Naive Bayes", "naive_bayes",
        "Naive Bayes over binarized (thresholded) features -- tests whether feature presence/absence alone is informative.",
        lambda nc, nf, s: BernoulliNB())

    add("cnb", "Complement Naive Bayes", "naive_bayes",
        "Naive Bayes variant designed for imbalanced classes; complements weights to correct majority-class skew.",
        lambda nc, nf, s: make_pipeline(MinMaxScaler(), ComplementNB()))

    add("qda", "Quadratic Discriminant Analysis", "naive_bayes",
        "Like LDA but allows each class its own covariance shape -- a quadratic, not linear, decision boundary.",
        lambda nc, nf, s: QuadraticDiscriminantAnalysis(reg_param=0.1))

    # ---------------- Single trees ----------------
    add("decision_tree", "Decision Tree", "tree",
        "A single greedy-split tree -- the interpretable building block every ensemble below is made of.",
        lambda nc, nf, s: DecisionTreeClassifier(**BALANCED, random_state=s))

    add("extra_tree", "Extra Tree (single)", "tree",
        "A single extremely-randomized tree: split thresholds are random rather than optimized, trading bias for variance.",
        lambda nc, nf, s: ExtraTreeClassifier(**BALANCED, random_state=s))

    # ---------------- Bagging ensembles ----------------
    add("random_forest", "Random Forest", "ensemble_bagging",
        "200 bagged decision trees voting together -- the workhorse ensemble baseline.",
        lambda nc, nf, s: RandomForestClassifier(n_estimators=200, **BALANCED, random_state=s, n_jobs=-1))

    add("extra_trees", "Extra Trees", "ensemble_bagging",
        "200 extremely-randomized trees bagged together -- usually faster to train than Random Forest, similar accuracy.",
        lambda nc, nf, s: ExtraTreesClassifier(n_estimators=200, **BALANCED, random_state=s, n_jobs=-1))

    add("bagging_dt", "Bagging (Decision Trees)", "ensemble_bagging",
        "Classic bootstrap-aggregated decision trees (the general recipe Random Forest specializes).",
        lambda nc, nf, s: BaggingClassifier(estimator=DecisionTreeClassifier(**BALANCED), n_estimators=50, random_state=s, n_jobs=-1),
        subsample_cap=MODERATE_CAP)

    add("bagging_knn", "Bagging (KNN)", "ensemble_bagging",
        "Bootstrap-aggregated k-nearest-neighbors -- tests whether bagging helps an already-low-bias model.",
        lambda nc, nf, s: BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=15, random_state=s, n_jobs=-1),
        subsample_cap=MODERATE_CAP)

    # ---------------- Boosting ----------------
    add("adaboost", "AdaBoost", "boosting",
        "Sequentially reweights misclassified samples across shallow trees -- the original boosting algorithm.",
        lambda nc, nf, s: AdaBoostClassifier(n_estimators=200, random_state=s))

    add("gbm_sklearn", "Gradient Boosting (sklearn)", "boosting",
        "Classic stagewise gradient-boosted trees, scikit-learn's original (pre-histogram) implementation.",
        lambda nc, nf, s: GradientBoostingClassifier(n_estimators=200, random_state=s))

    add("histgbm", "Hist Gradient Boosting", "boosting",
        "Histogram-binned gradient boosting (sklearn's LightGBM-style implementation) -- fast on large tabular data.",
        lambda nc, nf, s: HistGradientBoostingClassifier(max_iter=200, random_state=s))

    add("xgboost", "XGBoost", "boosting",
        "Industry-standard regularized gradient boosting library, widely used in IDS/network-security literature.",
        lambda nc, nf, s: _xgb(nc, s))

    add("xgboost_dart", "XGBoost (DART)", "boosting",
        "XGBoost with DART dropout regularization -- randomly drops trees during boosting to reduce overfitting.",
        lambda nc, nf, s: _xgb(nc, s, booster="dart"))

    add("lightgbm", "LightGBM", "boosting",
        "Microsoft's leaf-wise (rather than level-wise) gradient boosting framework, built for speed at scale.",
        lambda nc, nf, s: _lgbm(s))

    add("lightgbm_goss", "LightGBM (GOSS)", "boosting",
        "LightGBM with Gradient-based One-Side Sampling -- keeps high-gradient samples, subsamples the rest for speed.",
        lambda nc, nf, s: _lgbm(s, boosting_type="goss"))

    add("catboost", "CatBoost", "boosting",
        "Yandex's ordered-boosting gradient boosting library, designed to reduce prediction shift/target leakage.",
        lambda nc, nf, s: _catboost(s))

    add("ngboost", "NGBoost", "boosting",
        "Natural Gradient Boosting: boosts full predictive probability distributions, not just point predictions.",
        lambda nc, nf, s: _ngboost(nc, s))

    add("rgf", "Regularized Greedy Forest", "boosting",
        "Builds one shared tree structure that's fully re-optimized at every boosting step, rather than a growing ensemble.",
        lambda nc, nf, s: _rgf(s),
        subsample_cap=MODERATE_CAP)

    # ---------------- Kernel / margin-based ----------------
    add("svc_linear", "SVM (Linear Kernel)", "kernel",
        "Support vector machine with a linear decision boundary and margin-maximizing objective.",
        lambda nc, nf, s: SVC(kernel="linear", max_iter=20000, **BALANCED, random_state=s),
        subsample_cap=SVM_CAP)

    add("svc_rbf", "SVM (RBF Kernel)", "kernel",
        "Support vector machine with a Gaussian (RBF) kernel for nonlinear decision boundaries.",
        lambda nc, nf, s: SVC(kernel="rbf", max_iter=20000, **BALANCED, random_state=s),
        subsample_cap=SVM_CAP)

    add("svc_poly", "SVM (Polynomial Kernel)", "kernel",
        "Support vector machine with a degree-3 polynomial kernel.",
        lambda nc, nf, s: SVC(kernel="poly", degree=3, max_iter=20000, **BALANCED, random_state=s),
        subsample_cap=SVM_CAP)

    add("linear_svc", "Linear SVC (liblinear)", "kernel",
        "Linear support vector classifier via liblinear's coordinate-descent solver -- scales far better than kernel SVC.",
        lambda nc, nf, s: LinearSVC(dual=False, **BALANCED, random_state=s, max_iter=3000),
        subsample_cap=MODERATE_CAP)

    add("nystroem_sgd", "Nystroem + SGD", "kernel",
        "Approximates an RBF kernel with a low-rank Nystroem feature map, then a fast linear SGD classifier on top.",
        lambda nc, nf, s: make_pipeline(Nystroem(random_state=s, n_components=200),
                                         SGDClassifier(loss="log_loss", **BALANCED, random_state=s)))

    add("rbf_sampler_ridge", "RBF Sampler + Ridge", "kernel",
        "Random Fourier Features approximation of an RBF kernel feeding a closed-form ridge classifier.",
        lambda nc, nf, s: make_pipeline(RBFSampler(random_state=s, n_components=200),
                                         RidgeClassifier(**BALANCED, random_state=s)))

    # ---------------- Instance-based ----------------
    add("knn5", "K-Nearest Neighbors (k=5)", "instance",
        "Classifies by majority vote among the 5 nearest training points.",
        lambda nc, nf, s: KNeighborsClassifier(n_neighbors=5, n_jobs=-1))

    add("knn15_distance", "K-Nearest Neighbors (k=15, distance-weighted)", "instance",
        "15-neighbor vote weighted inversely by distance, so closer neighbors count more.",
        lambda nc, nf, s: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))

    add("nearest_centroid", "Nearest Centroid", "instance",
        "Assigns each point to the class whose training-set centroid (mean) is closest -- essentially 1-NN against class means.",
        lambda nc, nf, s: NearestCentroid())

    # ---------------- Semi-supervised / graph-based ----------------
    add("label_spreading", "Label Spreading", "graph",
        "Graph-based transductive classifier that propagates labels through a similarity graph over the data.",
        lambda nc, nf, s: LabelSpreading(kernel="knn", n_neighbors=7),
        subsample_cap=GRAPH_CAP)

    # ---------------- Gaussian Process ----------------
    add("gaussian_process", "Gaussian Process Classifier", "probabilistic",
        "Bayesian non-parametric classifier over an RBF-kernel Gaussian Process posterior (O(n^3) -- heavily subsampled).",
        lambda nc, nf, s: GaussianProcessClassifier(kernel=RBF(), random_state=s, n_jobs=-1),
        subsample_cap=HEAVY_CAP)

    # ---------------- Interpretable / rule-based ----------------
    add("ebm", "Explainable Boosting Machine", "interpretable",
        "Microsoft InterpretML's glass-box GAM-style booster -- near-blackbox accuracy with fully inspectable per-feature shape functions.",
        lambda nc, nf, s: _ebm(s),
        subsample_cap=MODERATE_CAP)

    add("figs", "FIGS (Fast Interpretable Greedy-Sums)", "interpretable",
        "Sums a handful of small, jointly-fit trees into one interpretable additive model instead of one deep tree.",
        lambda nc, nf, s: _figs())

    # ---------------- Exotic / uncommon architectures ----------------
    add("elm", "Extreme Learning Machine", "exotic",
        "Single hidden layer of fixed random (untrained) weights + closed-form ridge-regression readout -- no backprop.",
        lambda nc, nf, s: _elm(s))

    add("som_classifier", "Self-Organizing Map Classifier", "exotic",
        "Kohonen self-organizing map used as a nearest-prototype classifier via majority vote per map unit.",
        lambda nc, nf, s: _som(s),
        subsample_cap=MODERATE_CAP)

    # ---------------- Meta-ensembles ----------------
    add("voting_soft", "Voting Classifier (soft)", "meta",
        "Averages predicted-probability outputs of Random Forest, HistGBM, and Logistic Regression.",
        lambda nc, nf, s: VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=150, **BALANCED, random_state=s, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=150, random_state=s)),
            ("logreg", LogisticRegression(max_iter=1000, **BALANCED, random_state=s)),
        ], voting="soft", n_jobs=-1))

    add("stacking", "Stacking Classifier", "meta",
        "Random Forest, HistGBM, and KNN base predictions fed into a Logistic Regression meta-learner.",
        lambda nc, nf, s: StackingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=150, **BALANCED, random_state=s, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=150, random_state=s)),
            ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ], final_estimator=LogisticRegression(max_iter=1000, **BALANCED, random_state=s), n_jobs=-1),
        subsample_cap=MODERATE_CAP)

    # ---------------- Imbalance-aware ensembles (imbalanced-learn) ----------------
    add("balanced_random_forest", "Balanced Random Forest", "imbalanced",
        "Random Forest where every tree's bootstrap sample is rebalanced across classes -- built for exactly this kind of rare-attack-class skew.",
        lambda nc, nf, s: _imb_brf(s))

    add("rusboost", "RUSBoost", "imbalanced",
        "AdaBoost with random under-sampling of the majority class before each boosting round.",
        lambda nc, nf, s: _imb_rusboost(s),
        subsample_cap=MODERATE_CAP)

    add("easy_ensemble", "Easy Ensemble", "imbalanced",
        "Trains many AdaBoost ensembles, each on an independently balanced under-sample, then votes across all of them.",
        lambda nc, nf, s: _imb_easy_ensemble(s),
        subsample_cap=MODERATE_CAP)

    add("balanced_bagging", "Balanced Bagging", "imbalanced",
        "Bagged decision trees where every bootstrap sample is class-balanced before fitting.",
        lambda nc, nf, s: _imb_balanced_bagging(s),
        subsample_cap=MODERATE_CAP)

    return specs


# ---- factory helpers that need lazily-imported / conditionally-configured libs ----

def _xgb(n_classes, seed, **kw):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=seed,
        tree_method="hist", n_jobs=-1, **kw,
    )


def _lgbm(seed, **kw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1, n_jobs=-1, **kw)


def _catboost(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=200, random_state=seed, verbose=0, allow_writing_files=False)


def _ngboost(n_classes, seed):
    from ngboost import NGBClassifier
    from ngboost.distns import k_categorical
    return NGBClassifier(Dist=k_categorical(n_classes), n_estimators=150, verbose=False, random_state=seed)


def _rgf(seed):
    from rgf.sklearn import RGFClassifier
    return RGFClassifier(max_leaf=500, algorithm="RGF")


def _ebm(seed):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(interactions=0, random_state=seed, n_jobs=-1)


def _figs():
    from imodels import FIGSClassifier
    return FIGSClassifier(max_rules=30)


def _elm(seed):
    from custom_models import ExtremeLearningMachineClassifier
    return ExtremeLearningMachineClassifier(n_hidden=512, alpha=1.0, random_state=seed)


def _som(seed):
    from custom_models import SOMClassifier
    return SOMClassifier(grid_size=12, n_iter=3000, random_state=seed)


def _imb_brf(seed):
    from imblearn.ensemble import BalancedRandomForestClassifier
    return BalancedRandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                           sampling_strategy="all", replacement=True, bootstrap=False)


def _imb_rusboost(seed):
    from imblearn.ensemble import RUSBoostClassifier
    return RUSBoostClassifier(n_estimators=100, random_state=seed)


def _imb_easy_ensemble(seed):
    from imblearn.ensemble import EasyEnsembleClassifier
    return EasyEnsembleClassifier(n_estimators=15, random_state=seed, n_jobs=-1)


def _imb_balanced_bagging(seed):
    from imblearn.ensemble import BalancedBaggingClassifier
    return BalancedBaggingClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
